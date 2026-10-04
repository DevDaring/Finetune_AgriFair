"""E2: option-letter probabilities at the answer position, for every GPU system (Future_PLan.md).

    python -m Submission1_TMLR.logprobs --attention sdpa [--pilot 2] [--families ...]

For each prompt the chat-templated text is extended with the assistant prefix {"answer_choice_letter": "
and one forward pass gives the next-token distribution. The probabilities of the offered option
letters are renormalised over those letters. Also recorded: the raw probability mass that fell on
the letters, the overall top token, and the argmax letter scored like a generated answer.
Prompts: the verified and extended comparisons (standard and no-rule) and the numerical set.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

from Submission1_Code_Phase2 import common as C
from Submission1_TMLR import run_gpu as G

PREFIX = '{"answer_choice_letter": "'
EQUAL = "roughly equal"


def prompts_for_logprobs(pilot: int = 0) -> List[Dict]:
    main = [p for p in G.load_prompts("main") if p["experiment"] in ("verified", "extended", "numerical")]
    norule = [p for p in C.read_jsonl(C.CODES_ROOT / G.FOLLOWUPS) if p["experiment"] == "norule"]
    out = main + norule
    if pilot:
        out = [p for exp in sorted({q["experiment"] for q in out})
               for p in sorted((q for q in out if q["experiment"] == exp), key=lambda q: q["prompt_id"])[:pilot]]
    return out


def letter_ids(tok) -> Dict[str, int]:
    """Token id of each letter when it follows the JSON prefix (tokenisation is local to the boundary)."""
    base = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True) + PREFIX
    ids_base = tok(base, add_special_tokens=False)["input_ids"]
    out = {}
    for letter in "abcd":
        ids = tok(base + letter, add_special_tokens=False)["input_ids"]
        if ids[:len(ids_base)] == ids_base and len(ids) == len(ids_base) + 1:
            out[letter] = ids[-1]
        else:                                                             # fall back to the bare letter token
            out[letter] = tok(letter, add_special_tokens=False)["input_ids"][-1]
    return out


def run(pilot: int, families: List[str], attention: str, stage: str, all_arms: bool) -> None:
    import torch
    out = C.CODES_ROOT / G.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"gpu_{stage}_predictions.jsonl"
    done = {(r["prompt_id"], r["system"]) for r in C.read_jsonl(path)} if path.exists() else set()
    prompts = prompts_for_logprobs(pilot)
    arms = G.ARMS + G.DRAW_ARMS if all_arms else G.ARMS
    if attention == "sdpa":
        from GPU_Run.common import model_registry as MR
        MR._select_attention = lambda preferred: "eager" if preferred == "eager" else "sdpa"
    backend = G.HubBackend({"budget": {"attention": attention}})
    started, written, ids_used = time.time(), 0, {}
    for tier in families:
        backend.load(tier)
        tok = backend.tok
        lid = letter_ids(tok); ids_used[tier] = lid
        for method, seed in arms:
            sid = G.system_id(tier, method, seed)
            todo = [p for p in prompts if (p["prompt_id"], sid) not in done]
            if not todo:
                continue
            backend.attach(method, seed)
            model = backend.peft if backend.peft is not None else backend.model
            todo.sort(key=lambda p: len(p["prompt"]))
            bs, i = G.BATCH[tier], 0
            while i < len(todo):
                batch = todo[i:i + bs]
                texts = [tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False,
                                                 add_generation_prompt=True) + PREFIX for p in batch]
                enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                try:
                    with torch.no_grad():
                        logits = model(**enc).logits[:, -1, :].float()
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and bs > 1:
                        torch.cuda.empty_cache(); bs = max(1, bs // 2); continue
                    raise
                probs = torch.softmax(logits, dim=-1)
                top_p, top_i = probs.max(dim=-1)
                rows = []
                for k, p in enumerate(batch):
                    letters = "abcd"[:len(p["choices"])]
                    raw = {l: float(probs[k, lid[l]]) for l in letters}
                    mass = sum(raw.values())
                    norm = {l: (raw[l] / mass if mass > 0 else 1 / len(letters)) for l in letters}
                    arg = max(norm, key=norm.get)
                    picked = p["choices"][letters.index(arg)]
                    srt = sorted(norm.values(), reverse=True)
                    rows.append({**{kk: v for kk, v in p.items() if kk != "prompt"}, "system": sid, "tier": tier,
                                 "method": method, "seed": seed, "prompt_sha256": C.freeze(p["prompt"]),
                                 **{f"p_{l}": round(norm[l], 6) for l in letters}, "letter_mass": round(mass, 6),
                                 "top_token": tok.decode([int(top_i[k])]), "top_token_prob": round(float(top_p[k]), 6),
                                 "argmax_letter": arg, "argmax_choice": picked,
                                 "argmax_correct": picked.strip().lower() == str(p["gold_choice_text"]).strip().lower(),
                                 "p_equal": round(next((norm[l] for l in letters if EQUAL in p["choices"][letters.index(l)].lower()), 0.0), 6),
                                 "margin_top2": round(srt[0] - srt[1], 6) if len(srt) > 1 else None})
                with path.open("a", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                written += len(batch); i += bs
            backend.detach()
            print(f"  {sid}: done, {written} rows so far, {(time.time() - started) / 60:.1f} min", flush=True)
    backend.free()
    C.write_json(out / f"gpu_{stage}_manifest.json",
                 {"stage": stage, "rows_written": written, "minutes": round((time.time() - started) / 60, 2),
                  "prompts_per_system": len(prompts), "prefix": PREFIX, "letter_token_ids": ids_used,
                  "systems": [G.system_id(t, m, s) for t in families for m, s in arms],
                  "attention": attention, "environment": G.environment()})
    print(f"[logprobs] {written} rows -> {path}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--families", nargs="*", default=G.FAMILIES)
    ap.add_argument("--attention", choices=["auto", "sdpa"], default="sdpa")
    ap.add_argument("--stage-name", default="logprobs")
    ap.add_argument("--all-arms", action="store_true", help="include the two further random draws")
    a = ap.parse_args(argv)
    run(a.pilot, a.families, a.attention, a.stage_name, a.all_arms)


if __name__ == "__main__":
    main()
