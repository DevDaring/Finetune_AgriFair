"""Greedy decoding, shared extraction, and batched generation.

Evaluation batching (EVAL_BATCH_SIZE) uses left-padding plus an attention mask and
greedy decoding so a given batch size is reproducible and equivalent to the single-
example path up to floating-point reduction order. Every prompt is rendered through the
tokenizer's chat template (prompts.render_chat) so instruction-tuned models see the
format they were trained on; training uses the same rendering. Mamba hybrids are
decoded at batch size 1.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

from GPU_Run.common import prompts as P
from GPU_Run.common.parsing import extract_answer_letter, extract_rationale, fallback_letter_from_freetext

_SEQUENTIAL_MODEL_TYPES = {"nemotron_h", "mamba", "mamba2", "jamba", "zamba", "zamba2", "falcon_mamba"}


def eval_batch_size(default: int = 16) -> int:
    try:
        return int(os.environ.get("EVAL_BATCH_SIZE", default))
    except ValueError:
        return default


def _effective_batch_size(model, requested: Optional[int]) -> int:
    """The requested batch size, lowered by whatever the model needs and never raised.

    A state-space or hybrid backbone is decoded one sequence at a time for correctness.
    Larger models carry a cap set in the registry so a fixed EVAL_BATCH_SIZE does not have
    to be tuned per model to stay inside one device."""
    bs = requested or eval_batch_size()
    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    if model_type in _SEQUENTIAL_MODEL_TYPES:
        return 1
    cap = int(getattr(model, "_agrifair_max_eval_batch", 0) or 0)
    if cap > 0:
        bs = min(bs, cap)
    return max(1, bs)


def generate_batch(
    model,
    tokenizer,
    prompt_texts: Sequence[str],
    max_new_tokens: int = 256,
    batch_size: Optional[int] = None,
    render: bool = True,
) -> List[str]:
    """Greedy, temperature-0 generation for any number of prompts, chunked by the
    evaluation batch size (left-padded). Prompts are chat-rendered unless render=False."""
    import torch

    bs = _effective_batch_size(model, batch_size)
    texts = [P.render_chat(tokenizer, t) if render else t for t in prompt_texts]
    add_special = P.prompt_add_special_tokens(tokenizer) if render else True
    outputs: List[str] = []
    for start in range(0, len(texts), bs):
        chunk = texts[start : start + bs]
        enc = tokenizer(
            list(chunk), return_tensors="pt", padding=True, truncation=True, max_length=1024,
            add_special_tokens=add_special,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outputs


def run_mcq_eval(
    model,
    tokenizer,
    records: Sequence[Dict],
    rationale_mode: bool = False,
    batch_size: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    judge=None,
    rotation: int = 0,
) -> List[Dict]:
    """Return per-item prediction dicts with canonical letters for metrics.

    Answer extraction is deterministic JSON parsing; only on parse failure does it fall
    back to the judge (if provided), then to a free-text heuristic. Parse-failure rate is
    tracked per call. `rotation` cyclically rotates the displayed option order for the
    option-rotation robustness check; the canonical letters are unaffected."""
    if max_new_tokens is None:
        max_new_tokens = 384 if rationale_mode else 64
    built = [P.build_mcq_prompt(r, rationale_mode=rationale_mode, rotation=rotation) for r in records]
    texts = generate_batch(model, tokenizer, [b["prompt"] for b in built], max_new_tokens, batch_size)
    out: List[Dict] = []
    parse_failures = 0
    judge_recoveries = 0
    for rec, b, raw in zip(records, built, texts):
        disp_letter, parse_ok = extract_answer_letter(raw)
        answer_source = "deterministic_json_parse"
        if not parse_ok:
            parse_failures += 1
            # The judge sees the model's own output and nothing else: no question, no options,
            # no gold answer. It can read a letter out of prose; it cannot supply one.
            if judge is not None and judge.available():
                disp_letter = judge.extract_answer_letter(raw)
                if disp_letter is not None:
                    judge_recoveries += 1
                    answer_source = f"judge:{judge.last_route}"
            if disp_letter is None:
                disp_letter = fallback_letter_from_freetext(raw)
                answer_source = "free_text_heuristic" if disp_letter else "unresolved"
        canonical = b["display_to_canonical"].get(disp_letter) if disp_letter else None
        rationale = extract_rationale(raw)["rationale"] if rationale_mode else ""
        out.append(
            {
                "id": rec["id"],
                # The rationale judge grades factual consistency and needs the question
                # itself; it was previously handed the opaque item id.
                "question": rec.get("question", ""),
                "condition": rec["condition"],
                "category": rec.get("category"),
                "form": rec.get("form"),
                "gold_canonical": rec["correct_answer"],
                "pred_canonical": canonical if canonical else "unparsed",
                "pred_display_letter": disp_letter or "",
                "rotation": rotation,
                "raw_output": raw,
                "generated_rationale": rationale,
                "law_reference": rec.get("law_reference", ""),
                "reference_rationale": rec.get("rationale", ""),
                "parse_ok": parse_ok,
                "answer_extraction_source": answer_source,
            }
        )
    rate = 100.0 * parse_failures / max(1, len(out))
    for r in out:
        r["json_parse_failure_rate_percent"] = rate
        r["answers_recovered_by_judge"] = judge_recoveries
    return out


def run_capability_eval(model, tokenizer, items: Sequence[Dict], batch_size: Optional[int] = None,
                        judge=None) -> List[Dict]:
    """External capability probe (four-option MMLU subset). Exact match on the parsed
    letter; an unparseable output is scored wrong and stays in the denominator."""
    built = [P.build_capability_prompt(it) for it in items]
    texts = generate_batch(model, tokenizer, [b["prompt"] for b in built], 48, batch_size)
    out = []
    for it, b, raw in zip(items, built, texts):
        letter, ok = extract_answer_letter(raw, letters=tuple(P.CAPABILITY_LETTERS))
        if not ok:
            if judge is not None and judge.available():
                letter = judge.extract_answer_letter(raw, letters=tuple(P.CAPABILITY_LETTERS))
            if letter is None:
                letter = fallback_letter_from_freetext(raw, letters=tuple(P.CAPABILITY_LETTERS))
        out.append({
            "id": it.get("id"), "subject": it.get("subject", ""),
            "gold_letter": b["gold_letter"], "pred_letter": letter or "unparsed",
            "correct": int(letter == b["gold_letter"]), "parse_ok": ok, "raw_output": raw,
        })
    return out


def generate_freetext(model, tokenizer, prompt_text: str, max_new_tokens: int = 256) -> str:
    return generate_batch(model, tokenizer, [prompt_text], max_new_tokens)[0]
