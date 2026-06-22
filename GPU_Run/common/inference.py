"""Greedy decoding, shared extraction, and batched generation.

Evaluation batching (EVAL_BATCH_SIZE) uses left-padding plus an attention mask and
greedy decoding so a given batch size is reproducible and equivalent to the single-
example path up to floating-point reduction order (Instruction.md Section 14). Mamba
hybrids are clamped to batch size 1 by model_registry.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

from GPU_Run.common import prompts as P
from GPU_Run.common.parsing import extract_answer_letter, extract_rationale, fallback_letter_from_freetext


def eval_batch_size(default: int = 16) -> int:
    try:
        return int(os.environ.get("EVAL_BATCH_SIZE", default))
    except ValueError:
        return default


def generate_batch(model, tokenizer, prompt_texts: Sequence[str], max_new_tokens: int = 256) -> List[str]:
    """Greedy, temperature-0 generation for a batch of prompts (left-padded)."""
    import torch

    enc = tokenizer(list(prompt_texts), return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen = out[:, enc["input_ids"].shape[1]:]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def run_mcq_eval(
    model,
    tokenizer,
    records: Sequence[Dict],
    rationale_mode: bool = False,
    batch_size: Optional[int] = None,
    max_new_tokens: int = 256,
    judge=None,
) -> List[Dict]:
    """Return per-item prediction dicts with canonical letters for metrics.

    Answer extraction is deterministic JSON parsing; only on parse failure does it fall
    back to the judge (if provided), then to a free-text heuristic. Parse-failure rate is
    tracked per call (Instruction.md Section 9)."""
    bs = batch_size or eval_batch_size()
    built = [P.build_mcq_prompt(r, rationale_mode=rationale_mode) for r in records]
    out: List[Dict] = []
    parse_failures = 0
    for start in range(0, len(records), bs):
        chunk = list(range(start, min(start + bs, len(records))))
        texts = generate_batch(model, tokenizer, [built[i]["prompt"] for i in chunk], max_new_tokens)
        for j, i in enumerate(chunk):
            rec, b, raw = records[i], built[i], texts[j]
            disp_letter, parse_ok = extract_answer_letter(raw)
            if not parse_ok:
                parse_failures += 1
                if judge is not None and judge.available():
                    jtext = judge.complete(P.judge_answer_extraction_prompt(raw))
                    disp_letter, _ = extract_answer_letter(jtext or "")
                if disp_letter is None:
                    disp_letter = fallback_letter_from_freetext(raw)
            canonical = b["display_to_canonical"].get(disp_letter) if disp_letter else None
            rationale = extract_rationale(raw)["rationale"] if rationale_mode else ""
            out.append(
                {
                    "id": rec["id"],
                    "condition": rec["condition"],
                    "category": rec.get("category"),
                    "form": rec.get("form"),
                    "gold_canonical": rec["correct_answer"],
                    "pred_canonical": canonical if canonical else "unparsed",
                    "raw_output": raw,
                    "generated_rationale": rationale,
                    "law_reference": rec.get("law_reference", ""),
                    "reference_rationale": rec.get("rationale", ""),
                    "parse_ok": parse_ok,
                }
            )
    for r in out:
        r["json_parse_failure_rate_percent"] = 100.0 * parse_failures / max(1, len(out))
    return out


def generate_freetext(model, tokenizer, prompt_text: str, max_new_tokens: int = 256) -> str:
    return generate_batch(model, tokenizer, [prompt_text], max_new_tokens)[0]
