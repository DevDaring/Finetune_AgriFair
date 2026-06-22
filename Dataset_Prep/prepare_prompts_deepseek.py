"""DeepSeek prompt-engineering of the fixed instruction wrappers ONLY.

It never generates question, choice, answer, or rationale content (Instruction.md
Section 4.2; AgriFair rule: no label from a language model). With no DeepSeek key it is
a logged no-op. The canonical templates in GPU_Run/common/prompts.py remain authoritative;
this script only records a suggested wrapper edit for human review.

Run:  python Dataset_Prep/prepare_prompts_deepseek.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from GPU_Run.common import env_loader
from GPU_Run.common.clients import _RoundRobin, _call_openai_compatible
from GPU_Run.common.logging_utils import get_logger, log_run_metadata
from GPU_Run.common.paths import RESULTS_DIR
from GPU_Run.common.prompts import _MCQ_INSTRUCTION

logger = get_logger("prepare_prompts_deepseek")


def main():
    keys = _RoundRobin(env_loader.get_round_robin(["DEEPSEEK_KEY1", "DEEPSEEK_KEY2"]))
    out = RESULTS_DIR / "wrapper_suggestions.json"
    if not keys:
        logger.info("No DeepSeek key; no-op. Canonical wrappers remain authoritative.")
        out.write_text(json.dumps({"suggested_wrapper": _MCQ_INSTRUCTION, "source": "unchanged"}, indent=2), encoding="utf-8")
        return
    base = env_loader.get("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1")
    model = env_loader.get("DEEPSEEK_JUDGE_MODEL_NAME", "deepseek-chat")
    prompt = (
        "Improve ONLY the wording of this multiple-choice instruction wrapper for clarity and "
        "neutrality. Do not add any task content, examples, or answers. Reply with one JSON "
        "object only: {\"suggested_wrapper\": \"...\"}.\n\nWrapper:\n" + _MCQ_INSTRUCTION
    )
    try:
        text = _call_openai_compatible(base, keys.next(), model, prompt)
        from GPU_Run.common.parsing import json_repair_parse

        obj = json_repair_parse(text) or {}
        suggestion = obj.get("suggested_wrapper", _MCQ_INSTRUCTION)
        out.write_text(json.dumps({"suggested_wrapper": suggestion, "source": "deepseek_review_only"}, indent=2), encoding="utf-8")
        logger.info("Wrote wrapper suggestion for human review (canonical wrapper unchanged).")
    except Exception as e:
        logger.warning("DeepSeek call failed (%s); wrappers unchanged.", e)
    log_run_metadata("prepare_prompts_deepseek", {"keys_present": bool(keys)})


if __name__ == "__main__":
    main()
