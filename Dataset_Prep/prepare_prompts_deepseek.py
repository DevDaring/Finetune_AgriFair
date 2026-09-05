"""Light-edit the fixed instruction wrapper, and nothing else.

This script may only rephrase the standing instruction that wraps every multiple-choice
prompt. It never generates a question, a choice, an answer, or a rationale, because that
would violate the AgriFair rule that no label comes from a language model. Its output is a
suggestion written to results/wrapper_suggestions.json for human review; the canonical
wrapper in GPU_Run/common/prompts.py stays authoritative until a person changes it.

It uses the same ordered judge chain as everything else, DeepSeek key 1, DeepSeek key 2,
Mistral key 1, Mistral key 2, with no retries. With no keys at all it is a logged no-op.

Run:  python Dataset_Prep/prepare_prompts_deepseek.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from GPU_Run.common.clients import JudgeChain
from GPU_Run.common.logging_utils import get_logger, log_run_metadata
from GPU_Run.common.parsing import json_repair_parse
from GPU_Run.common.paths import RESULTS_DIR
from GPU_Run.common.prompts import _MCQ_INSTRUCTION

logger = get_logger("prepare_prompts_deepseek")

REVIEW_PROMPT = (
    "Improve ONLY the wording of this multiple-choice instruction wrapper, for clarity and "
    "neutrality. Do not add task content, examples, or answers, and do not change what it "
    "asks for. Reply with one JSON object only: {\"suggested_wrapper\": \"...\"}.\n\nWrapper:\n"
)


def main():
    out = RESULTS_DIR / "wrapper_suggestions.json"
    chain = JudgeChain()
    if not chain.available():
        logger.info("No judge keys available; no-op. The canonical wrapper remains authoritative.")
        out.write_text(json.dumps({"suggested_wrapper": _MCQ_INSTRUCTION, "source": "unchanged",
                                   "canonical_wrapper_applied": True}, indent=2), encoding="utf-8")
        log_run_metadata("prepare_prompts_deepseek", {"keys_present": False})
        return

    text = chain.complete(REVIEW_PROMPT + _MCQ_INSTRUCTION)
    suggestion = (json_repair_parse(text or "") or {}).get("suggested_wrapper", _MCQ_INSTRUCTION)
    out.write_text(json.dumps({
        "suggested_wrapper": suggestion,
        "canonical_wrapper": _MCQ_INSTRUCTION,
        "canonical_wrapper_applied": True,
        "source": f"review_only_via_{chain.last_route}" if text else "unchanged_call_failed",
    }, indent=2), encoding="utf-8")
    logger.info("Wrote a wrapper suggestion for human review via %s; the canonical wrapper is unchanged.",
                chain.last_route or "no route")
    log_run_metadata("prepare_prompts_deepseek", {"keys_present": True, "route": chain.last_route})


if __name__ == "__main__":
    main()
