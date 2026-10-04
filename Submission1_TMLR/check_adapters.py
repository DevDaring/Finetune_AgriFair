"""Pre-run check: do adapters attach, change the model, and detach cleanly?

    python -m Submission1_TMLR.check_adapters small-instruct general-instruct

For each family, one base model is loaded (as in run_gpu). For every adapter arm it records the
last-token logits on two prompts with the adapter attached, then again after detaching. A sound
run needs: attached != base (the adapter is active) and detached == base exactly (no residue that
would contaminate the next system). It also prints the attention implementation in use.
"""
from __future__ import annotations

import json
import sys

import torch

from Submission1_TMLR.run_gpu import ARMS, HubBackend, load_prompts


def logits(backend: HubBackend, prompts):
    model = backend.peft if backend.peft is not None else backend.model
    texts = [backend.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
             for p in prompts]
    enc = backend.tok(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        return model(**enc).logits[:, -1, :].float().cpu()


def main() -> None:
    prompts = [p["prompt"] for p in load_prompts() if p["experiment"] == "numerical"][:2]
    report = {}
    for tier in sys.argv[1:]:
        b = HubBackend({"budget": {"attention": "auto"}})
        b.load(tier)
        base = logits(b, prompts)
        rows = {"attention": b.attention, "arms": {}}
        for method, seed in ARMS[1:]:
            b.attach(method, seed)
            n_lora = sum(1 for n, _ in b.peft.named_modules() if n.endswith("lora_A"))
            on = logits(b, prompts)
            b.detach()
            off = logits(b, prompts)
            rows["arms"][f"{method}|seed{seed}"] = {
                "lora_modules": n_lora,
                "max_change_when_attached": round(float((on - base).abs().max()), 4),
                "max_residue_after_detach": round(float((off - base).abs().max()), 6),
                "same_top_token_as_base": bool((on.argmax(-1) == base.argmax(-1)).all())}
        report[tier] = rows
        b.free()
        print(json.dumps({tier: rows}, indent=1), flush=True)
    ok = all(a["max_change_when_attached"] > 0 and a["max_residue_after_detach"] == 0 and a["lora_modules"] > 0
             for r in report.values() for a in r["arms"].values())
    print("ADAPTER CHECK", "PASSED" if ok else "FAILED")


if __name__ == "__main__":
    main()
