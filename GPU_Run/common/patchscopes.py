"""The Patchscope patch primitive.

Adapted from Patchscopes (Ghandeharioun et al., ICML 2024; arXiv:2401.06102): patch a
source layer's last-position hidden state into a target prompt at the same model with an
identity mapping, run a single deterministic forward, and read a token probability. Used
by patchscope_bias_verification.py to read the renormalized probability of option c
("Roughly equal") at each attribution-targeted layer (Instruction.md Section 8.3).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _decoder_layers(model):
    """Locate the decoder layer list, unwrapping PEFT and nested .model stacks.

    Works for Llama/Qwen/Gemma-style stacks and for PeftModel wrappers
    (peft_model.base_model.model.model.layers)."""
    # PEFT wrappers expose the wrapped model at .base_model.model or .get_base_model()
    candidates = [model]
    for attr in ("base_model", "model", "transformer"):
        obj = getattr(model, attr, None)
        if obj is not None:
            candidates.append(obj)
    if hasattr(model, "get_base_model"):
        try:
            candidates.append(model.get_base_model())
        except Exception:
            pass

    seen = set()
    stack = list(candidates)
    while stack:
        obj = stack.pop(0)
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        layers = getattr(obj, "layers", None)
        if layers is None:
            layers = getattr(obj, "h", None)
        if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
            return layers
        for attr in ("base_model", "model", "transformer", "decoder"):
            child = getattr(obj, attr, None)
            if child is not None and id(child) not in seen:
                stack.append(child)
    raise AttributeError("Could not locate decoder layers for this model.")


def capture_last_hidden_states(model, tokenizer, prompt: str) -> List["np.ndarray"]:
    """Return the last-position hidden state at every layer for one prompt."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple(num_layers+1) of [1, seq, hidden]
    return [hs[0, -1, :].detach().float().cpu().numpy() for hs in out.hidden_states]


def patch_and_read_letter_prob(
    model,
    tokenizer,
    source_hidden: "np.ndarray",
    target_prompt: str,
    layer_index: int,
    letter_token_ids: Dict[str, int],
    read_letter: str = "c",
) -> float:
    """Patch source_hidden into the target prompt at layer_index (last position),
    one forward pass, return the renormalized probability of read_letter among a/b/c."""
    import torch

    layers = _decoder_layers(model)
    if layer_index >= len(layers):
        layer_index = len(layers) - 1
    inputs = tokenizer(target_prompt, return_tensors="pt").to(model.device)
    vec = torch.tensor(source_hidden, dtype=next(model.parameters()).dtype, device=model.device)

    handle = None

    def hook(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        hidden[:, -1, :] = vec
        return (hidden,) + tuple(out[1:]) if isinstance(out, tuple) else hidden

    try:
        handle = layers[layer_index].register_forward_hook(hook)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
    finally:
        if handle is not None:
            handle.remove()

    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    abc = {k: float(probs[v]) for k, v in letter_token_ids.items() if v is not None}
    total = sum(abc.values()) or 1.0
    return abc.get(read_letter, 0.0) / total


def letter_token_ids(tokenizer) -> Dict[str, int]:
    ids = {}
    for letter in ("a", "b", "c"):
        toks = tokenizer.encode(letter, add_special_tokens=False)
        ids[letter] = toks[0] if toks else None
    return ids
