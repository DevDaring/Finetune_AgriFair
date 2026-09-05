"""The Patchscope patch primitive.

Adapted from Patchscopes (Ghandeharioun et al., ICML 2024; arXiv:2401.06102): patch a
source layer's last-position hidden state into a target prompt at the same model with an
identity mapping, run a single deterministic forward, and read a token probability.
patchscope_bias_verification.py uses it to read the renormalized probability of the
"Roughly equal" option at each attribution-targeted layer.

Two details decide whether the number means anything:
  * hidden_states[i] is the input to block i, so the OUTPUT of block L is
    hidden_states[L + 1]. `layer_output_state` enforces that mapping.
  * the option letters must be scored with the token ids the tokenizer actually emits in
    the JSON answer context, not with a bare "a" that may tokenize differently.
    `letter_token_ids` resolves them in context.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


_NESTING_ATTRIBUTES = ("base_model", "model", "transformer", "decoder", "language_model", "text_model")


def _decoder_layers(model):
    """Locate the decoder layer list, unwrapping PEFT wrappers and nested stacks.

    Covers the plain `model.model.layers` shape, the PEFT wrapper, and multimodal
    checkpoints such as Gemma 3, whose text tower sits under `model.language_model`."""
    candidates = [model]
    for attr in _NESTING_ATTRIBUTES:
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
        for attr in _NESTING_ATTRIBUTES:
            child = getattr(obj, attr, None)
            if child is not None and id(child) not in seen:
                stack.append(child)
    raise AttributeError("Could not locate decoder layers for this model.")


def number_of_layers(model) -> int:
    return len(_decoder_layers(model))


def capture_last_hidden_states(model, tokenizer, prompt: str, add_special_tokens: bool = True) -> List[np.ndarray]:
    """Return the last-position hidden state at every layer boundary for one prompt.

    Index 0 is the embedding output; index i + 1 is the output of decoder block i."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens).to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return [hs[0, -1, :].detach().float().cpu().numpy() for hs in out.hidden_states]


def layer_output_state(hidden_states: List[np.ndarray], layer_index: int) -> np.ndarray:
    """The output of decoder block `layer_index`, clamped to the available range."""
    idx = min(max(0, layer_index + 1), len(hidden_states) - 1)
    return hidden_states[idx]


def patch_and_read_letter_prob(
    model,
    tokenizer,
    source_hidden: "np.ndarray",
    target_prompt: str,
    layer_index: int,
    letter_token_ids: Dict[str, int],
    read_letter: str = "c",
    add_special_tokens: bool = True,
) -> float:
    """Patch source_hidden into the target prompt at the output of block `layer_index`
    (last position), run one forward, and return the probability of `read_letter`
    renormalized over the option letters."""
    import torch

    layers = _decoder_layers(model)
    layer_index = max(0, min(layer_index, len(layers) - 1))
    inputs = tokenizer(target_prompt, return_tensors="pt", add_special_tokens=add_special_tokens).to(model.device)
    vec = torch.tensor(source_hidden, dtype=next(model.parameters()).dtype, device=model.device)

    handle = None

    def hook(module, inp, out):
        if isinstance(out, tuple):
            hidden = out[0].clone()
            hidden[:, -1, :] = vec
            return (hidden,) + tuple(out[1:])
        hidden = out.clone()
        hidden[:, -1, :] = vec
        return hidden

    try:
        handle = layers[layer_index].register_forward_hook(hook)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
    finally:
        if handle is not None:
            handle.remove()

    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    abc = {k: float(probs[v]) for k, v in letter_token_ids.items() if v is not None}
    total = sum(abc.values())
    if total <= 0:
        return float("nan")
    return abc.get(read_letter, 0.0) / total


def letter_token_ids(tokenizer, context: str = '{"answer_choice_letter": "', letters=("a", "b", "c")) -> Dict[str, int]:
    """Token id each letter takes in the JSON answer context.

    Tokenizers split '"a' differently from a bare 'a', so the id is read as the first
    token that appears beyond the context's own tokens."""
    base = tokenizer(context, add_special_tokens=False)["input_ids"]
    ids: Dict[str, Optional[int]] = {}
    for letter in letters:
        full = tokenizer(context + letter, add_special_tokens=False)["input_ids"]
        if len(full) > len(base) and full[: len(base)] == base:
            ids[letter] = full[len(base)]
        else:
            solo = tokenizer(letter, add_special_tokens=False)["input_ids"]
            ids[letter] = solo[0] if solo else None
    return ids
