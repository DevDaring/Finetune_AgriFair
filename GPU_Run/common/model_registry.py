"""The subject models, revision pinning, offline loading, LoRA targets, batch sizes.

Four models on a model-diversity axis (English; India-grounded agriculture domain).
SUBJECT_MODELS overrides the active tiers; SMOKE_MODEL_ID swaps in a tiny CPU-friendly
model for smoke/dry runs. Attention implementation is flash_attention_2 where supported,
sdpa otherwise (Windows/CPU -> sdpa), recorded per model (Instruction.md Section 5).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from GPU_Run.common import env_loader
from GPU_Run.common.logging_utils import get_logger
from GPU_Run.common.paths import MODELS_DIR, MODEL_REVISIONS

logger = get_logger("model_registry")

# attention + MLP projection modules, the LoRA target set (Instruction.md Section 5).
DEFAULT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class ModelSpec:
    tier: str
    hf_id: str
    family: str
    role: str  # "primary" or "transfer"
    preferred_attention: str  # flash_attention_2 | sdpa
    lora_targets: List[str] = field(default_factory=lambda: list(DEFAULT_LORA_TARGETS))
    notes: str = ""
    is_mamba_hybrid: bool = False


REGISTRY: Dict[str, ModelSpec] = {
    "compact-hybrid": ModelSpec(
        tier="compact-hybrid",
        hf_id="nvidia/NVIDIA-Nemotron-3-Nano-4B",
        family="nemotron",
        role="transfer",
        preferred_attention="sdpa",
        notes="Mamba-2 hybrid; restricted target set; safetensors, never GGUF.",
        is_mamba_hybrid=True,
    ),
    "small-instruct": ModelSpec(
        tier="small-instruct",
        hf_id="meta-llama/Llama-3.2-3B-Instruct",
        family="llama",
        role="primary",
        preferred_attention="flash_attention_2",
        notes="Gated repo; needs HF key at download.",
    ),
    "indic-specialized": ModelSpec(
        tier="indic-specialized",
        hf_id="Telugu-LLM-Labs/Indic-gemma-2b-finetuned-sft-Navarasa-2.0",
        family="gemma",
        role="transfer",
        preferred_attention="flash_attention_2",
        notes="Gemma-2 based; especially relevant to the India-grounded domain.",
    ),
    "broad-instruct": ModelSpec(
        tier="broad-instruct",
        hf_id="Qwen/Qwen3-4B-Instruct-2507",
        family="qwen",
        role="primary",
        preferred_attention="flash_attention_2",
        notes="bf16 repo, never FP8.",
    ),
}

PRIMARY_TIERS = [t for t, s in REGISTRY.items() if s.role == "primary"]
TRANSFER_TIERS = [t for t, s in REGISTRY.items() if s.role == "transfer"]


def active_tiers() -> List[str]:
    override = os.environ.get("SUBJECT_MODELS") or env_loader.get("SUBJECT_MODELS")
    if override:
        return [t.strip() for t in override.split(",") if t.strip()]
    return list(REGISTRY.keys())


def smoke_model_id() -> str:
    # default to a tiny LLaMA-architecture model so the smoke run exercises the same
    # module names (q_proj/k_proj/.../down_proj) and nn.Linear types as the real models.
    return (
        os.environ.get("SMOKE_MODEL_ID")
        or env_loader.get("SMOKE_MODEL_ID")
        or "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )


def get_spec(tier: str) -> ModelSpec:
    if tier in REGISTRY:
        return REGISTRY[tier]
    # allow a bare hf id as an ad-hoc tier (used for smoke runs)
    return ModelSpec(
        tier=tier,
        hf_id=tier,
        family="generic",
        role="transfer",
        preferred_attention="sdpa",
    )


def safe_eval_batch_size(spec: ModelSpec, requested: int) -> int:
    return 1 if spec.is_mamba_hybrid else max(1, requested)


def safe_train_micro_batch_size(spec: ModelSpec, requested: int) -> int:
    return 1 if spec.is_mamba_hybrid else max(1, requested)


def _local_dir(spec: ModelSpec) -> str:
    return str(MODELS_DIR / spec.hf_id.replace("/", "__"))


def record_revision(tier: str, revision: str) -> None:
    data = {}
    if MODEL_REVISIONS.exists():
        data = json.loads(MODEL_REVISIONS.read_text(encoding="utf-8"))
    data[tier] = revision
    MODEL_REVISIONS.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_revision(tier: str) -> str:
    if MODEL_REVISIONS.exists():
        return json.loads(MODEL_REVISIONS.read_text(encoding="utf-8")).get(tier, "unknown")
    return "unknown"


def _select_attention(preferred: str) -> str:
    try:
        import torch

        if preferred == "flash_attention_2" and torch.cuda.is_available():
            try:
                import flash_attn  # noqa: F401

                return "flash_attention_2"
            except Exception:
                return "sdpa"
        return "sdpa"
    except Exception:
        return "sdpa"


def load_model_and_tokenizer(
    tier: str,
    smoke: bool = False,
    quantized_4bit: bool = False,
    for_training: bool = False,
):
    """Load a model + tokenizer offline. Returns (model, tokenizer, meta).

    meta carries attention_implementation_used and model_revision_hash for the CSVs.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = get_spec(smoke_model_id() if smoke else tier)
    attn = _select_attention(spec.preferred_attention)
    local = _local_dir(spec)
    src = local if os.path.isdir(local) else spec.hf_id
    offline = os.path.isdir(local)

    tok = AutoTokenizer.from_pretrained(src, local_files_only=offline, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left-pad for batched greedy generation

    kwargs = dict(trust_remote_code=True, local_files_only=offline)
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["attn_implementation"] = attn
    else:
        kwargs["torch_dtype"] = torch.float32  # CPU smoke

    if quantized_4bit and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        except Exception as e:
            logger.warning("4-bit config unavailable (%s); loading full precision.", e)

    model = AutoModelForCausalLM.from_pretrained(src, **kwargs)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    meta = {
        "tier": "smoke" if smoke else tier,
        "hf_id": spec.hf_id,
        "attention_implementation_used": attn,
        "model_revision_hash": get_revision(tier),
        "quantization_setting": "nf4_4bit" if (quantized_4bit and torch.cuda.is_available()) else "bf16_full",
    }
    return model, tok, meta
