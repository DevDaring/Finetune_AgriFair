"""The subject models, revision pinning, offline loading, LoRA targets, batch sizes.

Four instruction-tuned models spanning two size classes and four pretraining recipes, so
that no finding can be one architecture's quirk and so the localization claim is tested
above the small-model regime as well as inside it. English, India-grounded agriculture
domain. All four are primary: each carries the full baseline set, the ablations, the rank
sweep and the multi-seed arms.

SUBJECT_MODELS overrides the active tiers, which is the lever for running a subset when
the full sweep does not fit the compute budget. SMOKE_MODEL_ID swaps in a tiny CPU-friendly
model for smoke and dry runs.

Two loading details this file exists to absorb:

  * Gemma 3 at 12B ships as `Gemma3ForConditionalGeneration`, a multimodal checkpoint whose
    text tower sits under `model.language_model`. `AutoModelForCausalLM` cannot load it, so
    the loader walks a chain of auto classes and records which one succeeded. Every
    downstream component addresses modules by name pattern or by walking to the decoder
    layer list, so the extra nesting changes nothing else.
  * Attention implementation is flash_attention_2 where the architecture supports it, sdpa
    otherwise, and eager where the vendor recommends it. A rejected implementation falls
    back down the chain and the one actually used is recorded in every results CSV.
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

# attention + MLP projection modules, the LoRA target set.
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
    preferred_attention: str  # flash_attention_2 | sdpa | eager
    parameter_count_billions: float = 0.0
    lora_targets: List[str] = field(default_factory=lambda: list(DEFAULT_LORA_TARGETS))
    notes: str = ""
    is_mamba_hybrid: bool = False
    is_multimodal_checkpoint: bool = False
    # Caps that keep the larger models inside a single 80 GB device. They only ever lower
    # the requested batch size, never raise it, and the value actually used is logged.
    max_eval_batch_size: int = 0        # 0 means no cap
    max_train_micro_batch_size: int = 0
    # Files that are a duplicate of the sharded weights and would double the download.
    download_ignore_patterns: List[str] = field(default_factory=list)


REGISTRY: Dict[str, ModelSpec] = {
    "small-instruct": ModelSpec(
        tier="small-instruct",
        hf_id="meta-llama/Llama-3.2-3B-Instruct",
        family="llama",
        role="primary",
        preferred_attention="flash_attention_2",
        parameter_count_billions=3.2,
        notes="Gated repo; needs the HF token at download.",
        # Sized for a 40 GB A100 (3.2B, 128k vocabulary); an uncapped micro-batch of 8 does not fit.
        # Evaluation batch, 3.2B: 6.4 GB of weights leaves most of a 40 GB card for KV cache. generate_batch halves this on OOM.
        max_eval_batch_size=48,
        max_train_micro_batch_size=4,
    ),
    "broad-instruct": ModelSpec(
        tier="broad-instruct",
        hf_id="Qwen/Qwen3-4B-Instruct-2507",
        family="qwen",
        role="primary",
        preferred_attention="flash_attention_2",
        parameter_count_billions=4.0,
        notes="bf16 repo, never FP8; the non-thinking instruct variant.",
        # Sized for a 40 GB A100 (4.0B, 152k vocabulary); an uncapped micro-batch of 8 does not fit.
        # Evaluation batch, 4.0B: 8 GB of weights. generate_batch halves this on OOM.
        max_eval_batch_size=48,
        max_train_micro_batch_size=4,
    ),
    "general-instruct": ModelSpec(
        tier="general-instruct",
        # Verified on the Hub 2026-09-03. Gemma 3 ships at 270m / 1b / 4b / 12b / 27b;
        # there is no 9b, so 12b-it is the smallest release above that size.
        hf_id="google/gemma-3-12b-it",
        family="gemma3",
        role="primary",
        preferred_attention="eager",
        parameter_count_billions=12.2,
        notes="Gemma3ForConditionalGeneration: a multimodal checkpoint whose text tower is "
              "48 layers of hidden size 3840 under model.language_model. Only text is used. "
              "Eager attention follows the vendor's own recommendation for Gemma 3, whose "
              "interleaved local and global attention is the part that matters here. Gated.",
        is_multimodal_checkpoint=True,
        # Sized for a 40 GB A100. Weights alone are 24.4 GB in bf16, and Gemma 3's 262k
        # vocabulary makes the loss tensor 2.7 GB per micro-batch item at length 1024.
        # Evaluation batch, 12.2B: 24.4 GB of weights leaves about 13 GB for KV cache. generate_batch halves this on OOM.
        max_eval_batch_size=16,
        max_train_micro_batch_size=1,
    ),
    # Not a subject model. A genuinely small but REAL instruction-tuned checkpoint, used to
    # rehearse the whole pipeline on CPU before any GPU is rented. The smoke model has random
    # weights and no chat template, so it cannot exercise template rendering, real tokenisation,
    # or answer parsing on real text; this one can, in minutes and for free.
    "cpu-preflight": ModelSpec(
        tier="cpu-preflight",
        hf_id="Qwen/Qwen2.5-0.5B-Instruct",
        family="qwen",
        role="preflight",
        preferred_attention="sdpa",
        parameter_count_billions=0.5,
        notes="Ungated, same family as broad-instruct, real chat template. Never a subject model.",
        max_eval_batch_size=4,
        max_train_micro_batch_size=1,
    ),
    "general-instruct-2": ModelSpec(
        tier="general-instruct-2",
        # Verified on the Hub 2026-09-03. There is no ...-2409 release; 2410 is the only
        # Ministral 8B Instruct, at 8.02B parameters.
        hf_id="mistralai/Ministral-8B-Instruct-2410",
        family="mistral",
        role="primary",
        preferred_attention="flash_attention_2",
        parameter_count_billions=8.0,
        notes="Interleaved sliding-window attention. The repo carries both sharded HF "
              "weights and a 16 GB consolidated copy of the same weights; the consolidated "
              "file is skipped at download. Gated under the Mistral research licence.",
        # Evaluation batch, 8.0B: 16 GB of weights. generate_batch halves this on OOM.
        max_eval_batch_size=24,
        max_train_micro_batch_size=2,
        download_ignore_patterns=["consolidated.safetensors", "params.json"],
    ),
}

PRIMARY_TIERS = [t for t, s in REGISTRY.items() if s.role == "primary"]
TRANSFER_TIERS = [t for t, s in REGISTRY.items() if s.role == "transfer"]
# The preflight tier is opt-in only: it must never be picked up by a default study run.
SUBJECT_TIERS = [t for t, s in REGISTRY.items() if s.role in ("primary", "transfer")]


def active_tiers() -> List[str]:
    override = os.environ.get("SUBJECT_MODELS") or env_loader.get("SUBJECT_MODELS")
    if override:
        return [t.strip() for t in override.split(",") if t.strip()]
    return list(SUBJECT_TIERS)


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


def is_primary(tier: str) -> bool:
    return tier in PRIMARY_TIERS


def hidden_size(model) -> int:
    """Text hidden size, whether the config is flat or nests a text_config."""
    cfg = getattr(model, "config", None)
    size = getattr(cfg, "hidden_size", None)
    if size is None:
        text_cfg = getattr(cfg, "text_config", None)
        size = getattr(text_cfg, "hidden_size", None)
    if size is None:
        raise AttributeError("Could not determine the hidden size for this model.")
    return int(size)


def safe_eval_batch_size(spec: ModelSpec, requested: int) -> int:
    if spec.is_mamba_hybrid:
        return 1
    cap = spec.max_eval_batch_size or requested
    return max(1, min(requested, cap))


def safe_train_micro_batch_size(spec: ModelSpec, requested: int) -> int:
    if spec.is_mamba_hybrid:
        return 1
    cap = spec.max_train_micro_batch_size or requested
    return max(1, min(requested, cap))


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

        if not torch.cuda.is_available():
            return "eager" if preferred == "eager" else "sdpa"
        if preferred == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401

                return "flash_attention_2"
            except Exception:
                return "sdpa"
        return preferred
    except Exception:
        return "sdpa"


def _attention_fallbacks(first: str) -> List[str]:
    order = [first] + [a for a in ("sdpa", "eager") if a != first]
    return order


def _auto_classes(spec: ModelSpec):
    """Auto classes to try, in order.

    A text-only checkpoint loads with AutoModelForCausalLM. A multimodal checkpoint such as
    google/gemma-3-12b-it registers only under the image-text-to-text mapping, so the causal
    class raises and the next entry takes over. AutoModel is the last resort; if it is what
    succeeded, the results CSV says so."""
    from transformers import AutoModel, AutoModelForCausalLM

    try:
        from transformers import AutoModelForImageTextToText

        multimodal = [AutoModelForImageTextToText]
    except ImportError:
        try:
            from transformers import AutoModelForVision2Seq

            multimodal = [AutoModelForVision2Seq]
        except ImportError:
            multimodal = []
    if spec.is_multimodal_checkpoint:
        return multimodal + [AutoModelForCausalLM, AutoModel]
    return [AutoModelForCausalLM] + multimodal + [AutoModel]


def gradient_checkpointing_enabled() -> bool:
    """On by default. Recomputing activations trades roughly 30% training speed for a large
    cut in activation memory, and it changes no result: the gradients are identical.

    The study targets a 40 GB A100, where a 12B backbone in bf16 already spends 24 GB on
    weights alone. Defaulting this off meant the largest tier ran out of memory partway
    through, which costs far more than the recompute. Set GRADIENT_CHECKPOINTING=0 on a
    card with memory to spare."""
    return os.environ.get("GRADIENT_CHECKPOINTING", "1") == "1"


def load_model_and_tokenizer(
    tier: str,
    smoke: bool = False,
    quantized_4bit: bool = False,
    for_training: bool = False,
):
    """Load a model + tokenizer offline. Returns (model, tokenizer, meta).

    meta carries attention_implementation_used and model_revision_hash for the CSVs.
    On CUDA the model is placed with device_map so 4-bit weights are never moved with
    .to(), which bitsandbytes forbids. If the preferred attention implementation is
    rejected by the architecture, the next one in (sdpa, eager) is tried and recorded.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = get_spec(smoke_model_id() if smoke else tier)
    local = _local_dir(spec)
    src = local if os.path.isdir(local) else spec.hf_id
    offline = os.path.isdir(local)

    tok = AutoTokenizer.from_pretrained(src, local_files_only=offline, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left-pad for batched greedy generation

    cuda = torch.cuda.is_available()
    quantized = bool(quantized_4bit and cuda)
    base_kwargs = dict(trust_remote_code=True, local_files_only=offline, pad_token_id=tok.pad_token_id)
    if cuda:
        base_kwargs["torch_dtype"] = torch.bfloat16
        base_kwargs["device_map"] = {"": 0}
    else:
        # fp32 was chosen for the tiny CPU smoke model. The post-teardown geometry stage
        # (parameter_space_geometry, H3) loads the full-size bases on CPU too, and at fp32 an
        # 8B model is ~32GB - more than this box has - which OOM-kills the process silently
        # mid-load. bf16 halves that and loses nothing here: the geometry maths already casts
        # each matrix bf16 -> fp32 on its own. Opt-in so the smoke path is untouched.
        cpu_dtype = os.environ.get("CPU_LOAD_DTYPE", "float32")
        base_kwargs["torch_dtype"] = torch.bfloat16 if cpu_dtype == "bfloat16" else torch.float32
    if quantized:
        from transformers import BitsAndBytesConfig

        base_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = None
    used_attention = None
    used_class = None
    last_error = None
    for auto_class in _auto_classes(spec):
        for attn in _attention_fallbacks(_select_attention(spec.preferred_attention)):
            kwargs = dict(base_kwargs)
            if cuda or attn == "eager":
                kwargs["attn_implementation"] = attn
            try:
                model = auto_class.from_pretrained(src, **kwargs)
                used_attention, used_class = attn, auto_class.__name__
                break
            except (ValueError, ImportError, RuntimeError, KeyError) as e:
                last_error = e
                logger.warning("%s with attn_implementation=%s rejected for %s (%s); trying the next option.",
                               auto_class.__name__, attn, spec.hf_id, type(e).__name__)
        if model is not None:
            break
    if model is None:
        raise last_error if last_error else RuntimeError(f"Could not load {spec.hf_id}")

    if quantized and for_training:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing_enabled()
        )
    elif for_training and gradient_checkpointing_enabled():
        try:
            model.gradient_checkpointing_enable()
        except Exception as e:
            logger.warning("Gradient checkpointing unavailable (%s); continuing without it.", e)
        else:
            # Must succeed once checkpointing is on. With checkpointing enabled but inputs not
            # requiring grad, the recomputed graph is detached and every LoRA parameter comes
            # back with grad None: the run trains adapters that never change, and reports
            # nothing wrong. Sharing one try with the call above turned that into a warning
            # that said checkpointing was unavailable while it was in fact active.
            try:
                model.enable_input_require_grads()
            except Exception as e:
                logger.error("Gradient checkpointing is on but enable_input_require_grads "
                             "failed (%s); disabling checkpointing so gradients still flow.", e)
                try:
                    model.gradient_checkpointing_disable()
                except Exception:
                    raise RuntimeError(
                        "Gradient checkpointing is active without input gradients; LoRA "
                        "parameters would receive no gradient. Refusing to train.") from e
    if model.generation_config is not None:
        model.generation_config.pad_token_id = tok.pad_token_id
    model.eval()
    # Batch caps travel on the model so that inference and training read them without
    # every call site having to know which tier it is working with.
    model._agrifair_max_eval_batch = spec.max_eval_batch_size or 0
    model._agrifair_max_train_micro_batch = spec.max_train_micro_batch_size or 0
    meta = {
        "tier": "smoke" if smoke else tier,
        "hf_id": spec.hf_id,
        "parameter_count_billions": spec.parameter_count_billions,
        "attention_implementation_used": used_attention,
        "model_class_used": used_class,
        "model_revision_hash": get_revision(tier),
        "quantization_setting": "nf4_4bit" if quantized else "bf16_full",
        "is_mamba_hybrid": spec.is_mamba_hybrid,
    }
    return model, tok, meta
