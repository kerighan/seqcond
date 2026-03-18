"""Keras 3 implementation of SeqCond models (backend-agnostic: JAX, TF, PyTorch)."""

from .norm import RMSNorm, gated_rmsnorm
from .rope import (
    TransformerDecoderBlock,
    RotarySelfAttention,
    precompute_freqs,
    get_rope_embeddings,
    apply_rope,
)
from .seqcond import SeqCondAttention, SeqCondBlock
from .model import SeqCondModel, create_seqcond_model
from .train import (
    MaskedSparseCategoricalCrossentropy,
    MaskedAccuracy,
    Perplexity,
    WarmupCosineDecay,
    create_optimizer,
    save_checkpoint,
    load_checkpoint,
    create_model_from_config,
    Trainer,
    CheckpointCallback,
)

__all__ = [
    "RMSNorm",
    "gated_rmsnorm",
    "TransformerDecoderBlock",
    "RotarySelfAttention",
    "precompute_freqs",
    "get_rope_embeddings",
    "apply_rope",
    "SeqCondAttention",
    "SeqCondBlock",
    "SeqCondModel",
    "create_seqcond_model",
    "MaskedSparseCategoricalCrossentropy",
    "MaskedAccuracy",
    "Perplexity",
    "WarmupCosineDecay",
    "create_optimizer",
    "save_checkpoint",
    "load_checkpoint",
    "create_model_from_config",
    "Trainer",
    "CheckpointCallback",
]
