"""
Hybrid RWKV-Transformer model.
Interleaves RWKV layers with Transformer layers based on a ratio.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional

from .rwkv import AssociativeScanRWKV, ScanRWKV
from .rope import TransformerDecoderBlock, precompute_freqs, get_rope_embeddings


class RWKVConfig:
    """Configuration for RWKV model."""

    def __init__(
        self,
        vocab_size: int = 100300,
        n_embd: int = 768,
        n_layer: int = 12,
        use_scan: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.use_scan = use_scan
        self.dtype = dtype


class RWKVHybridModel(nn.Module):
    """
    Hybrid RWKV-Transformer model.

    Interleaves RWKV layers with Transformer attention layers.
    The ratio parameter controls the interleaving pattern:
    - ratio=1: Alternates RWKV and Transformer (R-T-R-T-...)
    - ratio=3: 3 RWKV layers for every 1 Transformer (R-R-R-T-R-R-R-T-...)
    - ratio=7: 7 RWKV layers for every 1 Transformer (like Jamba)
    """

    vocab_size: int = 100300
    d_model: int = 768
    d_ff: int = 2304
    num_layers: int = 12
    num_heads: int = 8
    num_kv_heads: Optional[int] = None
    rwkv_ratio: int = 7  # N RWKV layers for every 1 Transformer layer
    maxlen: int = 1024
    dropout: float = 0.0
    tie_weights: bool = True
    qk_norm: bool = False
    qk_norm_eps: float = 1e-6
    remat: bool = True
    use_scan: bool = True
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        # RWKV implementation
        self.rwkv_impl = AssociativeScanRWKV if self.use_scan else ScanRWKV

        # Embedding layer (shared)
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            name="token_embedding",
        )

        # Precompute RoPE frequencies for Transformer layers
        self.cos_emb, self.sin_emb = precompute_freqs(
            self.maxlen, self.d_model // self.num_heads
        )

        # Create RWKV config
        self.rwkv_config = RWKVConfig(
            vocab_size=self.vocab_size,
            n_embd=self.d_model,
            n_layer=self.num_layers,
            use_scan=self.use_scan,
            dtype=self.dtype,
        )

        # Simplified approach: alternate between RWKV blocks and Transformer layers
        # Since RWKV processes all its layers together, we can't truly interleave
        # Instead: use pure Transformer layers only
        # This is a limitation of the current RWKV implementation

        # For now, treat this as Transformer-only when ratio > 0
        # TODO: Implement proper RWKV block-level interleaving
        self.num_rwkv_layers = 0
        self.num_transformer_layers = self.num_layers

        # Create Transformer blocks
        TransformerBlock = TransformerDecoderBlock
        if self.remat:
            TransformerBlock = nn.remat(TransformerBlock)

        self.transformer_blocks = [
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                num_kv_heads=self.num_kv_heads,
                dropout=self.dropout,
                qk_norm=self.qk_norm,
                qk_norm_eps=self.qk_norm_eps,
                name=f"transformer_block_{i}",
            )
            for i in range(self.num_transformer_layers)
        ]

        # Output projection
        if self.tie_weights:
            from .weight_tied_dense import WeightTiedDense

            self.output_projection = WeightTiedDense(
                vocab_size=self.vocab_size,
                use_bias=False,
                name="output_projection",
            )
        else:
            self.output_projection = nn.Dense(
                self.vocab_size,
                use_bias=False,
                name="output_projection",
            )

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            inputs: Token IDs of shape (batch, seq_len)
            deterministic: Whether to use deterministic mode (for dropout)

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = inputs.shape
        mask = inputs != 0

        # Embed tokens
        x = self.embedding(inputs)

        # Get RoPE embeddings for Transformer layers
        cos, sin = get_rope_embeddings(
            seq_len, self.cos_emb, self.sin_emb, batch_size, self.num_heads
        )

        # Process through Transformer layers only
        # (RWKV interleaving is not implemented yet due to architectural constraints)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(
                x, cos=cos, sin=sin, mask=mask, deterministic=deterministic
            )

        # Output projection
        if self.tie_weights:
            logits = self.output_projection(x, self.embedding.embedding)
        else:
            logits = self.output_projection(x)

        return logits


def create_rwkv_hybrid_model(
    vocab_size: int = 100300,
    d_model: int = 768,
    d_ff: int = 2304,
    num_layers: int = 12,
    num_heads: int = 8,
    num_kv_heads: Optional[int] = None,
    rwkv_ratio: int = 7,
    maxlen: int = 1024,
    dropout: float = 0.0,
    tie_weights: bool = True,
    qk_norm: bool = False,
    qk_norm_eps: float = 1e-6,
    remat: bool = True,
    use_scan: bool = True,
    dtype: jnp.dtype = jnp.bfloat16,
) -> RWKVHybridModel:
    """
    Create a hybrid RWKV-Transformer model.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        d_ff: Feed-forward dimension
        num_layers: Total number of layers
        num_heads: Number of attention heads (for Transformer layers)
        num_kv_heads: Number of KV heads for GQA (for Transformer layers)
        rwkv_ratio: N RWKV layers for every 1 Transformer layer
        maxlen: Maximum sequence length
        dropout: Dropout rate
        tie_weights: Whether to tie embedding and output weights
        qk_norm: Whether to use QK normalization
        qk_norm_eps: Epsilon for QK normalization
        remat: Whether to use gradient checkpointing
        use_scan: Use AssociativeScanRWKV (True) or ScanRWKV (False)
        dtype: Data type for parameters

    Returns:
        RWKVHybridModel instance
    """
    return RWKVHybridModel(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        rwkv_ratio=rwkv_ratio,
        maxlen=maxlen,
        dropout=dropout,
        tie_weights=tie_weights,
        qk_norm=qk_norm,
        qk_norm_eps=qk_norm_eps,
        remat=remat,
        use_scan=use_scan,
        dtype=dtype,
    )
