import math
from typing import Optional, Tuple, Any

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np

from .rope import TransformerDecoderBlock, precompute_freqs, get_rope_embeddings

# from .seqcond_decay import SeqCondBlock

from .seqcond_fast import SeqCondBlock

# from .seqcond_summary import SeqCondBlock
from .seqcond_2 import SeqCondBlockV2
from .mamba.mamba import Mamba2Block, Mamba2RMSNorm
from .mamba.config import Mamba2Config
from .weight_tied_dense import WeightTiedDense
from .bivector import TransformerDecoderBlock as BivectorBlock
from .rwkv_wrapper import RWKVModel, create_rwkv_model
from .rwkv_hybrid import RWKVHybridModel, create_rwkv_hybrid_model


class TransformerModel(nn.Module):
    d_model: int = 256
    d_ff: int = 768
    num_layers: int = 12
    num_heads: int = 8
    num_kv_heads: Optional[int] = None
    vocab_size: int = 100300
    maxlen: int = 1024
    dropout: float = 0.0
    tie_weights: bool = True
    qk_norm: bool = False
    qk_norm_eps: float = 1e-6
    remat: bool = True

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            name="token_embedding",
        )
        self.cos_emb, self.sin_emb = precompute_freqs(
            self.maxlen, self.d_model // self.num_heads
        )

        Block = TransformerDecoderBlock
        if self.remat:
            Block = nn.remat(Block)

        self.blocks = [
            Block(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                num_kv_heads=self.num_kv_heads,
                dropout=self.dropout,
                qk_norm=self.qk_norm,
                qk_norm_eps=self.qk_norm_eps,
                name=f"transformer_block_{i}",
            )
            for i in range(self.num_layers)
        ]
        if self.tie_weights:
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

    def __call__(
        self,
        inputs: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        b, l = inputs.shape
        mask = inputs != 0

        x = self.embedding(inputs)
        cos, sin = get_rope_embeddings(l, self.cos_emb, self.sin_emb, b, self.num_heads)

        for block in self.blocks:
            x = block(x, cos=cos, sin=sin, mask=mask, deterministic=deterministic)

        if self.tie_weights:
            logits = self.output_projection(x, self.embedding.embedding)
        else:
            logits = self.output_projection(x)

        return logits


class BivectorModel(nn.Module):
    d_model: int = 256
    d_ff: int = 768
    num_layers: int = 12
    num_heads: int = 8
    num_kv_heads: Optional[int] = None
    vocab_size: int = 100300
    maxlen: int = 1024
    dropout: float = 0.0
    tie_weights: bool = True
    qk_norm: bool = False
    qk_norm_eps: float = 1e-6
    remat: bool = True

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            name="token_embedding",
        )
        self.cos_emb, self.sin_emb = precompute_freqs(
            self.maxlen, self.d_model // self.num_heads
        )

        Block = BivectorBlock
        if self.remat:
            Block = nn.remat(Block)

        self.blocks = [
            Block(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                num_kv_heads=self.num_kv_heads,
                dropout=self.dropout,
                qk_norm=self.qk_norm,
                qk_norm_eps=self.qk_norm_eps,
                name=f"bivector_block_{i}",
            )
            for i in range(self.num_layers)
        ]
        if self.tie_weights:
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

    def __call__(
        self,
        inputs: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        b, l = inputs.shape
        mask = inputs != 0

        x = self.embedding(inputs)
        cos, sin = get_rope_embeddings(l, self.cos_emb, self.sin_emb, b, self.num_heads)

        for block in self.blocks:
            x = block(x, cos=cos, sin=sin, mask=mask, deterministic=deterministic)

        if self.tie_weights:
            logits = self.output_projection(x, self.embedding.embedding)
        else:
            logits = self.output_projection(x)

        return logits


class SeqCondModel(nn.Module):
    d_model: int = 256
    d_ff: int = 768
    num_layers: int = 12
    vocab_size: int = 100300
    maxlen: int = 1024
    use_positional_embedding: bool = False
    seqcond_ratio: int = 3
    num_heads: int = 8
    num_kv_heads: Optional[int] = None
    seqcond_heads: Optional[int] = None
    num_query_heads: int = 6
    num_anchor_heads: int = 0
    num_thetas: int = 4
    derivative_order: int = 0
    derivative_aggregation: str = "re_im"
    dropout: float = 0.0
    tie_weights: bool = True
    qk_norm: bool = False
    qk_norm_eps: float = 1e-6
    conv_kernel_size: int = 4
    expand_factor: float = 2.0
    out_expand_factor: int = 3
    remat: bool = True
    chunk_size: int = 0
    use_square_matrix: bool = False

    def setup(self):
        _seqcond_heads = (
            self.seqcond_heads if self.seqcond_heads is not None else self.num_heads
        )

        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            name="token_embedding",
        )

        if self.use_positional_embedding:
            if self.maxlen is None:
                raise ValueError(
                    "maxlen must be set when use_positional_embedding=True"
                )
            self.position_embedding = nn.Embed(
                num_embeddings=self.maxlen,
                features=self.d_model,
                name="position_embedding",
            )

        self.cos_emb, self.sin_emb = precompute_freqs(
            self.maxlen, self.d_model // self.num_heads
        )

        TransformerBlock = TransformerDecoderBlock
        SeqBlock = SeqCondBlock
        if self.remat:
            TransformerBlock = nn.remat(TransformerBlock)
            SeqBlock = nn.remat(SeqBlock)

        blocks = []
        transformer_idx = 0
        seqcond_idx = 0
        for i in range(self.num_layers):
            if (i + 1) % (self.seqcond_ratio + 1) == 0:
                block = TransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                    num_kv_heads=self.num_kv_heads,
                    dropout=self.dropout,
                    qk_norm=self.qk_norm,
                    qk_norm_eps=self.qk_norm_eps,
                    name=f"transformer_block_{transformer_idx}",
                )
                blocks.append(("transformer", block))
                transformer_idx += 1
            else:
                block = SeqBlock(
                    num_heads=_seqcond_heads,
                    num_query_heads=self.num_query_heads,
                    num_thetas=self.num_thetas,
                    num_anchor_heads=self.num_anchor_heads,
                    conv_kernel_size=self.conv_kernel_size,
                    expand_factor=self.expand_factor,
                    out_expand_factor=self.out_expand_factor,
                    dropout=self.dropout,
                    maxlen=self.maxlen,
                    name=f"seqcond_block_{seqcond_idx}",
                    use_square_matrix=self.use_square_matrix,
                )
                blocks.append(("seqcond", block))
                seqcond_idx += 1
        self.blocks = blocks

        if self.tie_weights:
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

    def __call__(
        self,
        inputs: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        b, l = inputs.shape
        mask = inputs != 0

        x = self.embedding(inputs)

        if self.use_positional_embedding:
            positions = jnp.arange(l, dtype=jnp.int32)[None, :]
            x = x + self.position_embedding(positions)

        cos, sin = get_rope_embeddings(l, self.cos_emb, self.sin_emb, b, self.num_heads)

        for block_type, block in self.blocks:
            if block_type == "transformer":
                x = block(x, cos=cos, sin=sin, mask=mask, deterministic=deterministic)
            else:
                x = block(x, mask=mask, deterministic=deterministic)

        if self.tie_weights:
            logits = self.output_projection(x, self.embedding.embedding)
        else:
            logits = self.output_projection(x)

        return logits

    def init_state(self, batch_size: int = 1):
        """
        Initialize states for all blocks for step-by-step generation.

        Returns:
            states: List of states, one per block
                - SeqCond: (den_acc, re_acc, im_acc, pos, conv_buffer)
                - Transformer: (k_cache, v_cache)
        """
        states = []
        _seqcond_heads = (
            self.seqcond_heads if self.seqcond_heads is not None else self.num_heads
        )

        for block_type, block in self.blocks:
            if block_type == "seqcond":
                # SeqCond state
                num_heads = _seqcond_heads
                num_query_heads = self.num_query_heads or num_heads
                num_thetas = self.num_thetas

                d_inner = int(self.d_model * self.expand_factor)
                H = max(1, d_inner // (num_heads * num_thetas))
                conv_kernel_size = self.conv_kernel_size

                dim_memory = num_heads * H
                dim_query_head = H * num_thetas * 2
                dim_query_total = num_query_heads * dim_query_head
                dim_mem_total = dim_memory + num_heads
                dim_conv_total = dim_mem_total + dim_query_total

                den_acc = jnp.zeros((batch_size, num_heads), dtype=jnp.float32)
                re_acc = jnp.zeros(
                    (batch_size, num_heads, H, num_thetas), dtype=jnp.float32
                )
                im_acc = jnp.zeros(
                    (batch_size, num_heads, H, num_thetas), dtype=jnp.float32
                )
                pos = jnp.zeros((batch_size,), dtype=jnp.int32)
                conv_buffer = jnp.zeros(
                    (batch_size, conv_kernel_size - 1, dim_conv_total),
                    dtype=jnp.float32,
                )
                states.append((den_acc, re_acc, im_acc, pos, conv_buffer))
            else:
                # Transformer KV cache - pre-allocate to maxlen
                num_kv_heads = self.num_kv_heads or self.num_heads
                head_dim = self.d_model // self.num_heads

                k_cache = jnp.zeros(
                    (batch_size, self.maxlen, num_kv_heads, head_dim),
                    dtype=jnp.bfloat16,
                )
                v_cache = jnp.zeros(
                    (batch_size, self.maxlen, num_kv_heads, head_dim),
                    dtype=jnp.bfloat16,
                )
                states.append((k_cache, v_cache))

        return states

    def step(
        self,
        token_id: jnp.ndarray,
        states: list,
        pos: int,
        deterministic: bool = True,
    ):
        """
        O(1) step for SeqCond blocks, O(L) step for Transformer blocks.

        Args:
            token_id: Token ID (B,) or (B, 1)
            states: List of states from init_state() or previous step()
            pos: Current position in sequence
            deterministic: Whether to use dropout

        Returns:
            logits: (B, vocab_size) logits for next token
            new_states: Updated states for next step
        """
        # Handle input shape
        if token_id.ndim == 1:
            token_id = token_id[:, None]  # (B,) -> (B, 1)

        b = token_id.shape[0]

        # Embed token
        x = self.embedding(token_id)[:, 0, :]  # (B, D)

        if self.use_positional_embedding:
            pos_emb = self.position_embedding(jnp.array([[pos]]))[:, 0, :]
            x = x + pos_emb

        # Get RoPE for this position (use dynamic_slice for JIT compatibility)
        head_dim_half = self.cos_emb.shape[1]
        cos_t = jax.lax.dynamic_slice(self.cos_emb, (pos, 0), (1, head_dim_half))
        sin_t = jax.lax.dynamic_slice(self.sin_emb, (pos, 0), (1, head_dim_half))
        cos_t = cos_t[None, :, None, :]  # (1, 1, 1, head_dim//2)
        sin_t = sin_t[None, :, None, :]
        cos_t = jnp.broadcast_to(cos_t, (b, 1, self.num_heads, head_dim_half))
        sin_t = jnp.broadcast_to(sin_t, (b, 1, self.num_heads, head_dim_half))

        new_states = []
        for i, (block_type, block) in enumerate(self.blocks):
            if block_type == "transformer":
                x, new_state = block.step(
                    x, states[i], pos, cos_t, sin_t, deterministic=deterministic
                )
                new_states.append(new_state)
            else:
                # SeqCond block step (uses block.step which handles norm + attn + residual)
                x, new_state = block.step(x, states[i], deterministic=deterministic)
                new_states.append(new_state)

        # Output projection
        if self.tie_weights:
            logits = self.output_projection(x[:, None, :], self.embedding.embedding)[
                :, 0, :
            ]
        else:
            logits = self.output_projection(x)

        return logits, new_states


class SeqCondModelV2(nn.Module):
    """SeqCond model with dynamic thetas projected from input."""

    d_model: int = 256
    d_ff: int = 768
    num_layers: int = 12
    vocab_size: int = 100300
    maxlen: int = 1024
    use_positional_embedding: bool = False
    seqcond_ratio: int = 3
    num_heads: int = 8
    num_kv_heads: Optional[int] = None
    seqcond_heads: Optional[int] = None
    num_anchor_heads: int = 0
    num_thetas: int = 4
    dropout: float = 0.0
    tie_weights: bool = True
    qk_norm: bool = False
    qk_norm_eps: float = 1e-6
    conv_kernel_size: int = 4
    expand_factor: float = 2.0
    remat: bool = True

    def setup(self):
        _seqcond_heads = (
            self.seqcond_heads if self.seqcond_heads is not None else self.num_heads
        )

        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            name="token_embedding",
        )

        if self.use_positional_embedding:
            if self.maxlen is None:
                raise ValueError(
                    "maxlen must be set when use_positional_embedding=True"
                )
            self.position_embedding = nn.Embed(
                num_embeddings=self.maxlen,
                features=self.d_model,
                name="position_embedding",
            )

        self.cos_emb, self.sin_emb = precompute_freqs(
            self.maxlen, self.d_model // self.num_heads
        )

        TransformerBlock = TransformerDecoderBlock
        SeqBlock = SeqCondBlockV2
        if self.remat:
            TransformerBlock = nn.remat(TransformerBlock)
            SeqBlock = nn.remat(SeqBlock)

        blocks = []
        transformer_idx = 0
        seqcond_idx = 0
        for i in range(self.num_layers):
            if (i + 1) % (self.seqcond_ratio + 1) == 0:
                block = TransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                    num_kv_heads=self.num_kv_heads,
                    dropout=self.dropout,
                    qk_norm=self.qk_norm,
                    qk_norm_eps=self.qk_norm_eps,
                    name=f"transformer_block_{transformer_idx}",
                )
                blocks.append(("transformer", block))
                transformer_idx += 1
            else:
                block = SeqBlock(
                    num_heads=_seqcond_heads,
                    num_thetas=self.num_thetas,
                    num_anchor_heads=self.num_anchor_heads,
                    expand_factor=self.expand_factor,
                    dropout=self.dropout,
                    conv_kernel_size=self.conv_kernel_size,
                    name=f"seqcond_block_{seqcond_idx}",
                )
                blocks.append(("seqcond", block))
                seqcond_idx += 1
        self.blocks = blocks

        if self.tie_weights:
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

    def __call__(
        self,
        inputs: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        b, l = inputs.shape
        mask = inputs != 0

        x = self.embedding(inputs)

        if self.use_positional_embedding:
            positions = jnp.arange(l, dtype=jnp.int32)[None, :]
            x = x + self.position_embedding(positions)

        cos, sin = get_rope_embeddings(l, self.cos_emb, self.sin_emb, b, self.num_heads)

        for block_type, block in self.blocks:
            if block_type == "transformer":
                x = block(x, cos=cos, sin=sin, mask=mask, deterministic=deterministic)
            else:
                x = block(x, mask=mask, deterministic=deterministic)

        if self.tie_weights:
            logits = self.output_projection(x, self.embedding.embedding)
        else:
            logits = self.output_projection(x)

        return logits


class MambaModel(nn.Module):
    """Mamba model mixed with Transformer layers."""

    d_model: int = 768
    d_ff: int = 2304
    num_layers: int = 12
    vocab_size: int = 100300
    maxlen: int = 1024
    use_positional_embedding: bool = False
    seqcond_ratio: int = 5
    num_heads: int = 8
    num_kv_heads: Optional[int] = None

    # Mamba specific
    state_size: int = 128
    expand_factor: float = 2.0
    conv_kernel_size: int = 4

    dropout: float = 0.0
    tie_weights: bool = True
    qk_norm: bool = False
    qk_norm_eps: float = 1e-6
    remat: bool = True

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            name="token_embedding",
            embedding_init=nn.initializers.normal(stddev=0.02),
        )

        # Mamba generally handles position via SSM, but if we mix Transformer,
        # the Transformer layers need position info (RoPE).
        # We precompute RoPE frequencies for the Transformer layers.
        self.cos_emb, self.sin_emb = precompute_freqs(
            self.maxlen, self.d_model // self.num_heads
        )

        TransformerBlock = TransformerDecoderBlock
        MBlock = Mamba2Block

        if self.remat:
            TransformerBlock = nn.remat(TransformerBlock)
            MBlock = nn.remat(MBlock)

        # Mamba configuration
        mamba_config = Mamba2Config(
            hidden_size=self.d_model,
            state_size=self.state_size,
            expand=int(self.expand_factor),
            conv_kernel=self.conv_kernel_size,
            vocab_size=self.vocab_size,
            num_hidden_layers=self.num_layers,
            # Use defaults for other Mamba params or expose them if needed
        )

        blocks = []
        transformer_idx = 0
        mamba_idx = 0

        for i in range(self.num_layers):
            if (i + 1) % (self.seqcond_ratio + 1) == 0:
                block = TransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                    num_kv_heads=self.num_kv_heads,
                    dropout=self.dropout,
                    qk_norm=self.qk_norm,
                    qk_norm_eps=self.qk_norm_eps,
                    name=f"transformer_block_{transformer_idx}",
                )
                blocks.append(("transformer", block))
                transformer_idx += 1
            else:
                block = MBlock(
                    config=mamba_config,
                    layer_idx=i,
                    name=f"mamba_block_{mamba_idx}",
                )
                blocks.append(("mamba", block))
                mamba_idx += 1

        self.blocks = blocks

        if self.tie_weights:
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

    def __call__(
        self,
        inputs: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        b, l = inputs.shape
        mask = inputs != 0

        x = self.embedding(inputs)

        # RoPE embeddings for Transformer layers
        cos, sin = get_rope_embeddings(l, self.cos_emb, self.sin_emb, b, self.num_heads)

        for block_type, block in self.blocks:
            if block_type == "transformer":
                x = block(x, cos=cos, sin=sin, mask=mask, deterministic=deterministic)
            else:
                # Mamba block: returns (hidden_states, last_state)
                # We discard last_state during training/simple forward
                x, _ = block(x)

        if self.tie_weights:
            logits = self.output_projection(x, self.embedding.embedding)
        else:
            logits = self.output_projection(x)

        return logits


def create_transformer_model(
    d_model: int = 256,
    d_ff: int = 768,
    num_layers: int = 12,
    num_heads: int = 8,
    num_kv_heads: Optional[int] = None,
    vocab_size: int = 100300,
    maxlen: int = 1024,
    dropout: float = 0.0,
    tie_weights: bool = True,
    qk_norm: bool = False,
    qk_norm_eps: float = 1e-6,
    remat: bool = True,
) -> TransformerModel:
    """Create a Transformer model."""
    return TransformerModel(
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        vocab_size=vocab_size,
        maxlen=maxlen,
        dropout=dropout,
        tie_weights=tie_weights,
        qk_norm=qk_norm,
        qk_norm_eps=qk_norm_eps,
        remat=remat,
    )


def create_bivector_model(
    d_model: int = 256,
    d_ff: int = 768,
    num_layers: int = 12,
    num_heads: int = 8,
    num_kv_heads: Optional[int] = None,
    vocab_size: int = 100300,
    maxlen: int = 1024,
    dropout: float = 0.0,
    tie_weights: bool = True,
    qk_norm: bool = False,
    qk_norm_eps: float = 1e-6,
    remat: bool = True,
) -> BivectorModel:
    """Create a Bivector model."""
    return BivectorModel(
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        vocab_size=vocab_size,
        maxlen=maxlen,
        dropout=dropout,
        tie_weights=tie_weights,
        qk_norm=qk_norm,
        qk_norm_eps=qk_norm_eps,
        remat=remat,
    )


def create_seqcond_model(
    d_model: int = 256,
    d_ff: int = 768,
    num_layers: int = 12,
    vocab_size: int = 100300,
    maxlen: int = 1024,
    use_positional_embedding: bool = False,
    seqcond_ratio: int = 3,
    num_heads: int = 8,
    num_kv_heads: Optional[int] = None,
    seqcond_heads: Optional[int] = None,
    num_query_heads: int = 6,
    num_anchor_heads: int = 0,
    num_thetas: int = 4,
    derivative_order: int = 0,
    derivative_aggregation: str = "re_im",
    dropout: float = 0.0,
    tie_weights: bool = True,
    qk_norm: bool = False,
    qk_norm_eps: float = 1e-6,
    conv_kernel_size: int = 4,
    expand_factor: float = 2.0,
    out_expand_factor: int = 3,
    remat: bool = True,
    chunk_size: int = 0,
    use_square_matrix: bool = False,
) -> SeqCondModel:
    """Create a SeqCond model."""
    return SeqCondModel(
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        vocab_size=vocab_size,
        maxlen=maxlen,
        use_positional_embedding=use_positional_embedding,
        seqcond_ratio=seqcond_ratio,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        seqcond_heads=seqcond_heads,
        num_query_heads=num_query_heads,
        num_anchor_heads=num_anchor_heads,
        num_thetas=num_thetas,
        derivative_order=derivative_order,
        derivative_aggregation=derivative_aggregation,
        dropout=dropout,
        tie_weights=tie_weights,
        qk_norm=qk_norm,
        qk_norm_eps=qk_norm_eps,
        conv_kernel_size=conv_kernel_size,
        expand_factor=expand_factor,
        out_expand_factor=out_expand_factor,
        remat=remat,
        chunk_size=chunk_size,
        use_square_matrix=use_square_matrix,
    )


def create_seqcond_model_v2(
    d_model: int = 256,
    d_ff: int = 768,
    num_layers: int = 12,
    vocab_size: int = 100300,
    maxlen: int = 1024,
    use_positional_embedding: bool = False,
    seqcond_ratio: int = 3,
    num_heads: int = 8,
    num_kv_heads: Optional[int] = None,
    seqcond_heads: Optional[int] = None,
    num_anchor_heads: int = 0,
    num_thetas: int = 4,
    dropout: float = 0.0,
    tie_weights: bool = True,
    qk_norm: bool = False,
    qk_norm_eps: float = 1e-6,
    conv_kernel_size: int = 4,
    expand_factor: float = 2.0,
    remat: bool = True,
) -> SeqCondModelV2:
    """Create a SeqCond V2 model with dynamic thetas."""
    return SeqCondModelV2(
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        vocab_size=vocab_size,
        maxlen=maxlen,
        use_positional_embedding=use_positional_embedding,
        seqcond_ratio=seqcond_ratio,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        seqcond_heads=seqcond_heads,
        num_anchor_heads=num_anchor_heads,
        num_thetas=num_thetas,
        dropout=dropout,
        tie_weights=tie_weights,
        qk_norm=qk_norm,
        qk_norm_eps=qk_norm_eps,
        conv_kernel_size=conv_kernel_size,
        expand_factor=expand_factor,
        remat=remat,
    )


def create_mamba_model(
    d_model: int = 768,
    d_ff: int = 2304,
    num_layers: int = 12,
    vocab_size: int = 100300,
    maxlen: int = 1024,
    seqcond_ratio: int = 5,
    num_heads: int = 8,
    num_kv_heads: Optional[int] = None,
    state_size: int = 128,
    expand_factor: float = 2.0,
    conv_kernel_size: int = 4,
    dropout: float = 0.0,
    tie_weights: bool = True,
    qk_norm: bool = False,
    qk_norm_eps: float = 1e-6,
    remat: bool = True,
) -> MambaModel:
    """Create a Mamba model."""
    return MambaModel(
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        vocab_size=vocab_size,
        maxlen=maxlen,
        seqcond_ratio=seqcond_ratio,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        state_size=state_size,
        expand_factor=expand_factor,
        conv_kernel_size=conv_kernel_size,
        dropout=dropout,
        tie_weights=tie_weights,
        qk_norm=qk_norm,
        qk_norm_eps=qk_norm_eps,
        remat=remat,
    )


def warmup_cosine_decay_schedule(
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    alpha: float = 1e-5,
) -> optax.Schedule:
    """Create a warmup + cosine decay learning rate schedule."""
    decay_steps = total_steps - warmup_steps

    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_steps,
    )

    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=decay_steps,
        alpha=alpha / base_lr if base_lr > 0 else 0.0,
    )

    return optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps],
    )


def create_optimizer(
    base_lr: float = 1e-3,
    warmup_steps: int = 1000,
    total_steps: int = 1_000_000,
    weight_decay: float = 1e-2,
    clipnorm: float = 1.0,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    optimizer_type: str = "adamw",
) -> optax.GradientTransformation:
    """Create an optimizer with warmup + cosine decay schedule.

    Args:
        optimizer_type: 'adamw' or 'muon'. Muon applies Newton-Schulz orthogonalization
            to 2D weight matrices and AdamW to everything else (embeddings, biases, norms).
    """
    lr_schedule = warmup_cosine_decay_schedule(
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        alpha=1e-5,
    )

    if optimizer_type == "muon":
        return optax.chain(
            optax.clip_by_global_norm(clipnorm),
            optax.contrib.muon(
                learning_rate=lr_schedule,
                beta=beta_1,
                weight_decay=weight_decay,
                adam_b1=beta_1,
                adam_b2=beta_2,
                adam_weight_decay=weight_decay,
            ),
        )
    else:
        return optax.chain(
            optax.clip_by_global_norm(clipnorm),
            optax.adamw(
                learning_rate=lr_schedule,
                b1=beta_1,
                b2=beta_2,
                eps=1e-7,
                mu_dtype=jnp.float32,
                weight_decay=weight_decay,
            ),
        )


def sparse_categorical_crossentropy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    ignore_class: int = 0,
) -> jnp.ndarray:
    """Compute sparse categorical cross-entropy loss ignoring a specific class."""
    logits = logits.astype(jnp.float32)
    mask = labels != ignore_class
    labels_flat = labels.reshape(-1)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    mask_flat = mask.reshape(-1)

    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    indices = jnp.arange(labels_flat.shape[0])
    selected_log_probs = log_probs[indices, labels_flat]

    masked_log_probs = jnp.where(mask_flat, selected_log_probs, 0.0)
    total_log_prob = jnp.sum(masked_log_probs)
    count = jnp.sum(mask_flat.astype(jnp.float32))

    return jnp.where(count > 0, -total_log_prob / count, 0.0)


def init_model(
    model: nn.Module,
    rng: jax.random.PRNGKey,
    input_shape: Tuple[int, int] = (1, 128),
) -> Any:
    """Initialize model parameters."""
    dummy_input = jnp.ones(input_shape, dtype=jnp.int32)
    variables = model.init(rng, dummy_input, deterministic=True)
    return variables


def count_parameters(params) -> int:
    """Count the total number of parameters in a model."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
