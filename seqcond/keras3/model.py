"""SeqCond hybrid model for Keras 3 (ported from jax/model.py)."""

from typing import Optional

import numpy as np
import keras
from keras import ops, layers

from .norm import RMSNorm
from .rope import TransformerDecoderBlock, precompute_freqs, get_rope_embeddings
from .seqcond import SeqCondBlock


class SeqCondModel(keras.Model):
    """Hybrid SeqCond + Transformer language model.

    Interleaves SeqCond blocks (linear-time recurrence) with
    Transformer blocks (quadratic attention) at a configurable ratio.
    Supports weight-tied output projection.
    """

    def __init__(
        self,
        d_model=256,
        d_ff=768,
        num_layers=12,
        vocab_size=100300,
        maxlen=1024,
        seqcond_ratio=3,
        num_heads=8,
        num_kv_heads=None,
        seqcond_heads=None,
        num_query_heads=6,
        num_anchor_heads=0,
        num_thetas=4,
        dropout=0.0,
        tie_weights=True,
        qk_norm=False,
        qk_norm_eps=1e-6,
        conv_kernel_size=4,
        expand_factor=2.0,
        out_expand_factor=3,
        use_square_matrix=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers_total = num_layers
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.seqcond_ratio = seqcond_ratio
        self.num_heads = num_heads
        self.tie_weights = tie_weights

        _seqcond_heads = seqcond_heads if seqcond_heads is not None else num_heads

        # ── Embedding ───────────────────────────────────────────────────
        self.token_embedding = layers.Embedding(
            vocab_size, d_model, name="token_embedding"
        )

        # ── RoPE precomputed tables (numpy, converted on first call) ───
        self._cos_np, self._sin_np = precompute_freqs(maxlen, d_model // num_heads)

        # ── Blocks ──────────────────────────────────────────────────────
        self.block_types = []
        self.blocks_list = []
        transformer_idx = 0
        seqcond_idx = 0

        for i in range(num_layers):
            if (i + 1) % (seqcond_ratio + 1) == 0:
                block = TransformerDecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    num_kv_heads=num_kv_heads,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    qk_norm_eps=qk_norm_eps,
                    name=f"transformer_block_{transformer_idx}",
                )
                self.block_types.append("transformer")
                self.blocks_list.append(block)
                transformer_idx += 1
            else:
                block = SeqCondBlock(
                    num_heads=_seqcond_heads,
                    num_query_heads=num_query_heads,
                    num_thetas=num_thetas,
                    num_anchor_heads=num_anchor_heads,
                    conv_kernel_size=conv_kernel_size,
                    expand_factor=expand_factor,
                    out_expand_factor=out_expand_factor,
                    dropout=dropout,
                    maxlen=maxlen,
                    use_square_matrix=use_square_matrix,
                    name=f"seqcond_block_{seqcond_idx}",
                )
                self.block_types.append("seqcond")
                self.blocks_list.append(block)
                seqcond_idx += 1

        # ── Output projection ───────────────────────────────────────────
        if not tie_weights:
            self.output_dense = layers.Dense(
                vocab_size, use_bias=False, name="output_projection"
            )

    def call(self, inputs, training=False):
        b = ops.shape(inputs)[0]
        l = ops.shape(inputs)[1]
        mask = ops.cast(inputs != 0, "bool")

        x = self.token_embedding(inputs)

        cos, sin = get_rope_embeddings(l, self._cos_np, self._sin_np, b, self.num_heads)

        for btype, block in zip(self.block_types, self.blocks_list):
            if btype == "transformer":
                x = block(x, cos=cos, sin=sin, mask=mask, training=training)
            else:
                x = block(x, mask=mask, training=training)

        # Output logits
        if self.tie_weights:
            emb_w = self.token_embedding.embeddings  # (vocab, d_model)
            logits = ops.matmul(x, ops.transpose(emb_w))
        else:
            logits = self.output_dense(x)

        return logits

    def init_state(self, batch_size):
        """Initialize states for all blocks for autoregressive generation.

        Returns:
            List of states, one per block:
            - Transformer blocks: (k_cache, v_cache) both (B, maxlen, num_kv_heads, head_dim)
            - SeqCond blocks: (den_acc, re_acc, im_acc, pos, conv_buffer)
        """
        head_dim = self.d_model // self.num_heads

        states = []
        for btype, block in zip(self.block_types, self.blocks_list):
            if btype == "transformer":
                # Transformer: (k_cache, v_cache)
                # Get num_kv_heads from the block's attention layer
                num_kv_heads = block.attn._num_kv_heads
                k_cache = ops.zeros((batch_size, self.maxlen, num_kv_heads, head_dim))
                v_cache = ops.zeros_like(k_cache)
                states.append((k_cache, v_cache))
            else:
                # SeqCond: (den_acc, re_acc, im_acc, pos, conv_buffer)
                attn = block.attn
                K = attn.K
                H = attn.H
                M = attn.M
                den_acc = ops.zeros((batch_size, K))
                re_acc = ops.zeros((batch_size, K, H, M))
                im_acc = ops.zeros_like(re_acc)
                pos = ops.zeros((batch_size,))

                # Get dim_conv_total from built layer
                if hasattr(attn, "dim_mem_total"):
                    dim_conv_total = attn.dim_mem_total + attn.K_q * attn.H * attn.M * 2
                else:
                    # Fallback: compute from config
                    d_inner = int(
                        self.d_model
                        * (attn.expand_factor if hasattr(attn, "expand_factor") else 1)
                    )
                    H_calc = max(1, d_inner // (K * M))
                    dim_memory = K * H_calc
                    dim_query_total = attn.K_q * H_calc * M * 2
                    dim_conv_total = dim_memory + K + dim_query_total

                conv_buffer = ops.zeros(
                    (batch_size, attn.conv_kernel_size - 1, dim_conv_total),
                )
                states.append((den_acc, re_acc, im_acc, pos, conv_buffer))

        return states

    def step(self, token_id, states):
        """Single autoregressive step.

        Args:
            token_id: (B,) or (B, 1) token IDs
            states: List of states from init_state() or previous step()

        Returns:
            logits: (B, vocab_size) next token logits
            new_states: Updated states list
        """
        # Handle both (B,) and (B, 1) shapes
        if len(ops.shape(token_id)) == 2:
            token_id = ops.squeeze(token_id, axis=1)

        B = ops.shape(token_id)[0]

        # Get position from first SeqCond state
        pos = None
        for state, btype in zip(states, self.block_types):
            if btype == "seqcond":
                pos = state[3]  # pos is 4th element
                break

        if pos is None:
            # Pure transformer model
            pos = ops.zeros((B,), dtype="float32")

        # Embedding
        x = self.token_embedding(token_id)  # (B, d_model)

        # Clamp pos to valid range
        pos = ops.minimum(pos, float(self.maxlen - 1))

        # Get RoPE for current position
        pos_int = ops.cast(pos, "int32")
        cos_t = ops.take(self._cos_np, pos_int, axis=0)  # (B, head_dim)
        sin_t = ops.take(self._sin_np, pos_int, axis=0)

        # Expand for multi-head: (B, 1, num_heads, head_dim)
        cos_t = ops.expand_dims(ops.expand_dims(cos_t, 1), 1)
        sin_t = ops.expand_dims(ops.expand_dims(sin_t, 1), 1)
        cos_t = ops.tile(cos_t, [1, 1, self.num_heads, 1])
        sin_t = ops.tile(sin_t, [1, 1, self.num_heads, 1])

        # Save initial pos for transformer KV cache writes.
        # SeqCond blocks increment their own pos in-place during step(),
        # but all transformer blocks should write at the SAME position
        # (the current token's position), matching the forward pass behavior.
        transformer_pos = pos_int

        # Process through blocks
        new_states = []
        for btype, block, state in zip(self.block_types, self.blocks_list, states):
            if btype == "transformer":
                x, new_state = block.step(x, state, transformer_pos, cos_t, sin_t)
            else:
                x, new_state = block.step(x, state)
            new_states.append(new_state)

        # Output logits
        if self.tie_weights:
            emb_w = self.token_embedding.embeddings
            logits = ops.matmul(x, ops.transpose(emb_w))
        else:
            logits = self.output_dense(x)

        return logits, new_states


def create_seqcond_model(
    d_model=256,
    d_ff=768,
    num_layers=12,
    vocab_size=100300,
    maxlen=1024,
    seqcond_ratio=3,
    num_heads=8,
    num_kv_heads=None,
    seqcond_heads=None,
    num_query_heads=6,
    num_anchor_heads=0,
    num_thetas=4,
    dropout=0.0,
    tie_weights=True,
    qk_norm=False,
    qk_norm_eps=1e-6,
    conv_kernel_size=4,
    expand_factor=2.0,
    out_expand_factor=3,
    use_square_matrix=False,
):
    """Factory function for SeqCondModel."""
    return SeqCondModel(
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        vocab_size=vocab_size,
        maxlen=maxlen,
        seqcond_ratio=seqcond_ratio,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        seqcond_heads=seqcond_heads,
        num_query_heads=num_query_heads,
        num_anchor_heads=num_anchor_heads,
        num_thetas=num_thetas,
        dropout=dropout,
        tie_weights=tie_weights,
        qk_norm=qk_norm,
        qk_norm_eps=qk_norm_eps,
        conv_kernel_size=conv_kernel_size,
        expand_factor=expand_factor,
        out_expand_factor=out_expand_factor,
        use_square_matrix=use_square_matrix,
    )
