"""Depth-SeqCond: a SeqCond layer applied across the depth (layer) dimension.

The base model is frozen. We collect intermediate outputs [o_0, ..., o_N],
optionally compute deltas (o_{i+1} - o_i) or use raw outputs, feed them
through a single SeqCond block, and add the result to the final
representation before the LM head.

This is parameter-efficient: only the depth SeqCond block is trained.
"""

import keras
from keras import ops, layers

from .seqcond import SeqCondBlock
from .norm import RMSNorm
from .rope import get_rope_embeddings


class DepthSeqCondModel(keras.Model):
    """Wrapper that adds a depth-SeqCond layer on top of a frozen base model.

    Args:
        base_model: Frozen SeqCondModel instance.
        use_deltas: If True (default), the depth sequence is the per-layer
            residual deltas (o_{i+1} - o_i). If False, use the raw layer
            outputs [o_0, ..., o_{N-1}] (excluding final output to keep
            the same sequence length N).
        depth_num_heads, ...: SeqCond block hyperparameters.
        depth_scale_init: Initial value for the gating sigmoid logit.
            0.0 → sigmoid = 0.5 (moderate contribution at init).
    """

    def __init__(
        self,
        base_model,
        use_deltas=True,
        depth_num_heads=8,
        depth_num_query_heads=4,
        depth_num_thetas=4,
        depth_expand_factor=1.0,
        depth_out_expand_factor=3,
        depth_conv_kernel_size=4,
        depth_dropout=0.0,
        depth_scale_init=0.0,
        chunk_size=512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base = base_model
        self.d_model = base_model.d_model
        self.num_depth_layers = base_model.num_layers_total
        self.use_deltas = use_deltas
        self.chunk_size = chunk_size

        # Freeze base model
        for w in self.base.weights:
            w.trainable = False

        # The single depth SeqCond block (the only trainable part)
        self.depth_block = SeqCondBlock(
            num_heads=depth_num_heads,
            num_query_heads=depth_num_query_heads,
            num_thetas=depth_num_thetas,
            expand_factor=depth_expand_factor,
            out_expand_factor=depth_out_expand_factor,
            conv_kernel_size=depth_conv_kernel_size,
            dropout=depth_dropout,
            maxlen=self.num_depth_layers + 8,  # depth seq len + margin
            name="depth_seqcond",
        )

        # Learnable scalar to gate the depth residual (init near zero)
        self.depth_gate = self.add_weight(
            name="depth_gate",
            shape=(1,),
            initializer=keras.initializers.Constant(depth_scale_init),
        )

    def _collect_intermediates(self, inputs, training=False):
        """Run base model and return (final_output, intermediates list).

        IMPORTANT: We detach/stop_gradient on all intermediates to prevent
        the gradient graph from being retained through the frozen base model.
        This is critical for memory efficiency.
        """
        import keras

        base = self.base
        mask = ops.cast(inputs != 0, "bool")
        x = base.token_embedding(inputs)
        b = ops.shape(inputs)[0]
        l = ops.shape(inputs)[1]
        cos, sin = get_rope_embeddings(l, base._cos_np, base._sin_np, b, base.num_heads)

        # Detach embedding output - no gradients needed through frozen base
        if keras.backend.backend() == "torch":
            x = x.detach()
        else:
            x = ops.stop_gradient(x)

        intermediates = [x]  # o_0 = embedding output

        for btype, block in zip(base.block_types, base.blocks_list):
            if btype == "transformer":
                x = block(x, cos=cos, sin=sin, mask=mask, training=False)
            else:
                x = block(x, mask=mask, training=False)

            # Detach each intermediate to cut gradient graph
            if keras.backend.backend() == "torch":
                x = x.detach()
            else:
                x = ops.stop_gradient(x)

            intermediates.append(x)

        # intermediates: [o_0, o_1, ..., o_N]  (N+1 elements, N = num_layers)
        return x, intermediates

    def call(self, inputs, training=False):
        base = self.base
        b = ops.shape(inputs)[0]
        l = ops.shape(inputs)[1]

        # ── 1. Run base model, collecting intermediate outputs ────────
        x, intermediates = self._collect_intermediates(inputs)

        # ── 2. Build depth sequence ───────────────────────────────────
        if self.use_deltas:
            # delta_i = o_{i+1} - o_i  →  N deltas
            seq_items = []
            for i in range(len(intermediates) - 1):
                seq_items.append(intermediates[i + 1] - intermediates[i])
        else:
            # Raw outputs [o_0, ..., o_{N-1}]  (N items, exclude final)
            seq_items = intermediates[:-1]

        depth_seq = ops.stack(seq_items, axis=2)  # (B, T, N, d_model)

        # ── 3. Process in chunks to avoid OOM ─────────────────────────
        # Instead of (B*T, N, d_model), process chunks of (B*chunk_size, N, d_model)
        chunk_size = self.chunk_size
        depth_summaries = []

        for start_t in range(0, l, chunk_size):
            end_t = min(start_t + chunk_size, l)
            chunk = depth_seq[:, start_t:end_t, :, :]  # (B, chunk_len, N, d_model)
            chunk_len = end_t - start_t

            # Reshape chunk: (B*chunk_len, N, d_model)
            chunk_flat = ops.reshape(
                chunk, (b * chunk_len, self.num_depth_layers, self.d_model)
            )

            # Apply depth SeqCond block
            chunk_out = self.depth_block(chunk_flat, training=training)

            # Take last depth position: (B*chunk_len, d_model)
            chunk_summary = chunk_out[:, -1, :]

            # Reshape back: (B, chunk_len, d_model)
            chunk_summary = ops.reshape(chunk_summary, (b, chunk_len, self.d_model))
            depth_summaries.append(chunk_summary)

        # Concatenate all chunks: (B, T, d_model)
        depth_summary = ops.concatenate(depth_summaries, axis=1)

        # ── 4. Add gated residual ─────────────────────────────────────
        gate = ops.sigmoid(self.depth_gate)
        x = x + gate * depth_summary

        # ── 5. Project to logits (frozen weights) ─────────────────────
        if base.tie_weights:
            emb_w = base.token_embedding.embeddings
            logits = ops.matmul(x, ops.transpose(emb_w))
        else:
            logits = base.output_dense(x)

        return logits

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, **kwargs):
        """Masked cross-entropy loss (ignore padding token 0)."""
        mask = ops.cast(y != 0, "float32")
        per_token = keras.losses.sparse_categorical_crossentropy(
            y, y_pred, from_logits=True
        )
        return ops.sum(per_token * mask) / ops.maximum(ops.sum(mask), 1.0)
