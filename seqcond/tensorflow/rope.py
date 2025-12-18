import math

import numpy as np
import tensorflow as tf

from .norm import RMSNorm


class RotaryPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen: int, head_dim: int, **kw):
        if head_dim % 2:
            raise ValueError("head_dim must be even")
        super().__init__(**kw)
        self.maxlen = maxlen
        self.head_dim = head_dim

    def build(self, _):
        half_d = self.head_dim // 2

        pos = np.arange(self.maxlen)[:, None]
        dim = np.arange(half_d)[None, :]
        inv = 1.0 / (10000 ** (dim / half_d))
        angles = pos * inv

        cos = np.cos(angles).astype(np.float32)
        sin = np.sin(angles).astype(np.float32)

        self.cos_emb = self.add_weight(
            name="cos",
            shape=cos.shape,
            initializer=tf.constant_initializer(cos),
            trainable=False,
        )
        self.sin_emb = self.add_weight(
            name="sin",
            shape=sin.shape,
            initializer=tf.constant_initializer(sin),
            trainable=False,
        )

    def call(self, x, *, n_heads: int | None = None):
        b = tf.shape(x)[0]
        l = tf.shape(x)[1]
        cos, sin = self.cos_emb[:l], self.sin_emb[:l]
        if n_heads is None:
            cos = tf.tile(cos[None, ...], [b, 1, 1])
            sin = tf.tile(sin[None, ...], [b, 1, 1])
        else:
            cos = tf.tile(cos[None, :, None, :], [b, 1, n_heads, 1])
            sin = tf.tile(sin[None, :, None, :], [b, 1, n_heads, 1])
        return cos, sin


def apply_rope(tensor, cos, sin):
    dim = tensor.shape[-1] // 2
    x1, x2 = tensor[..., :dim], tensor[..., dim:]
    rot = tf.concat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return tf.reshape(rot, tf.shape(tensor))


class RotarySelfAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model,
        num_heads,
        num_kv_heads=None,
        dropout=0.0,
        qk_norm=False,
        qk_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_groups = num_heads // self.num_kv_heads
        self.head_dim = d_model // num_heads
        self.qk_norm = bool(qk_norm)
        self.qk_norm_eps = float(qk_norm_eps)

        self.q_proj = tf.keras.layers.Dense(d_model, use_bias=False)
        self.k_proj = tf.keras.layers.Dense(
            self.num_kv_heads * self.head_dim, use_bias=False
        )
        self.v_proj = tf.keras.layers.Dense(
            self.num_kv_heads * self.head_dim, use_bias=False
        )
        self.out_proj = tf.keras.layers.Dense(d_model, use_bias=False)
        self.attn_dropout = tf.keras.layers.Dropout(dropout)

    def _repeat_kv(self, x):
        """Repeat KV heads to match Q heads for GQA."""
        if self.num_groups == 1:
            return x
        b = tf.shape(x)[0]
        l = tf.shape(x)[1]
        x = tf.reshape(x, [b, l, self.num_kv_heads, 1, self.head_dim])
        x = tf.tile(x, [1, 1, 1, self.num_groups, 1])
        return tf.reshape(x, [b, l, self.num_heads, self.head_dim])

    def call(self, x, cos, sin, training=False, mask=None):
        b = tf.shape(x)[0]
        l = tf.shape(x)[1]

        q = tf.reshape(self.q_proj(x), [b, l, self.num_heads, self.head_dim])
        k = tf.reshape(self.k_proj(x), [b, l, self.num_kv_heads, self.head_dim])
        v = tf.reshape(self.v_proj(x), [b, l, self.num_kv_heads, self.head_dim])

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.qk_norm:
            q_f32 = tf.cast(q, tf.float32)
            k_f32 = tf.cast(k, tf.float32)
            q_ms = tf.reduce_mean(tf.square(q_f32), axis=-1, keepdims=True)
            k_ms = tf.reduce_mean(tf.square(k_f32), axis=-1, keepdims=True)
            q = tf.cast(q_f32 * tf.math.rsqrt(q_ms + self.qk_norm_eps), q.dtype)
            k = tf.cast(k_f32 * tf.math.rsqrt(k_ms + self.qk_norm_eps), k.dtype)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        scores = tf.einsum("blhd,bmhd->bhlm", q, k) / tf.math.sqrt(
            tf.cast(self.head_dim, tf.float32)
        )

        causal = tf.linalg.band_part(tf.ones((l, l)), -1, 0)
        causal_mask = tf.cast(causal, tf.bool)[None, None, :, :]

        if mask is not None:
            key_mask = tf.cast(mask, tf.float32)[:, None, None, :]
            scores = scores + (1.0 - key_mask) * -1e9

        scores = tf.where(causal_mask, scores, tf.fill(tf.shape(scores), -1e9))
        attn = tf.nn.softmax(scores, axis=-1)
        attn = self.attn_dropout(attn, training=training)

        out = tf.einsum("bhql,blhd->bqhd", attn, v)
        out = tf.reshape(out, [b, l, self.d_model])
        return self.out_proj(out)


class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        num_kv_heads=None,
        dropout=0.0,
        norm_eps=1e-6,
        qk_norm=False,
        qk_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm1 = RMSNorm(epsilon=norm_eps)
        self.attn = RotarySelfAttention(
            d_model,
            num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
        )
        self.drop1 = tf.keras.layers.Dropout(dropout)

        self.norm2 = RMSNorm(epsilon=norm_eps)
        self.ff_in = tf.keras.layers.Dense(2 * d_ff)
        self.ff_out = tf.keras.layers.Dense(d_model)
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, cos, sin, training=False, mask=None):
        y = self.norm1(x)
        y = self.attn(y, cos=cos, sin=sin, training=training, mask=mask)
        x = x + self.drop1(y, training=training)

        y = self.norm2(x)
        u, v = tf.split(self.ff_in(y), 2, axis=-1)
        y = tf.nn.swish(v) * u
        y = self.ff_out(y)
        return x + self.drop2(y, training=training)
