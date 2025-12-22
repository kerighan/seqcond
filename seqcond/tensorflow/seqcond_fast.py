import numpy as np
import tensorflow as tf

from .norm import RMSNorm


class SeqCondAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads=32,
        key_heads=None,
        num_anchor_heads=4,
        num_thetas=4,
        derivative_order=2,
        derivative_aggregation="re_im",
        dropout=0.0,
        use_conv=True,
        conv_kernel_size=4,
        conv_kernel=None,
        **layer_kwargs,
    ):
        super().__init__(**layer_kwargs)

        if key_heads is not None and int(key_heads) != int(num_heads):
            raise ValueError(
                "key_heads and num_heads must match when both are provided"
            )
        if conv_kernel is not None:
            conv_kernel_size = conv_kernel_size if conv_kernel_size is not None else 4
            if int(conv_kernel) != int(conv_kernel_size):
                raise ValueError(
                    "conv_kernel and conv_kernel_size must match when both are provided"
                )

        self.K = int(num_heads)
        self.num_anchor_heads = int(num_anchor_heads)
        self.M = int(num_thetas)
        self.derivative_order = int(derivative_order)
        self.derivative_aggregation = derivative_aggregation
        self.dropout = float(dropout)
        self.use_conv = bool(use_conv)
        self.conv_kernel = int(conv_kernel_size)

        if self.K <= 0:
            raise ValueError("num_heads must be > 0")
        if self.num_anchor_heads < 0:
            raise ValueError("num_anchor_heads must be >= 0")
        if self.num_anchor_heads > self.K:
            raise ValueError(
                f"num_anchor_heads ({self.num_anchor_heads}) > num_heads ({self.K})"
            )
        if self.M <= 0:
            raise ValueError("num_thetas must be > 0")
        if self.derivative_order < 0:
            raise ValueError("derivative_order must be >= 0")
        if self.derivative_order > 2:
            raise ValueError("derivative_order > 2 is not supported")
        if self.derivative_aggregation not in ("re_im", "cos_sin"):
            raise ValueError("derivative_aggregation must be 're_im' or 'cos_sin'")
        if self.conv_kernel <= 0:
            raise ValueError("conv_kernel_size must be > 0")
        if self.dropout < 0.0:
            raise ValueError("dropout must be >= 0")

        self.num_decay_heads = self.K - self.num_anchor_heads

    def build(self, input_shape):
        d_model = input_shape[-1]
        self.d_inner = d_model
        self.H = max(1, self.d_inner // self.K)

        total_dim = self.d_inner * 2 + self.K
        self.in_proj = tf.keras.layers.Dense(total_dim, name="in_proj", use_bias=False)

        if self.use_conv:
            self.causal_pad = tf.keras.layers.ZeroPadding1D(
                padding=(self.conv_kernel - 1, 0)
            )
            self.conv1d = tf.keras.layers.DepthwiseConv1D(
                kernel_size=self.conv_kernel, padding="valid", use_bias=True
            )

        # Theta initialization: [-π/3, π/3] to avoid gradient dead zones
        # cos/sin have zero gradients at multiples of π/2, so we stay within safe range
        grid = np.linspace(-np.pi / 3, np.pi / 3, self.M, dtype=np.float32)

        head_scale = np.ones((1, 1, self.K, 1, 1), dtype=np.float32)
        base = np.tile(grid.reshape(1, 1, 1, 1, self.M), (1, 1, self.K, self.H, 1))
        init = head_scale * base

        self.theta = self.add_weight(
            "theta",
            shape=(1, 1, self.K, self.H, self.M),
            initializer=tf.constant_initializer(init),
            trainable=True,
        )

        if self.num_decay_heads > 0:
            rates = np.geomspace(0.001, 0.1, self.num_decay_heads).astype(np.float32)
            self.decay_slopes = self.add_weight(
                "decay_slopes",
                shape=(self.num_decay_heads,),
                initializer=tf.constant_initializer(np.log(np.exp(rates) - 1)),
                trainable=True,
            )
        if self.num_anchor_heads > 0:
            rates = np.geomspace(0.01, 0.1, self.num_anchor_heads).astype(np.float32)
            self.anchor_slopes = self.add_weight(
                "anchor_slopes",
                shape=(self.num_anchor_heads,),
                initializer=tf.constant_initializer(np.log(np.exp(rates) - 1)),
                trainable=True,
            )

        self.score_scale = self.add_weight(
            "score_scale", shape=(self.K,), initializer="ones", trainable=True
        )
        if self.derivative_order > 0:
            self.deriv_logits = self.add_weight(
                "deriv_logits",
                shape=(self.derivative_order + 1,),
                initializer=tf.constant_initializer(
                    [5.0] + [0.0] * self.derivative_order
                ),
                trainable=True,
            )

        norm_dim = self.H * 2 * self.M
        self.norm_scale = self.add_weight(
            "norm_scale", shape=(norm_dim,), initializer="ones", trainable=True
        )
        self.norm_eps = 1e-5

        split_dim = self.H * self.M
        self.W_re = self.add_weight(
            "W_re",
            shape=(split_dim, self.H),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.W_im = self.add_weight(
            "W_im",
            shape=(split_dim, self.H),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.out_proj = tf.keras.layers.Dense(d_model, use_bias=False, name="out_proj")

        if self.dropout:
            self.drop = tf.keras.layers.Dropout(self.dropout)
        super().build(input_shape)

    def call(self, x, mask=None, training=False):
        b, l = tf.shape(x)[0], tf.shape(x)[1]
        k, h = self.K, self.H

        z = self.in_proj(x)
        if self.use_conv:
            z = self.conv1d(self.causal_pad(z))
        else:
            z = tf.nn.silu(z)

        x_val = z[..., : self.d_inner]
        x_gate = z[..., self.d_inner : 2 * self.d_inner]
        s_raw = z[..., -self.K :]

        x_val = tf.reshape(x_val, [b, l, k, h])
        s_raw = tf.reshape(s_raw, [b, l, k, 1])
        x_gate = tf.nn.silu(x_gate)

        if mask is not None:
            m = tf.cast(mask, x.dtype)[:, :, None, None]
            s_raw = s_raw * m
            x_val = x_val * m

        pos_i32 = tf.range(l, dtype=tf.int32)
        pos_f32 = tf.cast(pos_i32, tf.float32)
        w_list = []
        if self.num_decay_heads > 0:
            slopes = tf.nn.softplus(
                tf.reshape(self.decay_slopes, [1, 1, self.num_decay_heads, 1])
            )
            # Use fixed sequence length l (not mask-based lengths) for train/inference consistency
            dist = tf.cast(l - 1, tf.float32) - pos_f32
            dist = tf.maximum(dist, 0.0)
            dist = dist[None, :, None, None]
            slopes = tf.cast(slopes, tf.float32)
            w_list.append(tf.exp(-slopes * dist))
        if self.num_anchor_heads > 0:
            slopes = tf.nn.softplus(
                tf.reshape(self.anchor_slopes, [1, 1, self.num_anchor_heads, 1])
            )
            slopes = tf.cast(slopes, tf.float32)
            dist = pos_f32[None, :, None, None]
            w_list.append(tf.exp(-slopes * dist))
        time_weight = tf.concat(w_list, axis=2)
        time_weight = tf.cast(time_weight, x.dtype)

        p = tf.exp(
            tf.clip_by_value(self.score_scale[None, None, :, None] * s_raw, -20.0, 20.0)
        )
        if mask is not None:
            p = p * tf.cast(mask, x.dtype)[:, :, None, None]

        x_val5 = tf.reshape(x_val, [b, l, k, h, 1])
        phi = x_val5 * tf.cast(self.theta, x.dtype)
        phi_f32 = tf.cast(phi, tf.float32)
        cos_b = tf.cast(tf.cos(phi_f32), x.dtype)
        sin_b = tf.cast(tf.sin(phi_f32), x.dtype)

        if self.derivative_order == 0:
            re_m, im_m = cos_b, sin_b
        else:
            w = tf.nn.softmax(self.deriv_logits)
            acc = -tf.square(x_val5) if self.derivative_order == 2 else 0.0

            if self.derivative_aggregation == "re_im":
                poly = (
                    w[0]
                    + w[1] * x_val5
                    + (w[2] * acc if self.derivative_order > 1 else 0.0)
                )
                re_m = poly * cos_b
                im_m = poly * sin_b
            else:
                mod = (
                    w[0]
                    + w[1] * x_val5
                    + (w[2] * acc if self.derivative_order > 1 else 0.0)
                )
                re_m = mod * cos_b
                im_mod = (
                    w[0]
                    - w[1] * x_val5
                    + (w[2] * acc if self.derivative_order > 1 else 0.0)
                )
                im_m = im_mod * sin_b

        p_w = (p * time_weight)[..., None]
        flat_shape = [b, l, k * h * self.M]
        merged = tf.concat(
            [
                tf.reshape(p_w * re_m, flat_shape),
                tf.reshape(p_w * im_m, flat_shape),
                tf.reshape(tf.broadcast_to(p_w, [b, l, k, h, self.M]), flat_shape),
            ],
            axis=-1,
        )

        cumsum = tf.cumsum(merged, axis=1)
        num_re, num_im, den = tf.split(cumsum, 3, axis=-1)

        inv_den = tf.math.reciprocal(tf.maximum(den, tf.cast(1e-4, den.dtype)))
        re = num_re * inv_den
        im = num_im * inv_den

        re_flat = tf.reshape(re, [b, l, k, h * self.M])
        im_flat = tf.reshape(im, [b, l, k, h * self.M])

        re_flat_f32 = tf.cast(re_flat, tf.float32)
        im_flat_f32 = tf.cast(im_flat, tf.float32)
        mean_sq_re = tf.reduce_sum(tf.square(re_flat_f32), axis=-1)
        mean_sq_im = tf.reduce_sum(tf.square(im_flat_f32), axis=-1)

        inv_total_dim = 1.0 / (2.0 * float(h * self.M))
        mean_sq = (mean_sq_re + mean_sq_im) * inv_total_dim
        rsqrt = tf.cast(tf.math.rsqrt(mean_sq[..., None] + self.norm_eps), x.dtype)

        split_idx = self.H * self.M
        scale_re = self.norm_scale[:split_idx]
        scale_im = self.norm_scale[split_idx:]

        re_norm = re_flat * rsqrt * scale_re
        y_re = tf.matmul(re_norm, self.W_re)

        im_norm = im_flat * rsqrt * scale_im
        y_im = tf.matmul(im_norm, self.W_im)

        y_per_head = y_re + y_im
        y = tf.reshape(y_per_head, [b, l, self.d_inner])

        out = self.out_proj(y * x_gate)
        if self.dropout:
            out = self.drop(out, training=training)
        return out


class SeqCondBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads=8,
        key_heads=None,
        num_thetas=4,
        num_anchor_heads=0,
        derivative_order=0,
        dropout=0.0,
        use_conv=True,
        conv_kernel_size=4,
        conv_kernel=None,
        norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.norm = RMSNorm(epsilon=norm_eps)
        self.mixer = SeqCondAttention(
            num_heads=num_heads,
            key_heads=key_heads,
            num_thetas=num_thetas,
            num_anchor_heads=num_anchor_heads,
            derivative_order=derivative_order,
            dropout=dropout,
            use_conv=use_conv,
            conv_kernel_size=conv_kernel_size,
            conv_kernel=conv_kernel,
        )

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, training=False, mask=None):
        residual = x
        x = self.norm(x)
        x = self.mixer(x, mask=mask, training=training)
        return x + residual
