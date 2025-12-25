from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from .norm import RMSNorm


# class SeqCondAttention(nn.Module):
#     num_heads: int = 32
#     key_heads: Optional[int] = None
#     num_anchor_heads: int = 4
#     num_thetas: int = 4
#     derivative_order: int = 2
#     derivative_aggregation: str = "re_im"
#     dropout: float = 0.0
#     use_conv: bool = True
#     conv_kernel_size: int = 4
#     conv_kernel: Optional[int] = None
#     maxlen: Optional[int] = None  # if set, fixes distance calc to a constant length

#     def setup(self):
#         if self.key_heads is not None and int(self.key_heads) != int(self.num_heads):
#             raise ValueError(
#                 "key_heads and num_heads must match when both are provided"
#             )
#         conv_kernel_size = self.conv_kernel_size
#         if self.conv_kernel is not None:
#             if int(self.conv_kernel) != int(conv_kernel_size):
#                 raise ValueError(
#                     "conv_kernel and conv_kernel_size must match when both are provided"
#                 )
#             conv_kernel_size = self.conv_kernel

#         self.K = int(self.num_heads)
#         self._num_anchor_heads = int(self.num_anchor_heads)
#         self.M = int(self.num_thetas)
#         self._derivative_order = int(self.derivative_order)
#         self._derivative_aggregation = self.derivative_aggregation
#         self._dropout = float(self.dropout)
#         self._use_conv = bool(self.use_conv)
#         self._conv_kernel = int(conv_kernel_size)

#         if self.K <= 0:
#             raise ValueError("num_heads must be > 0")
#         if self._num_anchor_heads < 0:
#             raise ValueError("num_anchor_heads must be >= 0")
#         if self._num_anchor_heads > self.K:
#             raise ValueError(
#                 f"num_anchor_heads ({self._num_anchor_heads}) > num_heads ({self.K})"
#             )
#         if self.M <= 0:
#             raise ValueError("num_thetas must be > 0")
#         if self._derivative_order < 0:
#             raise ValueError("derivative_order must be >= 0")
#         if self._derivative_order > 2:
#             raise ValueError("derivative_order > 2 is not supported")
#         if self._derivative_aggregation not in ("re_im", "cos_sin"):
#             raise ValueError("derivative_aggregation must be 're_im' or 'cos_sin'")
#         if self._conv_kernel <= 0:
#             raise ValueError("conv_kernel_size must be > 0")
#         if self._dropout < 0.0:
#             raise ValueError("dropout must be >= 0")

#         self.num_decay_heads = self.K - self._num_anchor_heads

#     @nn.compact
#     def __call__(
#         self,
#         x: jnp.ndarray,
#         mask: Optional[jnp.ndarray] = None,
#         deterministic: bool = True,
#     ) -> jnp.ndarray:
#         b, l, d_model = x.shape
#         d_inner = d_model
#         H = max(1, d_inner // self.K)
#         k, h = self.K, H

#         total_dim = d_inner * 2 + self.K
#         in_proj = nn.Dense(total_dim, use_bias=False, name="in_proj")
#         z = in_proj(x)

#         if self._use_conv:
#             pad_width = ((0, 0), (self._conv_kernel - 1, 0), (0, 0))
#             z_padded = jnp.pad(z, pad_width, mode="constant", constant_values=0)
#             conv1d = nn.Conv(
#                 features=z.shape[-1],
#                 kernel_size=(self._conv_kernel,),
#                 padding="VALID",
#                 use_bias=True,
#                 feature_group_count=z.shape[-1],
#                 name="conv1d",
#             )
#             z = conv1d(z_padded)
#         else:
#             z = jax.nn.silu(z)

#         x_val = z[..., :d_inner]
#         x_gate = z[..., d_inner : 2 * d_inner]
#         s_raw = z[..., -self.K :]

#         x_val = x_val.reshape(b, l, k, h)
#         s_raw = s_raw.reshape(b, l, k, 1)
#         x_gate = jax.nn.silu(x_gate)

#         if mask is not None:
#             m = mask.astype(x.dtype)[:, :, None, None]
#             s_raw = s_raw * m
#             x_val = x_val * m

#         pos_f32 = jnp.arange(l, dtype=jnp.float32)

#         # Theta initialization: [-π/3, π/3] to avoid gradient dead zones
#         # cos/sin have zero gradients at multiples of π/2, so we stay within safe range
#         grid = np.linspace(-np.pi / 3, np.pi / 3, self.M, dtype=np.float32)

#         head_scale = np.ones((1, 1, self.K, 1, 1), dtype=np.float32)
#         base = np.tile(grid.reshape(1, 1, 1, 1, self.M), (1, 1, self.K, H, 1))
#         init = head_scale * base

#         theta = self.param(
#             "theta",
#             lambda rng, shape: jnp.array(init),
#             (1, 1, self.K, H, self.M),
#         )

#         w_list = []
#         if self.num_decay_heads > 0:
#             rates = np.geomspace(0.001, 0.1, self.num_decay_heads).astype(np.float32)
#             decay_slopes = self.param(
#                 "decay_slopes",
#                 lambda rng, shape: jnp.array(np.log(np.exp(rates) - 1)),
#                 (self.num_decay_heads,),
#             )
#             slopes = jax.nn.softplus(
#                 decay_slopes.reshape(1, 1, self.num_decay_heads, 1)
#             )
#             # Use fixed sequence length for train/inference consistency (avoids length-dependent shifts)
#             length_const = self.maxlen if self.maxlen is not None else l
#             dist = jnp.float32(length_const - 1) - pos_f32
#             dist = jnp.maximum(dist, 0.0)
#             dist = dist[None, :, None, None]
#             slopes = slopes.astype(jnp.float32)
#             w_list.append(jnp.exp(-slopes * dist))

#         if self._num_anchor_heads > 0:
#             rates = np.geomspace(0.01, 0.1, self._num_anchor_heads).astype(np.float32)
#             anchor_slopes = self.param(
#                 "anchor_slopes",
#                 lambda rng, shape: jnp.array(np.log(np.exp(rates) - 1)),
#                 (self._num_anchor_heads,),
#             )
#             slopes = jax.nn.softplus(
#                 anchor_slopes.reshape(1, 1, self._num_anchor_heads, 1)
#             )
#             slopes = slopes.astype(jnp.float32)
#             dist = pos_f32[None, :, None, None]
#             w_list.append(jnp.exp(-slopes * dist))

#         time_weight = jnp.concatenate(w_list, axis=2)
#         time_weight = time_weight.astype(x.dtype)

#         score_scale = self.param("score_scale", nn.initializers.ones, (self.K,))
#         p = jnp.exp(jnp.clip(score_scale[None, None, :, None] * s_raw, -20.0, 20.0))
#         if mask is not None:
#             p = p * mask.astype(x.dtype)[:, :, None, None]

#         x_val5 = x_val.reshape(b, l, k, h, 1)
#         phi = x_val5 * theta.astype(x.dtype)
#         phi_f32 = phi.astype(jnp.float32)
#         cos_b = jnp.cos(phi_f32).astype(x.dtype)
#         sin_b = jnp.sin(phi_f32).astype(x.dtype)

#         if self._derivative_order == 0:
#             re_m, im_m = cos_b, sin_b
#         elif self._derivative_order == 1:
#             re_m, im_m = x_val5 * sin_b, x_val5 * cos_b
#         else:
#             deriv_logits = self.param(
#                 "deriv_logits",
#                 lambda rng, shape: jnp.array([5.0] + [0.0] * self._derivative_order),
#                 (self._derivative_order + 1,),
#             )
#             w = jax.nn.softmax(deriv_logits)
#             acc = -jnp.square(x_val5) if self._derivative_order == 2 else 0.0

#             if self._derivative_aggregation == "re_im":
#                 poly = (
#                     w[0]
#                     + w[1] * x_val5
#                     + (w[2] * acc if self._derivative_order > 1 else 0.0)
#                 )
#                 re_m = poly * cos_b
#                 im_m = poly * sin_b
#             else:
#                 mod = (
#                     w[0]
#                     + w[1] * x_val5
#                     + (w[2] * acc if self._derivative_order > 1 else 0.0)
#                 )
#                 re_m = mod * cos_b
#                 im_mod = (
#                     w[0]
#                     - w[1] * x_val5
#                     + (w[2] * acc if self._derivative_order > 1 else 0.0)
#                 )
#                 im_m = im_mod * sin_b

#         p_w = (p * time_weight)[..., None]
#         flat_shape = (b, l, k * h * self.M)
#         merged = jnp.concatenate(
#             [
#                 (p_w * re_m).reshape(flat_shape),
#                 (p_w * im_m).reshape(flat_shape),
#                 jnp.broadcast_to(p_w, (b, l, k, h, self.M)).reshape(flat_shape),
#             ],
#             axis=-1,
#         )

#         cumsum = jnp.cumsum(merged, axis=1)
#         num_re, num_im, den = jnp.split(cumsum, 3, axis=-1)

#         inv_den = 1.0 / jnp.maximum(den, jnp.float32(1e-4))
#         re = num_re * inv_den
#         im = num_im * inv_den

#         re_flat = re.reshape(b, l, k, h * self.M)
#         im_flat = im.reshape(b, l, k, h * self.M)

#         re_flat_f32 = re_flat.astype(jnp.float32)
#         im_flat_f32 = im_flat.astype(jnp.float32)
#         mean_sq_re = jnp.sum(jnp.square(re_flat_f32), axis=-1)
#         mean_sq_im = jnp.sum(jnp.square(im_flat_f32), axis=-1)

#         norm_dim = H * 2 * self.M
#         norm_scale = self.param("norm_scale", nn.initializers.ones, (norm_dim,))
#         norm_eps = 1e-5

#         inv_total_dim = 1.0 / (2.0 * float(h * self.M))
#         mean_sq = (mean_sq_re + mean_sq_im) * inv_total_dim
#         rsqrt = jax.lax.rsqrt(mean_sq[..., None] + norm_eps).astype(x.dtype)

#         split_idx = H * self.M
#         scale_re = norm_scale[:split_idx]
#         scale_im = norm_scale[split_idx:]

#         W_re = self.param(
#             "W_re",
#             nn.initializers.glorot_uniform(),
#             (H * self.M, H),
#         )
#         W_im = self.param(
#             "W_im",
#             nn.initializers.glorot_uniform(),
#             (H * self.M, H),
#         )

#         re_norm = re_flat * rsqrt * scale_re
#         y_re = jnp.matmul(re_norm, W_re)

#         im_norm = im_flat * rsqrt * scale_im
#         y_im = jnp.matmul(im_norm, W_im)

#         y_per_head = y_re + y_im
#         y = y_per_head.reshape(b, l, d_inner)

#         out_proj = nn.Dense(d_model, use_bias=False, name="out_proj")
#         out = out_proj(y * x_gate)

#         if self._dropout > 0.0:
#             drop = nn.Dropout(self._dropout)
#             out = drop(out, deterministic=deterministic)

#         return out

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Optional

class SeqCondAttention(nn.Module):
    num_heads: int = 32
    key_heads: Optional[int] = None
    num_anchor_heads: int = 4
    num_thetas: int = 1
    derivative_order: int = 2
    conv_kernel_size: int = 4
    expand_factor: int = 2
    dropout: float = 0.0
    maxlen: Optional[int] = None

    def setup(self):
        self.K = self.key_heads if self.key_heads is not None else self.num_heads
        self.M = self.num_thetas
        self.num_decay_heads = self.K - self.num_anchor_heads

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        b, l, d_model = x.shape
        d_inner = int(d_model * self.expand_factor)
        H = max(1, d_inner // self.K)

        # --- 1. PROJECTION & CONVOLUTION ---
        z = nn.Dense(d_inner * 2 + self.K, use_bias=False, name="in_proj")(x)
        
        # Padding natif dans la conv (évite la copie mémoire)
        z = nn.Conv(
            features=z.shape[-1], 
            kernel_size=(self.conv_kernel_size,), 
            padding=((self.conv_kernel_size - 1, 0),), 
            feature_group_count=z.shape[-1], 
            name="conv"
        )(z)

        x_val = z[..., :d_inner].reshape(b, l, self.K, H)
        x_gate = jax.nn.silu(z[..., d_inner : 2 * d_inner])
        s_raw = z[..., -self.K:].reshape(b, l, self.K, 1)

        if mask is not None:
            m = mask.astype(x.dtype)[:, :, None, None]
            s_raw *= m
            x_val *= m

        # --- 2. PARAMETERS (Time & Frequency) ---
        # Theta Init (Robuste M=1)
        def init_theta(key, shape):
            if self.M == 1:
                # Force shape explicite (1, 1, K, H, 1)
                grid = np.linspace(0.1, 1.5, self.K * H).reshape(1, 1, self.K, H, 1)
                signs = np.resize([1., -1.], grid.shape)
                return jnp.array(grid * signs, dtype=jnp.float32)
            else:
                grid = np.linspace(-np.pi/3, np.pi/3, self.M)
                base = np.tile(grid.reshape(1, 1, 1, 1, self.M), (1, 1, self.K, H, 1))
                return jnp.array(base, dtype=jnp.float32)

        theta = self.param("theta", init_theta, (1, 1, self.K, H, self.M))
        
        # Decay Weights
        pos = jnp.arange(l, dtype=jnp.float32)
        w_list = []
        
        if self.num_decay_heads > 0:
            d_slopes = self.param("decay_slopes", lambda r,s: jnp.log(jnp.exp(np.geomspace(0.001, 0.1, s[0]))-1), (self.num_decay_heads,))
            slopes = jax.nn.softplus(d_slopes).reshape(1, 1, -1, 1)
            dist = jnp.maximum(jnp.float32((self.maxlen or l) - 1) - pos, 0.)
            # Shape: (1, L, num_decay, 1)
            w_list.append(jnp.exp(-slopes * dist[None, :, None, None]))

        if self.num_anchor_heads > 0:
            a_slopes = self.param("anchor_slopes", lambda r,s: jnp.log(jnp.exp(np.geomspace(0.01, 0.1, s[0]))-1), (self.num_anchor_heads,))
            slopes_a = jax.nn.softplus(a_slopes).reshape(1, 1, -1, 1)
            # Shape: (1, L, num_anchor, 1)
            w_list.append(jnp.exp(-slopes_a * pos[None, :, None, None]))

        # Concatenate heads -> (1, L, K, 1)
        time_weight = jnp.concatenate(w_list, axis=2).astype(jnp.float32)
        
        score_scale = self.param("score_scale", nn.initializers.ones, (self.K,))
        p = jnp.exp(jnp.clip(score_scale[None, None, :, None] * s_raw, -20., 20.))
        
        # --- FIX DIMENSION p_w ---
        # On force explicitement le Rank 5: (B, L, K, 1, M)
        # p * time_weight -> (B, L, K, 1)
        p_w = (p * time_weight).reshape(b, l, self.K, 1, 1)
        
        # Si M > 1, on broadcast p_w sur la dernière dimension pour qu'il matche theta
        if self.M > 1:
            p_w = jnp.broadcast_to(p_w, (b, l, self.K, 1, self.M))

        # --- 3. SPECTRAL MODULATION ---
        # x_val (B, L, K, H) -> (B, L, K, H, 1)
        phi = (x_val[..., None] * theta).astype(jnp.float32) 
        cos_b, sin_b = jnp.cos(phi), jnp.sin(phi)

        if self.derivative_order == 0:
            re_m, im_m = cos_b, sin_b
        else:
            w = jax.nn.softmax(self.param("deriv_logits", lambda r,s: jnp.array([5.] + [0.]*self.derivative_order), (self.derivative_order+1,)))
            term_0 = w[0]
            term_1 = w[1] * x_val[..., None]
            term_2 = (w[2] * -jnp.square(x_val[..., None])) if self.derivative_order == 2 else 0.
            poly = term_0 + term_1 + term_2
            re_m, im_m = poly * cos_b, poly * sin_b

        # --- 4. SCANS OPTIMISÉS (Correct Dimensions) ---
        
        # 1. Denominator (Scan sur p_w directement, très léger)
        # Shape p_w: (B, L, K, 1, M) -> Scan axis 1
        den_small = jnp.cumsum(p_w, axis=1)
        
        # 2. Numerators (Scan sur le produit)
        # p_w (..., 1, M) * re_m (..., H, M) -> Broadcast OK (1 -> H)
        # Result (B, L, K, H, M)
        num_re = jnp.cumsum(p_w * re_m, axis=1)
        num_im = jnp.cumsum(p_w * im_m, axis=1)

        # --- 5. NORMALIZATION ---
        inv_den = 1.0 / jnp.maximum(den_small, 1e-4)
        
        # Broadcast implicite de la division (..., 1, M) vers (..., H, M)
        re = num_re * inv_den
        im = num_im * inv_den

        # --- 6. PROJECTION ---
        re_flat = re.reshape(b, l, self.K, H * self.M)
        im_flat = im.reshape(b, l, self.K, H * self.M)

        mean_sq = (jnp.sum(jnp.square(re_flat), -1) + jnp.sum(jnp.square(im_flat), -1)) / (2 * H * self.M)
        rsqrt = jax.lax.rsqrt(mean_sq[..., None] + 1e-5).astype(x.dtype)

        split_dim = H * self.M
        W_re = self.param("W_re", nn.initializers.glorot_uniform(), (split_dim, H))
        W_im = self.param("W_im", nn.initializers.glorot_uniform(), (split_dim, H))
        scale = self.param("norm_scale", nn.initializers.ones, (2 * split_dim,))

        y_re = jnp.dot(re_flat * rsqrt * scale[:split_dim], W_re)
        y_im = jnp.dot(im_flat * rsqrt * scale[split_dim:], W_im)

        y = (y_re + y_im).reshape(b, l, d_inner)
        out = nn.Dense(d_model, use_bias=False, name="out_proj")(y * x_gate)

        if self.dropout > 0:
            out = nn.Dropout(self.dropout)(out, deterministic=deterministic)
        
        return out


class SeqCondBlock(nn.Module):
    # Wrapper simple avec Pre-Norm et Residual
    num_heads: int = 32
    key_heads: Optional[int] = None
    expand_factor: int = 2
    num_thetas: int = 4
    num_anchor_heads: int = 0
    derivative_order: int = 0
    dropout: float = 0.0
    conv_kernel_size: int = 4
    norm_eps: float = 1e-5
    maxlen: Optional[int] = None  # forwarded to SeqCondAttention for fixed-length dist

    # ... passer tous les autres args ...
    
    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        # Passage d'args dynamique via **kwargs possible ou explicite
        # Ici simplifié pour l'exemple
        normed = nn.RMSNorm(epsilon=1e-5)(x)
        mixed = SeqCondAttention(
            num_heads=self.num_heads,
            key_heads=self.key_heads,
            expand_factor=self.expand_factor,
            num_thetas=self.num_thetas,
            num_anchor_heads=self.num_anchor_heads,
            derivative_order=self.derivative_order,
            dropout=self.dropout,
            conv_kernel_size=self.conv_kernel_size,
            maxlen=self.maxlen,
        )(normed, mask, deterministic)
        return x + mixed


# class SeqCondBlock(nn.Module):
#     num_heads: int = 8
#     key_heads: Optional[int] = None
#     num_thetas: int = 4
#     num_anchor_heads: int = 0
#     derivative_order: int = 0
#     dropout: float = 0.0
#     use_conv: bool = True
#     conv_kernel_size: int = 4
#     conv_kernel: Optional[int] = None
#     norm_eps: float = 1e-5
#     maxlen: Optional[int] = None  # forwarded to SeqCondAttention for fixed-length dist

#     def setup(self):
#         self.norm = RMSNorm(epsilon=self.norm_eps)
#         self.mixer = SeqCondAttention(
#             num_heads=self.num_heads,
#             key_heads=self.key_heads,
#             num_thetas=self.num_thetas,
#             num_anchor_heads=self.num_anchor_heads,
#             derivative_order=self.derivative_order,
#             dropout=self.dropout,
#             use_conv=self.use_conv,
#             conv_kernel_size=self.conv_kernel_size,
#             conv_kernel=self.conv_kernel,
#             maxlen=self.maxlen,
#         )

#     def __call__(
#         self,
#         x: jnp.ndarray,
#         mask: Optional[jnp.ndarray] = None,
#         deterministic: bool = True,
#     ) -> jnp.ndarray:
#         residual = x
#         x = self.norm(x)
#         x = self.mixer(x, mask=mask, deterministic=deterministic)
#         return x + residual
