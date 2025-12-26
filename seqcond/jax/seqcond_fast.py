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

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from typing import Optional

class SeqCondAttention(nn.Module):
    # Dimensions
    num_heads: int = 32          # K (Mémoire)
    num_query_heads: int = 4     # K' (Recherche - GQA)
    num_anchor_heads: int = 4    # Long terme
    num_thetas: int = 8          # M (Résolution Spectrale)
    
    # Paramètres
    conv_kernel_size: int = 4
    expand_factor: int = 1       # Speed Mode
    dropout: float = 0.0
    maxlen: Optional[int] = None
    
    compute_dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert self.num_heads % self.num_query_heads == 0
        self.n_rep = self.num_heads // self.num_query_heads
        
        self.K = self.num_heads
        self.K_q = self.num_query_heads
        self.M = self.num_thetas
        self.num_decay_heads = self.K - self.num_anchor_heads

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        b, l, d_model = x.shape
        d_inner = int(d_model * self.expand_factor)
        H = max(1, d_inner // self.K)

        # Dims
        dim_k = d_inner
        dim_decay = self.K
        dim_q = self.K_q * H * self.M * 2
        
        total_dim = dim_k + dim_decay + dim_q

        # =================================================================
        # 1. ENCODAGE MÉMOIRE (SIGNAL)
        # =================================================================
        z = nn.Dense(total_dim, use_bias=False, name="in_proj_mem")(x)
        z = z.astype(self.compute_dtype)
        # On ne veut conv que sur la mémoire (K_val) et le decay, pas sur Q (qui est "futur")
        # Split rapide
        z_mem = z[..., :dim_k + dim_decay]
        q_raw = z[..., dim_k + dim_decay:]

        z_mem = nn.Conv(
            features=z_mem.shape[-1], 
            kernel_size=(self.conv_kernel_size,), 
            padding=((self.conv_kernel_size - 1, 0),),
            feature_group_count=z_mem.shape[-1], 
            use_bias=False, name="conv_mem"
        )(z_mem)

        k_val = z_mem[..., :d_inner].reshape(b, l, self.K, H)
        s_raw = z_mem[..., -self.K:].reshape(b, l, self.K, 1)
        # Query Reshape (Directement en complexe, pas de modulation inutile)
        q_raw = q_raw.reshape(b, l, self.K_q, H_state, self.M, 2)
        q_re, q_im = q_raw[..., 0], q_raw[..., 1]

        if mask is not None:
            m = mask.astype(x.dtype)[:, :, None, None]
            s_raw *= m
            k_val *= m

        # =================================================================
        # 3. ÉTAT & SCAN (ECF)
        # =================================================================
        # Base Spectrale (Pour la mémoire uniquement)
        def init_theta(key, shape):
            if self.M == 1:
                grid = np.linspace(0.1, 1.5, self.K * H).reshape(1, 1, self.K, H, 1)
                signs = np.resize([1., -1.], grid.shape)
                return jnp.array(grid * signs, dtype=jnp.float32)
            else:
                grid = np.linspace(-np.pi/3, np.pi/3, self.M)
                base = np.tile(grid.reshape(1, 1, 1, 1, self.M), (1, 1, self.K, H, 1))
                return jnp.array(base, dtype=jnp.float32)

        theta = self.param("theta", init_theta, (1, 1, self.K, H, self.M))
        theta = theta.astype(self.compute_dtype)

        # Decay Log-Space
        pos = jnp.arange(l, dtype=jnp.float32)
        log_w_list = []
        if self.num_decay_heads > 0:
            d_slopes = self.param("decay_slopes", lambda r,s: jnp.log(jnp.exp(np.geomspace(0.001, 0.1, s[0]))-1), (self.num_decay_heads,))
            slopes = jax.nn.softplus(d_slopes).reshape(1, 1, -1, 1)
            dist = jnp.maximum(jnp.float32((self.maxlen or l) - 1) - pos, 0.)
            log_w_list.append(-slopes * dist[None, :, None, None])
        if self.num_anchor_heads > 0:
            a_slopes = self.param("anchor_slopes", lambda r,s: jnp.log(jnp.exp(np.geomspace(0.01, 0.1, s[0]))-1), (self.num_anchor_heads,))
            slopes_a = jax.nn.softplus(a_slopes).reshape(1, 1, -1, 1)
            log_w_list.append(-slopes_a * pos[None, :, None, None])
            
        log_time_weight = jnp.concatenate(log_w_list, axis=2).astype(jnp.float32)
        score_scale = self.param("score_scale", nn.initializers.ones, (self.K,))
        log_p = jnp.clip(score_scale[None, None, :, None] * s_raw, -20., 20.)
        p_w = jnp.exp(log_p + log_time_weight).astype(self.compute_dtype)

        # Modulation Mémoire (Indispensable pour encoder la position/temps)
        phi_k = (k_val[..., None].astype(self.compute_dtype) * theta)
        re_k = jnp.cos(phi_k).astype(self.compute_dtype)
        im_k = jnp.sin(phi_k).astype(self.compute_dtype)

        # Scan
        den_in = p_w.squeeze(-1)
        p_w_broad = p_w[..., None]
        
        # flat_dim = self.K * H * self.M
        # num_re_in = (p_w_broad * re_k).reshape(b, l, flat_dim)
        # num_im_in = (p_w_broad * im_k).reshape(b, l, flat_dim)
        # merged = jnp.concatenate([den_in, num_re_in, num_im_in], axis=-1)
        # cumsum = jnp.cumsum(merged, axis=1)
        # den, num_re, num_im = jnp.split(cumsum, [self.K, self.K + flat_dim], axis=-1)
        den = jnp.cumsum(den_in, axis=1)  # (B,L,K)
        # (B,L,K,H,M) directement, pas besoin de flatten
        num_re = jnp.cumsum(p_w_broad * re_k, axis=1)  # (B,L,K,H,M)
        num_im = jnp.cumsum(p_w_broad * im_k, axis=1)
        # num_re = jax.lax.associative_scan(jnp.add, p_w_broad * re_k, axis=1)
        # num_im = jax.lax.associative_scan(jnp.add, p_w_broad * im_k, axis=1)

        # inv_den = 1.0 / jnp.maximum(den[..., None], 1e-4)
        # state_re = (num_re.reshape(b, l, self.K, H, self.M) * inv_den[..., None])
        # state_im = (num_im.reshape(b, l, self.K, H, self.M) * inv_den[..., None])
        inv_den = jax.lax.rsqrt(jnp.maximum(den, 1e-4))  # ou 1/den si tu veux strictement pareil
        inv_den = (inv_den * inv_den)[..., None, None]  # revient à 1/den, stable (rsqrt+square)
        state_re = num_re * inv_den
        state_im = num_im * inv_den

        # =================================================================
        # 4. RÉSONANCE & INTÉGRATION (OPTIMISÉE MEMOIRE)
        # =================================================================
        
        # Astuce : On reshape pour isoler les groupes GQA
        # State : (B, L, K, H, M) -> (B, L, K_q, N_rep, H, M)
        # Cela ne coûte RIEN (c'est juste des métadonnées)
        state_re = state_re.reshape(b, l, self.K_q, self.n_rep, H, self.M)
        state_im = state_im.reshape(b, l, self.K_q, self.n_rep, H, self.M)

        # Query : (B, L, K_q, H, M) -> (B, L, K_q, 1, H, M)
        # On ajoute une dimension 1 pour le broadcast. 
        # ON NE FAIT PAS DE REPEAT EXPLICITE !
        q_re = q_re.reshape(b, l, self.K_q, 1, H, self.M)
        q_im = q_im.reshape(b, l, self.K_q, 1, H, self.M)

        # PRODUIT HERMITIEN + INTÉGRATION FUSIONNÉE
        # XLA va voir : "Je dois multiplier un tenseur (..., N_rep, ...) par un (..., 1, ...)"
        # Il va broadcaster à la volée dans les registres et sommer immédiatement sur M.
        # Aucune écriture intermédiaire en VRAM du tenseur (B, L, K, H, M).

        # (State_re * Q_re + State_im * Q_im) -> Somme sur M (axis=-1)
        # Note : Le broadcast (1 vs N_rep) se fait automatiquement ici
        out_re = jnp.sum(state_re * q_re + state_im * q_im, axis=-1) 
        out_im = jnp.sum(state_im * q_re - state_re * q_im, axis=-1)
        
        # Résultat : out_re est (B, L, K_q, N_rep, H). On remet à plat vers (K, H)
        out_re = out_re.reshape(b, l, self.K, H)
        out_im = out_im.reshape(b, l, self.K, H)

        # =================================================================
        # 5. SORTIE
        # =================================================================
        split_dim = H
        W_re = self.param("W_re", nn.initializers.glorot_uniform(), (split_dim, H))
        W_im = self.param("W_im", nn.initializers.glorot_uniform(), (split_dim, H))
        scale = self.param("norm_scale", nn.initializers.ones, (2 * split_dim,))

        # y_re = jnp.dot(out_re * scale[:split_dim], W_re)
        # y_im = jnp.dot(out_im * scale[split_dim:], W_im)
        y_re = jnp.einsum("blkh,hh->blkh", out_re * scale[:split_dim], W_re)
        y_im = jnp.einsum("blkh,hh->blkh", out_im * scale[split_dim:], W_im)

        y = jax.nn.silu((y_re + y_im).reshape(b, l, d_inner))
        out = nn.Dense(d_model, use_bias=False, name="out_proj")(y)

        if self.dropout > 0:
            out = nn.Dropout(self.dropout)(out, deterministic=deterministic)
        
        return out


class SeqCondBlock(nn.Module):
    num_heads: int = 32
    num_query_heads: Optional[int] = 4
    expand_factor: float = 1.0
    num_thetas: int = 1
    num_anchor_heads: int = 0
    conv_kernel_size: int = 4
    dropout: float = 0.0
    norm_eps: float = 1e-5
    maxlen: Optional[int] = None
    derivative_order: Optional[int] = 0

    compute_dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        h = nn.RMSNorm(epsilon=self.norm_eps, dtype=self.compute_dtype, param_dtype=self.param_dtype)(x)
        h = SeqCondAttention(
            num_heads=self.num_heads,
            num_query_heads=self.num_query_heads,
            expand_factor=self.expand_factor,
            num_thetas=self.num_thetas,
            num_anchor_heads=self.num_anchor_heads,
            conv_kernel_size=self.conv_kernel_size,
            dropout=self.dropout,
            maxlen=self.maxlen,
            compute_dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )(h, mask=mask, deterministic=deterministic)
        return x + h


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
