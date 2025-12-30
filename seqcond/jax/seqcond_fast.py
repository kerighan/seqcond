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
from flax import linen as nn
import numpy as np
from typing import Optional


def cumsum_chunked(x: jnp.ndarray, axis: int = 1, chunk: int = 128) -> jnp.ndarray:
    """
    Drop-in replacement for jnp.cumsum(x, axis=axis) using chunked scan.

    Requirements:
      - axis dimension length must be static or at least known at compile.
      - best when L is large (>= 512). For small L, plain cumsum may be faster.

    Semantics:
      - Exact prefix sum along `axis`, same as jnp.cumsum.
    """
    if axis < 0:
        axis = x.ndim + axis

    # Move target axis to 1 for convenience: (B, L, ...)
    x_perm = jnp.moveaxis(x, axis, 1)
    B = x_perm.shape[0]
    L = x_perm.shape[1]
    tail_shape = x_perm.shape[2:]

    # Pad L to multiple of chunk
    pad = (-L) % chunk
    if pad:
        pad_width = [(0, 0), (0, pad)] + [(0, 0)] * (x_perm.ndim - 2)
        x_pad = jnp.pad(x_perm, pad_width)
    else:
        x_pad = x_perm

    Lp = x_pad.shape[1]
    n_chunks = Lp // chunk

    # Reshape into chunks: (B, n_chunks, chunk, ...)
    x_chunks = x_pad.reshape((B, n_chunks, chunk) + tail_shape)

    # 1) cumsum inside each chunk
    intra = jnp.cumsum(x_chunks, axis=2)

    # 2) carry = sum of each chunk (last element of intra)
    chunk_sums = intra[:, :, -1, ...]  # (B, n_chunks, ...)

    # 3) prefix sums of chunk sums => offset for each chunk
    # offsets[c] = sum_{i < c} chunk_sums[i]
    offsets = jnp.cumsum(chunk_sums, axis=1)
    offsets = jnp.concatenate(
        [jnp.zeros_like(offsets[:, :1, ...]), offsets[:, :-1, ...]],
        axis=1
    )  # (B, n_chunks, ...)

    # 4) add offsets to every element in chunk
    out_chunks = intra + offsets[:, :, None, ...]

    # Unchunk + unpad + restore axis
    out = out_chunks.reshape((B, Lp) + tail_shape)
    out = out[:, :L, ...]
    out = jnp.moveaxis(out, 1, axis)
    return out


class SeqCondAttention(nn.Module):
    # Dimensions Architecture
    num_heads: int = 12          # K
    num_query_heads: int = 6    # K'
    num_anchor_heads: int = 0
    num_thetas: int = 1          # M
    
    # Paramètres Locaux
    conv_kernel_size: int = 4
    expand_factor: int = 1       # Input Slim (Scan Rapide)
    out_expand_factor: int = 3   # Output Fat (Cerveau SwiGLU) - Ajustable selon VRAM
    
    dropout: float = 0.0
    maxlen: Optional[int] = None
    chunk_size: int = 0

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
        
        # 1. Budget Constant : On adapte H pour garder la taille du Scan fixe
        # Si M augmente, H diminue.
        H = max(1, d_inner // (self.K * self.M))
        
        # Dimension réelle projetée pour la mémoire
        memory_dim = self.K * H
        # Dimension flatten des phi'
        flat_dim = self.K * H * self.M
        # Expansion du cerveau logique
        dim_expand = H * self.out_expand_factor
        dim_swiglu_per_head = dim_expand * 2
        dim_swiglu_total = self.K * dim_swiglu_per_head
        latent_dim = d_model // 4   # latent skip
        # =================================================================
        # 1. ENCODAGE MÉMOIRE
        # =================================================================
        z_mem = nn.Dense(memory_dim + self.K, use_bias=False, name="in_proj_mem")(x)
        z_mem = z_mem.astype(self.compute_dtype)
        
        z_mem = nn.Conv(
            features=z_mem.shape[-1], 
            kernel_size=(self.conv_kernel_size,), 
            padding=((self.conv_kernel_size - 1, 0),),
            feature_group_count=z_mem.shape[-1], 
            use_bias=False, name="conv_mem"
        )(z_mem)

        k_val = z_mem[..., :memory_dim].reshape(b, l, self.K, H)
        s_raw = z_mem[..., -self.K:].reshape(b, l, self.K, 1)

        # Stabilisation (RMSNorm sur K conseillé mais désactivé selon ton choix)
        # k_val = nn.RMSNorm(dtype=self.compute_dtype, name="k_norm")(k_val)

        if mask is not None:
            m = mask.astype(x.dtype)[:, :, None, None]
            s_raw *= m
            k_val *= m

        # =================================================================
        # 2. QUERY
        # =================================================================
        q_raw = nn.Dense(self.K_q * H * self.M * 2, use_bias=False, name="in_proj_query")(x)
        q_raw = q_raw.reshape(b, l, self.K_q, H, self.M, 2)
        q_re, q_im = q_raw[..., 0], q_raw[..., 1]
        
        # =================================================================
        # 3. GRILLE SPECTRALE (REFACTORISÉE)
        # =================================================================
        # Bornes physiques du spectre
        theta_min, theta_max = 0.001, 3.0

        if self.M == 1:
            # --- CAS OPTIMISÉ M=1 (Single Frequency per Head) ---
            # Ici, on veut que chaque tête (K) capture une fréquence différente.
            # Pas d'intégrale, pas de cumsum. Juste un paramètre par tête.
            
            def init_theta_m1(key, shape):
                # shape: (1, 1, K, H, 1)
                # 1. On crée une gamme géométrique étalée sur les K têtes
                grid_k = np.geomspace(theta_min, theta_max, self.K)
                
                # 2. Reshape pour matcher (1, 1, K, 1, 1)
                grid_k = grid_k.reshape(1, 1, self.K, 1, 1)
                
                # 3. Broadcast sur H (chaque canal H d'une même tête partage la fréquence de base)
                # Ou on peut ajouter du bruit si on veut que H varie aussi.
                base = np.tile(grid_k, (1, 1, 1, shape[3], 1))
                
                # 4. Inverse Softplus pour l'initialisation
                return jnp.log(jnp.exp(base) - 1.0 + 1e-4)

            # Paramètre direct (pas de delta)
            theta_raw = self.param("theta_raw", init_theta_m1, (1, 1, self.K, H, 1))
            
            # Softplus pour garantir la positivité. 
            # On ne borne pas forcément à 3.0, on laisse le modèle aller chercher plus haut si besoin.
            theta = jax.nn.softplus(theta_raw).astype(jnp.float32) + theta_min
            
            # Poids unitaire pour l'intégrale (produit simple)
            w_int = jnp.ones((1, 1, self.K, H, 1), dtype=jnp.float32)
        else:
            # --- CAS RIEMANN M>1 (Adaptive Integration) ---
            # Ici, chaque tête possède un spectre interne de M fréquences.
            def init_theta_deltas(key, shape):
                # shape: (1, 1, K, H, M)
                # On étale sur M (axe fréquentiel interne)
                grid_m = np.geomspace(theta_min, theta_max, self.M)
                base = np.tile(grid_m.reshape(1, 1, 1, 1, self.M), (1, 1, self.K, shape[3], 1))
                return jnp.log(jnp.exp(base) - 1.0 + 1e-4)

            theta_d_raw = self.param("theta_d_raw", init_theta_deltas, (1, 1, self.K, H, self.M))
            
            # Construction Monotone
            theta_d = jax.nn.softplus(theta_d_raw).astype(jnp.float32) + 1e-4
            theta_accum = jnp.cumsum(theta_d, axis=-1)
            
            # Normalisation dans la plage [min, max]
            scale_range = theta_max - theta_min
            total_sum = theta_accum[..., -1:] 
            theta = theta_min + (theta_accum / total_sum) * scale_range
            
            # Calcul des poids Trapèze (dtheta)
            dtheta_raw = theta_accum[..., 1:] - theta_accum[..., :-1]
            dtheta = dtheta_raw * (scale_range / total_sum) # Correction d'échelle
            
            w0 = dtheta[..., :1] * 0.5
            w_mid = 0.5 * (dtheta[..., :-1] + dtheta[..., 1:])
            wL = dtheta[..., -1:] * 0.5
            
            w_int = jnp.concatenate([w0, w_mid, wL], axis=-1)

        # =================================================================
        # 4. SCAN & INTEGRALE
        # =================================================================
        # A. Decay (Log-Space)
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
            
        log_time_weight = jnp.concatenate(log_w_list, axis=2)
        score_scale = self.param("score_scale", nn.initializers.ones, (self.K,))
        log_p = jnp.clip(score_scale[None, None, :, None] * s_raw, -20., 20.)
        p_w = jnp.exp(log_p + log_time_weight)

        # B. Modulation & Scan
        k_val_expand = k_val[..., None]
        kvw = k_val_expand * p_w[..., None]
        # phi_k = (k_val_expand * theta)
        phi_k = jnp.tanh(k_val_expand) * theta  # cheap norm
        re_k, im_k = kvw * jnp.cos(phi_k), kvw * jnp.sin(phi_k)

        den_in = p_w.squeeze(-1)
        num_re_in = re_k.reshape(b, l, flat_dim)
        num_im_in = im_k.reshape(b, l, flat_dim)
        
        merged = jnp.concatenate([den_in, num_re_in, num_im_in], axis=-1)
        if self.chunk_size > 0:
            cumsum = cumsum_chunked(merged, axis=1, chunk=self.chunk_size)
        else:
            cumsum = jnp.cumsum(merged, axis=1)
        cumsum = cumsum.astype(self.compute_dtype)
        den, num_re, num_im = jnp.split(cumsum, [self.K, self.K + flat_dim], axis=-1)
        
        # State: (B, L, K, H, M)
        inv_den = 1.0 / jnp.maximum(den[..., None], 1e-4)
        state_re = (num_re.reshape(b, l, self.K, H, self.M) * inv_den[..., None])
        state_im = (num_im.reshape(b, l, self.K, H, self.M) * inv_den[..., None])

        # =================================================================
        # 4. RÉSONANCE & INTÉGRALE PONDÉRÉE (ZERO-COPY GQA)
        # =================================================================

        # 1. On reshape l'État (Mémoire) pour exposer les groupes GQA
        # State: (B, L, K, H, M) -> (B, L, K', n_rep, H, M)
        state_re_g = state_re.reshape(b, l, self.K_q, self.n_rep, H, self.M)
        state_im_g = state_im.reshape(b, l, self.K_q, self.n_rep, H, self.M)
        # 2. Reshape w_int pour matcher GQA
        # w_int est (1, 1, K, H, M) -> On doit le splitter en (1, 1, K', n_rep, H, M)
        w_int_g = w_int.reshape(1, 1, self.K_q, self.n_rep, H, self.M)
        # 2. On prépare le Query (Filtre) pour le broadcast
        # Query: (B, L, K', H, M) -> (B, L, K', 1, H, M)
        q_re_g = q_re[:, :, :, None, :, :]
        q_im_g = q_im[:, :, :, None, :, :]

        # 3. Produit Hermitien (Broadcasting automatique sur n_rep)
        match_re = state_re_g * q_re_g + state_im_g * q_im_g
        match_im = state_im_g * q_re_g - state_re_g * q_im_g

        # 4. Intégrale pondérée par w_int
        # w_int est (1, 1, 1, 1, M). Le broadcast fonctionne car il aligne la droite (M).
        # Somme sur M (Dernier axe)
        # Résultat : (B, L, K', n_rep, H)
        out_re_g = jnp.sum(match_re * w_int_g, axis=-1) 
        out_im_g = jnp.sum(match_im * w_int_g, axis=-1)
        
        # 5. On remet à plat vers K pour le reste du réseau
        # (B, L, K', n_rep, H) -> (B, L, K, H)
        out_re = out_re_g.reshape(b, l, self.K, H)
        out_im = out_im_g.reshape(b, l, self.K, H)

        # =================================================================
        # 5. READOUT/SWIGLU + SKIP CONNECTION (HIGHWAY)
        # =================================================================        
        # --- A. Chemin Spectral (Scan) ---
        # Readout indépendant par tête (Einsum)
        W_re = self.param("W_re", nn.initializers.glorot_uniform(), (self.K, H, dim_swiglu_per_head))
        W_im = self.param("W_im", nn.initializers.glorot_uniform(), (self.K, H, dim_swiglu_per_head))
        scale = self.param("norm_scale", nn.initializers.ones, (1, 1, self.K, H))
        # --- DYNAMIC GATING ---
        # Au lieu d'un paramètre scalaire, on apprend une projection depuis x
        # On projette vers K (une gate par tête) pour être économe.
        # On initialise le bias à -3.0 pour fermer la porte au début.
        gate_proj = nn.Dense(self.K, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.constant(-3.0), name="spectral_gate_proj")        
        # Calcul de la gate dynamique (B, L, K)
        # Sigmoid pour borner entre 0 (Ignorer mémoire) et 1 (Utiliser mémoire)
        spectral_gate = jax.nn.sigmoid(gate_proj(x)).astype(self.compute_dtype)
        # On reshape pour broadcaster sur la dimension SwiGLU: (B, L, K, 1)
        spectral_gate = spectral_gate[..., None]
        y_re = jnp.einsum('blkh,khn->blkn', out_re * scale, W_re)
        y_im = jnp.einsum('blkh,khn->blkn', out_im * scale, W_im)
        # (B, L, K, dim_swiglu_per_head)
        y_spectral = (y_re + y_im) * spectral_gate

        # --- B. Chemin Direct (Skip Connection) - NOUVEAU ---
        # Down-Project (Compression)
        c_skip = nn.Dense(latent_dim, use_bias=False, name="skip_down")(x)
        # Up-Project (Expansion)
        y_direct_raw = nn.Dense(dim_swiglu_total, use_bias=False, name="skip_up")(c_skip)
        y_direct_raw = y_direct_raw.reshape(b, l, self.K, dim_swiglu_per_head)

        # =================================================================
        # 3. FUSION (ADDITION PURE)
        # =================================================================
        # On splitte les deux branches en Gate/Value
        y_val_spec, y_gate_spec = jnp.split(y_spectral, 2, axis=-1)
        y_val_direct, y_gate_direct = jnp.split(y_direct_raw, 2, axis=-1)
        # On peut scaler le skip pour un démarrage doux (optionnel)
        highway_scale = self.param("highway_scale", nn.initializers.constant(1.0), (1, 1, self.K, 1))
        # Value = Contenu mémoriel + Contenu actuel
        y_val_fused = y_val_spec + (y_val_direct * highway_scale)
        # Gate = Commande mémorielle + Commande actuelle
        y_gate_fused = y_gate_spec + (y_gate_direct * highway_scale)        
        # SwiGLU Activation
        y_activated = y_val_fused * jax.nn.silu(y_gate_fused)
        # Flatten
        y_flat = y_activated.reshape(b, l, -1) 
        # Projection Finale
        out = nn.Dense(d_model, use_bias=False, name="out_proj")(y_flat)
        if self.dropout > 0:
            out = nn.Dropout(self.dropout)(out, deterministic=deterministic)
        
        return out

# VERSION LATEST SANS CONVOLUTION SPECTRALE
# class SeqCondAttention(nn.Module):
#     # Dimensions Architecture
#     num_heads: int = 32          # K
#     num_anchor_heads: int = 4
#     num_thetas: int = 8          # M
    
#     # Paramètres Locaux
#     conv_kernel_size: int = 4
#     expand_factor: int = 1       # Input Slim (Scan)
#     out_expand_factor: int = 2   # Output Fat (SwiGLU)

#     dropout: float = 0.0
#     maxlen: Optional[int] = None
    
#     compute_dtype: jnp.dtype = jnp.bfloat16
#     param_dtype: jnp.dtype = jnp.float32

#     def setup(self):
#         self.K = self.num_heads
#         self.M = self.num_thetas
#         self.num_decay_heads = self.K - self.num_anchor_heads

#     @nn.compact
#     def __call__(self, x, mask=None, deterministic=True):
#         b, l, d_model = x.shape
#         d_inner = int(d_model * self.expand_factor)
#         H = max(1, d_inner // self.K)

#         # =================================================================
#         # 1. ENCODAGE MÉMOIRE (Slim & Shared)
#         # =================================================================
#         # Ici on garde le partage pour l'extraction de features bas niveau
#         z = nn.Dense(d_inner + self.K, use_bias=False, name="in_proj")(x)
#         z = z.astype(self.compute_dtype)
        
#         z = nn.Conv(
#             features=z.shape[-1], 
#             kernel_size=(self.conv_kernel_size,), 
#             padding=((self.conv_kernel_size - 1, 0),),
#             feature_group_count=z.shape[-1], 
#             use_bias=False, name="conv"
#         )(z)

#         k_val = z[..., :d_inner].reshape(b, l, self.K, H)
#         s_raw = z[..., -self.K:].reshape(b, l, self.K, 1)

#         if mask is not None:
#             m = mask.astype(x.dtype)[:, :, None, None]
#             s_raw *= m
#             k_val *= m

#         # =================================================================
#         # 2. SCAN SPECTRAL (ECF Core)
#         # =================================================================
#         def init_theta(key, shape):
#             grid = np.linspace(-np.pi/3, np.pi/3, self.M)
#             base = np.tile(grid.reshape(1, 1, 1, 1, self.M), (1, 1, self.K, H, 1))
#             return jnp.array(base, dtype=jnp.float32)

#         theta = self.param("theta", init_theta, (1, 1, self.K, H, self.M))

#         # Decay Log-Space
#         pos = jnp.arange(l, dtype=jnp.float32)
#         log_w_list = []
#         if self.num_decay_heads > 0:
#             d_slopes = self.param("decay_slopes", lambda r,s: jnp.log(jnp.exp(np.geomspace(0.001, 0.1, s[0]))-1), (self.num_decay_heads,))
#             slopes = jax.nn.softplus(d_slopes).reshape(1, 1, -1, 1)
#             dist = jnp.maximum(jnp.float32((self.maxlen or l) - 1) - pos, 0.)
#             log_w_list.append(-slopes * dist[None, :, None, None])
#         if self.num_anchor_heads > 0:
#             a_slopes = self.param("anchor_slopes", lambda r,s: jnp.log(jnp.exp(np.geomspace(0.01, 0.1, s[0]))-1), (self.num_anchor_heads,))
#             slopes_a = jax.nn.softplus(a_slopes).reshape(1, 1, -1, 1)
#             log_w_list.append(-slopes_a * pos[None, :, None, None])
            
#         log_time_weight = jnp.concatenate(log_w_list, axis=2)
#         score_scale = self.param("score_scale", nn.initializers.ones, (self.K,))
#         log_p = jnp.clip(score_scale[None, None, :, None] * s_raw, -20., 20.)
#         p_w = jnp.exp(log_p + log_time_weight)

#         phi_k = (k_val[..., None] * theta)
#         re_k, im_k = jnp.cos(phi_k), jnp.sin(phi_k)

#         # Scan
#         flat_dim = self.K * H * self.M
#         den_in = p_w.squeeze(-1)
#         p_w_broad = p_w[..., None]
#         num_re_in = (p_w_broad * re_k).reshape(b, l, flat_dim)
#         num_im_in = (p_w_broad * im_k).reshape(b, l, flat_dim)
        
#         merged = jnp.concatenate([den_in, num_re_in, num_im_in], axis=-1)
#         cumsum = jnp.cumsum(merged, axis=1)
#         den, num_re, num_im = jnp.split(cumsum, [self.K, self.K + flat_dim], axis=-1)
        
#         # CORRECTION 1 : inv_den est déjà (B, L, K, 1)
#         inv_den = 1.0 / jnp.maximum(den[..., None], 1e-4)
        
#         # On multiplie directement. Pas de [..., None] supplémentaire !
#         # (B, L, K, H*M) * (B, L, K, 1) -> (B, L, K, H*M)
#         state_re = (num_re.reshape(b, l, self.K, H * self.M) * inv_den)
#         state_im = (num_im.reshape(b, l, self.K, H * self.M) * inv_den)

#         # =================================================================
#         # 4. READOUT INDÉPENDANT PAR TÊTE (EINSUM SWIGLU)
#         # =================================================================
        
#         dim_expand = H * self.out_expand_factor
#         dim_swiglu_per_head = dim_expand * 2
#         input_dim = H * self.M 
        
#         W_re = self.param("W_re", nn.initializers.glorot_uniform(), (self.K, input_dim, dim_swiglu))
#         W_im = self.param("W_im", nn.initializers.glorot_uniform(), (self.K, input_dim, dim_swiglu))
        
#         # CORRECTION 2 : Shape du scale pour broadcaster sur (B, L, K, D)
#         # Il faut que K soit à l'avant-dernière position ou géré par broadcast
#         # Le plus simple : (1, 1, K, D)
#         scale = self.param("norm_scale", nn.initializers.ones, (1, 1, self.K, 2 * input_dim))
        
#         # On applique le scale
#         scale_re = scale[..., :input_dim]
#         scale_im = scale[..., input_dim:]
        
#         state_re_scaled = state_re * scale_re
#         state_im_scaled = state_im * scale_im

#         # EINSUM (Maintenant state_re_scaled est bien 4D : blkm)
#         y_re = jnp.einsum('blkm,kmn->blkn', state_re_scaled, W_re)
#         y_im = jnp.einsum('blkm,kmn->blkn', state_im_scaled, W_im)
        
#         y_raw = y_re + y_im # (B, L, K, dim_swiglu)

#         # SwiGLU
#         y_val, y_gate = jnp.split(y_raw, 2, axis=-1)
#         y_activated = y_val * jax.nn.silu(y_gate) # (B, L, K, dim_expand)
        
#         # Flatten pour le mélange final
#         y_flat = y_activated.reshape(b, l, -1) # (B, L, K * dim_expand)

#         # Projection Finale (Mélange global des têtes)
#         # C'est la seule fois où les têtes communiquent entre elles
#         out = nn.Dense(d_model, use_bias=False, name="out_proj")(y_flat)

#         if self.dropout > 0:
#             out = nn.Dropout(self.dropout)(out, deterministic=deterministic)
        
#         return out


class SeqCondBlock(nn.Module):
    num_heads: int = 32
    expand_factor: float = 1.0
    num_thetas: int = 1
    num_anchor_heads: int = 0
    conv_kernel_size: int = 4
    dropout: float = 0.0
    norm_eps: float = 1e-5
    maxlen: Optional[int] = None
    derivative_order: Optional[int] = 0
    chunk_size: int = 0

    compute_dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        h = nn.RMSNorm(epsilon=self.norm_eps, dtype=self.compute_dtype, param_dtype=self.param_dtype)(x)
        h = SeqCondAttention(
            num_heads=self.num_heads,
            expand_factor=self.expand_factor,
            num_thetas=self.num_thetas,
            num_anchor_heads=self.num_anchor_heads,
            conv_kernel_size=self.conv_kernel_size,
            dropout=self.dropout,
            maxlen=self.maxlen,
            chunk_size=self.chunk_size,
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
