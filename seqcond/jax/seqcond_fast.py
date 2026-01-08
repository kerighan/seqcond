from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


class GatedRMSNorm(nn.Module):
    """RMSNorm with gating on residual (Mamba2 style).

    Applies: x = rmsnorm(x * sigmoid(residual))
    This allows the residual to gate what information passes through.
    """

    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, residual: jnp.ndarray) -> jnp.ndarray:
        hidden_size = x.shape[-1]
        weight = self.param(
            "weight",
            nn.initializers.ones,
            (hidden_size,),
        )

        x = x.astype(jnp.float32)
        res = residual.astype(jnp.float32)

        # Gate with sigmoid (bounded, stable)
        x = x * jax.nn.sigmoid(res)

        # RMSNorm
        variance = jnp.mean(x**2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.epsilon)
        x = x * weight

        return x.astype(self.dtype)


# class SeqCondAttention(nn.Module):
#     # Architecture
#     num_heads: int = 12
#     num_query_heads: int = 6
#     num_anchor_heads: int = 0
#     num_thetas: int = 1  # Strict M=1

#     conv_kernel_size: int = 4
#     expand_factor: float = 1.0       # Input Slim
#     out_expand_factor: int = 3       # Output Moderate

#     chunk_size: int = 128             # Chunk Size

#     dropout: float = 0.0
#     maxlen: Optional[int] = None
#     compute_dtype: jnp.dtype = jnp.bfloat16
#     param_dtype: jnp.dtype = jnp.float32

#     def setup(self):
#         assert self.num_heads % self.num_query_heads == 0
#         assert self.num_thetas == 1

#         self.K = self.num_heads
#         self.K_q = self.num_query_heads
#         self.n_rep = self.K // self.K_q
#         self.M = self.num_thetas
#         self.num_decay_heads = self.K - self.num_anchor_heads

#     @nn.compact
#     def __call__(self, x, mask=None, deterministic=True):
#         b, l, d_model = x.shape
#         d_inner = int(d_model * self.expand_factor)

#         # Dimensions
#         H = max(1, d_inner // (self.K * self.M))
#         memory_dim = self.K * H

#         dim_expand = H * self.out_expand_factor
#         dim_swiglu_per_head = dim_expand * 2
#         dim_swiglu_total = self.K * dim_swiglu_per_head

#         # DeepSeek Latent Dim (Petit goulot d'étranglement)
#         latent_dim = d_model // 4

#         # ------------------------------------------------------------
#         # 1. PRE-CALCULS LÉGERS (Outside Scan)
#         # ------------------------------------------------------------
#         # On ne calcule ICI que ce qui est "Cheap" en mémoire ou difficile à fusionner.

#         # A. Memory Encoding (Conv nécessite le contexte global)
#         z_mem = nn.Dense(memory_dim + self.K, use_bias=False, name="in_proj_mem")(x)
#         z_mem = z_mem.astype(self.compute_dtype)
#         z_mem = nn.Conv(
#             features=z_mem.shape[-1], kernel_size=(self.conv_kernel_size,),
#             padding=((self.conv_kernel_size - 1, 0),), feature_group_count=z_mem.shape[-1],
#             use_bias=False, name="conv_mem"
#         )(z_mem)

#         k_val = z_mem[..., :memory_dim].reshape(b, l, self.K, H)
#         s_raw = z_mem[..., -self.K:].reshape(b, l, self.K, 1)

#         if mask is not None:
#             m = mask.astype(x.dtype)[:, :, None, None]
#             s_raw *= m
#             k_val *= m

#         # B. Query (Pré-calcul OK car 2*H est raisonnable)
#         # Fix Shape: (B, L, Kq, 1, H) pour broadcaster correctement sur n_rep
#         q_raw = nn.Dense(self.K_q * H * 2, use_bias=False, name="in_proj_query")(x)
#         q_raw = nn.RMSNorm(dtype=self.compute_dtype, name="q_norm")(q_raw)
#         q_raw = q_raw.reshape(b, l, self.K_q, 1, H, 2)
#         q_re_seq, q_im_seq = q_raw[..., 0], q_raw[..., 1]

#         # C. Params (Theta, Decay, Tanh Scale)
#         theta_min, theta_max = 0.001, 3.0
#         def init_theta_m1(key, shape):
#             grid_k = np.geomspace(theta_min, theta_max, self.K).reshape(1, 1, self.K, 1, 1)
#             base = np.tile(grid_k, (1, 1, 1, shape[3], 1))
#             u = (base - theta_min) / max(theta_max - theta_min, 1e-6)
#             u = np.clip(u, 1e-4, 1 - 1e-4)
#             return jnp.array(np.log(u) - np.log(1 - u), dtype=jnp.float32)

#         theta_raw = self.param("theta_raw", init_theta_m1, (1, 1, self.K, H, 1))
#         theta = theta_min + (theta_max - theta_min) * jax.nn.sigmoid(theta_raw).astype(jnp.float32)

#         # Decay Logic
#         pos = jnp.arange(l, dtype=jnp.float32)
#         log_w_list = []
#         if self.num_decay_heads > 0:
#             d_slopes = self.param("decay_slopes", lambda r, s: jnp.log(jnp.exp(np.geomspace(0.001, 0.1, s[0])) - 1), (self.num_decay_heads,))
#             slopes = jax.nn.softplus(d_slopes).reshape(1, 1, -1, 1)
#             dist = jnp.maximum(jnp.float32((self.maxlen or l) - 1) - pos, 0.)
#             log_w_list.append(-slopes * dist[None, :, None, None])
#         if self.num_anchor_heads > 0:
#             a_slopes = self.param("anchor_slopes", lambda r, s: jnp.log(jnp.exp(np.geomspace(0.01, 0.1, s[0])) - 1), (self.num_anchor_heads,))
#             slopes_a = jax.nn.softplus(a_slopes).reshape(1, 1, -1, 1)
#             log_w_list.append(-slopes_a * pos[None, :, None, None])

#         log_time_weight = (jnp.concatenate(log_w_list, axis=2) if log_w_list else jnp.zeros((1, l, self.K, 1), jnp.float32))
#         score_scale = self.param("score_scale", nn.initializers.ones, (self.K,))
#         log_p = jnp.clip(score_scale[None, None, :, None] * s_raw.astype(jnp.float32), -20., 20.)
#         p_w = jnp.exp(log_p + log_time_weight)

#         # D. Pre-Scan Projections (OPTIMISATION MÉMOIRE CRITIQUE)
#         # On ne calcule QUE les petits tenseurs ici. Les gros GEMMs se feront dans le scan.

#         # 1. Gate Logits (B, L, K) -> Petit
#         gate_proj_w = self.param("gate_proj_w", nn.initializers.zeros, (d_model, self.K))
#         gate_proj_b = self.param("gate_proj_b", nn.initializers.constant(-3.0), (self.K,))
#         spec_gate_logits = jnp.dot(x, gate_proj_w) + gate_proj_b

#         # 2. Latent Skip (B, L, D/4) -> Petit
#         # On ne calcule PAS y_direct (B, L, 6*D) ici ! C'est ça qui tuait la mémoire.
#         c_skip_seq = nn.Dense(latent_dim, use_bias=False, name="skip_down")(x)

#         # 3. Tanh Scale
#         tanh_scale = self.param("tanh_scale", nn.initializers.ones, (self.K,))
#         tanh_scale_broad = tanh_scale[None, None, :, None, None] # 5D pour broadcasting

#         # ------------------------------------------------------------
#         # 2. DÉFINITION DES POIDS POUR LE SCAN
#         # ------------------------------------------------------------
#         # Ces poids seront utilisés DANS la boucle pour générer les gros tenseurs à la volée.

#         W_readout = self.param("W_readout", nn.initializers.glorot_uniform(), (self.K, 2 * H, dim_swiglu_per_head))
#         scale_readout = self.param("norm_scale", nn.initializers.ones, (1, 1, self.K, 2 * H))
#         highway_scale = self.param("highway_scale", nn.initializers.constant(1.0), (1, 1, self.K, 1))

#         # Poids "Lourds" utilisés uniquement en local
#         W_skip_up = self.param("skip_up_w", nn.initializers.glorot_uniform(), (latent_dim, dim_swiglu_total))
#         W_out = self.param("out_proj_w", nn.initializers.glorot_uniform(), (self.K * dim_expand, d_model))

#         # ------------------------------------------------------------
#         # 3. CHUNKING
#         # ------------------------------------------------------------
#         C = self.chunk_size
#         pad = (-l) % C
#         if pad:
#             def p(t, val=0.):
#                 pad_width = ((0,0),(0,pad)) + ((0,0),)*(t.ndim-2)
#                 return jnp.pad(t, pad_width, constant_values=val)

#             k_val = p(k_val)
#             p_w   = p(p_w, 0.)
#             q_re_seq = p(q_re_seq)
#             q_im_seq = p(q_im_seq)
#             c_skip_seq = p(c_skip_seq) # Pad du latent
#             spec_gate_logits = p(spec_gate_logits, -1e9)
#             Lp = l + pad
#         else:
#             Lp = l

#         n_chunks = Lp // C

#         def to_chunks(t):
#             sh = t.shape
#             return t.reshape(b, n_chunks, C, *sh[2:]).swapaxes(0, 1)

#         k_chunks = to_chunks(k_val)
#         p_chunks = to_chunks(p_w)
#         qre_chunks = to_chunks(q_re_seq)
#         qim_chunks = to_chunks(q_im_seq)
#         c_skip_chunks = to_chunks(c_skip_seq) # On passe le latent chunké
#         gate_chunks = to_chunks(spec_gate_logits)

#         # ------------------------------------------------------------
#         # 4. KERNEL HYBRIDE (Pre-Computed Inputs + Fused GEMMs)
#         # ------------------------------------------------------------
#         def step(carry, xs):
#             den_c, re_c, im_c = carry
#             k_x, p_x, qre_x, qim_x, c_skip_x, gate_logits_x = xs

#             # --- A. ON-THE-FLY MODULATION ---
#             k_x_f32 = k_x.astype(jnp.float32)[..., None]
#             phi = jnp.tanh(k_x_f32 * tanh_scale_broad) * theta
#             kvw = k_x_f32 * p_x[..., None]

#             re_x = (kvw * jnp.cos(phi))[..., 0]
#             im_x = (kvw * jnp.sin(phi))[..., 0]
#             den_x = p_x[..., 0].astype(jnp.float32)

#             # --- B. SCAN LOCAL ---
#             den_ps = jnp.cumsum(den_x, axis=1) + den_c[:, None, :]
#             re_ps  = jnp.cumsum(re_x,  axis=1) + re_c[:, None, :, :]
#             im_ps  = jnp.cumsum(im_x,  axis=1) + im_c[:, None, :, :]

#             # --- C. READOUT (FUSED) ---
#             inv_den = 1.0 / jnp.maximum(den_ps, 1e-4)
#             state_re = re_ps * inv_den[..., None]
#             state_im = im_ps * inv_den[..., None]

#             # GQA View
#             state_re_g = state_re.reshape(b, C, self.K_q, self.n_rep, H)
#             state_im_g = state_im.reshape(b, C, self.K_q, self.n_rep, H)

#             # Query (B, C, Kq, 1, H) - Déjà chunké proprement
#             qre_g = qre_x
#             qim_g = qim_x

#             # Hermitien
#             out_re_g = state_re_g * qre_g + state_im_g * qim_g
#             out_im_g = state_im_g * qre_g - state_re_g * qim_g

#             # Flatten & Concat
#             out_re = out_re_g.reshape(b, C, self.K, H)
#             out_im = out_im_g.reshape(b, C, self.K, H)
#             out_complex = jnp.concatenate([out_re, out_im], axis=-1).astype(self.compute_dtype)

#             # Proj Readout (Manuel)
#             y_spectral_raw = jnp.einsum('bckf,kfn->bckn', out_complex * scale_readout, W_readout)

#             # --- D. FUSION & SWIGLU (HEAVY GEMMS INSIDE SCAN) ---

#             # 1. Gate
#             spec_gate = jax.nn.sigmoid(gate_logits_x).astype(self.compute_dtype)[..., None]
#             y_spectral = y_spectral_raw * spec_gate

#             # 2. Skip Up-Projection (LOURD mais LOCAL)
#             # On projette le Latent (petit) vers le SwiGLU (énorme) ICI
#             # (B, C, Latent) @ (Latent, SwiGLU) -> (B, C, SwiGLU)
#             # Pas de stockage VRAM, le résultat est consommé tout de suite
#             y_dir_x = jnp.dot(c_skip_x, W_skip_up).reshape(b, C, self.K, dim_swiglu_per_head)

#             # 3. Fusion
#             y_val_dir, y_gate_dir = jnp.split(y_dir_x, 2, axis=-1)
#             y_val_spec, y_gate_spec = jnp.split(y_spectral, 2, axis=-1)

#             y_val = y_val_spec + y_val_dir * highway_scale
#             y_gate = y_gate_spec + y_gate_dir * highway_scale

#             y_act = y_val * jax.nn.silu(y_gate)

#             # 4. Out Projection (LOURD mais LOCAL)
#             y_flat = y_act.reshape(b, C, -1)
#             out_chunk = jnp.dot(y_flat, W_out) # (B, C, D)

#             new_carry = (den_ps[:, -1, :], re_ps[:, -1, :, :], im_ps[:, -1, :, :])
#             return new_carry, out_chunk

#         init = (
#             jnp.zeros((b, self.K), dtype=jnp.float32),
#             jnp.zeros((b, self.K, H), dtype=jnp.float32),
#             jnp.zeros((b, self.K, H), dtype=jnp.float32),
#         )

#         _, out_chunks = jax.lax.scan(
#             step, init,
#             (k_chunks, p_chunks, qre_chunks, qim_chunks, c_skip_chunks, gate_chunks),
#             length=n_chunks
#         )

#         out = out_chunks.swapaxes(0, 1).reshape(b, Lp, d_model)[:, :l, :]

#         if self.dropout > 0:
#             out = nn.Dropout(self.dropout)(out, deterministic=deterministic)

#         return out


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
        [jnp.zeros_like(offsets[:, :1, ...]), offsets[:, :-1, ...]], axis=1
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
    num_heads: int = 12  # K
    num_query_heads: int = 6  # K'
    num_anchor_heads: int = 0
    num_thetas: int = 1  # M

    # Paramètres Locaux
    conv_kernel_size: int = 4
    expand_factor: int = 1  # Input Slim (Scan Rapide)
    out_expand_factor: int = 3  # Output Fat (Cerveau SwiGLU) - Ajustable selon VRAM
    skip_low_rank: bool = (
        True  # If True, use D//4 latent skip; if False, use full dim_swiglu_total
    )

    dropout: float = 0.0
    maxlen: Optional[int] = None
    chunk_size: int = 0

    use_square_matrix: bool = False

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
        # ======================================================================
        # 0. SETUP DIMENSIONS
        # ======================================================================
        B, L, D = x.shape
        d_inner = int(D * self.expand_factor)

        # Iso-Param H scaling
        H = max(1, d_inner // (self.K * self.M))

        dim_memory = self.K * H

        dim_query_head = H * self.M * 2
        dim_query_total = self.K_q * dim_query_head

        dim_expand = H * self.out_expand_factor
        dim_swiglu_head = dim_expand * 2
        dim_swiglu_total = self.K * dim_swiglu_head

        # Skip dimension: low-rank (D//4) or full (dim_swiglu_total)
        dim_skip = D // 4 if self.skip_low_rank else dim_swiglu_total
        dim_gate = self.K

        # ======================================================================
        # 1. FUSED INPUT PROJECTION (Mamba-style single projection + conv)
        # ======================================================================
        # All inputs in one projection: conv_branch (mem + query) + skip + gate
        dim_mem_total = dim_memory + self.K  # k_val + s_raw
        dim_conv_total = (
            dim_mem_total + dim_query_total
        )  # Everything that goes through conv
        dim_total = dim_conv_total + dim_skip + dim_gate

        z_all = nn.Dense(dim_total, use_bias=False, name="in_proj")(x)
        z_all = z_all.astype(self.compute_dtype)

        # Split: conv_branch vs non-conv branches
        z_conv = z_all[..., :dim_conv_total]
        c_skip = z_all[..., dim_conv_total : dim_conv_total + dim_skip]
        gate_logits = z_all[..., dim_conv_total + dim_skip :]

        # Single fused conv on mem+query (Mamba style)
        z_conv = nn.Conv(
            features=z_conv.shape[-1],
            kernel_size=(self.conv_kernel_size,),
            padding=((self.conv_kernel_size - 1, 0),),
            feature_group_count=z_conv.shape[-1],
            use_bias=False,
            name="conv",
        )(z_conv)
        z_conv = jax.nn.silu(z_conv)

        # Split conv output into memory and query
        z_mem = z_conv[..., :dim_mem_total]
        q_raw = z_conv[..., dim_mem_total:]

        # Process memory branch
        k_val = z_mem[..., :dim_memory].reshape(B, L, self.K, H)
        k_val = nn.RMSNorm(dtype=self.compute_dtype, name="k_norm")(k_val)
        s_raw = z_mem[..., dim_memory:]

        if mask is not None:
            m = mask.astype(x.dtype)[:, :, None, None]
            s_raw = s_raw * mask.astype(x.dtype)[..., None]
            k_val = k_val * m

        # Process query branch
        q_raw = nn.RMSNorm(dtype=self.compute_dtype, name="q_norm")(q_raw)
        q_raw = q_raw.reshape(B, L, self.K_q, 1, H, self.M, 2)
        q_re, q_im = q_raw[..., 0], q_raw[..., 1]

        # ======================================================================
        # 2. GRILLE SPECTRALE (THETA & W_INT)
        # ======================================================================
        theta_min, theta_max = 0.001, 3.0
        if self.M == 1:

            def init_theta_m1(key, shape):
                grid = np.geomspace(theta_min, theta_max, self.K).reshape(
                    1, 1, self.K, 1, 1
                )
                base = np.tile(grid, (1, 1, 1, H, 1))
                # Inverse Softplus (Mamba2 style robust)
                u = (base - theta_min) / max(theta_max - theta_min, 1e-6)
                u = np.clip(u, 1e-4, 1 - 1e-4)
                raw = np.log(u) - np.log(1 - u)
                return jnp.array(raw, dtype=jnp.float32)

            theta_raw = self.param("theta_raw", init_theta_m1, (1, 1, self.K, H, 1))
            theta = theta_min + (theta_max - theta_min) * jax.nn.sigmoid(
                theta_raw
            ).astype(jnp.float32)

            # Poids d'intégration appris (initialisés à 1)
            w_int_raw = self.param(
                "w_int_raw", nn.initializers.zeros, (1, 1, self.K_q, self.n_rep, H, 1)
            )
            w_int_raw_clipped = jnp.clip(w_int_raw, -5.0, 5.0)  # Prevent exp overflow
            w_int = jnp.exp(w_int_raw_clipped).astype(jnp.float32)  # Toujours positif
        else:

            def init_theta_deltas(key, shape):
                grid_m = np.geomspace(theta_min, theta_max, self.M)
                base = np.tile(grid_m.reshape(1, 1, 1, 1, self.M), (1, 1, self.K, H, 1))
                return jnp.log(jnp.exp(base) - 1.0 + 1e-4)

            theta_d_raw = self.param(
                "theta_d_raw", init_theta_deltas, (1, 1, self.K, H, self.M)
            )

            theta_d = jax.nn.softplus(theta_d_raw).astype(jnp.float32) + 1e-4
            theta_accum = jnp.cumsum(theta_d, axis=-1)

            scale_range = theta_max - theta_min
            total_sum = theta_accum[..., -1:]
            theta = theta_min + (theta_accum / total_sum) * scale_range

            dtheta_raw = theta_accum[..., 1:] - theta_accum[..., :-1]
            dtheta = dtheta_raw * (scale_range / total_sum)

            w0 = dtheta[..., :1] * 0.5
            w_mid = 0.5 * (dtheta[..., :-1] + dtheta[..., 1:])
            wL = dtheta[..., -1:] * 0.5

            w_int = jnp.concatenate([w0, w_mid, wL], axis=-1)
            w_int = w_int.reshape(1, 1, self.K_q, self.n_rep, H, self.M)

        # ======================================================================
        # 3. MODULATION & SCAN UNIFIÉ (MATRIX vs LINEAR)
        # ======================================================================
        # Decay (Log-Space)
        pos = jnp.arange(L, dtype=jnp.float32)
        log_w_list = []
        if self.num_decay_heads > 0:
            d_slopes = self.param(
                "decay_slopes",
                lambda r, s: jnp.log(jnp.exp(np.geomspace(0.001, 0.1, s[0])) - 1),
                (self.num_decay_heads,),
            )
            slopes = jax.nn.softplus(d_slopes).reshape(1, 1, -1)
            dist = jnp.maximum(jnp.float32((self.maxlen or L) - 1) - pos, 0.0)
            log_w_list.append(-slopes * dist[None, :, None])
        if self.num_anchor_heads > 0:
            a_slopes = self.param(
                "anchor_slopes",
                lambda r, s: jnp.log(jnp.exp(np.geomspace(0.01, 0.1, s[0])) - 1),
                (self.num_anchor_heads,),
            )
            slopes_a = jax.nn.softplus(a_slopes).reshape(1, 1, -1)
            log_w_list.append(-slopes_a * pos[None, :, None])

        log_time_weight = (
            jnp.concatenate(log_w_list, axis=2)
            if log_w_list
            else jnp.zeros((1, L, self.K), dtype=jnp.float32)
        )
        score_scale = self.param("score_scale", nn.initializers.ones, (self.K,))
        score_bias = self.param("score_bias", nn.initializers.zeros, (self.K,))
        score_raw = (
            score_scale[None, None, :] * s_raw.astype(jnp.float32)
            + score_bias[None, None, :]
        )

        # Softplus instead of exp for stability (Mamba-like)
        p_w_content = jax.nn.softplus(score_raw)

        # Temporal weight also with softplus
        temporal_weight = jax.nn.softplus(log_time_weight)

        # Combine and clip for safety
        p_w = p_w_content * temporal_weight
        p_w = jnp.clip(p_w, 1e-6, 1000.0)  # (B, L, K)

        # Modulation (with tanh to bound phase)
        k_f32 = k_val.astype(jnp.float32)[..., None]
        p_w_b = p_w[..., None, None]

        tanh_scale = self.param("tanh_scale", nn.initializers.ones, (self.K,))
        tanh_scale_b = tanh_scale[None, None, :, None, None]
        phi = jnp.tanh(k_f32 * tanh_scale_b) * theta  # Bounded modulation
        kvw = k_f32 * p_w_b

        # re, im: (B, L, K, H, M)
        re = kvw * jnp.cos(phi)
        im = kvw * jnp.sin(phi)

        # --- DYNAMIC SWITCH ---
        # use_square_matrix = L <= self.matrix_threshold
        if self.use_square_matrix:
            # PATH 1: O(L^2) Matrix Multiply (Tensor Core Optimized)
            # Causal Mask (L, L)
            # trues below diagonal
            mask_idx = jnp.arange(L)[:, None] >= jnp.arange(L)[None, :]
            causal_mask = mask_idx.astype(self.compute_dtype)

            # Einsum Accumulation
            # p_w: (B, L, K) -> den_acc: (B, L, K) via mask (L, L)
            den_acc = jnp.einsum(
                "ts,bsk->btk", causal_mask, p_w.astype(self.compute_dtype)
            )

            # Stack RE/IM for single einsum
            # (B, L, K, H, M, 2)
            stack_ri = jnp.stack([re, im], axis=-1).astype(self.compute_dtype)
            # acc_ri: (B, L, K, H, M, 2)
            acc_ri = jnp.einsum("ts,bskhmc->btkhmc", causal_mask, stack_ri)

            re_acc = acc_ri[..., 0]
            im_acc = acc_ri[..., 1]

        else:
            # PATH 2: O(L) Linear Scan (Memory Optimized)
            flat_size = self.K * H * self.M
            re_flat = re.reshape(B, L, flat_size)
            im_flat = im.reshape(B, L, flat_size)
            den_flat = p_w  # (B, L, K)

            # Stack & Scan
            stack = jnp.concatenate([den_flat, re_flat, im_flat], axis=-1)
            cumsum = jnp.cumsum(stack, axis=1)

            # Unpack
            den_acc, re_acc_flat, im_acc_flat = jnp.split(
                cumsum, [self.K, self.K + flat_size], axis=-1
            )
            re_acc = re_acc_flat.reshape(B, L, self.K, H, self.M)
            im_acc = im_acc_flat.reshape(B, L, self.K, H, self.M)

        # Normalisation
        inv_den = 1.0 / jnp.maximum(den_acc, 1e-4)
        inv_den = inv_den[..., None, None]  # (B, L, K, 1, 1)

        state_re = re_acc * inv_den
        state_im = im_acc * inv_den

        # ======================================================================
        # 4. READOUT & GQA & INTEGRATION
        # ======================================================================

        state_re_g = state_re.reshape(B, L, self.K_q, self.n_rep, H, self.M)
        state_im_g = state_im.reshape(B, L, self.K_q, self.n_rep, H, self.M)

        match_re = state_re_g * q_re + state_im_g * q_im
        match_im = state_im_g * q_re - state_re_g * q_im

        out_re_g = jnp.sum(match_re * w_int, axis=-1)
        out_im_g = jnp.sum(match_im * w_int, axis=-1)

        out_re = out_re_g.reshape(B, L, self.K, H).astype(self.compute_dtype)
        out_im = out_im_g.reshape(B, L, self.K, H).astype(self.compute_dtype)

        out_complex = jnp.concatenate([out_re, out_im], axis=-1)

        # ======================================================================
        # 5. FUSION FINALE with GatedRMSNorm
        # ======================================================================
        W_readout = self.param(
            "W_readout",
            nn.initializers.glorot_uniform(),
            (self.K, 2 * H, dim_swiglu_head),
        )

        # Apply GatedRMSNorm to spectral output (Mamba2 style)
        # gate_logits acts as the residual gate
        out_complex_flat = out_complex.reshape(B, L, -1)
        gate_for_norm = nn.Dense(
            out_complex_flat.shape[-1], use_bias=False, name="gate_proj"
        )(gate_logits)
        out_normed = GatedRMSNorm(dtype=self.compute_dtype, name="gated_norm")(
            out_complex_flat, gate_for_norm
        )
        out_complex = out_normed.reshape(B, L, self.K, 2 * H)

        y_spec_raw = jnp.einsum("blkf,kfn->blkn", out_complex, W_readout)

        # Skip connection: low-rank or full
        if self.skip_low_rank:
            y_skip_raw = nn.Dense(dim_swiglu_total, use_bias=False, name="skip_up")(
                c_skip
            )
        else:
            y_skip_raw = c_skip  # Already dim_swiglu_total
        y_skip = y_skip_raw.reshape(B, L, self.K, dim_swiglu_head)

        y_spec_val, y_spec_gate = jnp.split(y_spec_raw, 2, axis=-1)
        y_skip_val, y_skip_gate = jnp.split(y_skip, 2, axis=-1)

        highway_scale = self.param(
            "highway_scale", nn.initializers.constant(1.0), (1, 1, self.K, 1)
        )

        y_val = y_spec_val + (y_skip_val * highway_scale)
        y_gate = y_spec_gate + (y_skip_gate * highway_scale)

        y_act = y_val * jax.nn.sigmoid(y_gate)

        y_flat = y_act.reshape(B, L, -1)
        out = nn.Dense(D, use_bias=False, name="out_proj")(y_flat)

        if self.dropout > 0:
            out = nn.Dropout(self.dropout)(out, deterministic=deterministic)

        return out

    @nn.compact
    def step(self, x_t, state, deterministic=True):
        """
        O(1) autoregressive decoding step for seqcond_fast.

        Args:
            x_t: Input token embedding (B, D)
            state: Tuple of (den_acc, re_acc, im_acc, pos, conv_buffer) where:
                - den_acc: (B, K) - accumulated denominator
                - re_acc: (B, K, H, M) - accumulated real part
                - im_acc: (B, K, H, M) - accumulated imaginary part
                - pos: (B,) - current position
                - conv_buffer: (B, conv_kernel_size-1, dim_conv_total) - fused conv history
            deterministic: Whether to use dropout

        Returns:
            out: (B, D) - output for this step
            new_state: Updated state tuple
        """
        B, D = x_t.shape
        d_inner = int(D * self.expand_factor)

        H = max(1, d_inner // (self.K * self.M))
        dim_memory = self.K * H

        dim_query_head = H * self.M * 2
        dim_query_total = self.K_q * dim_query_head

        dim_expand = H * self.out_expand_factor
        dim_swiglu_head = dim_expand * 2
        dim_swiglu_total = self.K * dim_swiglu_head

        # Skip dimension: low-rank (D//4) or full (dim_swiglu_total)
        dim_skip = D // 4 if self.skip_low_rank else dim_swiglu_total
        dim_gate = self.K

        # Unpack state (now with single fused conv buffer)
        den_acc, re_acc, im_acc, pos, conv_buffer = state

        # ======================================================================
        # FUSED INPUT PROJECTION + CONV (matches __call__)
        # ======================================================================
        dim_mem_total = dim_memory + self.K
        dim_conv_total = dim_mem_total + dim_query_total
        dim_total = dim_conv_total + dim_skip + dim_gate

        z_all = nn.Dense(dim_total, use_bias=False, name="in_proj")(x_t)
        z_all = z_all.astype(self.compute_dtype)

        # Split: conv_branch vs non-conv branches
        z_conv = z_all[..., :dim_conv_total]
        c_skip = z_all[..., dim_conv_total : dim_conv_total + dim_skip]
        gate_logits = z_all[..., dim_conv_total + dim_skip :]

        # Single fused conv with buffer
        z_conv_expanded = z_conv[:, None, :]
        conv_input = jnp.concatenate([conv_buffer, z_conv_expanded], axis=1)

        z_conv_out = nn.Conv(
            features=conv_input.shape[-1],
            kernel_size=(self.conv_kernel_size,),
            padding="VALID",
            feature_group_count=conv_input.shape[-1],
            use_bias=False,
            name="conv",
        )(conv_input)

        z_conv = jax.nn.silu(z_conv_out[:, 0, :])
        conv_buffer_new = jnp.concatenate(
            [conv_buffer[:, 1:, :], z_conv_expanded], axis=1
        )

        # Split conv output into memory and query
        z_mem = z_conv[..., :dim_mem_total]
        q_raw = z_conv[..., dim_mem_total:]

        # Process memory branch
        k_val = z_mem[..., :dim_memory].reshape(B, self.K, H)
        k_val = nn.RMSNorm(dtype=self.compute_dtype, name="k_norm")(k_val)
        s_raw = z_mem[..., dim_memory:]

        # Process query branch (matches __call__)
        q_raw = nn.RMSNorm(dtype=self.compute_dtype, name="q_norm")(q_raw)
        q_raw = q_raw.reshape(B, self.K_q, 1, H, self.M, 2)
        q_re, q_im = q_raw[..., 0], q_raw[..., 1]  # (B, K_q, 1, H, M)

        # Theta grid
        theta_min, theta_max = 0.001, 3.0

        if self.M == 1:

            def init_theta_m1(key, shape):
                grid = np.geomspace(theta_min, theta_max, self.K).reshape(
                    1, 1, self.K, 1, 1
                )
                base = np.tile(grid, (1, 1, 1, H, 1))
                u = (base - theta_min) / max(theta_max - theta_min, 1e-6)
                u = np.clip(u, 1e-4, 1 - 1e-4)
                raw = np.log(u) - np.log(1 - u)
                return jnp.array(raw, dtype=jnp.float32)

            theta_raw = self.param("theta_raw", init_theta_m1, (1, 1, self.K, H, 1))
            theta = theta_min + (theta_max - theta_min) * jax.nn.sigmoid(
                theta_raw
            ).astype(jnp.float32)
        else:

            def init_theta_deltas(key, shape):
                grid_m = np.geomspace(theta_min, theta_max, self.M)
                base = np.tile(grid_m.reshape(1, 1, 1, 1, self.M), (1, 1, self.K, H, 1))
                return jnp.log(jnp.exp(base) - 1.0 + 1e-4)

            theta_d_raw = self.param(
                "theta_d_raw", init_theta_deltas, (1, 1, self.K, H, self.M)
            )
            theta_d = jax.nn.softplus(theta_d_raw).astype(jnp.float32) + 1e-4
            theta_accum = jnp.cumsum(theta_d, axis=-1)

            scale_range = theta_max - theta_min
            total_sum = theta_accum[..., -1:]
            theta = theta_min + (theta_accum / total_sum) * scale_range

        theta = theta[0, 0]  # (K, H, M)

        # Decay weights
        log_w_list = []
        if self.num_decay_heads > 0:
            d_slopes = self.param(
                "decay_slopes",
                lambda r, s: jnp.log(jnp.exp(np.geomspace(0.001, 0.1, s[0])) - 1),
                (self.num_decay_heads,),
            )
            slopes = jax.nn.softplus(d_slopes).reshape(1, -1)
            dist = jnp.maximum(
                jnp.float32((self.maxlen or 2048) - 1) - pos[:, None], 0.0
            )
            log_w_list.append(-slopes * dist)
        if self.num_anchor_heads > 0:
            a_slopes = self.param(
                "anchor_slopes",
                lambda r, s: jnp.log(jnp.exp(np.geomspace(0.01, 0.1, s[0])) - 1),
                (self.num_anchor_heads,),
            )
            slopes_a = jax.nn.softplus(a_slopes).reshape(1, -1)
            log_w_list.append(-slopes_a * pos[:, None])

        log_time_weight = (
            jnp.concatenate(log_w_list, axis=1)
            if log_w_list
            else jnp.zeros((B, self.K), dtype=jnp.float32)
        )

        score_scale = self.param("score_scale", nn.initializers.ones, (self.K,))
        score_bias = self.param("score_bias", nn.initializers.zeros, (self.K,))
        score_raw = (
            score_scale[None, :] * s_raw.astype(jnp.float32) + score_bias[None, :]
        )
        score_raw = jnp.clip(score_raw, -20.0, 20.0)

        # Softplus instead of exp for stability
        p_w_content = jax.nn.softplus(score_raw)
        temporal_weight = jax.nn.softplus(log_time_weight)
        p_w = p_w_content * temporal_weight
        p_w = jnp.clip(p_w, 1e-6, 100.0)

        # Modulation
        k_f32 = k_val[..., None].astype(jnp.float32)  # (B, K, H, 1)
        p_w_b = p_w[..., None, None]  # (B, K, 1, 1)

        tanh_scale = self.param("tanh_scale", nn.initializers.ones, (self.K,))
        tanh_scale_b = tanh_scale.reshape(1, self.K, 1, 1)
        phi = jnp.tanh(k_f32 * tanh_scale_b) * theta  # (B, K, H, M)
        kvw = k_f32 * p_w_b

        re = kvw * jnp.cos(phi)
        im = kvw * jnp.sin(phi)

        # Update accumulation
        den_acc_new = den_acc + p_w
        re_acc_new = re_acc + re
        im_acc_new = im_acc + im

        # Normalize
        inv_den = 1.0 / jnp.maximum(den_acc_new, 1e-4)
        inv_den = inv_den[..., None, None]

        state_re = re_acc_new * inv_den
        state_im = im_acc_new * inv_den

        # Repeat state for query heads
        state_re_rep = jnp.repeat(
            state_re[:, None, :, :, :], self.n_rep, axis=1
        ).reshape(B, self.K, H, self.M)
        state_im_rep = jnp.repeat(
            state_im[:, None, :, :, :], self.n_rep, axis=1
        ).reshape(B, self.K, H, self.M)

        # Group queries
        state_re_grouped = state_re_rep.reshape(B, self.K_q, self.n_rep, H, self.M)
        state_im_grouped = state_im_rep.reshape(B, self.K_q, self.n_rep, H, self.M)

        # Compute attention
        q_re_b = q_re[:, :, None, :, :]
        q_im_b = q_im[:, :, None, :, :]

        out_re = q_re_b * state_re_grouped - q_im_b * state_im_grouped
        out_im = q_re_b * state_im_grouped + q_im_b * state_re_grouped

        # Flatten and apply GatedRMSNorm
        out_complex = jnp.stack([out_re, out_im], axis=-1).reshape(B, -1)
        out_complex = out_complex.astype(self.compute_dtype)

        # GatedRMSNorm (matches __call__)
        gate_for_norm = nn.Dense(
            out_complex.shape[-1], use_bias=False, name="gate_proj"
        )(gate_logits)
        out_normed = GatedRMSNorm(dtype=self.compute_dtype, name="gated_norm")(
            out_complex, gate_for_norm
        )
        out_complex = out_normed.reshape(B, self.K, 2 * H)

        # Readout
        W_readout = self.param(
            "W_readout",
            nn.initializers.glorot_uniform(),
            (self.K, 2 * H, dim_swiglu_head),
        )
        y_spec_raw = jnp.einsum("bkf,kfn->bkn", out_complex, W_readout)

        # Skip connection
        if self.skip_low_rank:
            y_skip_raw = nn.Dense(dim_swiglu_total, use_bias=False, name="skip_up")(
                c_skip
            )
        else:
            y_skip_raw = c_skip
        y_skip = y_skip_raw.reshape(B, self.K, dim_swiglu_head)

        y_spec_val, y_spec_gate = jnp.split(y_spec_raw, 2, axis=-1)
        y_skip_val, y_skip_gate = jnp.split(y_skip, 2, axis=-1)

        highway_scale = self.param(
            "highway_scale", nn.initializers.constant(1.0), (1, 1, self.K, 1)
        )
        highway_scale_step = highway_scale[0, 0]  # (K, 1)

        y_val = y_spec_val + (y_skip_val * highway_scale_step)
        y_gate = y_spec_gate + (y_skip_gate * highway_scale_step)

        y_act = y_val * jax.nn.sigmoid(y_gate)

        # Output
        y_flat = y_act.reshape(B, -1)
        out = nn.Dense(D, use_bias=False, name="out_proj")(y_flat)

        if self.dropout > 0:
            out = nn.Dropout(self.dropout)(out, deterministic=deterministic)

        # Update position
        pos_new = pos + 1

        return out, (
            den_acc_new,
            re_acc_new,
            im_acc_new,
            pos_new,
            conv_buffer_new,
        )


class SeqCondBlock(nn.Module):
    num_heads: int = 32
    num_query_heads: int = 6
    expand_factor: float = 1.0
    out_expand_factor: int = 3  # SwiGLU expansion factor
    num_thetas: int = 1
    num_anchor_heads: int = 0
    conv_kernel_size: int = 4
    skip_low_rank: bool = False  # If True, use D//4 latent skip; if False, use full dim
    dropout: float = 0.0
    norm_eps: float = 1e-5
    maxlen: Optional[int] = None
    derivative_order: Optional[int] = 0
    chunk_size: int = 32
    use_square_matrix: bool = False

    compute_dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        h = nn.RMSNorm(
            epsilon=self.norm_eps,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )(x)
        h = SeqCondAttention(
            num_heads=self.num_heads,
            num_query_heads=self.num_query_heads,
            expand_factor=self.expand_factor,
            out_expand_factor=self.out_expand_factor,
            num_thetas=self.num_thetas,
            num_anchor_heads=self.num_anchor_heads,
            conv_kernel_size=self.conv_kernel_size,
            skip_low_rank=self.skip_low_rank,
            dropout=self.dropout,
            maxlen=self.maxlen,
            chunk_size=self.chunk_size,
            use_square_matrix=self.use_square_matrix,
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
