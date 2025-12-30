from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Optional


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
        H = max(1, d_inner // self.K)
        
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
        tanh_scale = self.param("tanh_scale", nn.initializers.ones, (self.K,))
        log_p = jnp.clip(score_scale[None, None, :, None] * s_raw, -20., 20.)
        p_w = jnp.exp(log_p + log_time_weight)

        # B. Modulation & Scan
        k_val_expand = k_val[..., None, None]
        kvw = k_val_expand * p_w[..., None]
        tanh_scale_broad = tanh_scale[None, None, :, None, None]
        phi_k = jnp.tanh(tanh_scale_broad * k_val_expand) * theta  # cheap norm
        re_k, im_k = kvw * jnp.cos(phi_k), kvw * jnp.sin(phi_k)

        den_in = p_w.squeeze(-1)
        num_re_in = re_k.reshape(b, l, flat_dim)
        num_im_in = im_k.reshape(b, l, flat_dim)
        
        merged = jnp.concatenate([den_in, num_re_in, num_im_in], axis=-1)
        if self.chunk_size > 0:
            cumsum = cumsum_chunked(merged, axis=1, chunk=self.chunk_size)
        else:
            cumsum = jnp.cumsum(merged, axis=1)
        # cumsum = cumsum.astype(self.compute_dtype)
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
    chunk_size: int = 32

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
