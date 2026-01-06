import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from typing import Optional

from .rope import RMSNorm, apply_rope

class BivectorRotarySelfAttention(nn.Module):
    d_model: int
    num_heads: int
    num_kv_heads: Optional[int] = None
    dropout: float = 0.0
    qk_norm: bool = False
    qk_norm_eps: float = 1e-5
    
    # Paramètres spécifiques Bivector
    init_q_scale: float = 0.5

    def setup(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        
        self._num_kv_heads = (
            self.num_kv_heads if self.num_kv_heads is not None else self.num_heads
        )
        
        # Le Bivector attention divise la dimension de tête en 2 sous-espaces
        self.head_dim = self.d_model // self.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be divisible by 2 for Bivector Attention")

        self.num_groups = self.num_heads // self._num_kv_heads

        # Projections
        self.q_proj = nn.Dense(self.d_model, use_bias=False)
        self.k_proj = nn.Dense(self._num_kv_heads * self.head_dim, use_bias=False)
        self.v_proj = nn.Dense(self._num_kv_heads * self.head_dim, use_bias=False)
        self.out_proj = nn.Dense(self.d_model, use_bias=False)
        
        self.attn_dropout = nn.Dropout(self.dropout)

        # STABILISATION 1: RMSNorm (Equivalent à la logique TF RMSNormalization)
        self.q_norm = nn.RMSNorm(epsilon=self.qk_norm_eps, dtype=jnp.float32)
        self.k_norm = nn.RMSNorm(epsilon=self.qk_norm_eps, dtype=jnp.float32)

        # Initialisation du paramètre 'q' (facteur de mélange bivectoriel)
        def init_q_param(key, shape):
            # Reproduction du pattern TF: linspace(0.5, -0.5) -> arctanh
            pattern = np.linspace(self.init_q_scale, -self.init_q_scale, self.num_heads)
            q_values = [pattern[i % len(pattern)] for i in range(self.num_heads)]
            # Shape: [1, H, 1, 1] pour broadcaster sur [B, H, L, M]
            arr = np.array(q_values, dtype=np.float32).reshape(1, self.num_heads, 1, 1)
            return jnp.array(np.arctanh(arr))

        self.q_param = self.param("q", init_q_param, (1, self.num_heads, 1, 1))

        # STABILISATION 2: Température apprenable
        def init_log_scale(key, shape):
            return jnp.full(shape, jnp.log(1.0))
        
        self.log_scale = self.param("log_scale", init_log_scale, (1, self.num_heads, 1, 1))

    def _repeat_kv(self, x: jnp.ndarray) -> jnp.ndarray:
        """Repeat KV heads to match Q heads for GQA."""
        if self.num_groups == 1:
            return x
        # x input shape: [B, L, H_kv, 2, D/2] (puisque nous avons splité avant)
        b, l, h_kv, splits, d_split = x.shape
        x = x[:, :, :, None, :, :] # Expand dim for groups
        x = jnp.tile(x, [1, 1, 1, self.num_groups, 1, 1])
        # Reshape to merge groups into heads
        return x.reshape(b, l, self.num_heads, splits, d_split)

    def __call__(
        self,
        x: jnp.ndarray,
        cos: jnp.ndarray, # Attendu shape compatible avec D/2 ou D complet géré par apply_rope
        sin: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        B, L = x.shape[0], x.shape[1]

        # 1. Projections et Reshape pour séparer les deux composantes du bivecteur
        # Output shape desire: [B, L, H, 2, D//2]
        q = self.q_proj(x).reshape(B, L, self.num_heads, 2, self.head_dim // 2)
        k = self.k_proj(x).reshape(B, L, self._num_kv_heads, 2, self.head_dim // 2)
        v = self.v_proj(x).reshape(B, L, self._num_kv_heads, self.head_dim) # V reste standard

        # 2. STABILISATION: Normalisation pré-interaction
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 3. RoPE
        # On applique RoPE sur chaque sous-vecteur individuellement (index 0 et index 1)
        # Supposons que 'apply_rope' gère le broadcast ou l'input [B, L, H, D_head]
        # On extrait les sous-espaces
        q0, q1 = q[..., 0, :], q[..., 1, :]
        k0, k1 = k[..., 0, :], k[..., 1, :]

        # Adaptation du RoPE pour GQA si nécessaire sur les dimensions K
        if self._num_kv_heads < self.num_heads:
             cos_kv, sin_kv = cos[:, :, :self._num_kv_heads, :], sin[:, :, :self._num_kv_heads, :]
        else:
             cos_kv, sin_kv = cos, sin

        q0, q1 = apply_rope(q0, cos, sin), apply_rope(q1, cos, sin)
        k0, k1 = apply_rope(k0, cos_kv, sin_kv), apply_rope(k1, cos_kv, sin_kv)

        # Re-stack [B, L, H_kv, 2, D/2]
        q_rope = jnp.stack([q0, q1], axis=3)
        k_rope = jnp.stack([k0, k1], axis=3)

        # GQA: Répétition des têtes KV pour matcher Q
        k_rope = self._repeat_kv(k_rope) # -> [B, L, H_q, 2, D/2]
        v = self._repeat_kv(v.reshape(B, L, self._num_kv_heads, 1, self.head_dim)).reshape(B, L, self.num_heads, self.head_dim)

        # 4. Interaction Bivecteur (Force Float32 pour éviter Catastrophic Cancellation)
        q_rope = q_rope.astype(jnp.float32)
        k_rope = k_rope.astype(jnp.float32)

        # Extraire les composantes du bivecteur après GQA
        # q0, q1 shape: [B, L, H, D/2]
        # k0, k1 shape: [B, L, H, D/2]
        q0, q1 = q_rope[..., 0, :], q_rope[..., 1, :]
        k0, k1 = k_rope[..., 0, :], k_rope[..., 1, :]

        # Calculer les 4 produits scalaires matriciels directement
        # Chaque résultat a la forme [B, H, L, M]
        q0k0 = jnp.einsum("blhd,bmhd->bhlm", q0, k0)
        q1k1 = jnp.einsum("blhd,bmhd->bhlm", q1, k1)
        q0k1 = jnp.einsum("blhd,bmhd->bhlm", q0, k1)
        q1k0 = jnp.einsum("blhd,bmhd->bhlm", q1, k0)

        # Calcul du "Déterminant" généralisé
        diag = q0k0 * q1k1
        cross = q0k1 * q1k0

        # Scores bruts
        raw_scores = diag - 2 * jnp.tanh(self.q_param) * cross

        # 5. Scaling
        scale = jnp.exp(self.log_scale)
        # Division par head_dim explicite comme dans le code TF original
        # Attention: scale est appliqué APRÈS le produit quadratique
        scores = raw_scores * scale / jnp.float32(self.head_dim)

        # 6. Masking & Softmax
        large_neg = -1e9 # JAX préfère des valeurs finies mais grandes
        
        # Causal Mask
        causal_mask = jnp.tril(jnp.ones((L, L), dtype=jnp.bool_))[None, None, :, :]
        scores = jnp.where(causal_mask, scores, large_neg)

        # Padding Mask (si fourni)
        if mask is not None:
             # mask shape [B, M] -> [B, 1, 1, M]
             key_mask = mask.astype(jnp.bool_)[:, None, None, :]
             scores = jnp.where(key_mask, scores, large_neg)

        attn = jax.nn.softmax(scores, axis=-1)
        attn = attn.astype(v.dtype) # Retour au dtype original (ex: bfloat16)
        attn = self.attn_dropout(attn, deterministic=deterministic)

        # Output projection
        out = jnp.einsum("bhql,blhd->bqhd", attn, v)
        out = out.reshape(B, L, self.d_model)
        
        return self.out_proj(out)


class TransformerDecoderBlock(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    num_kv_heads: Optional[int] = None
    dropout: float = 0.0
    norm_eps: float = 1e-6
    qk_norm: bool = False
    qk_norm_eps: float = 1e-6

    def setup(self):
        self.norm1 = RMSNorm(epsilon=self.norm_eps)
        self.attn = BivectorRotarySelfAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            dropout=self.dropout,
            qk_norm=self.qk_norm,
            qk_norm_eps=self.qk_norm_eps,
        )
        self.drop1 = nn.Dropout(self.dropout)

        self.norm2 = RMSNorm(epsilon=self.norm_eps)
        self.ff_in = nn.Dense(2 * self.d_ff)
        self.ff_out = nn.Dense(self.d_model)
        self.drop2 = nn.Dropout(self.dropout)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        cos: jnp.ndarray,
        sin: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        y = self.norm1(x)
        y = self.attn(y, cos=cos, sin=sin, mask=mask, deterministic=deterministic)
        x = x + self.drop1(y, deterministic=deterministic)

        y = self.norm2(x)
        u, v = jnp.split(self.ff_in(y), 2, axis=-1)
        y = jax.nn.swish(v) * u
        y = self.ff_out(y)
        return x + self.drop2(y, deterministic=deterministic)
