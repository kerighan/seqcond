# Model Upscaling — Head-Aware Weight Duplication

## TL;DR

Upscale SeqCond 370M → 1.26B **instantanément** (pas d'optimisation, pas de GPU, pas de données).
Le modèle upscalé est fonctionnellement équivalent au petit : `downsample(y_large) == y_small` (loss = 0).
Il génère du texte cohérent dès l'init, et sert de point de départ pour continuer le training sur le gros modèle.

## Commande pour reproduire

```bash
# 1. Générer la config cible (une seule fois)
python -c "
from seqcond.config import ModelConfig
import json
cfg = ModelConfig(
    model_type='seqcond',
    d_model=2048, d_ff=5460, num_layers=24,
    vocab_size=100300, maxlen=2048,
    num_heads=32, num_kv_heads=8, qk_norm=True,
    seqcond_heads=32, num_query_heads=32,
    num_thetas=2, conv_kernel_size=4,
    expand_factor=2.0, out_expand_factor=3, seqcond_ratio=2,
)
with open('target_config_xlarge.json', 'w') as f:
    json.dump(cfg.to_dict(), f, indent=2)
"

# 2. Upscaler le modèle (< 1 min sur CPU, pas de GPU nécessaire)
python upscale_head_aware.py \
  --checkpoint checkpoints/seqcond_torch_640k.pt \
  --target-config target_config_xlarge.json \
  --output checkpoints/seqcond_xlarge_init.pt

# 3. Convertir en format JAX (.pkl) pour l'entraînement
python convert_torch_to_jax.py \
  --input checkpoints/seqcond_xlarge_init.pt \
  --output checkpoints/seqcond_xlarge_init.pkl

# 4. Tester la génération (torch, optionnel)
python generate.py \
  --checkpoint checkpoints/seqcond_xlarge_init.pt \
  --prompt "What is blood pressure" \
  --max_tokens 100 --temp 0.6 --cpu
```

## Config cible

Sauvée dans `target_config_xlarge.json`. Tout est ×2 en largeur, profondeur iso :

| Paramètre | Source (370M) | Cible (1.26B) | Ratio |
|---|---|---|---|
| d_model | 1024 | 2048 | ×2 |
| d_ff | 2730 | 5460 | ×2 |
| num_heads | 16 | 32 | ×2 |
| num_kv_heads | 4 | 8 | ×2 |
| seqcond_heads | 16 | 32 | ×2 |
| num_query_heads | 16 | 32 | ×2 |
| seqcond_ratio | 2 | 2 | iso |
| num_layers | 24 | 24 | iso |

Block pattern : `SS T SS T SS T SS T SS T SS T SS T SS T` (16 SeqCond + 8 Transformer).
289/289 paramètres sont upscalés. Résultat : 1,256,000,832 params (1.26B).

## Fichiers

| Fichier | Rôle |
|---|---|
| `upscale_head_aware.py` | **Script principal** — head-aware weight duplication (< 1 min) |
| `convert_torch_to_jax.py` | Convertit le .pt upscalé en .pkl (format JAX) pour l'entraînement |
| `target_config_xlarge.json` | Config cible (1.26B) |
| `create_target_config.py` | Génère une config cible (presets 1.5x/2x/3x ou interactif) |
| `upscale_functional.py` | Ancienne approche (déconv fonctionnelle, déprécié) |
| `upscale_model.py` | Ancienne approche (weight-only super-résolution, déprécié) |
| `verify_upscaled_model.py` | Vérifie reconstruction + inférence |

---

## Comment ça marche — Description technique détaillée

### Principe fondamental

L'architecture SeqCond (comme les Transformers) est **structurée par heads**. Chaque head
est un module quasi-indépendant avec ses propres poids. Quand on passe de 16 à 32 heads,
on peut simplement **dupliquer** chaque head. Le modèle résultant produit exactement les
mêmes activations (à un facteur de répétition près), d'où **loss = 0** à l'init.

C'est fondamentalement différent de l'interpolation bilinéaire (qui mélange les heads
entre elles et détruit la structure).

### Trois types de poids et leur traitement

#### 1. Embedding (`embedding.weight`)

```
Source: (100300, 1024)  →  Cible: (100300, 2048)
```

L'embedding n'a pas de structure per-head. On utilise une **interpolation bilinéaire**
pour passer de d_model=1024 à d_model=2048. Chaque vecteur d'embedding est "étiré"
de manière lisse (comme un zoom d'image 1D).

```python
# embedding.weight: (vocab_size, d_model_small) → (vocab_size, d_model_large)
emb_up = F.interpolate(emb, size=(100300, 2048), mode='bilinear')
```

#### 2. Poids des Transformer blocks (attention + FFN)

**Attention — q_proj, k_proj, v_proj, out_proj :**

La matrice `q_proj` a la forme `(num_heads * head_dim, d_model)`. Elle est structurée
par heads : chaque head a un bloc `(head_dim, d_model)` indépendant.

```
q_proj.weight source: (1024, 1024)
  → reshape: (16 heads, 64 head_dim, 1024 d_model)
  → repeat_interleave(2, dim=0): (32 heads, 64 head_dim, 1024 d_model)
  → reshape: (2048, 1024)
  → repeat_interleave(2, dim=1) / 2: (2048, 2048)
q_proj.weight cible: (2048, 2048)
```

Le `/2` sur la dimension input est crucial : quand x_large = repeat(x_small), la somme
`W @ x` double. On compense en divisant W par 2 sur cette dimension.

Même logique pour `k_proj` (avec `num_kv_heads` au lieu de `num_heads`),
`v_proj`, et `out_proj` (heads sur dim 1, d_model sur dim 0).

**FFN — ff_in, ff_out :**

```
ff_in.weight source: (5460, 1024)  →  (10920, 2048)
  → repeat_interleave(2, dim=0)             # doubler les neurones
  → repeat_interleave(2, dim=1) / 2         # doubler d_model, compenser
ff_in.bias source: (5460,) → (10920,)
  → repeat_interleave(2)

ff_out.weight source: (1024, 2730)  →  (2048, 5460)
  → repeat_interleave(2, dim=0) / 2         # d_model output (résiduel)
  → repeat_interleave(2, dim=1)             # d_ff input
```

#### 3. Poids des SeqCond blocks

Les SeqCond blocks sont plus complexes. Leur structure interne :

```
in_proj.weight: (dim_conv_total, d_model) = (5136, 1024)
│
├─ [0:1024]    K×H = 16×64     memory values   (per-head blocks de 64)
├─ [1024:1040] K = 16          score values     (1 par head)
└─ [1040:5136] K_q×query_dim = 16×256  query values (per-head blocks de 256)
```

**Point crucial :** `H = d_inner / (K × M) = 2048 / (16 × 2) = 64` reste **identique**
entre source et cible ! On double K (16→32) mais H reste à 64. Chaque head est un
module de taille identique — on les duplique simplement.

```
in_proj.weight source: (5136, 1024)
  → split par head structure:
    memory: (16, 64, 1024) → repeat(2, dim=0) → (32, 64, 1024) → reshape (2048, 1024)
    scores: (16, 1024)     → repeat(2, dim=0) → (32, 1024)
    queries: (16, 256, 1024) → repeat(2, dim=0) → (32, 256, 1024) → reshape (8192, 1024)
  → concat: (10272, 1024)
  → repeat_interleave(2, dim=1) / 2: (10272, 2048)
in_proj.weight cible: (10272, 2048)
```

**Paramètres per-head scalaires** (decay_slopes, score_scale, score_bias, phase_scale) :

```
decay_slopes source: (16,)  →  cible: (32,)
  → repeat_interleave(2)
  Chaque head a son propre taux de decay. On le duplique tel quel.
```

**Paramètres per-head multi-dimensionnels** :

```
theta_d_raw source: (1, 1, 16, 64, 2)  →  cible: (1, 1, 32, 64, 2)
  → repeat_interleave(2, dim=2)  # dupliquer sur la dimension K (heads)

w_int_raw source: (1, 1, 16, 1, 64, 2)  →  cible: (1, 1, 32, 1, 64, 2)
  → repeat_interleave(2, dim=2)  # idem

W_readout source: (16, 128, 384)  →  cible: (32, 128, 384)
  → repeat_interleave(2, dim=0)  # dupliquer les heads

conv_weight source: (5136, 1, 4)  →  cible: (10272, 1, 4)
  → même split per-head que in_proj, repeat sur dim 0
```

### Pourquoi ça donne loss = 0

Soit le forward d'un block : `y = block(x)`.

Si on définit :
- `x_large = repeat_interleave(x_small, 2, dim=-1)` — chaque feature doublée
- `W_large = repeat(W_small) / 2` sur la dim input — compense le doublement
- `W_large = repeat(W_small)` sur les dims heads — double le nombre de heads

Alors la sortie `y_large` a la même structure que `x_large` : chaque valeur de
`y_small` est simplement répétée 2 fois. Donc `y_large.reshape(..., 2).mean(-1) == y_small`
exactement (aux erreurs flottantes près).

Vérifié expérimentalement :
```
MSE(downsample(block_large(upsample(x))), block_small(x)) = 0.00000000
```

Pour SeqCond ET Transformer blocks.

### Résumé des opérations par type de paramètre

| Type | Opération | Pourquoi |
|---|---|---|
| Embedding `(V, d)` | Interpolation bilinéaire | Pas de structure per-head |
| Linear input dim `d_model` | `repeat(2, dim) / 2` | Compenser x_large = repeat(x) |
| Linear output dim heads | `repeat(2, dim)` par head | Dupliquer les heads |
| Scalaires per-head | `repeat(2)` | 1 valeur par head, on duplique |
| Tenseurs per-head (N-D) | `repeat(2, dim_heads)` | Idem sur la dim des heads |
| Norm scale | `repeat(2)` | Per-feature, suit d_model |
| FFN output dim `d_model` | `repeat(2, dim) / 2` | Résiduel, compenser |
| conv_weight | Split per-head + repeat | Même structure que in_proj |
| final_norm.scale | `ones(d_model_large)` | Pas dans le source, init défaut |
| cos_emb, sin_emb | Recalculés par le modèle | Buffers RoPE, pas des poids |

---

## Approches alternatives (dans le code mais non recommandées)

### Approche weight-only (`upscale_model.py`)

Optimise `W_large` directement tel que `downsample(W_large) ≈ W_small` via Adam.
Fonctionne mais lent (4h CPU) et ne bénéficie pas de la structure per-head.

### Approche déconv fonctionnelle (`upscale_functional.py`)

Apprend des noyaux de déconv (64 params/layer) pour transformer `W_small → W_large`,
vérifie fonctionnellement avec des inputs aléatoires. Fonctionne sur GPU mais
la loss SeqCond est élevée (0.5) car l'interpolation détruit la structure per-head.
Rendu obsolète par le head-aware init.

## Résultats (mars 2025)

- **370M → 1.26B** en < 1 minute (CPU)
- **Loss = 0** à l'init (reconstruction fonctionnelle parfaite)
- **Génération cohérente** dès l'init (testé avec generate.py)
- Le modèle upscalé est prêt pour continuer le training sur une plus grande capacité
