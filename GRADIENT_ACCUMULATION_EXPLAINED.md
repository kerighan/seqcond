# Gradient Accumulation : Comment ça marche ?

## Exemple concret : `--batch-size 100 --grad-accum-steps 5`

### Configuration TPU v4-16

Supposons un TPU v4-16 avec **16 devices** (4 hosts × 4 chips).

### Ce qui se passe exactement

#### 1. **Calcul du micro-batch par device**

```
batch_size = 100
num_devices = 16
grad_accum_steps = 5

per_device_batch = batch_size / num_devices = 100 / 16 = 6.25
```

⚠️ **PROBLÈME** : 100 n'est pas divisible par 16 !

Le code va **échouer** avec cette erreur :
```
ValueError: Batch size must be divisible by the number of devices for FSDP training.
```

**Solution** : Utiliser `batch_size = 96` ou `batch_size = 112` (multiples de 16).

---

### Exemple corrigé : `--batch-size 96 --grad-accum-steps 5`

#### 1. **Micro-batch par device**

```
batch_size = 96
per_device_batch = 96 / 16 = 6 samples par device
```

#### 2. **Boucle d'entraînement**

Le code fait **5 forward/backward passes** avant de mettre à jour les poids :

```python
# Ligne 857 dans train.py
micro_steps = total_steps * grad_accum_steps
# Si total_steps = 10000, alors micro_steps = 50000

# Ligne 917-932 dans train.py
for step in range(1, micro_steps + 1):
    # Forward + backward
    grads, metrics = fsdp_grad_step(params, x, y)
    
    # Accumulation : divise par grad_accum_steps
    grads_accum = accumulate_grads(grads_accum, grads, grad_accum_steps)
    # grads_accum += grads / 5
    
    accum_count += 1
    
    # Tous les 5 micro-steps, on met à jour les poids
    if accum_count >= grad_accum_steps:  # Tous les 5 steps
        params, opt_state = fsdp_update_step(params, opt_state, grads_accum)
        # Reset
        grads_accum = None
        accum_count = 0
        macro_step += 1  # C'est le "vrai" step
```

#### 3. **Timeline détaillée**

| Micro-step | Action | Batch par device | Gradients | Update ? |
|------------|--------|------------------|-----------|----------|
| 1 | Forward+Backward | 6 samples | `g1/5` | ❌ |
| 2 | Forward+Backward | 6 samples | `g1/5 + g2/5` | ❌ |
| 3 | Forward+Backward | 6 samples | `g1/5 + g2/5 + g3/5` | ❌ |
| 4 | Forward+Backward | 6 samples | `g1/5 + g2/5 + g3/5 + g4/5` | ❌ |
| 5 | Forward+Backward | 6 samples | `g1/5 + g2/5 + g3/5 + g4/5 + g5/5` | ✅ **UPDATE** |

**Résultat** : Après 5 micro-steps, on a accumulé les gradients de **5 × 96 = 480 samples** !

#### 4. **Batch effectif**

```
Effective batch size = batch_size × grad_accum_steps
                     = 96 × 5
                     = 480 samples
```

C'est comme si tu entraînais avec un batch de 480, mais en utilisant seulement la mémoire pour 96 !

---

## Pourquoi utiliser gradient accumulation ?

### 1. **Simuler de gros batch avec peu de mémoire**

```bash
# Sans accumulation : OOM (Out of Memory)
--batch-size 480  # ❌ Trop gros pour la mémoire

# Avec accumulation : OK !
--batch-size 96 --grad-accum-steps 5  # ✅ Même résultat, moins de mémoire
```

### 2. **Stabilité de l'entraînement**

Les gros batch donnent des gradients plus stables :
- Moins de bruit dans les mises à jour
- Convergence plus lisse
- Peut permettre des learning rates plus élevés

### 3. **Trade-off : Vitesse vs Stabilité**

```
Sans accumulation (grad-accum-steps = 1):
- ✅ Updates fréquents (tous les steps)
- ✅ Plus rapide en wall-clock time
- ❌ Gradients plus bruités
- ❌ Batch effectif = 96

Avec accumulation (grad-accum-steps = 5):
- ✅ Gradients plus stables
- ✅ Batch effectif = 480
- ❌ Updates moins fréquents (tous les 5 steps)
- ❌ Légèrement plus lent (overhead d'accumulation)
```

---

## Compatibilité avec FSDP

**OUI, gradient accumulation fonctionne avec FSDP !**

Le code le gère correctement :

```python
# Ligne 718 dans train.py
if self.use_fsdp:
    if grad_accum_steps > 1:
        # Crée des steps séparés pour grad et update
        self._fsdp_grad_step = pjit(...)  # Forward + backward
        self._fsdp_update_step = pjit(...) # Apply gradients
```

**Chaque device** :
1. Calcule ses gradients locaux sur son micro-batch (6 samples)
2. Les gradients sont **automatiquement moyennés** entre devices via FSDP
3. Accumule ces gradients moyennés
4. Après 5 accumulations, applique l'update

---

## Recommandations pour TPU v4-16

### Configuration optimale

```bash
# Pour 16 devices
--batch-size 64 \           # 4 samples/device (petit, rapide)
--grad-accum-steps 8 \      # Batch effectif = 512
--lr 6e-4

# OU

--batch-size 96 \           # 6 samples/device (moyen)
--grad-accum-steps 5 \      # Batch effectif = 480
--lr 6e-4

# OU

--batch-size 128 \          # 8 samples/device (gros)
--grad-accum-steps 4 \      # Batch effectif = 512
--lr 6e-4
```

### Règles de choix

1. **batch_size** doit être divisible par **num_devices** (16 pour v4-16)
2. **Batch effectif** = batch_size × grad_accum_steps
3. Plus le batch effectif est gros, plus le LR peut être élevé (règle : `lr ∝ √batch_size`)

### Calcul du learning rate

```python
# Règle empirique
base_lr = 3e-4  # LR pour batch 256
target_batch = 480
scale_factor = sqrt(target_batch / 256) = sqrt(480/256) ≈ 1.37

adjusted_lr = base_lr * scale_factor = 3e-4 * 1.37 ≈ 4e-4
```

Donc avec `--batch-size 96 --grad-accum-steps 5` (batch effectif 480), un bon LR serait `--lr 4e-4` ou `--lr 5e-4`.

---

## Vérification dans les logs

Quand tu lances l'entraînement, tu devrais voir :

```
Using FSDP across 16 devices (per-device batch size = 6).
Using FSDP with gradient accumulation: 5 steps
```

Et dans les logs :
```
Step 1/10000 | Loss: 2.45 | Tokens/sec: 50000
Step 2/10000 | Loss: 2.42 | Tokens/sec: 51000
...
```

**Attention** : Le "Step" affiché est le **macro-step** (après accumulation), pas le micro-step !

---

## Code source

La logique d'accumulation est dans `seqcond/jax/train.py` :

```python
# Ligne 389-395
def accumulate_grads(grads_accum, grads, accum_steps: int):
    """Accumulate gradients."""
    if grads_accum is None:
        return jax.tree_util.tree_map(lambda g: g / accum_steps, grads)
    return jax.tree_util.tree_map(
        lambda acc, g: acc + g / accum_steps, grads_accum, grads
    )
```

**Important** : Les gradients sont **divisés par accum_steps** à chaque accumulation, donc la moyenne finale est correcte.

---

## Résumé

Avec `--batch-size 96 --grad-accum-steps 5` sur TPU v4-16 :

| Paramètre | Valeur |
|-----------|--------|
| Devices | 16 |
| Batch par device | 6 |
| Batch total par step | 96 |
| Micro-steps avant update | 5 |
| **Batch effectif** | **480** |
| Samples vus avant update | 480 |
| Mémoire utilisée | Comme batch 96 |
| Stabilité | Comme batch 480 |

✅ **Gradient accumulation fonctionne parfaitement avec FSDP sur TPU multi-host !**
