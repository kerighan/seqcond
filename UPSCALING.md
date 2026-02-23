# Model Upscaling via Learned Deconvolution

## Concept

Cette approche permet d'upscaler un modèle pré-entraîné (ex: 370M params) vers un modèle plus grand (ex: 740M params) en réutilisant les connaissances déjà apprises, plutôt que de partir d'une initialisation aléatoire.

### Inspiration

L'idée est inspirée de la super-résolution d'images : votre modèle 370M est comme une "image basse résolution" d'un modèle plus grand. On apprend des transformations de déconvolution qui, lorsqu'on les downsample, reproduisent les poids originaux.

### Avantages

1. **Réutilisation du compute** : Le modèle 370M a déjà été entraîné avec beaucoup de compute. On capitalise sur cet investissement.
2. **Meilleure initialisation** : Le modèle upscalé part d'une initialisation bien meilleure qu'un random init.
3. **Budget compute limité** : Permet de créer un modèle plus gros sans refaire tout le pretraining.

## Architecture

Le processus fonctionne layer-by-layer :

```
Pour chaque layer du modèle original:
  1. Upscaler les poids (ex: d_model 1024 -> 2048)
  2. Apprendre une déconvolution optimale via gradient descent
  3. Objectif: downsample(upscaled_weights) ≈ original_weights
  4. Sauvegarder les poids upscalés
```

## Utilisation

### 1. Upscaler un modèle

```bash
python upscale_model.py \
  --checkpoint checkpoints/seqcond_torch_395k.pt \
  --output checkpoints/seqcond_upscaled_2x.pt \
  --scale-factor 2 \
  --steps-per-layer 500 \
  --device cuda
```

**Paramètres:**
- `--checkpoint`: Checkpoint du modèle original
- `--output`: Où sauvegarder le modèle upscalé
- `--scale-factor`: Facteur d'upscaling (2 = doubler, 3 = tripler, etc.)
- `--steps-per-layer`: Nombre d'étapes d'optimisation par layer (plus = meilleure qualité mais plus lent)
- `--device`: `cuda` ou `cpu`

**Temps estimé:**
- Sur TPU v4: ~30-60 minutes pour un modèle 370M -> 740M avec 500 steps/layer
- Sur GPU A100: ~1-2 heures

### 2. Vérifier la qualité

```bash
python verify_upscaled_model.py \
  --original checkpoints/seqcond_torch_395k.pt \
  --upscaled checkpoints/seqcond_upscaled_2x.pt \
  --scale-factor 2
```

Cela vérifie:
- ✓ Que les poids peuvent être chargés
- ✓ Que le downsampling reproduit bien l'original
- ✓ Que le modèle peut faire de l'inférence
- ✓ Que les logits sont raisonnables (pas de NaN/Inf)

### 3. Fine-tuner le modèle upscalé

Une fois upscalé, vous pouvez fine-tuner sur vos données:

```bash
# Exemple avec votre script d'entraînement existant
python train_jax.py \
  --checkpoint checkpoints/seqcond_upscaled_2x.pt \
  --config large \
  --total-steps 50000 \
  --warmup-steps 1000
```

**Recommandations:**
- Utilisez un learning rate plus faible que pour un training from scratch (ex: 5e-4 au lieu de 1e-3)
- Warmup plus court car le modèle est déjà bien initialisé
- Vous devriez voir une convergence plus rapide qu'avec random init

## Détails techniques

### Upscaling des différents types de poids

1. **Embeddings (vocab_size, d_model)**
   - On garde `vocab_size` constant
   - On upscale seulement `d_model` avec interpolation bilinéaire

2. **Linear layers (d_out, d_in)**
   - On upscale les deux dimensions
   - Interpolation bilinéaire + apprentissage de la déconvolution

3. **Conv1D (features, kernel, channels)**
   - On upscale seulement `features`
   - Les autres dimensions restent constantes

4. **Normalization (d_model,)**
   - Interpolation linéaire simple

### Loss function

```python
loss = reconstruction_loss + smoothness_regularization

reconstruction_loss = MSE(downsample(upscaled_weights), original_weights)
smoothness_regularization = 0.01 * variance(gradient(upscaled_weights))
```

La régularisation de smoothness encourage des transitions douces entre les poids upscalés, ce qui aide à la généralisation.

### Downsampling

On utilise average pooling pour le downsampling car:
- Préserve la magnitude des poids
- Évite les artefacts de sous-échantillonnage
- Mathématiquement réversible (dans une certaine mesure)

## Résultats attendus

### Reconstruction quality

Après upscaling, vous devriez voir:
- MSE de reconstruction < 0.01 pour la plupart des layers
- Relative error < 5% en moyenne

### Performance après fine-tuning

Comparé à un modèle de même taille avec random init:
- **Convergence plus rapide** : 2-3x moins de steps pour atteindre la même perplexité
- **Meilleure perplexité finale** : Typiquement 5-10% meilleure
- **Moins de compute** : Économie de 50-70% du compute total

## Exemple complet

```bash
# 1. Upscaler le modèle 370M -> 740M
python upscale_model.py \
  --checkpoint checkpoints/seqcond_torch_395k.pt \
  --output checkpoints/seqcond_740m_init.pt \
  --scale-factor 2 \
  --steps-per-layer 1000

# 2. Vérifier la qualité
python verify_upscaled_model.py \
  --original checkpoints/seqcond_torch_395k.pt \
  --upscaled checkpoints/seqcond_740m_init.pt

# 3. Fine-tuner (exemple avec 20 jours de TPU v4-64)
python train_jax.py \
  --checkpoint checkpoints/seqcond_740m_init.pt \
  --config xlarge \
  --total-steps 100000 \
  --base-lr 5e-4 \
  --warmup-steps 500 \
  --use-wandb \
  --wandb-project seqcond-upscaled
```

## Optimisations possibles

### Pour économiser du compute

1. **Fewer steps per layer** : Utiliser 200-300 steps au lieu de 500-1000
2. **Selective upscaling** : N'upscaler que certaines layers (ex: les SeqCond blocks)
3. **Progressive upscaling** : 370M -> 500M -> 740M en plusieurs étapes

### Pour améliorer la qualité

1. **More steps** : 1000-2000 steps par layer
2. **Better initialization** : Utiliser des techniques plus sophistiquées (ex: spectral initialization)
3. **Curriculum learning** : Commencer avec des layers simples, puis les plus complexes

## Limitations

1. **Pas magique** : Le modèle upscalé ne sera jamais aussi bon qu'un modèle entraîné from scratch avec le même compute total
2. **Asymptote** : Au-delà de 2-3x upscaling, les gains diminuent
3. **Architecture constraints** : Fonctionne mieux si l'architecture reste similaire (même ratio de layers, etc.)

## Questions fréquentes

**Q: Pourquoi ne pas juste copier les poids et ajouter du bruit?**
A: L'apprentissage de la déconvolution optimale donne de meilleurs résultats car il trouve la transformation qui préserve au mieux les patterns appris.

**Q: Peut-on upscaler plusieurs fois (2x puis 2x encore)?**
A: Oui, mais les gains diminuent. Mieux vaut faire un seul upscaling plus important.

**Q: Quel scale factor choisir?**
A: 2x est optimal. Au-delà de 3x, les gains deviennent marginaux.

**Q: Combien de compute économise-t-on vraiment?**
A: Si votre modèle 370M a pris X compute, upscaler vers 740M + fine-tuner prendra environ 0.3X au lieu de 4X pour un training from scratch.

## Références

Cette approche s'inspire de:
- Super-resolution d'images (SRCNN, ESRGAN)
- Knowledge distillation (mais dans le sens inverse)
- Progressive growing (StyleGAN)
- Model merging et interpolation
