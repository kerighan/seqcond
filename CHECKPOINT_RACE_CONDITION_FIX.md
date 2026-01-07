# ⚠️ RACE CONDITION CRITIQUE : Sauvegarde des checkpoints

## Problème identifié

**OUI, il y a une race condition sur la sauvegarde des checkpoints en multi-host !**

### Code actuel (BUGUÉ)

```python
# train.py ligne 1143-1157
def _save_checkpoint(self, step: int, final: bool = False):
    """Save a checkpoint."""
    if final:
        path = f"{self.train_config.checkpoint_dir}/{self.model_name}.pkl"
    else:
        path = f"{self.train_config.checkpoint_dir}/{self.model_name}_step{step}.pkl"
    
    # Get params and opt_state from devices before saving
    params_to_save = self._params_for_host()
    opt_state_to_save = self._opt_state_for_host()
    
    save_checkpoint(params_to_save, opt_state_to_save, self.config, path, step)
    print(f"Checkpoint saved: {path}")
```

**Problème** : Sur un TPU v4-16 avec 4 hosts, **les 4 processes vont TOUS essayer de sauvegarder au même fichier en même temps !**

### Conséquences

1. **Corruption de fichier** : 4 processes écrivent simultanément → fichier corrompu
2. **Perte de checkpoint** : Le dernier process à écrire écrase les autres
3. **Ralentissement** : 4× plus d'I/O que nécessaire
4. **Crash potentiel** : Conflits d'accès fichier

---

## Solution : Sauvegarder uniquement depuis le process 0

### Fix pour `_save_checkpoint`

```python
def _save_checkpoint(self, step: int, final: bool = False):
    """Save a checkpoint (only from process 0 to avoid race conditions)."""
    import jax
    
    # Only process 0 should save checkpoints
    if jax.process_index() != 0:
        return
    
    if final:
        path = f"{self.train_config.checkpoint_dir}/{self.model_name}.pkl"
    else:
        path = f"{self.train_config.checkpoint_dir}/{self.model_name}_step{step}.pkl"
    
    # Get params and opt_state from devices before saving
    params_to_save = self._params_for_host()
    opt_state_to_save = self._opt_state_for_host()
    
    save_checkpoint(params_to_save, opt_state_to_save, self.config, path, step)
    print(f"[Process 0] Checkpoint saved: {path}")
```

### Fix pour `_generate_sample`

Le même problème existe pour la génération de texte :

```python
def _generate_sample(self, step: int):
    """Generate a sample (only from process 0)."""
    import jax
    
    # Only process 0 should generate samples
    if jax.process_index() != 0:
        return
    
    if self.tokenizer is None:
        return
    
    # Use a complete prompt that expects assistant response with thinking
    prompt = "<|im_start|>user\n"
    
    # Ensure params are on the host for generation
    host_params = self._params_for_host()
    
    print(f"\n[Process 0] --- Generation at step {step} (fixed padding) ---")
    # ... rest of generation code
```

### Fix pour `_log_progress`

Les logs aussi devraient être limités au process 0 :

```python
def _log_progress(
    self,
    macro_step: int,
    total_steps: int,
    metrics: MetricsAccumulator,
    start_time: float,
    last_log_time: float,
    tokens_delta: int = 0,
    tokens_seen: int = 0,
) -> float:
    """Log training progress (only from process 0)."""
    import jax
    
    # Only process 0 should log
    if jax.process_index() != 0:
        return last_log_time
    
    # ... rest of logging code
```

---

## Pourquoi `_params_for_host()` fonctionne quand même ?

Même si tous les processes appellent `_params_for_host()`, chaque process récupère **sa propre copie** des paramètres depuis ses devices locaux.

Avec FSDP :
- Process 0 récupère les shards des devices 0-3
- Process 1 récupère les shards des devices 4-7
- Process 2 récupère les shards des devices 8-11
- Process 3 récupère les shards des devices 12-15

Mais grâce à `_unshard_tree()` (ligne 470-475), chaque process peut reconstruire les paramètres complets :

```python
def _unshard_tree(self, tree):
    """Unshard a tree from devices to host."""
    return jax.tree_util.tree_map(
        lambda x: x.addressable_data(0),  # Récupère les données du premier device local
        tree,
    )
```

**Donc tous les processes PEUVENT sauvegarder, mais un seul DEVRAIT le faire !**

---

## Autres endroits à vérifier

### 1. Wandb logging

```python
def _init_wandb(self, num_params: int):
    """Initialize wandb logging."""
    import jax
    
    # Only process 0 should initialize wandb
    if jax.process_index() != 0:
        self._wandb = None
        return
    
    try:
        import wandb
        # ... rest of init
```

### 2. Callback dans `callback.py`

Le `StepwiseGenerationCallback` devrait aussi vérifier le process :

```python
def on_train_batch_end(self, params: Any, batch: int):
    """Called at the end of a training batch."""
    import jax
    
    # Only process 0 should generate
    if jax.process_count() > 1 and jax.process_index() != 0:
        return
    
    self.step_counter += 1
    # ... rest of generation
```

---

## Vérification après fix

Après avoir appliqué les fixes, tu devrais voir dans les logs :

```
# Process 0
[Process 0] Step 100/10000 | Loss: 2.45 | Tokens/sec: 50000
[Process 0] Checkpoint saved: checkpoints/model_step100.pkl
[Process 0] --- Generation at step 100 ---

# Process 1, 2, 3
(silence - pas de logs)
```

---

## Impact sur les performances

**Avant fix** :
- 4 processes sauvent → 4× I/O
- Risque de corruption
- Logs dupliqués × 4

**Après fix** :
- 1 process sauve → 1× I/O
- Pas de corruption
- Logs clairs

**Gain** : ~75% de réduction d'I/O + stabilité garantie

---

## Checklist de fix

- [ ] Ajouter `if jax.process_index() != 0: return` dans `_save_checkpoint`
- [ ] Ajouter `if jax.process_index() != 0: return` dans `_generate_sample`
- [ ] Ajouter `if jax.process_index() != 0: return` dans `_log_progress`
- [ ] Ajouter check dans `_init_wandb`
- [ ] Ajouter check dans `StepwiseGenerationCallback.on_train_batch_end`
- [ ] Tester sur TPU multi-host

---

## Résumé

**Race condition** : ✅ **CONFIRMÉE**

**Cause** : Tous les processes sauvent au même fichier simultanément

**Solution** : Sauvegarder uniquement depuis `jax.process_index() == 0`

**Urgence** : 🔴 **CRITIQUE** - À fixer avant de lancer sur TPU multi-host !
