# TPU Multi-Host Readiness Check

## Status: ⚠️ NEEDS FIXES

Votre code est **presque prêt** pour TPU multi-host (v4-16, etc.), mais il y a **un problème critique** à corriger.

---

## ✅ Ce qui fonctionne déjà

### 1. **Initialisation JAX distribuée**
```python
# train_jax.py ligne 216
jax.distributed.initialize()
```
✅ Correct - initialise automatiquement la communication multi-host

### 2. **FSDP (Fully Sharded Data Parallel)**
Le code implémente correctement FSDP avec :
- ✅ Mesh JAX sur tous les devices : `Mesh(jax.devices(), axis_names=("dp",))`
- ✅ Sharding des paramètres : première dimension shardée si divisible par num_devices
- ✅ Sharding de l'optimizer state
- ✅ Sharding des données : `NamedSharding(self.mesh, PartitionSpec("dp"))`
- ✅ pjit pour les train steps avec in/out shardings corrects

**Localisation** : `seqcond/jax/train.py` lignes 553-792

### 3. **Scripts de déploiement**
- ✅ `run_tpu.sh` : lance la commande sur tous les workers avec `--worker=all`
- ✅ `setup_workers.sh` : installe les dépendances sur tous les workers
- ✅ Git pull automatique avant chaque run

---

## ❌ PROBLÈME CRITIQUE : Dataset Sharding

### Le problème

**Tous les workers lisent les MÊMES données !**

```python
# seqcond/dataset.py ligne 82
dataset = load_dataset("PleIAs/SYNTH", split="train", streaming=True)

# Pas de sharding par process_index !
for i, item in enumerate(dataset):
    # Tous les workers voient les mêmes items
```

**Conséquence** : Sur un TPU v4-16 (4 hosts × 4 chips = 16 devices), les 4 hosts vont :
1. Lire exactement les mêmes exemples du dataset
2. Calculer les mêmes gradients
3. Les moyenner (ce qui ne change rien)
4. **Résultat : vous n'entraînez qu'avec 1/4 des données effectives !**

### La solution

Il faut **sharder le dataset par process** :

```python
import jax

def iterate_synth_sharded(
    max_samples: int = None,
    tokenize: bool = True,
    tok: Tokenizer = None,
) -> Iterator:
    """
    Iterate over PleIAs/SYNTH with per-process sharding.
    Each process gets a different subset of the data.
    """
    if tok is None:
        tok = tokenizer
    
    # Get process info
    process_index = jax.process_index()
    process_count = jax.process_count()
    
    print(f"[Process {process_index}/{process_count}] Loading dataset shard...")
    
    dataset = load_dataset("PleIAs/SYNTH", split="train", streaming=True)
    
    # Skip to this process's starting position
    # Each process takes every Nth sample
    for i, item in enumerate(dataset):
        # Only process samples assigned to this worker
        if i % process_count != process_index:
            continue
            
        if max_samples is not None and i >= max_samples * process_count:
            break
        
        text = format_synth_item(item)
        
        if tokenize:
            try:
                tokens = tok.encode(text)
                yield tokens
            except ValueError as e:
                if "disallowed special token" in str(e):
                    print(f"[Process {process_index}] Skipping sample with disallowed token")
                    continue
                else:
                    raise
        else:
            yield text
```

**Principe** :
- Process 0 prend les samples 0, 4, 8, 12, ...
- Process 1 prend les samples 1, 5, 9, 13, ...
- Process 2 prend les samples 2, 6, 10, 14, ...
- Process 3 prend les samples 3, 7, 11, 15, ...

---

## 📋 Checklist pour TPU v4-16

### Avant de lancer

- [ ] **Implémenter le dataset sharding** (voir solution ci-dessus)
- [ ] Modifier `iterate_synth()` pour utiliser `jax.process_index()` et `jax.process_count()`
- [ ] Modifier `DataLoader.__iter__()` pour appeler la version shardée
- [ ] Modifier `data_generator()` pour appeler la version shardée
- [ ] Tester localement avec `JAX_PROCESS_COUNT=4` simulé

### Configuration recommandée

```bash
# Pour TPU v4-16 (4 hosts × 4 chips)
python train_jax.py \
    --model seqcond \
    --size medium \
    --num-layers 48 \
    --expand 2 \
    --seqcond-heads 15 \
    --seqcond-query-heads 15 \
    --batch-size 64 \          # Total batch = 64
    --fsdp \                    # Active FSDP
    --grad-accum-steps 4 \      # Accumulation si besoin
    --prefetch-batches 100
```

**Calcul** :
- 16 devices total
- Batch size 64 → 4 samples per device
- Avec grad accumulation 4 → 1 sample per device per micro-step

### Vérifications post-lancement

```bash
# Sur le TPU, vérifier que chaque process voit des données différentes
gcloud compute tpus tpu-vm ssh seqcond-tpu --zone=us-central2-b --worker=0 \
    --command "tail -f /tmp/train.log | grep 'Process'"

# Devrait afficher :
# [Process 0/4] Loading dataset shard...
# [Process 1/4] Loading dataset shard...
# [Process 2/4] Loading dataset shard...
# [Process 3/4] Loading dataset shard...
```

---

## 🔧 Modifications nécessaires

### 1. `seqcond/dataset.py`

Ajouter la fonction `iterate_synth_sharded()` et modifier :

```python
def iterate_synth(
    max_samples: int = None,
    tokenize: bool = True,
    tok: Tokenizer = None,
    shard_data: bool = True,  # NOUVEAU
) -> Iterator:
    """
    Iterate over PleIAs/SYNTH dataset in streaming mode.
    
    Args:
        shard_data: If True, shard data across JAX processes (for multi-host)
    """
    if shard_data:
        try:
            import jax
            if jax.process_count() > 1:
                return iterate_synth_sharded(max_samples, tokenize, tok)
        except:
            pass  # JAX not initialized or single process
    
    # Original implementation (single process)
    ...
```

### 2. `run_tpu.sh`

Le script est déjà correct ! Il utilise `--worker=all` ce qui est parfait.

**Optionnel** : Ajouter des logs par worker :

```bash
# Modifier la ligne de lancement pour rediriger les logs
--command "cd $REMOTE_DIR && source ~/.bashrc && $CMD 2>&1 | tee /tmp/train_worker_\$(hostname).log"
```

---

## 🚀 Ordre de lancement

```bash
# 1. Setup (une seule fois)
./setup_workers.sh

# 2. Lancer l'entraînement
./run_tpu.sh "python train_jax.py --model seqcond --size medium --num-layers 48 --expand 2 --batch-size 64 --fsdp --seqcond-heads 15 --seqcond-query-heads 15"

# 3. Monitorer
gcloud compute tpus tpu-vm ssh seqcond-tpu --zone=us-central2-b --worker=0 \
    --command "tail -f /tmp/train_worker_*.log"
```

---

## 📊 Performance attendue

Sur TPU v4-16 :
- **16 chips** (4 hosts × 4 chips/host)
- **Throughput** : ~16x single chip (avec dataset sharding correct)
- **Batch size effectif** : 64 global = 4 per device
- **Memory** : Paramètres shardés via FSDP

**Sans dataset sharding** : vous n'utilisez que 25% de la puissance de calcul ! 😱

---

## ⚠️ Points d'attention

1. **Streaming dataset** : `load_dataset(..., streaming=True)` est bon pour TPU
2. **Prefetch** : `prefetch_batches=100` aide à cacher la latence réseau
3. **Checkpointing** : Seul le process 0 devrait sauvegarder (déjà le cas dans le code)
4. **Wandb** : Seul le process 0 devrait logger (vérifier dans `train.py`)

---

## 🎯 Résumé

| Item | Status | Action |
|------|--------|--------|
| JAX distributed init | ✅ | Rien |
| FSDP implementation | ✅ | Rien |
| Dataset sharding | ❌ | **IMPLÉMENTER** |
| run_tpu.sh | ✅ | Rien |
| setup_workers.sh | ✅ | Rien |

**Action immédiate** : Implémenter le dataset sharding par process avant de lancer sur TPU !
