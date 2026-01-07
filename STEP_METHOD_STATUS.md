# Status des méthodes `step` pour génération auto-régressive

## ⚠️ PROBLÈME IDENTIFIÉ

Les méthodes `step()` dans les trois versions de SeqCond produisent des sorties **différentes** de la méthode `__call__()`, même pour le premier token.

### Tests effectués

```bash
python check_generation.py
```

**Résultats** :
- `seqcond_light` : ❌ FAIL (max diff: 1.85, mean diff: 0.246)
- `seqcond_summary` : ❌ FAIL (max diff: 1.95, mean diff: 0.251)  
- `seqcond_fast` : ❌ FAIL (erreur de scope de paramètres)

### Cause probable

Le problème apparaît dès le **premier token**, ce qui indique que ce n'est pas un problème d'accumulation de state, mais plutôt :

1. **Gestion incorrecte de la convolution causale** dans `step`
2. **Différence dans l'application des layers** entre `__call__` (séquence) et `step` (token unique)
3. **Possible incompatibilité dans le partage des paramètres** entre les deux méthodes

### Modifications récentes

Les modifications suivantes ont été appliquées aux trois versions :
- ✅ Remplacement de `jnp.exp` par `jax.nn.softplus` pour stabilité
- ✅ Ajout de `score_bias` pour contrôle fin
- ✅ Ajout de clipping de `p_w` à [1e-6, 100.0]

Ces modifications ont été appliquées **à la fois dans `__call__` et `step`**, donc elles ne devraient pas causer de divergence.

## 🔧 TODO

1. **Déboguer la méthode `step`** :
   - Comparer ligne par ligne la logique entre `__call__` et `step`
   - Vérifier que la convolution causale est gérée identiquement
   - S'assurer que tous les paramètres sont correctement partagés

2. **Vérifier la version originale** :
   - Tester si les méthodes `step` originales (avant modifications) fonctionnaient
   - Si oui, identifier quelle modification a cassé la compatibilité
   - Si non, réécrire les méthodes `step` from scratch

3. **Tests de régression** :
   - Ajouter `check_generation.py` aux tests CI
   - S'assurer que toute modification future maintient l'équivalence

## 📝 Recommandation

**Pour l'instant, NE PAS utiliser les méthodes `step` pour la génération auto-régressive.**

Utiliser plutôt la méthode `__call__()` avec des séquences progressivement plus longues, même si c'est O(L²) au lieu de O(L).

## 🚀 Prochaines étapes

1. Déboguer `seqcond_light` en priorité (version la plus simple)
2. Une fois corrigé, appliquer le même fix à `seqcond_summary`
3. Corriger `seqcond_fast` (plus complexe avec queries)
4. Valider avec `check_generation.py`
