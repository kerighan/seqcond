"""
Vérifie que keras_model et torch_gen.model donnent les mêmes log probs
sur les mêmes séquences, avant et après un faux gradient step.

Usage:
    KERAS_BACKEND=torch python check_keras_torch_equiv.py
"""
import os, numpy as np, torch
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras import ops
from seqcond.torch.generator import TorchGenerator
from convert_torch_to_keras import build_keras_model, convert_weights
from train_grpo import _seq_token_log_probs, sync_keras_to_torch

CHECKPOINT = "./checkpoints/seqcond_lin5.pt"

def compare(lps_k, lps_t, label):
    a = lps_k.float().detach().cpu()
    b = lps_t.float().detach().cpu()
    diff = (a - b).abs()
    print(f"  [{label}]  max={diff.max():.6f}  mean={diff.mean():.6f}  "
          f"tokens={len(a)}  match_top1={a.argmax()==b.argmax()}")

print("Loading torch_gen...")
torch_gen = TorchGenerator(CHECKPOINT, device="cuda")
config = torch_gen.config
state_dict = {k: v.detach().cpu().float().numpy()
              for k, v in torch_gen.model.state_dict().items()}

print("Building keras model (float32)...")
keras.mixed_precision.set_global_policy("float32")
keras_model = build_keras_model(config)
convert_weights(config, state_dict, keras_model)
keras_model.cuda()

# ── Test sequences ─────────────────────────────────────────────────────────────
seqs = [
    ([1, 2, 3, 4, 5, 100, 200, 300], 5),          # short
    (list(range(1, 51)) + list(range(100, 130)), 50),  # medium 80 tok
    (list(range(1, 251)) + list(range(100, 351)), 250),  # long 600 tok
]

print("\n=== Avant sync (poids identiques à l'init) ===")
for ids_list, pl in seqs:
    ids_np  = np.array([ids_list], dtype=np.int32)
    ids_th  = torch.tensor([ids_list], dtype=torch.long, device="cuda")
    with torch.no_grad():
        lps_k = _seq_token_log_probs(keras_model, ids_np, pl)
        lps_t = _seq_token_log_probs(torch_gen.model, ids_th, pl)
    compare(lps_k, lps_t, f"len={len(ids_list)} pl={pl}")

# ── Faux gradient step sur keras ──────────────────────────────────────────────
print("\nFaux gradient step sur keras_model...")
optimizer = torch.optim.AdamW(keras_model.parameters(), lr=1e-4)
optimizer.zero_grad()
ids_np = np.array([seqs[0][0]], dtype=np.int32)
pl = seqs[0][1]
lps = _seq_token_log_probs(keras_model, ids_np, pl)
fake_loss = -ops.mean(lps)
fake_loss.backward()
optimizer.step()

print("Sync keras → torch...")
sync_keras_to_torch(keras_model, torch_gen.model, config)

print("\n=== Après faux step + sync ===")
for ids_list, pl in seqs:
    ids_np  = np.array([ids_list], dtype=np.int32)
    ids_th  = torch.tensor([ids_list], dtype=torch.long, device="cuda")
    with torch.no_grad():
        lps_k = _seq_token_log_probs(keras_model, ids_np, pl)
        lps_t = _seq_token_log_probs(torch_gen.model, ids_th, pl)
    compare(lps_k, lps_t, f"len={len(ids_list)} pl={pl}")

# ── Vérifie que keras a bien changé ──────────────────────────────────────────
print("\n=== keras a-t-il changé ? ===")
keras_model2 = build_keras_model(config)
convert_weights(config, state_dict, keras_model2)
keras_model2.cuda()
ids_np = np.array([seqs[0][0]], dtype=np.int32)
pl = seqs[0][1]
with torch.no_grad():
    lps_orig  = _seq_token_log_probs(keras_model2, ids_np, pl)
    lps_after = _seq_token_log_probs(keras_model,  ids_np, pl)
diff = (lps_after.float() - lps_orig.float()).abs()
print(f"  keras avant vs après step: max={diff.max():.6f}  mean={diff.mean():.6f}")
ids_th = torch.tensor([seqs[0][0]], dtype=torch.long, device="cuda")
with torch.no_grad():
    lps_torch = _seq_token_log_probs(torch_gen.model, ids_th, pl)
diff2 = (lps_after.float() - lps_torch.float()).abs()
print(f"  keras après vs torch après sync: max={diff2.max():.6f}  mean={diff2.mean():.6f}")

print("\nSi max > 0.01 dans 'après sync', le sync est cassé.")
print("Si keras 'avant vs après' est ~0, le gradient ne s'applique pas.")
