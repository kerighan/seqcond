"""
Test minimal : prend 1 completion correcte connue, vérifie que
après 1 gradient step (adv=+1), sa log-prob MONTE dans keras ET dans torch.

Si keras monte mais torch descend → sync cassé.
If keras descend → gradient dans la mauvaise direction.
If rien ne bouge → gradient ne s'applique pas.

Usage:
    KERAS_BACKEND=torch python debug_gradient.py
"""
import os, numpy as np, torch
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras import ops

from seqcond.torch.generator import TorchGenerator
from convert_torch_to_keras import build_keras_model, convert_weights
from train_grpo import _seq_token_log_probs, sync_keras_to_torch
from train_rft import load_train_test, score_group, _compute_advantages, select_pairs, _apply_gradient

CHECKPOINT = "./checkpoints/seqcond_lin5.pt"
LR = 1e-5
N_GRAD_STEPS = 5   # repeat to see trend

# ── Load models ────────────────────────────────────────────────────────────────
print("Loading torch_gen …")
torch_gen = TorchGenerator(CHECKPOINT, device="cuda")
config     = torch_gen.config
state_dict = {k: v.detach().cpu().float().numpy()
              for k, v in torch_gen.model.state_dict().items()}

print("Building keras model (float32) …")
keras.mixed_precision.set_global_policy("float32")
keras_model = build_keras_model(config)
convert_weights(config, state_dict, keras_model)
keras_model.cuda()

optimizer = torch.optim.AdamW(
    [p for p in keras_model.parameters() if p.requires_grad],
    lr=LR, betas=(0.9, 0.99),
)

# ── Find 1 example with at least 1 correct + 1 wrong completion ───────────────
print("\nSearching for a good example (≥1 correct, ≥1 wrong at temp=0.7) …")
train, _ = load_train_test()

found_ex = None
for ex in train[:100]:
    texts, comp_ids = torch_gen.generate_group(
        ex["prompt"], n=16, max_new_tokens=600,
        temperature=0.7, use_synth_template=False,
    )
    rewards, binary = score_group({"texts": texts, "comp_ids": comp_ids, "example": ex})
    if sum(binary) >= 1 and sum(b == 0 for b in binary) >= 1:
        found_ex = ex
        break

assert found_ex is not None, "No suitable example found in first 100 train items"
print(f"  → '{found_ex['question'][:80]}…'")
print(f"     gt={found_ex['ground_truth']}  solve={int(sum(binary))}/16")

prompt_tokens = torch_gen.tokenizer([found_ex["prompt"]])[0]
advantages    = list(_compute_advantages(rewards, normalize_std=False))
positives, negatives = select_pairs(texts, comp_ids, advantages, binary)

scored_groups = [{
    "example":       found_ex,
    "prompt_tokens": prompt_tokens,
    "texts":         texts,
    "binary":        binary,
    "rewards":       rewards,
    "advantages":    advantages,
    "positives":     positives,
    "negatives":     negatives,
}]

# pick the best positive and worst negative for tracking
track_pos = positives[0]   # (idx, text, comp, adv)
track_neg = negatives[0]

def lp(model, is_torch, comp):
    ids = prompt_tokens + list(comp)
    if is_torch:
        t = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            return float(ops.mean(_seq_token_log_probs(model, t, len(prompt_tokens))))
    else:
        a = np.array([ids], dtype=np.int32)
        with torch.no_grad():
            return float(ops.mean(_seq_token_log_probs(model, a, len(prompt_tokens))))

# ── Baseline ──────────────────────────────────────────────────────────────────
print(f"\n{'step':>4}  {'keras_pos':>10} {'Δkpos':>7}  {'torch_pos':>10} {'Δtpos':>7}  "
      f"{'keras_neg':>10} {'Δkneg':>7}  {'torch_neg':>10} {'Δtneg':>7}  "
      f"{'loss':>9}  {'gnorm':>6}")
print("─" * 110)

kpos0 = lp(keras_model, False, track_pos[2])
tpos0 = lp(torch_gen.model, True,  track_pos[2])
kneg0 = lp(keras_model, False, track_neg[2])
tneg0 = lp(torch_gen.model, True,  track_neg[2])

print(f"{'base':>4}  {kpos0:>10.4f} {'':>7}  {tpos0:>10.4f} {'':>7}  "
      f"{kneg0:>10.4f} {'':>7}  {tneg0:>10.4f} {'':>7}")

# ── Gradient steps ────────────────────────────────────────────────────────────
prev_kpos, prev_tpos = kpos0, tpos0
prev_kneg, prev_tneg = kneg0, tneg0

for step in range(1, N_GRAD_STEPS + 1):
    loss, gnorm = _apply_gradient(
        keras_model, optimizer, scored_groups,
        neg_weight=1.0, kl_beta=0.0,
    )
    sync_keras_to_torch(keras_model, torch_gen.model, config)

    kpos = lp(keras_model, False, track_pos[2])
    tpos = lp(torch_gen.model, True,  track_pos[2])
    kneg = lp(keras_model, False, track_neg[2])
    tneg = lp(torch_gen.model, True,  track_neg[2])

    dkp = kpos - prev_kpos
    dtp = tpos - prev_tpos
    dkn = kneg - prev_kneg
    dtn = tneg - prev_tneg

    kp_ok = "✓" if dkp > 0 else "✗"
    tp_ok = "✓" if dtp > 0 else "✗"
    kn_ok = "✓" if dkn < 0 else "✗"
    tn_ok = "✓" if dtn < 0 else "✗"

    print(f"{step:>4}  {kpos:>10.4f} {dkp:>+6.4f}{kp_ok}  {tpos:>10.4f} {dtp:>+6.4f}{tp_ok}  "
          f"{kneg:>10.4f} {dkn:>+6.4f}{kn_ok}  {tneg:>10.4f} {dtn:>+6.4f}{tn_ok}  "
          f"{loss:>9.5f}  {gnorm:>6.3f}")

    prev_kpos, prev_tpos = kpos, tpos
    prev_kneg, prev_tneg = kneg, tneg

print("─" * 110)
print(f"\nTotal Δ pos:  keras={kpos-kpos0:+.4f}  torch={tpos-tpos0:+.4f}  "
      f"(positive = log-prob went UP ✓)")
print(f"Total Δ neg:  keras={kneg-kneg0:+.4f}  torch={tneg-tneg0:+.4f}  "
      f"(negative = log-prob went DOWN ✓)")
print(f"\nExpected: pos should go UP, neg should go DOWN.")
print(f"adv_pos={track_pos[3]:+.4f}  adv_neg={track_neg[3]:+.4f}")
