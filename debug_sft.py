"""
SFT debug avec contrôle des négatifs.

Collecte une fois les completions correctes ET incorrectes, puis fait du SFT
en mélangeant les deux selon --neg_weight (0 = pur SFT, >0 = gradient ascent sur les negs).

Usage:
    KERAS_BACKEND=torch python debug_sft.py                  # pur SFT (baseline)
    KERAS_BACKEND=torch python debug_sft.py --neg_weight 0.5 # 50% de la force SFT sur les negs
    KERAS_BACKEND=torch python debug_sft.py --neg_weight 1.0 # force égale pos/neg
    KERAS_BACKEND=torch python debug_sft.py --neg_weight 2.0 # negs dominent
"""
import argparse, os, random
import numpy as np, torch
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras import ops

from seqcond.torch.generator import TorchGenerator
from convert_torch_to_keras import build_keras_model, convert_weights
from train_grpo import _seq_token_log_probs, sync_keras_to_torch, load_gsm8k, check_answer

p = argparse.ArgumentParser()
p.add_argument("--checkpoint",  default="./checkpoints/seqcond_lin5.pt")
p.add_argument("--n_examples",  type=int,   default=50)
p.add_argument("--g",           type=int,   default=8)
p.add_argument("--eval_batch",  type=int,   default=20,  help="prompts par batch pendant l'eval")
p.add_argument("--temperature", type=float, default=0.2)
p.add_argument("--max_tokens",  type=int,   default=800)
p.add_argument("--lr",          type=float, default=5e-6)
p.add_argument("--steps",       type=int,   default=200)
p.add_argument("--eval_every",  type=int,   default=40)
p.add_argument("--neg_weight",       type=float, default=0.0,
               help="poids du gradient ascent sur les négatifs (0=pur SFT)")
p.add_argument("--neg_ratio",        type=float, default=0.5,
               help="fraction des steps consacrés aux négatifs quand neg_weight>0")
p.add_argument("--softmax_weighted", action="store_true",
               help="pondération softmax des rewards par groupe (SFT pur si rewards égaux)")
p.add_argument("--softmax_temp",     type=float, default=1.0,
               help="température du softmax (haut=uniforme, bas=winner-take-all)")
p.add_argument("--on_policy",        action="store_true",
               help="IS correction : multiplie la loss par ratio = exp(lp_sum_θ - lp_sum_old)")
p.add_argument("--seed",             type=int,   default=42)
args = p.parse_args()

random.seed(args.seed); np.random.seed(args.seed)

# ── Load ──────────────────────────────────────────────────────────────────────
mode = "softmax_weighted" if args.softmax_weighted else (f"neg_weight={args.neg_weight}" if args.neg_weight > 0 else "pure_sft")
print(f"Loading…  mode={mode}")
torch_gen  = TorchGenerator(args.checkpoint, device="cuda")
config     = torch_gen.config
state_dict = {k: v.detach().cpu().float().numpy()
              for k, v in torch_gen.model.state_dict().items()}

keras.mixed_precision.set_global_policy("float32")
keras_model = build_keras_model(config)
convert_weights(config, state_dict, keras_model)
keras_model.cuda()

trainable = [p for p in keras_model.parameters() if p.requires_grad]
print(f"  {sum(p.numel() for p in trainable):,} trainable params")
optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.99))

# ── Data ──────────────────────────────────────────────────────────────────────
all_train = load_gsm8k(split="train", seed=args.seed)
train      = all_train[:args.n_examples]           # exemples vus (overfit)
test       = load_gsm8k(split="test",  seed=args.seed)[:args.n_examples]  # exemples inconnus

# ── Collect completions once ──────────────────────────────────────────────────
print(f"\nCollecting completions (n={args.n_examples}, G={args.g}, temp={args.temperature}) …")

# Structure plate pour SFT/neg modes
pos_items = []   # {"ids", "pl"}
neg_items = []

# Structure par groupe pour softmax_weighted : liste de {"items": [...], "rewards": [...]}
groups = []      # un groupe = 1 exemple qui a ≥1 positive

from train_grpo import _partial_answer_score  # pour avoir le reward continu

for ex in train:
    prompt_tokens = torch_gen.tokenizer([ex["prompt"]])[0]
    texts, comp_ids = torch_gen.generate_group(
        ex["prompt"], n=args.g, max_new_tokens=args.max_tokens,
        temperature=args.temperature, use_synth_template=False,
    )
    group_pos_items   = []
    group_pos_rewards = []
    for text, comp in zip(texts, comp_ids):
        if not comp:
            continue
        base_item = {"ids": prompt_tokens + comp, "pl": len(prompt_tokens)}
        if args.on_policy:
            ids_np = np.array([base_item["ids"]], dtype=np.int32)
            with torch.no_grad():
                lp_old_sum = float(ops.sum(
                    _seq_token_log_probs(keras_model, ids_np, base_item["pl"])
                ))
            base_item["lp_old_sum"] = lp_old_sum
        correct = check_answer(text, ex["ground_truth"])
        if correct:
            reward = 1.0
            pos_items.append(base_item)
            group_pos_items.append(base_item)
            group_pos_rewards.append(reward)
        else:
            reward = _partial_answer_score(text, ex["ground_truth"])
            if reward > 0:
                group_pos_items.append(base_item)
                group_pos_rewards.append(reward)
            neg_items.append(base_item)

    if group_pos_items:
        groups.append({"items": group_pos_items, "rewards": group_pos_rewards})

total = args.n_examples * args.g
n_pos_flat = len(pos_items)
print(f"  pos={n_pos_flat} ({100*n_pos_flat/total:.1f}%)  "
      f"neg={len(neg_items)} ({100*len(neg_items)/total:.1f}%)  "
      f"groupes avec ≥1 pos={len(groups)}/{args.n_examples}")

if not pos_items:
    raise RuntimeError("Aucune completion correcte — baisse la température ou augmente G")

# ── Corpus log-prob (positifs seulement, mesure de progression SFT) ───────────
def mean_corpus_lp():
    total = 0.0
    for item in pos_items:
        ids = np.array([item["ids"]], dtype=np.int32)
        with torch.no_grad():
            total += float(ops.mean(_seq_token_log_probs(keras_model, ids, item["pl"])))
    return total / len(pos_items)

# ── Eval greedy sur un set d'exemples ────────────────────────────────────────
def eval_solve(examples):
    keras_model.cpu(); torch.cuda.empty_cache()
    correct = 0
    for start in range(0, len(examples), args.eval_batch):
        batch = examples[start : start + args.eval_batch]
        texts = torch_gen.generate_batch(
            [ex["prompt"] for ex in batch],
            max_new_tokens=args.max_tokens,
            temperature=0.0, use_synth_template=False,
        )
        for text, ex in zip(texts, batch):
            if check_answer(text, ex["ground_truth"]):
                correct += 1
    keras_model.cuda()
    with torch.no_grad(): keras_model(np.ones((1, 4), dtype=np.int32), training=False)
    return correct / len(examples)

baseline_lp       = mean_corpus_lp()
baseline_train    = eval_solve(train)
baseline_test     = eval_solve(test)
print(f"\n  Baseline  corpus_lp={baseline_lp:.4f}  "
      f"train={baseline_train:.3f}  test={baseline_test:.3f}  "
      f"(gap={baseline_train - baseline_test:+.3f})")

# ── Construire le schedule de steps ──────────────────────────────────────────
if args.softmax_weighted:
    schedule = ["grp"] * args.steps
    grp_buf  = (groups * (args.steps // len(groups) + 2))
    random.shuffle(grp_buf)
    grp_idx  = 0
    print(f"  schedule: {args.steps} steps softmax-weighted  "
          f"(temp={args.softmax_temp}  {len(groups)} groupes)")
elif args.neg_weight > 0 and neg_items:
    n_neg_steps = int(args.steps * args.neg_ratio)
    n_pos_steps = args.steps - n_neg_steps
    schedule = (["pos"] * n_pos_steps + ["neg"] * n_neg_steps)
    random.shuffle(schedule)
    print(f"  schedule: {n_pos_steps} pos steps + {n_neg_steps} neg steps  "
          f"(neg_weight={args.neg_weight})")
else:
    schedule = ["pos"] * args.steps
    if args.neg_weight > 0:
        print("  (aucun négatif disponible, pur SFT)")

# cycles infinis sur chaque pool (modes SFT/neg)
pos_buf = (pos_items * (args.steps // len(pos_items) + 2))
neg_buf = (neg_items * (args.steps // max(len(neg_items), 1) + 2)) if neg_items else []
random.shuffle(pos_buf); random.shuffle(neg_buf)
pos_idx = neg_idx = 0

# ── Header ───────────────────────────────────────────────────────────────────
print(f"\n{'step':>5}  {'type':>4}  {'loss':>9}  {'gnorm':>6}  "
      f"{'corpus_lp':>10}  {'Δlp':>7}  "
      f"{'train':>7}  {'Δtrain':>7}  {'test':>7}  {'Δtest':>7}  {'gap':>6}")
print("─" * 90)

prev_lp    = baseline_lp
prev_train = baseline_train
prev_test  = baseline_test

# ── IS ratio helper ──────────────────────────────────────────────────────────
def is_ratio(lps, item):
    """exp(sum_lp_current - lp_old_sum) — ratio π_θ / π_old au niveau séquence."""
    if not args.on_policy:
        return 1.0
    lp_current_sum = float(ops.sum(lps).detach())
    return float(torch.exp(torch.tensor(lp_current_sum - item["lp_old_sum"])).clamp(0.0, 10.0))

# ── Training loop ─────────────────────────────────────────────────────────────
for step in range(1, args.steps + 1):
    kind = schedule[step - 1]
    optimizer.zero_grad()

    if kind == "grp":
        grp  = grp_buf[grp_idx % len(grp_buf)]; grp_idx += 1
        n    = len(grp["items"])
        r    = np.array(grp["rewards"], dtype=np.float32) / args.softmax_temp
        r   -= r.max()
        w    = n * np.exp(r) / np.exp(r).sum()   # w_i, somme = n
        step_loss = 0.0
        for item, wi in zip(grp["items"], w):
            ids  = np.array([item["ids"]], dtype=np.int32)
            lps  = _seq_token_log_probs(keras_model, ids, item["pl"])
            r_is = is_ratio(lps, item)
            l    = (-wi * r_is * ops.mean(lps)) / n
            l.backward()
            step_loss += float(l.detach())
        loss = step_loss
    elif kind == "pos":
        item = pos_buf[pos_idx % len(pos_buf)]; pos_idx += 1
        ids  = np.array([item["ids"]], dtype=np.int32)
        lps  = _seq_token_log_probs(keras_model, ids, item["pl"])
        r_is = is_ratio(lps, item)
        loss = -r_is * ops.mean(lps)
        loss.backward()
        loss = float(loss.detach())
    else:
        item = neg_buf[neg_idx % len(neg_buf)]; neg_idx += 1
        ids  = np.array([item["ids"]], dtype=np.int32)
        lps  = _seq_token_log_probs(keras_model, ids, item["pl"])
        r_is = is_ratio(lps, item)
        loss = args.neg_weight * r_is * ops.mean(lps)
        loss.backward()
        loss = float(loss.detach())

    gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0))
    optimizer.step()
    sync_keras_to_torch(keras_model, torch_gen.model, config)

    if step % args.eval_every == 0:
        cur_lp    = mean_corpus_lp()
        cur_train = eval_solve(train)
        cur_test  = eval_solve(test)
        dlp     = cur_lp    - prev_lp
        dtrain  = cur_train - prev_train
        dtest   = cur_test  - prev_test
        gap     = cur_train - cur_test
        print(f"{step:>5}  {kind:>4}  {float(loss):>9.5f}  {gnorm:>6.3f}  "
              f"{cur_lp:>10.4f}  {dlp:>+7.4f}  "
              f"{cur_train:>7.3f}  {dtrain:>+7.3f}  "
              f"{cur_test:>7.3f}  {dtest:>+7.3f}  {gap:>+6.3f}")
        prev_lp    = cur_lp
        prev_train = cur_train
        prev_test  = cur_test
    else:
        print(f"{step:>5}  {kind:>4}  {float(loss):>9.5f}  {gnorm:>6.3f}", flush=True)

print("─" * 90)
final_train = eval_solve(train)
final_test  = eval_solve(test)
print(f"\nFinal  corpus_lp={mean_corpus_lp():.4f} (Δ={mean_corpus_lp()-baseline_lp:+.4f})")
print(f"       train={final_train:.3f} (Δ={final_train-baseline_train:+.3f})  "
      f"test={final_test:.3f} (Δ={final_test-baseline_test:+.3f})  "
      f"gap={final_train-final_test:+.3f}")
