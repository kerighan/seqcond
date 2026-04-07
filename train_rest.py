"""
train_rest.py — ReST (Reinforced Self-Training) pour GSM8K.

Algorithme par cycle :
  1. COLLECT : génère G completions pour N problèmes, garde les correctes → buffer
  2. TRAIN   : fait train_steps étapes de SFT pur sur le buffer (shufflé)
  3. EVAL    : pass@1 greedy sur eval_n exemples de test
  4. Répète avec les N prochains problèmes (plusieurs epochs possibles)

Usage:
    KERAS_BACKEND=torch python -u train_rest.py --checkpoint checkpoints/seqcond_lin5.pt
"""
import argparse, os, random, time
import numpy as np
import torch

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras import ops

from seqcond.torch.generator import TorchGenerator
from convert_torch_to_keras import build_keras_model, convert_weights
from train_grpo import (
    load_gsm8k, check_answer,
    _seq_token_log_probs, sync_keras_to_torch,
)

W = 90

def sep(c="─"): print(c * W)
def dsep():     print("═" * W)


# ── Collect phase ─────────────────────────────────────────────────────────────

def collect(torch_gen, examples, G, max_tokens, temperature):
    """
    Génère G completions pour chaque exemple, garde uniquement les correctes.
    Retourne une liste de dicts {"ids": List[int], "pl": int, "question": str}.
    """
    items = []
    n_total = len(examples) * G
    n_correct = 0
    t0 = time.perf_counter()

    for i, ex in enumerate(examples):
        prompt_tokens = torch_gen.tokenizer([ex["prompt"]])[0]
        texts, comp_ids = torch_gen.generate_group(
            ex["prompt"], n=G, max_new_tokens=max_tokens,
            temperature=temperature, use_synth_template=False,
        )
        for text, comp in zip(texts, comp_ids):
            if check_answer(text, ex["ground_truth"]) and comp:
                items.append({
                    "ids":      prompt_tokens + comp,
                    "pl":       len(prompt_tokens),
                    "question": ex["question"][:60],
                    "gt":       ex["ground_truth"],
                })
                n_correct += 1

        # Progress every 20 examples
        if (i + 1) % 20 == 0 or (i + 1) == len(examples):
            elapsed = time.perf_counter() - t0
            rate = n_correct / ((i + 1) * G) * 100
            print(f"    [{i+1:3d}/{len(examples)}]  correct={n_correct}  "
                  f"solve={rate:.1f}%  {elapsed:.0f}s", flush=True)

    elapsed = time.perf_counter() - t0
    solve_pct = 100 * n_correct / n_total
    print(f"  → {n_correct}/{n_total} correct ({solve_pct:.1f}%)  in {elapsed:.0f}s")
    return items


# ── Buffer corpus log-prob ─────────────────────────────────────────────────────

def corpus_lp(keras_model, items, max_items=200):
    """Mean per-token log-prob on a random subset of buffer items (no grad)."""
    sample = random.sample(items, min(max_items, len(items)))
    total = 0.0
    for it in sample:
        ids = np.array([it["ids"]], dtype=np.int32)
        with torch.no_grad():
            total += float(ops.mean(_seq_token_log_probs(keras_model, ids, it["pl"])))
    return total / len(sample)


# ── Train phase ───────────────────────────────────────────────────────────────

def train_phase(keras_model, optimizer, trainable, buffer, train_steps, log_every=10):
    """
    Si train_steps == 1 : 1 seul gradient step avec accumulation sur tout le buffer
                          (gros batch virtuel → gradient cohérent, meilleure généralisation).
    Sinon              : train_steps passes individuelles sur le buffer (shufflé).
    Retourne (mean_loss, mean_gnorm).
    """
    if not buffer:
        print("  (buffer vide, skip)")
        return 0.0, 0.0

    t0 = time.perf_counter()

    if train_steps == 1:
        # ── Accumulation sur tout le buffer → 1 optimizer step ────────────────
        optimizer.zero_grad()
        buf = list(buffer)
        random.shuffle(buf)
        n = len(buf)
        total_loss_val = 0.0
        for item in buf:
            ids  = np.array([item["ids"]], dtype=np.int32)
            lps  = _seq_token_log_probs(keras_model, ids, item["pl"])
            loss = -ops.mean(lps) / n       # normalise pour gradient invariant à la taille
            loss.backward()
            total_loss_val += float(loss.detach())
        gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0))
        optimizer.step()
        elapsed = time.perf_counter() - t0
        print(f"    [1 step / {n} items]  loss={total_loss_val:.4f}  gnorm={gnorm:.2f}  {elapsed:.0f}s")
        return total_loss_val, gnorm

    # ── N passes individuelles ─────────────────────────────────────────────────
    buf = list(buffer)
    random.shuffle(buf)
    items_cycle = (buf * (train_steps // len(buf) + 2))[:train_steps]

    losses, gnorms = [], []
    for step, item in enumerate(items_cycle, 1):
        optimizer.zero_grad()
        ids  = np.array([item["ids"]], dtype=np.int32)
        lps  = _seq_token_log_probs(keras_model, ids, item["pl"])
        loss = -ops.mean(lps)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0))
        optimizer.step()
        losses.append(float(loss.detach()))
        gnorms.append(gnorm)

        if step % log_every == 0 or step == train_steps:
            elapsed = time.perf_counter() - t0
            print(f"    [step {step:4d}/{train_steps}]  "
                  f"loss={np.mean(losses[-log_every:]):.4f}  "
                  f"gnorm={np.mean(gnorms[-log_every:]):.2f}  "
                  f"{elapsed:.0f}s", flush=True)

    return float(np.mean(losses)), float(np.mean(gnorms))


# ── Eval phase ────────────────────────────────────────────────────────────────

def eval_phase(torch_gen, keras_model, eval_examples, max_tokens, batch_size=8):
    """pass@1 greedy sur eval_examples. Déplace keras sur CPU pendant la génération."""
    keras_model.cpu()
    torch.cuda.empty_cache()

    n_correct = 0
    prompts = [ex["prompt"] for ex in eval_examples]
    t0 = time.perf_counter()

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        texts = torch_gen.generate_batch(
            batch_prompts, max_new_tokens=max_tokens,
            temperature=0.0, use_synth_template=False,
        )
        for text, ex in zip(texts, eval_examples[start:start + batch_size]):
            if check_answer(text, ex["ground_truth"]):
                n_correct += 1

    keras_model.cuda()
    dummy = np.ones((1, 4), dtype=np.int32)
    with torch.no_grad():
        keras_model(dummy, training=False)

    elapsed = time.perf_counter() - t0
    score = n_correct / len(eval_examples)
    print(f"  → {n_correct}/{len(eval_examples)} correct  score={score:.4f}  {elapsed:.0f}s")
    return score


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   default="./checkpoints/seqcond_lin5.pt")
    p.add_argument("--chunk_size",   type=int,   default=300,   help="exemples par cycle de collecte")
    p.add_argument("--g",            type=int,   default=8,     help="completions par prompt")
    p.add_argument("--temperature",  type=float, default=0.7)
    p.add_argument("--max_tokens",   type=int,   default=800)
    p.add_argument("--train_steps",  type=int,   default=0,     help="étapes SFT par cycle (0 = 1 epoch sur le buffer)")
    p.add_argument("--lr",           type=float, default=3e-6)
    p.add_argument("--eval_n",       type=int,   default=200)
    p.add_argument("--eval_batch",   type=int,   default=8)
    p.add_argument("--epochs",       type=float, default=2.0,   help="epochs sur train set")
    p.add_argument("--save_every",   type=int,   default=1,     help="sauvegarder tous les N cycles")
    p.add_argument("--save_path",    type=str,   default=None)
    p.add_argument("--seed",         type=int,   default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.save_path is None:
        base = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.save_path = os.path.join("checkpoints", "rest", f"{base}_rest.pt")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading torch_gen …")
    torch_gen  = TorchGenerator(args.checkpoint, device="cuda")
    config     = torch_gen.config
    state_dict = {k: v.detach().cpu().float().numpy()
                  for k, v in torch_gen.model.state_dict().items()}

    print("Building keras model (float32) …")
    keras.mixed_precision.set_global_policy("float32")
    keras_model = build_keras_model(config)
    convert_weights(config, state_dict, keras_model)
    keras_model.cuda()

    trainable = [p for p in keras_model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total     = sum(p.numel() for p in keras_model.parameters())
    print(f"  {n_trainable:,} / {n_total:,} trainable params")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.99))

    # ── Data ──────────────────────────────────────────────────────────────────
    train_data = load_gsm8k(split="train", seed=args.seed)
    test_data  = load_gsm8k(split="test",  seed=args.seed)
    eval_examples = test_data[:args.eval_n]

    total_steps  = int(len(train_data) / args.chunk_size * args.epochs)
    chunks = []
    for _ in range(int(np.ceil(args.epochs))):
        d = list(train_data)
        random.shuffle(d)
        for i in range(0, len(d) - args.chunk_size + 1, args.chunk_size):
            chunks.append(d[i:i + args.chunk_size])
    chunks = chunks[:total_steps]

    print(f"\n  {len(train_data)} train examples  |  chunk={args.chunk_size}  "
          f"→  {len(chunks)} cycles\n")

    # ── Buffer ────────────────────────────────────────────────────────────────
    buffer = []

    # ── Baseline eval ─────────────────────────────────────────────────────────
    dsep()
    print(f"  BASELINE EVAL  n={args.eval_n}")
    sep()
    baseline_score = eval_phase(
        torch_gen, keras_model, eval_examples,
        max_tokens=args.max_tokens, batch_size=args.eval_batch,
    )
    dsep()
    prev_score = baseline_score
    print()

    # ── Main loop ─────────────────────────────────────────────────────────────
    for cycle, chunk in enumerate(chunks, 1):
        dsep()
        print(f"  CYCLE {cycle}/{len(chunks)}  "
              f"(exemples {(cycle-1)*args.chunk_size}–{cycle*args.chunk_size-1})")
        dsep()

        # ── 1. Collect ────────────────────────────────────────────────────────
        print(f"\n  [COLLECT]  {len(chunk)} problèmes × G={args.g}  temp={args.temperature}")
        sep()
        buffer = []
        new_items = collect(
            torch_gen, chunk, G=args.g,
            max_tokens=args.max_tokens, temperature=args.temperature,
        )
        buffer.extend(new_items)
        print(f"  buffer: {len(buffer)}  (+{len(new_items)} ce cycle)")

        if not new_items:
            print("  Aucune completion correcte — skip train")
            print()
            continue

        # ── 2. Train ──────────────────────────────────────────────────────────
        n_steps = args.train_steps if args.train_steps > 0 else len(buffer)
        print(f"\n  [TRAIN]  {n_steps} steps {'(1 epoch)' if args.train_steps == 0 else ''}  lr={args.lr}  buffer={len(buffer)}")
        sep()

        lp_before = corpus_lp(keras_model, list(buffer))
        print(f"  corpus_lp avant : {lp_before:.4f}")
        sep()

        mean_loss, mean_gnorm = train_phase(
            keras_model, optimizer, trainable,
            buffer=list(buffer),
            train_steps=n_steps,
            log_every=max(1, n_steps // 10),
        )

        # sync après tous les gradient steps
        sync_keras_to_torch(keras_model, torch_gen.model, config)

        lp_after = corpus_lp(keras_model, list(buffer))
        dlp = lp_after - lp_before
        sep()
        print(f"  corpus_lp après : {lp_after:.4f}  (Δ={dlp:+.4f})  "
              f"loss̄={mean_loss:.4f}  gnorm̄={mean_gnorm:.2f}")

        if dlp < 0:
            print("  ⚠ corpus_lp a baissé — learning rate trop élevé ?")

        # ── 3. Eval ───────────────────────────────────────────────────────────
        print(f"\n  [EVAL]  pass@1 greedy  n={args.eval_n}")
        sep()
        score = eval_phase(
            torch_gen, keras_model, eval_examples,
            max_tokens=args.max_tokens, batch_size=args.eval_batch,
        )
        d = score - prev_score
        tag = f"Δ={d:+.4f} {'↑' if d > 0 else ('↓' if d < 0 else '=')}"
        sep()
        print(f"  score={score:.4f}  {tag}  (baseline={baseline_score:.4f})")

        # ── 4. Save ───────────────────────────────────────────────────────────
        if cycle % args.save_every == 0:
            base, ext = os.path.splitext(args.save_path)
            path = f"{base}_cycle{cycle:03d}{ext}"
            sd = {k: v.cpu() for k, v in torch_gen.model.state_dict().items()}
            torch.save({"state_dict": sd, "config": config, "cycle": cycle}, path)
            print(f"  checkpoint → {path}")

        prev_score = score
        print()

    dsep()
    print(f"  FIN  baseline={baseline_score:.4f}  final={prev_score:.4f}  "
          f"Δ={prev_score - baseline_score:+.4f}")
    dsep()


if __name__ == "__main__":
    main()
