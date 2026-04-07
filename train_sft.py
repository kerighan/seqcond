"""
train_sft.py — SFT supervisé sur un dataset JSONL de CoT collectées.

Chaque ligne JSONL doit avoir : "query", "reasoning", "reward", "advantage".
Généré typiquement par collect_cot.py.

Usage:
    KERAS_BACKEND=torch python -u train_sft.py \
        --data data/cot_dataset.jsonl \
        --checkpoint checkpoints/seqcond_lin5.pt
"""
import argparse, json, os, random, time
import numpy as np
import torch

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras import ops

from seqcond.torch.generator import TorchGenerator
from convert_torch_to_keras import build_keras_model, convert_weights
from train_grpo import load_gsm8k, check_answer, _seq_token_log_probs, sync_keras_to_torch

W = 88
def sep(c="─"): print(c * W)
def dsep():     print("═" * W)


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_dataset(path, tokenizer, weighting):
    """
    Charge le JSONL, tokenise chaque item.
    weighting : "none" | "reward" | "advantage"
    Retourne une liste de dicts {ids, pl, weight}.
    """
    items = []
    skipped = 0
    with open(path) as f:
        for line in f:
            if not line:
                continue
            try:
                rec = json.loads(line)
            except:
                print(line, "line")
                continue
            question  = rec["query"]
            reasoning = rec["reasoning"]
            # Reconstituer le prompt formaté (même template que load_gsm8k / format_synth_item)
            prompt_text = (
                "<|im_start|>user\n" + question
                + "\n<|im_end|><|im_start|>assistant\n<|think_start|>"
            )
            prompt_tokens = tokenizer([prompt_text])[0]

            # Completion = reasoning + <|im_end|> (le modèle doit apprendre à fermer)
            completion = reasoning.strip()
            if not completion.endswith("<|im_end|>"):
                completion = completion + "\n<|im_end|>"

            full_tokens = tokenizer([prompt_text + completion])[0]
            comp_tokens = full_tokens[len(prompt_tokens):]

            if not comp_tokens:
                skipped += 1
                continue

            if weighting == "reward":
                w = float(rec.get("reward", 1.0))
            elif weighting == "advantage":
                w = float(rec.get("advantage", 1.0))
                if w <= 0:          # ignore negative advantages
                    skipped += 1
                    continue
            else:
                w = 1.0

            items.append({
                "ids":    prompt_tokens + comp_tokens,
                "pl":     len(prompt_tokens),
                "weight": w,
            })

    print(f"  {len(items)} items chargés  ({skipped} ignorés)")
    return items


# ── Gradient step (accumulation sur batch_size items) ─────────────────────────

def gradient_step(keras_model, optimizer, trainable, batch, kl_beta, ref_model,
                  micro_batch_size=1):
    """
    Accumule les gradients sur `batch` items puis fait 1 optimizer.step().
    Normalise par la somme des poids du batch.

    micro_batch_size : nombre d'items par appel .backward()
      - 1  : 1 backward par item (défaut, moins de mémoire)
      - N  : accumule N items dans un seul tenseur avant backward
             (moins d'appels backward, potentiellement plus rapide)
    Résultat mathématique identique dans les deux cas.
    """
    optimizer.zero_grad()
    total_w      = sum(it["weight"] for it in batch)
    total_tokens = sum(len(it["ids"]) - it["pl"] for it in batch)
    total_loss   = 0.0

    for i in range(0, len(batch), micro_batch_size):
        micro     = batch[i : i + micro_batch_size]
        micro_loss = 0
        for item in micro:
            ids       = np.array([item["ids"]], dtype=np.int32)
            token_lps = _seq_token_log_probs(keras_model, ids, item["pl"])
            w         = item["weight"] / total_w
            # Global mean: weight by token count so every token has equal contribution
            loss      = -w * ops.sum(token_lps) / total_tokens

            if kl_beta > 0 and ref_model is not None:
                with torch.no_grad():
                    ref_lps = _seq_token_log_probs(ref_model, ids, item["pl"])
                kl   = ops.sum(token_lps.float() - ref_lps.float()) / total_tokens
                loss = loss + kl_beta * w * kl

            micro_loss += loss
            total_loss += float(loss.detach())

        micro_loss.backward()

    gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0))
    optimizer.step()
    return total_loss, gnorm


# ── Eval ──────────────────────────────────────────────────────────────────────

def eval_pass1(torch_gen, keras_model, examples, max_tokens, batch_size):
    keras_model.cpu(); torch.cuda.empty_cache()
    correct = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        texts = torch_gen.generate_batch(
            [ex["prompt"] for ex in batch],
            max_new_tokens=max_tokens, temperature=0.0,
            use_synth_template=False,
        )
        for text, ex in zip(texts, batch):
            if check_answer(text, ex["ground_truth"]):
                correct += 1
    keras_model.cuda()
    with torch.no_grad():
        keras_model(np.ones((1, 4), dtype=np.int32), training=False)
    return correct / len(examples)


# ── Checkpoint ────────────────────────────────────────────────────────────────

_CACHE_PREFIXES = (
    '_conv_kernel_t', '_decay_slopes_cached', '_phase_scale_b',
    '_score_bias_b', '_score_scale_b', '_theta_cached', '_w_int_cached',
)

def save_ckpt(torch_gen, config, path, step):
    sd = {k: v.cpu() for k, v in torch_gen.model.state_dict().items()
          if not any(k.endswith(s) for s in _CACHE_PREFIXES)}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": sd, "config": config, "step": step}, path)
    print(f"  checkpoint → {path}  ({len(sd)} tensors)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",          required=True,              help="chemin vers le JSONL")
    p.add_argument("--checkpoint",    default="./checkpoints/seqcond_lin5.pt")
    p.add_argument("--epochs",        type=float, default=3.0)
    p.add_argument("--batch_size",     type=int,   default=16,    help="items par gradient step (gradient accumulation)")
    p.add_argument("--micro_batch_size", type=int, default=1,    help="items par appel .backward() (1=un par un, N=accumulation)")
    p.add_argument("--lr",            type=float, default=3e-6)
    p.add_argument("--kl_beta",       type=float, default=0.0,    help="poids KL vs référence (0=désactivé)")
    p.add_argument("--weighting",     default="none",             help="none | reward | advantage")
    p.add_argument("--eval_every",    type=int,   default=50,     help="steps entre deux evals")
    p.add_argument("--eval_n",        type=int,   default=300)
    p.add_argument("--eval_batch",    type=int,   default=16)
    p.add_argument("--max_tokens",    type=int,   default=1000)
    p.add_argument("--train",         default="all",              help="all | transformer | seqcond — which block types to train (others + embeddings frozen)")
    p.add_argument("--save_every",    type=int,   default=1,      help="sauvegarder tous les N evals (0=désactivé)")
    p.add_argument("--save_path",     default=None)
    p.add_argument("--seed",          type=int,   default=42)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    if args.save_path is None:
        base = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.save_path = os.path.join("checkpoints", "sft", f"{base}_sft.pt")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading torch_gen …")
    torch_gen  = TorchGenerator(args.checkpoint, device="cuda")
    config     = torch_gen.config
    state_dict = {k: v.detach().cpu().float().numpy()
                  for k, v in torch_gen.model.state_dict().items()}

    print("Building keras model (mixed_bfloat16) …")
    keras.mixed_precision.set_global_policy("mixed_bfloat16")
    keras_model = build_keras_model(config)
    convert_weights(config, state_dict, keras_model)
    keras_model.cuda()

    ref_model = None
    if args.kl_beta > 0:
        ref_model = build_keras_model(config)
        convert_weights(config, state_dict, ref_model)
        ref_model.cuda()
        for param in ref_model.parameters():
            param.requires_grad = False
        print(f"  KL ref model chargé  β={args.kl_beta}")

    # ── Selective freeze ──────────────────────────────────────────────────────
    # --train options: all | transformer | seqcond | mlp | transformer-mlp | seqcond-mlp
    if args.train != "all":
        # Freeze everything first, then selectively unfreeze
        for param in keras_model.parameters():
            param.requires_grad = False

        target = args.train
        unfrozen_names = []

        for btype, block in zip(keras_model.block_types, keras_model.blocks_list):
            if target in ("transformer", "seqcond") and btype == target:
                # Unfreeze entire block
                for param in block.parameters():
                    param.requires_grad = True
                unfrozen_names.append(f"{btype}:{block.name}")

            elif target == "mlp" or target == f"{btype}-mlp":
                if btype == "transformer":
                    # MLP = norm2 + ff_in + ff_out
                    for sub in (block.norm2, block.ff_in, block.ff_out):
                        for param in sub.parameters():
                            param.requires_grad = True
                    unfrozen_names.append(f"transformer-mlp:{block.name}")
                elif btype == "seqcond":
                    # Output projection = gate_proj + out_proj + W_readout + gated_norm_weight
                    attn = block.attn
                    for sub in (attn.gate_proj, attn.out_proj):
                        for param in sub.parameters():
                            param.requires_grad = True
                    attn.W_readout.requires_grad = True
                    attn.gated_norm_weight.requires_grad = True
                    unfrozen_names.append(f"seqcond-mlp:{block.name}")

        print(f"  [freeze] --train {target}: unfroze {len(unfrozen_names)} block groups, embeddings frozen")
        for n in unfrozen_names:
            print(f"    ✓ {n}")

    trainable   = [p for p in keras_model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  {n_trainable:,} paramètres entraînables")
    optimizer   = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.99))

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"\nChargement dataset : {args.data}  (weighting={args.weighting})")
    dataset = load_dataset(args.data, torch_gen.tokenizer, args.weighting)
    if not dataset:
        raise RuntimeError("Dataset vide !")

    test_data     = load_gsm8k(split="test", seed=args.seed)
    eval_examples = test_data[:args.eval_n]

    total_steps = max(1, int(len(dataset) / args.batch_size * args.epochs))
    print(f"  {len(dataset)} items  batch={args.batch_size}  "
          f"→ {len(dataset)//args.batch_size} steps/epoch × {args.epochs} = {total_steps} steps")

    # Cosine schedule: lr → lr/10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr / 10,
    )
    print(f"  LR schedule: cosine {args.lr} → {args.lr/10:.1e}")

    # ── Baseline eval ─────────────────────────────────────────────────────────
    dsep()
    print(f"  BASELINE EVAL  n={args.eval_n}")
    sep()
    t_eval = time.perf_counter()
    baseline = eval_pass1(torch_gen, keras_model, eval_examples, args.max_tokens, args.eval_batch)
    print(f"  score={baseline:.4f}  ({time.perf_counter()-t_eval:.0f}s)")
    dsep(); print()

    prev_score = baseline
    eval_count = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"{'step':>6}  {'loss':>8}  {'gnorm':>6}  (eval tous les {args.eval_every} steps)\n")

    # Construit un buffer epoch par epoch, chacune shufflée indépendamment
    def make_buf():
        buf = []
        for _ in range(int(np.ceil(args.epochs))):
            epoch = list(dataset)
            random.shuffle(epoch)
            buf.extend(epoch)
        return buf

    data_buf   = make_buf()
    step_losses, step_gnorms = [], []
    t0 = time.perf_counter()

    for step in range(1, total_steps + 1):
        start = (step - 1) * args.batch_size
        batch = data_buf[start : start + args.batch_size]

        loss, gnorm = gradient_step(
            keras_model, optimizer, trainable, batch,
            kl_beta=args.kl_beta, ref_model=ref_model,
            micro_batch_size=args.micro_batch_size,
        )
        sync_keras_to_torch(keras_model, torch_gen.model, config)
        scheduler.step()

        step_losses.append(loss); step_gnorms.append(gnorm)
        elapsed = time.perf_counter() - t0

        print(f"{step:>6}  {loss:>8.5f}  {gnorm:>6.3f}  {elapsed:>6.0f}s", flush=True)

        # ── Eval ──────────────────────────────────────────────────────────────
        if step % args.eval_every == 0:
            sep()
            print(f"  last {args.eval_every} steps:  "
                  f"loss̄={np.mean(step_losses[-args.eval_every:]):.5f}  "
                  f"gnorm̄={np.mean(step_gnorms[-args.eval_every:]):.3f}")
            sep()
            t_eval = time.perf_counter()
            score  = eval_pass1(torch_gen, keras_model, eval_examples, args.max_tokens, args.eval_batch)
            d      = score - prev_score
            tag    = f"Δ={d:+.4f} {'↑' if d > 0 else ('↓' if d < 0 else '=')}"
            dsep()
            print(f"  EVAL step={step}/{total_steps}  score={score:.4f}  {tag}  "
                  f"(baseline={baseline:.4f})  {time.perf_counter()-t_eval:.0f}s")
            dsep(); print()
            prev_score = score
            eval_count += 1

            if args.save_every > 0 and eval_count % args.save_every == 0:
                base, ext = os.path.splitext(args.save_path)
                save_ckpt(torch_gen, config, f"{base}_step{step:05d}{ext}", step)

    # ── Final eval ────────────────────────────────────────────────────────────
    dsep()
    score = eval_pass1(torch_gen, keras_model, eval_examples, args.max_tokens, args.eval_batch)
    print(f"  FINAL  score={score:.4f}  Δ={score-baseline:+.4f}  (baseline={baseline:.4f})")
    dsep()
    save_ckpt(torch_gen, config, args.save_path, total_steps)


if __name__ == "__main__":
    main()
