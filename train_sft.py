"""
train_sft.py — SFT supervisé sur un dataset JSONL de CoT collectées.

Chaque ligne JSONL doit avoir : "query", "reasoning", "reward", "advantage".
Généré typiquement par collect_cot.py.

Usage:
    python -u train_sft.py \
        --data data/cot_dataset.jsonl \
        --checkpoint checkpoints/seqcond_lin5.pt
"""
import argparse, copy, json, os, random, time
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F

from seqcond.torch.generator import TorchGenerator
from train_grpo import load_gsm8k, check_answer
from collect_cot import _extract_answer_after_thinking, _parse_choice

W = 88
def sep(c="─"): print(c * W)
def dsep():     print("═" * W)


# ── Dataset ───────────────────────────────────────────────────────────────────

def _item_sort_key(item):
    return (
        float(item.get("advantage", 0.0)),  # meilleur sample du groupe en premier
        float(item.get("reward", 0.0)),
        float(item.get("weight", 0.0)),
        -(len(item["ids"]) - item["pl"]),
    )

def _limit_per_query(items, max_per_query):
    if max_per_query <= 0:
        return items
    grouped = {}
    for item in items:
        grouped.setdefault(item["query"], []).append(item)
    limited = []
    for query_items in grouped.values():
        limited.extend(
            sorted(query_items, key=_item_sort_key, reverse=True)[:max_per_query]
        )
    return limited

def build_eval_examples_from_items(items):
    eval_examples = []
    seen = set()
    for item in items:
        query = item["query"]
        if query in seen:
            continue
        seen.add(query)
        eval_examples.append(
            {
                "question":     query,
                "ground_truth": item["ground_truth"],
                "dataset":      item.get("dataset", "gsm8k"),
                "prompt":       item["prompt"],
            }
        )
    return eval_examples

def _fmt_metric(value):
    return "n/a" if value is None else f"{value:.4f}"

def _fmt_delta(value):
    return "n/a" if value is None else f"{value:+.4f}"

def load_dataset(path, tokenizer, weighting, min_reward, max_per_query, max_prompt_len=1024, max_seq_len=4096):
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
            except Exception:
                skipped += 1
                continue
            question  = rec["query"]
            reasoning = rec["reasoning"]
            reward    = float(rec.get("reward", 1.0))
            advantage = float(rec.get("advantage", 0.0))
            prompt_text = (
                "<|im_start|>user\n" + question
                + "\n<|im_end|><|im_start|>assistant\n<|think_start|>"
            )
            prompt_tokens = tokenizer([prompt_text])[0]

            completion = reasoning.strip()
            if not completion.endswith("<|im_end|>"):
                completion = completion + "\n<|im_end|>"

            full_tokens = tokenizer([prompt_text + completion])[0]
            comp_tokens = full_tokens[len(prompt_tokens):]

            if not comp_tokens:
                skipped += 1
                continue
            if max_prompt_len > 0 and len(prompt_tokens) > max_prompt_len:
                skipped += 1
                continue
            if max_seq_len > 0 and len(full_tokens) > max_seq_len:
                skipped += 1
                continue
            if reward < min_reward:
                skipped += 1
                continue

            if weighting == "reward":
                w = reward
            elif weighting == "advantage":
                w = advantage
                if w <= 0:
                    skipped += 1
                    continue
            else:
                w = 1.0

            items.append({
                "ids":          prompt_tokens + comp_tokens,
                "pl":           len(prompt_tokens),
                "weight":       w,
                "query":        question,
                "ground_truth": rec.get("ground_truth", ""),
                "dataset":      rec.get("dataset", "gsm8k"),
                "prompt":       prompt_text,
                "reward":       reward,
                "advantage":    advantage,
            })

    items = _limit_per_query(items, max_per_query)
    n_queries = len({it["query"] for it in items})
    print(
        f"  {len(items)} items chargés  ({skipped} ignorés)  "
        f"queries={n_queries}  max_per_query={max_per_query}"
    )
    if weighting == "advantage" and not items:
        raise RuntimeError(
            "Aucun item avec advantage > 0. Ce dataset ne contient pas de signal SFT utilisable en mode weighting=advantage."
        )
    return items


# ── Torch helpers ─────────────────────────────────────────────────────────────

def _model_device(model):
    return next(model.parameters()).device


def _autocast_context(use_mixed_bfloat16):
    if use_mixed_bfloat16 and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _seq_token_log_probs(model, input_ids, prompt_len, use_mixed_bfloat16=False):
    device = _model_device(model)
    if torch.is_tensor(input_ids):
        ids = input_ids.to(device=device, dtype=torch.long)
    else:
        ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    with _autocast_context(use_mixed_bfloat16):
        logits = model(ids)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    shift = log_probs[0, prompt_len - 1 : -1, :]
    targets = ids[0, prompt_len:].unsqueeze(-1)
    return torch.gather(shift, dim=-1, index=targets).squeeze(-1)


def _set_trainable_parameters(model, target):
    if target == "all":
        for param in model.parameters():
            param.requires_grad = True
        return []

    for param in model.parameters():
        param.requires_grad = False

    unfrozen_names = []
    for idx, (btype, block) in enumerate(zip(model.block_types, model.blocks)):
        block_name = f"block{idx:02d}"
        if target in ("transformer", "seqcond") and btype == target:
            for param in block.parameters():
                param.requires_grad = True
            unfrozen_names.append(f"{btype}:{block_name}")
        elif target == "mlp" or target == f"{btype}-mlp":
            if btype == "transformer":
                for sub in (block.norm2, block.ff_in, block.ff_out):
                    for param in sub.parameters():
                        param.requires_grad = True
                unfrozen_names.append(f"transformer-mlp:{block_name}")
            elif btype == "seqcond":
                attn = block.attn
                for sub in (attn.gate_proj, attn.out_proj, attn.gated_norm):
                    for param in sub.parameters():
                        param.requires_grad = True
                attn.W_readout.requires_grad = True
                unfrozen_names.append(f"seqcond-mlp:{block_name}")
    return unfrozen_names


# ── Gradient step (accumulation sur batch_size items) ─────────────────────────

def cache_old_log_probs(model, batch, use_mixed_bfloat16=False):
    """Calcule et stocke mean log-prob par item avant l'update (pour on_policy)."""
    model.eval()
    with torch.no_grad():
        for item in batch:
            ids = torch.tensor([item["ids"]], dtype=torch.long, device=_model_device(model))
            token_lps = _seq_token_log_probs(model, ids, item["pl"], use_mixed_bfloat16=use_mixed_bfloat16)
            item["_old_mean_lp"] = float(torch.mean(token_lps))
    model.train()


def gradient_step(model, optimizer, trainable, batch, kl_beta, ref_model,
                  micro_batch_size=1, use_mixed_bfloat16=False, on_policy=False):
    """
    Accumule les gradients sur `batch` items puis fait 1 optimizer.step().
    Normalise par la somme des poids du batch.

    on_policy : divise chaque item par son ratio π_cur / π_old (clampé) pour
                lisser le gradient quand la politique a déjà beaucoup bougé.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_w = sum(it["weight"] for it in batch)
    total_tokens = sum(len(it["ids"]) - it["pl"] for it in batch)
    total_loss = 0.0

    for i in range(0, len(batch), micro_batch_size):
        micro = batch[i : i + micro_batch_size]
        micro_loss = None
        for item in micro:
            ids = torch.tensor([item["ids"]], dtype=torch.long, device=_model_device(model))
            token_lps = _seq_token_log_probs(
                model, ids, item["pl"], use_mixed_bfloat16=use_mixed_bfloat16
            )
            w = item["weight"] / total_w

            if on_policy and "_old_mean_lp" in item:
                cur_mean_lp = torch.mean(token_lps)
                log_ratio = cur_mean_lp - item["_old_mean_lp"]
                ratio = torch.exp(torch.clamp(log_ratio, min=-0.2, max=0.2))
                loss = -w * ratio * torch.sum(token_lps) / total_tokens
            else:
                loss = -w * torch.sum(token_lps) / total_tokens

            if kl_beta > 0 and ref_model is not None:
                with torch.no_grad():
                    ref_lps = _seq_token_log_probs(
                        ref_model, ids, item["pl"], use_mixed_bfloat16=use_mixed_bfloat16
                    )
                kl = torch.sum(token_lps.float() - ref_lps.float()) / total_tokens
                loss = loss + kl_beta * w * kl

            micro_loss = loss if micro_loss is None else micro_loss + loss
            total_loss += float(loss.detach())

        if micro_loss is not None:
            micro_loss.backward()

    gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0))
    optimizer.step()
    # SeqCondAttention lazily caches tensors derived from trainable params
    # (_theta_cached, _w_int_cached, _decay_slopes_cached, _anchor_slopes_cached,
    #  _score_scale_b, _score_bias_b, _phase_scale_b, _conv_kernel_t). They are
    # never recomputed once set, so after an optimizer.step() they are stale
    # while step() (used by torch_gen.generate_batch / eval_pass1) keeps using
    # them. Invalidate so they get recomputed from the updated params — this
    # also ensures reloaded checkpoints behave identically to the live model.
    for sub in model.modules():
        for a in ("_theta_cached", "_w_int_cached", "_decay_slopes_cached",
                  "_anchor_slopes_cached", "_score_scale_b", "_score_bias_b",
                  "_phase_scale_b", "_conv_kernel_t"):
            if hasattr(sub, a) and getattr(sub, a) is not None:
                setattr(sub, a, None)
    return total_loss, gnorm


# ── Eval ──────────────────────────────────────────────────────────────────────

def corpus_lp(model, items, max_items=200, use_mixed_bfloat16=False):
    if not items:
        return float("nan")
    sample = random.sample(items, min(max_items, len(items)))
    was_training = model.training
    model.eval()
    total = 0.0
    with torch.no_grad():
        for item in sample:
            ids = torch.tensor([item["ids"]], dtype=torch.long, device=_model_device(model))
            total += float(
                torch.mean(
                    _seq_token_log_probs(
                        model, ids, item["pl"], use_mixed_bfloat16=use_mixed_bfloat16
                    )
                )
            )
    if was_training:
        model.train()
    return total / len(sample)


def _check_answer_for_dataset(text, ex):
    dataset = ex.get("dataset", "gsm8k")
    gt = ex["ground_truth"]
    if dataset in ("winogrande", "piqa"):
        answer_text = _extract_answer_after_thinking(text)
        return _parse_choice(answer_text, {"A", "B"}) == gt
    elif dataset in ("commonsenseqa", "gpqa_diamond", "mmlu", "mmlu_pro"):
        if dataset in ("commonsenseqa", "mmlu_pro"):
            valid = set("ABCDEFGHIJ")
        else:
            valid = set("ABCD")
        answer_text = _extract_answer_after_thinking(text)
        return _parse_choice(answer_text, valid) == gt
    elif dataset in ("creative_writing", "hello"):
        return True  # pas de ground truth, pas d'eval pass@1
    elif dataset == "triviaqa":
        answer_text = _extract_answer_after_thinking(text).lower()
        return gt.lower() in answer_text
    else:  # gsm8k, math500, local_math, default
        answer_text = _extract_answer_after_thinking(text)
        return check_answer(answer_text, gt)


def eval_pass1(torch_gen, examples, max_tokens, batch_size):
    model = torch_gen.model
    was_training = model.training
    model.eval()
    correct = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        texts = torch_gen.generate_batch(
            [ex["prompt"] for ex in batch],
            max_new_tokens=max_tokens, temperature=0.0,
            use_synth_template=False,
        )
        for text, ex in zip(texts, batch):
            if _check_answer_for_dataset(text, ex):
                correct += 1
    if was_training:
        model.train()
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
    p.add_argument("--warmup_steps",  type=int,   default=0,     help="warmup steps avec LR linéaire de 0 à --lr")
    p.add_argument("--kl_beta",       type=float, default=0.0,    help="poids KL vs référence (0=désactivé)")
    p.add_argument("--weighting",     default="none",             help="none | reward | advantage")
    p.add_argument("--min_reward",    type=float, default=1.0)
    p.add_argument("--max_per_query", type=int, default=1)
    p.add_argument("--max_prompt_len", type=int, default=1024, help="longueur max du prompt en tokens (0=désactivé)")
    p.add_argument("--max_seq_len",    type=int, default=4096, help="longueur max de la séquence complète en tokens (0=désactivé)")
    p.add_argument("--eval_every",    type=int,   default=50,     help="steps entre deux evals")
    p.add_argument("--eval_n",        type=int,   default=300)
    p.add_argument("--eval_train_n",  type=int,   default=100)
    p.add_argument("--eval_batch",    type=int,   default=16)
    p.add_argument("--max_tokens",    type=int,   default=1000)
    p.add_argument("--train",         default="all",              help="all | transformer | seqcond | mlp | transformer-mlp | seqcond-mlp")
    p.add_argument("--save_every",    type=int,   default=1,      help="sauvegarder tous les N evals (0=désactivé)")
    p.add_argument("--save_path",     default=None)
    p.add_argument("--on_policy",      action="store_true",  help="divise le gradient par π_old/π_cur pour lisser les updates")
    p.add_argument("--mixed_bfloat16", action="store_true")
    p.add_argument("--seed",          type=int,   default=42)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    if args.save_path is None:
        base = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.save_path = os.path.join("checkpoints", "sft", f"{base}_sft.pt")

    print("Loading torch_gen …")
    torch_gen = TorchGenerator(args.checkpoint, device="cuda")
    config = torch_gen.config
    model = torch_gen.model
    precision = "mixed_bfloat16 autocast" if args.mixed_bfloat16 else "float32"
    print(f"Using native Torch model ({precision}) …")

    ref_model = None
    if args.kl_beta > 0:
        ref_model = copy.deepcopy(model).cuda().eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        print(f"  KL ref model chargé  β={args.kl_beta}")

    unfrozen_names = _set_trainable_parameters(model, args.train)
    if args.train != "all":
        print(f"  [freeze] --train {args.train}: unfroze {len(unfrozen_names)} block groups, embeddings frozen")
        for name in unfrozen_names:
            print(f"    ✓ {name}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  {n_trainable:,} paramètres entraînables")
    if n_trainable == 0:
        raise RuntimeError("No trainable parameters selected.")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.99))

    print(
        f"\nChargement dataset : {args.data}  "
        f"(weighting={args.weighting}  min_reward={args.min_reward}  max_per_query={args.max_per_query})"
    )
    dataset = load_dataset(
        args.data,
        torch_gen.tokenizer,
        args.weighting,
        args.min_reward,
        args.max_per_query,
        args.max_prompt_len,
        args.max_seq_len,
    )
    if not dataset:
        raise RuntimeError("Dataset vide !")

    train_eval_examples = build_eval_examples_from_items(dataset)
    if args.eval_train_n > 0:
        train_eval_examples = train_eval_examples[:args.eval_train_n]
    else:
        train_eval_examples = []

    if args.eval_n > 0:
        test_data = load_gsm8k(split="test", seed=args.seed)
        eval_examples = test_data[:args.eval_n]
    else:
        eval_examples = []

    total_steps = max(1, int(len(dataset) / args.batch_size * args.epochs))
    print(f"  {len(dataset)} items  batch={args.batch_size}  "
          f"→ {len(dataset)//args.batch_size} steps/epoch × {args.epochs} = {total_steps} steps")

    def _lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        decay_steps = total_steps - args.warmup_steps
        decay_step = step - args.warmup_steps
        if decay_steps <= 0:
            return 1.0
        return 0.3 + 0.7 * (1 + np.cos(np.pi * decay_step / decay_steps)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    print(f"  LR schedule: warmup {args.warmup_steps} steps → cosine {args.lr} → {args.lr*0.3:.1e}")

    dsep()
    print(f"  BASELINE EVAL  train_n={len(train_eval_examples)}  test_n={args.eval_n}")
    sep()
    t_eval = time.perf_counter()
    baseline_lp = corpus_lp(model, dataset, use_mixed_bfloat16=args.mixed_bfloat16)
    baseline_train = (
        eval_pass1(torch_gen, train_eval_examples, args.max_tokens, args.eval_batch)
        if train_eval_examples
        else None
    )
    baseline_test = eval_pass1(torch_gen, eval_examples, args.max_tokens, args.eval_batch)
    print(
        f"  corpus_lp={baseline_lp:.4f}  train={_fmt_metric(baseline_train)}  "
        f"test={baseline_test:.4f}  ({time.perf_counter()-t_eval:.0f}s)"
    )
    dsep(); print()

    prev_lp = baseline_lp
    prev_train = baseline_train
    prev_test = baseline_test
    eval_count = 0

    print(f"{'step':>6}  {'loss':>8}  {'gnorm':>6}  {'tok/s':>8}  (eval tous les {args.eval_every} steps)\n")

    def make_buf():
        buf = []
        for _ in range(int(np.ceil(args.epochs))):
            epoch = list(dataset)
            random.shuffle(epoch)
            buf.extend(epoch)
        return buf

    data_buf = make_buf()
    step_losses, step_gnorms = [], []
    t0 = time.perf_counter()
    last_step_time = t0

    for step in range(1, total_steps + 1):
        start = (step - 1) * args.batch_size
        batch = data_buf[start : start + args.batch_size]
        step_tokens = sum(len(item["ids"]) - item["pl"] for item in batch)

        if args.on_policy:
            cache_old_log_probs(model, batch, use_mixed_bfloat16=args.mixed_bfloat16)

        loss, gnorm = gradient_step(
            model, optimizer, trainable, batch,
            kl_beta=args.kl_beta, ref_model=ref_model,
            micro_batch_size=args.micro_batch_size,
            use_mixed_bfloat16=args.mixed_bfloat16,
            on_policy=args.on_policy,
        )
        scheduler.step()

        step_losses.append(loss); step_gnorms.append(gnorm)
        now = time.perf_counter()
        elapsed = now - t0
        step_dt = max(now - last_step_time, 1e-9)
        toks_per_sec = step_tokens / step_dt
        last_step_time = now
        avg_step = elapsed / step
        eta_s = avg_step * (total_steps - step)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_s))

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"{step:>6}/{total_steps}  {loss:>8.5f}  {gnorm:>6.3f}  {toks_per_sec:>8.1f} tok/s  {elapsed:>6.0f}s  ETA {eta_str}  lr={lr:.2e}",
            flush=True,
        )

        if step % args.eval_every == 0:
            sep()
            print(f"  last {args.eval_every} steps:  "
                  f"loss̄={np.mean(step_losses[-args.eval_every:]):.5f}  "
                  f"gnorm̄={np.mean(step_gnorms[-args.eval_every:]):.3f}")
            sep()
            t_eval = time.perf_counter()
            cur_lp = corpus_lp(model, dataset, use_mixed_bfloat16=args.mixed_bfloat16)
            train_score = (
                eval_pass1(torch_gen, train_eval_examples, args.max_tokens, args.eval_batch)
                if train_eval_examples
                else None
            )
            test_score = eval_pass1(torch_gen, eval_examples, args.max_tokens, args.eval_batch)
            dlp = cur_lp - prev_lp
            dtrain = None if train_score is None or prev_train is None else train_score - prev_train
            dtest = test_score - prev_test
            dsep()
            print(
                f"  EVAL step={step}/{total_steps}  "
                f"corpus_lp={cur_lp:.4f} (Δ={_fmt_delta(dlp)})  "
                f"train={_fmt_metric(train_score)} (Δ={_fmt_delta(dtrain)})  "
                f"test={test_score:.4f} (Δ={_fmt_delta(dtest)})  "
                f"(baseline_test={baseline_test:.4f})  {time.perf_counter()-t_eval:.0f}s"
            )
            dsep(); print()
            prev_lp = cur_lp
            prev_train = train_score
            prev_test = test_score
            eval_count += 1

            if args.save_every > 0 and eval_count % args.save_every == 0:
                base, ext = os.path.splitext(args.save_path)
                save_ckpt(torch_gen, config, f"{base}_step{step:05d}{ext}", step)

    dsep()
    final_lp = corpus_lp(model, dataset, use_mixed_bfloat16=args.mixed_bfloat16)
    final_train = (
        eval_pass1(torch_gen, train_eval_examples, args.max_tokens, args.eval_batch)
        if train_eval_examples
        else None
    )
    final_test = eval_pass1(torch_gen, eval_examples, args.max_tokens, args.eval_batch)
    print(
        f"  FINAL  corpus_lp={final_lp:.4f}  Δlp={final_lp-baseline_lp:+.4f}  "
        f"train={_fmt_metric(final_train)}  "
        f"Δtrain={_fmt_delta(None if final_train is None or baseline_train is None else final_train-baseline_train)}  "
        f"test={final_test:.4f}  Δtest={final_test-baseline_test:+.4f}  (baseline_test={baseline_test:.4f})"
    )
    dsep()
    save_ckpt(torch_gen, config, args.save_path, total_steps)


if __name__ == "__main__":
    main()
