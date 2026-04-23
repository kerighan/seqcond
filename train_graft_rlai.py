"""
train_graft_rlai.py — GRAFT on the RLAI mix with LLM-based multi-criteria scoring.

Based on train_rft.py architecture (native Torch, neg_scale balancing, on_policy,
KL penalty, rollback, save_best, selective freezing) but with:
  - LLM judge scoring instead of GSM8K check_answer
  - RLAI mix dataset instead of GSM8K
  - Mean LLM reward eval instead of pass@k

Usage:
    KERAS_BACKEND=torch python train_graft_rlai.py \
        --checkpoint checkpoints/seqcond_lin5.pt \
        --openai_api_key sk-... \
        --dataset_path data/rlai_mix.jsonl
"""
import argparse, asyncio, copy, os, random, re, textwrap
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F

from seqcond.torch.generator import TorchGenerator
from train_grpo import _compute_advantages, _repetition_penalty
from train_grpo_rlai import (
    load_rlai_examples, _llm_judgments, _judgment_reward, _score_one,
)
import time


# ── Data ──────────────────────────────────────────────────────────────────────

def load_train_eval(eval_frac=0.05, seed=42, **kwargs):
    """Load RLAI examples and split into train / eval.

    eval_frac: fraction reserved for eval (default 5 %).
    kwargs forwarded to load_rlai_examples (dataset_path, dataset_name, …).
    """
    examples = load_rlai_examples(seed=seed, **kwargs)
    n_eval   = max(10, int(len(examples) * eval_frac))
    # First n_eval are eval (load_rlai_examples already shuffled with seed)
    return examples[n_eval:], examples[:n_eval]


# ── Generation ────────────────────────────────────────────────────────────────

def generate_groups(torch_gen: TorchGenerator, examples, G, max_tokens, temperature):
    results = []
    for ex in examples:
        prompt_tokens = torch_gen.tokenizer([ex["prompt"]])[0]
        texts, comp_ids = torch_gen.generate_group(
            ex["prompt"], n=G,
            max_new_tokens=max_tokens,
            temperature=temperature,
            use_synth_template=False,
        )
        results.append({
            "example":       ex,
            "prompt_tokens": prompt_tokens,
            "texts":         texts,
            "comp_ids":      comp_ids,
        })
    return results


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_group_llm(group, api_key, judge_model="gpt-4.1-mini"):
    """LLM multi-criteria scoring.  Returns continuous rewards in [0, 1].

    No binary / conciseness scaling — the LLM already judges concision.
    Repetition penalty still applied on top.
    """
    ex       = group["example"]
    texts    = group["texts"]
    comp_ids = group["comp_ids"]

    judgments = _llm_judgments(
        ex["instruction"], texts, api_key,
        reference_answer=ex.get("reference_answer", ""),
        judge_model=judge_model,
    )
    rewards = [_judgment_reward(j) for j in judgments]

    for i, ids in enumerate(comp_ids):
        pen = _repetition_penalty(ids)
        if pen:
            rewards[i] = max(0.0, rewards[i] + pen)

    return rewards


# ── Selection ─────────────────────────────────────────────────────────────────

def select_pairs(texts, comp_ids, advantages):
    """Use advantages (mean-centred rewards) to define pos/neg.

    With continuous LLM rewards there's no binary threshold —
    anything above mean is positive, anything below is negative.
    """
    positives = sorted(
        [(i, texts[i], comp_ids[i], advantages[i])
         for i, a in enumerate(advantages) if a > 0],
        key=lambda x: x[3], reverse=True,
    )
    negatives = sorted(
        [(i, texts[i], comp_ids[i], advantages[i])
         for i, a in enumerate(advantages) if a < 0],
        key=lambda x: abs(x[3]), reverse=True,
    )
    return positives, negatives


# ── Log-prob helpers ──────────────────────────────────────────────────────────

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


def seq_avg_logprob(model, prompt_tokens, comp_tokens, use_mixed_bfloat16=False):
    """Mean per-token log-prob for comp_tokens (no grad, purely cosmetic)."""
    ids = torch.tensor([prompt_tokens + comp_tokens], dtype=torch.long, device=_model_device(model))
    with torch.no_grad():
        token_lps = _seq_token_log_probs(
            model, ids, len(prompt_tokens), use_mixed_bfloat16=use_mixed_bfloat16
        )
    return float(torch.mean(token_lps))


def _cache_old_policy_lps(model, scored_groups, use_mixed_bfloat16=False):
    for g in scored_groups:
        pt = g["prompt_tokens"]
        old_mean_lps = {}
        for idx, comp in enumerate(g["comp_ids"]):
            if comp:
                old_mean_lps[idx] = seq_avg_logprob(
                    model, pt, comp, use_mixed_bfloat16=use_mixed_bfloat16
                )
        g["old_mean_lps"] = old_mean_lps


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


# ── Gradient step ─────────────────────────────────────────────────────────────

_GRAD_CHECKED = False  # print grad diagnostics once


def _apply_gradient(model, optimizer, scored_groups, micro_batch_size=2,
                    neg_scale=0.0, kl_beta=0.0, ref_model=None,
                    use_mixed_bfloat16=False, on_policy=False):
    global _GRAD_CHECKED
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]

    # Build item lists
    pos_items, neg_items = [], []
    for g in scored_groups:
        pt = g["prompt_tokens"]
        for (idx, _, comp, adv) in g["positives"] + g["negatives"]:
            if not comp:
                continue
            if adv > 0:
                pos_items.append({
                    "ids": pt + comp,
                    "pl": len(pt),
                    "adv": adv,
                    "old_mean_lp": g.get("old_mean_lps", {}).get(idx),
                })
            elif adv < 0 and neg_scale > 0:
                neg_items.append({
                    "ids": pt + comp,
                    "pl": len(pt),
                    "adv": adv,
                    "old_mean_lp": g.get("old_mean_lps", {}).get(idx),
                })

    if not pos_items and not neg_items:
        return 0.0, 0.0, 0.0

    all_items = pos_items + neg_items
    n_terms = len(all_items)
    total_tokens = sum(len(it["ids"]) - it["pl"] for it in all_items)

    # ── Estimate effective neg weight via micro-batched grad norms ────────────
    effective_neg_weight = 0.0
    if neg_scale > 0 and neg_items and pos_items:
        def _micro_grad_norm(items):
            optimizer.zero_grad(set_to_none=True)
            for i in range(0, len(items), micro_batch_size):
                batch_loss = None
                for item in items[i : i + micro_batch_size]:
                    ids = torch.tensor([item["ids"]], dtype=torch.long,
                                       device=_model_device(model))
                    token_lps = _seq_token_log_probs(
                        model, ids, item["pl"], use_mixed_bfloat16=use_mixed_bfloat16
                    )
                    token_weight = (len(item["ids"]) - item["pl"]) / total_tokens
                    if on_policy and item["old_mean_lp"] is not None:
                        cur_mean_lp = torch.mean(token_lps)
                        log_ratio = cur_mean_lp - float(item["old_mean_lp"])
                        ratio = torch.exp(torch.clamp(log_ratio, min=-0.2, max=0.2))
                        item_loss = (-item["adv"] * ratio * token_weight) / n_terms
                    else:
                        item_loss = (-item["adv"] * torch.sum(token_lps) / total_tokens) / n_terms
                    batch_loss = item_loss if batch_loss is None else batch_loss + item_loss
                if batch_loss is not None:
                    batch_loss.backward()
            return sum(float(torch.sum(p.grad.float() ** 2))
                       for p in trainable if p.grad is not None) ** 0.5

        pos_norm = _micro_grad_norm(pos_items)
        neg_norm = _micro_grad_norm(neg_items)
        if pos_norm > 1e-8 and neg_norm > 1e-8:
            effective_neg_weight = min(neg_scale * pos_norm / neg_norm, 10.0)

    # ── Final gradient step ───────────────────────────────────────────────────
    optimizer.zero_grad(set_to_none=True)
    total_loss_val = 0.0
    for i in range(0, n_terms, micro_batch_size):
        batch_loss = None
        for item in all_items[i : i + micro_batch_size]:
            ids = torch.tensor([item["ids"]], dtype=torch.long, device=_model_device(model))
            token_lps = _seq_token_log_probs(
                model, ids, item["pl"], use_mixed_bfloat16=use_mixed_bfloat16
            )
            w = effective_neg_weight if item["adv"] < 0 else 1.0
            if on_policy and item["old_mean_lp"] is not None:
                cur_mean_lp = torch.mean(token_lps)
                token_weight = (len(item["ids"]) - item["pl"]) / total_tokens
                log_ratio = cur_mean_lp - float(item["old_mean_lp"])
                ratio = torch.exp(torch.clamp(log_ratio, min=-0.2, max=0.2))
                loss = (-w * item["adv"] * ratio * token_weight) / n_terms
            else:
                loss = (-w * item["adv"] * torch.sum(token_lps) / total_tokens) / n_terms
            if kl_beta > 0 and ref_model is not None:
                with torch.no_grad():
                    ref_lps = _seq_token_log_probs(
                        ref_model, ids, item["pl"], use_mixed_bfloat16=use_mixed_bfloat16
                    )
                log_ratio = token_lps.float() - ref_lps.float()
                kl_k3 = torch.exp(log_ratio) - 1 - log_ratio
                loss = loss + (kl_beta * torch.sum(kl_k3) / total_tokens) / n_terms
            batch_loss = loss if batch_loss is None else batch_loss + loss
            total_loss_val += float(loss.detach())
        if batch_loss is not None:
            batch_loss.backward()

    gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0))

    if not _GRAD_CHECKED:
        _GRAD_CHECKED = True
        has_grad = [p for p in trainable if p.grad is not None]
        nonzero  = [p for p in has_grad  if p.grad.abs().max() > 0]
        ref_param = trainable[0]
        val_before = ref_param.data.float().mean().item()
        optimizer.step()
        val_after  = ref_param.data.float().mean().item()
        print(f"\n  [GRAD CHECK] trainable={len(trainable)} | has_grad={len(has_grad)} | nonzero_grad={len(nonzero)}")
        print(f"  [GRAD CHECK] param[0] mean: {val_before:.6f} → {val_after:.6f}  (changed={val_before != val_after})\n")
    else:
        optimizer.step()

    return total_loss_val, gnorm, effective_neg_weight


# ── Checkpoint ───────────────────────────────────────────────────────────────

_CACHE_PREFIXES = (
    '_conv_kernel_t', '_decay_slopes_cached', '_phase_scale_b',
    '_score_bias_b', '_score_scale_b', '_theta_cached', '_w_int_cached',
)

def save_checkpoint(torch_gen: TorchGenerator, config, save_path, step):
    sd = {k: v.cpu() for k, v in torch_gen.model.state_dict().items()
          if not any(k.endswith(sfx) for sfx in _CACHE_PREFIXES)}
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({"state_dict": sd, "config": config}, save_path)
    print(f"  checkpoint saved → {save_path}  ({len(sd)} tensors, step={step})")


def refresh_reference_model(ref_model, model):
    sd = {k: v for k, v in model.state_dict().items()
          if not any(k.endswith(sfx) for sfx in _CACHE_PREFIXES)}
    ref_model.load_state_dict(sd, strict=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False


def snapshot_training_state(model, optimizer):
    model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    for state in optimizer_state.get("state", {}).values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.detach().cpu().clone()
    return model_state, optimizer_state


def _optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def restore_training_state(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    _optimizer_to_device(optimizer, _model_device(model))


def eval_score_sigma(score, n_examples):
    if n_examples <= 0:
        return 0.0
    p = min(max(float(score), 0.0), 1.0)
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / n_examples))


# ── Eval ──────────────────────────────────────────────────────────────────────

def eval_mean_reward(torch_gen: TorchGenerator, examples, api_key, judge_model,
                     max_tokens, temperature, gen_batch_size=8):
    """Mean LLM reward on eval examples (1 completion each).

    All LLM calls for a generation batch are issued concurrently.
    """
    model = torch_gen.model
    was_training = model.training
    model.eval()

    all_rewards = []
    for start in range(0, len(examples), gen_batch_size):
        batch    = examples[start : start + gen_batch_size]
        prompts  = [ex["prompt"] for ex in batch]
        texts    = torch_gen.generate_batch(
            prompts, max_new_tokens=max_tokens, temperature=temperature,
            use_synth_template=False,
        )
        # Parallel LLM calls
        async def _score_batch():
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key)
            sem    = asyncio.Semaphore(len(batch))
            tasks  = [
                _score_one(client, ex["instruction"], text,
                           ex.get("reference_answer", ""), sem, judge_model)
                for ex, text in zip(batch, texts)
            ]
            results = await asyncio.gather(*tasks)
            await client.close()
            return results

        judgments = asyncio.run(_score_batch())
        all_rewards.extend(_judgment_reward(j) for j in judgments)

    if was_training:
        model.train()

    return float(np.mean(all_rewards))


# ── Pretty print ──────────────────────────────────────────────────────────────

W = 88

def box(text, width=W):
    return "\n".join("  " + l for l in textwrap.wrap(text, width - 2))

def separator(char="─"):
    print(char * W)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     default="./checkpoints/seqcond_lin5.pt")
    p.add_argument("--openai_api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--judge_model",    default="gpt-4.1-mini")
    # Dataset — mutually exclusive: local file or HF dataset
    p.add_argument("--dataset_path",   default=None, help="path to .jsonl or .json file")
    p.add_argument("--dataset_name",   default=None, help="HuggingFace dataset name")
    p.add_argument("--dataset_config", default=None)
    p.add_argument("--dataset_split",  default="train")
    # Training
    p.add_argument("--n",              type=int,   default=2,    help="prompts per step")
    p.add_argument("--g",              type=int,   default=8,    help="completions per prompt")
    p.add_argument("--temperature",    type=float, default=0.7)
    p.add_argument("--eval_temperature", type=float, default=None, help="eval temperature (default: same as --temperature)")
    p.add_argument("--max_tokens",     type=int,   default=1000)
    p.add_argument("--lr",             type=float, default=1e-5)
    p.add_argument("--sft_steps",      type=int,   default=1,    help="gradient steps per batch")
    p.add_argument("--micro_batch_size", type=int, default=24,   help="sequences per backward call")
    p.add_argument("--neg_scale",      type=float, default=0.0,  help="negative gradient magnitude relative to positive (0=off, 0.5=half, 1=balanced)")
    p.add_argument("--kl_beta",        type=float, default=0.0,  help="KL penalty weight against reference policy (0 = disabled)")
    p.add_argument("--float32",        action="store_true",       help="use float32 compute instead of mixed bfloat16")
    p.add_argument("--on_policy",      action="store_true",       help="apply an old-policy importance ratio")
    p.add_argument("--refresh_kl_ref_on_best_eval", action="store_true")
    p.add_argument("--rollback_on_eval_drop", action="store_true")
    p.add_argument("--rollback_sigma", type=float, default=2.0)
    p.add_argument("--train",          default="all",             help="all | transformer | seqcond | mlp | transformer-mlp | seqcond-mlp")
    # Epochs / steps
    p.add_argument("--epochs",         type=float, default=1.0,  help="epochs over train set (float ok)")
    p.add_argument("--steps",          type=int,   default=None, help="override total steps (ignores --epochs)")
    # Eval
    p.add_argument("--eval_every",     type=int,   default=20)
    p.add_argument("--eval_n",         type=int,   default=50,   help="eval examples (LLM calls)")
    p.add_argument("--eval_batch",     type=int,   default=8)
    p.add_argument("--eval_frac",      type=float, default=0.05, help="fraction of dataset held out for eval")
    # Save
    p.add_argument("--save_path",      default=None)
    p.add_argument("--save_best",      action="store_true")
    p.add_argument("--seed",           type=int,   default=42)
    args = p.parse_args()

    if not args.openai_api_key:
        raise ValueError("--openai_api_key required (or set OPENAI_API_KEY env var)")

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.save_path is None:
        base = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.save_path = os.path.join("checkpoints", "graft_rlai", f"{base}_graft_rlai.pt")
    save_base, save_ext = os.path.splitext(args.save_path)
    best_save_path = f"{save_base}_best{save_ext}"

    # ── Load models ───────────────────────────────────────────────────────────
    torch_gen = TorchGenerator(args.checkpoint, device="cuda")
    config = torch_gen.config
    model = torch_gen.model
    use_mixed_bfloat16 = not args.float32
    precision = "mixed_bfloat16 autocast" if use_mixed_bfloat16 else "float32"
    print(f"  using native Torch model ({precision})")

    # ── Reference model for KL penalty (frozen, kept on CUDA) ────────────────
    ref_model = None
    if args.kl_beta > 0:
        ref_model = copy.deepcopy(model).cuda().eval()
        for p_param in ref_model.parameters():
            p_param.requires_grad = False
        print(f"  KL penalty enabled  β={args.kl_beta}")

    unfrozen_names = _set_trainable_parameters(model, args.train)
    if args.train != "all":
        print(f"  [freeze] --train {args.train}: unfroze {len(unfrozen_names)} block groups, embeddings frozen")
        for name in unfrozen_names:
            print(f"    ✓ {name}")

    trainable_params = [p_param for p_param in model.parameters() if p_param.requires_grad]
    n_trainable = sum(p_param.numel() for p_param in trainable_params)
    n_total     = sum(p_param.numel() for p_param in model.parameters())
    print(f"  params: {n_trainable:,} trainable / {n_total:,} total")
    if n_trainable == 0:
        raise RuntimeError("No trainable parameters!")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.99))

    # ── Data ──────────────────────────────────────────────────────────────────
    train, eval_all = load_train_eval(
        eval_frac=args.eval_frac,
        seed=args.seed,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
    )
    eval_examples = eval_all[:args.eval_n]
    print(f"Dataset: {len(train)} train  {len(eval_examples)} eval")

    # ── Baseline eval ─────────────────────────────────────────────────────────
    eval_temp = args.eval_temperature if args.eval_temperature is not None else args.temperature
    prev_eval_score = None
    n_eval_label = len(eval_examples)
    print(f"\n  Baseline eval (mean LLM reward) on {n_eval_label} examples …")
    score = eval_mean_reward(
        torch_gen, eval_examples,
        api_key=args.openai_api_key, judge_model=args.judge_model,
        max_tokens=args.max_tokens, temperature=eval_temp,
        gen_batch_size=args.eval_batch,
    )
    separator("═")
    print(f"  BASELINE EVAL  mean_reward={score:.4f}  n={n_eval_label}")
    separator("═")
    print()
    prev_eval_score = score
    best_eval_score = score
    best_model_state, best_optimizer_state = snapshot_training_state(model, optimizer)
    if args.save_best:
        save_checkpoint(torch_gen, config, best_save_path, 0)

    # ── Training loop ─────────────────────────────────────────────────────────
    steps_per_epoch = len(train) // args.n
    if args.steps is not None:
        total_steps = args.steps
    else:
        total_steps = max(1, int(steps_per_epoch * args.epochs))
    print(f"Train: {len(train)} examples  →  {steps_per_epoch} steps/epoch  ×  {args.epochs} = {total_steps} steps\n")

    train_buf  = []
    step       = 0
    last_step_time = time.perf_counter()

    # ── Inter-eval accumulators ───────────────────────────────────────────────
    _acc = dict(reward_sum=0.0, reward_n=0,
                loss_sum=0.0, gnorm_sum=0.0, grad_steps=0, steps=0)

    while step < total_steps:
        if not train_buf:
            epoch_data = list(train)
            random.shuffle(epoch_data)
            train_buf = [epoch_data[i : i + args.n]
                         for i in range(0, len(epoch_data) - args.n + 1, args.n)]

        batch = train_buf.pop(0)
        step += 1

        raw_groups = generate_groups(
            torch_gen, batch,
            G=args.g, max_tokens=args.max_tokens, temperature=args.temperature,
        )

        # Score (LLM)
        scored_groups = []
        for g in raw_groups:
            rewards = score_group_llm(g, args.openai_api_key, args.judge_model)
            advantages = list(_compute_advantages(rewards, normalize_std=True))
            positives, negatives = select_pairs(g["texts"], g["comp_ids"], advantages)
            scored_groups.append({
                "example":       g["example"],
                "prompt_tokens": g["prompt_tokens"],
                "comp_ids":      g["comp_ids"],
                "texts":         g["texts"],
                "rewards":       rewards,
                "advantages":    advantages,
                "positives":     positives,
                "negatives":     negatives,
            })
        if args.on_policy:
            _cache_old_policy_lps(
                model, scored_groups, use_mixed_bfloat16=use_mixed_bfloat16
            )

        # ── Gradient step ─────────────────────────────────────────────────────
        has_signal = any(g["positives"] or g["negatives"] for g in scored_groups)
        loss = 0.0
        gnorm = 0.0
        effective_neg_weight = args.neg_scale
        n_pos = sum(len(g["positives"]) for g in scored_groups)
        n_neg = sum(len(g["negatives"]) for g in scored_groups)
        if has_signal:
            for _ in range(args.sft_steps):
                loss, gnorm, effective_neg_weight = _apply_gradient(
                    model, optimizer, scored_groups,
                    micro_batch_size=args.micro_batch_size,
                    neg_scale=args.neg_scale,
                    kl_beta=args.kl_beta,
                    ref_model=ref_model,
                    use_mixed_bfloat16=use_mixed_bfloat16,
                    on_policy=args.on_policy,
                )

        # ── Per-step summary ──────────────────────────────────────────────────
        n_comp   = sum(len(g["rewards"]) for g in scored_groups)
        mean_r   = float(np.mean([r for g in scored_groups for r in g["rewards"]]))
        skip_tag = "  (skipped)" if not has_signal else ""
        step_duration = time.perf_counter() - last_step_time
        last_step_time = time.perf_counter()
        print(f"[step {step:4d}/{total_steps}]  "
              f"n={n_comp}  r̄={mean_r:.4f}  n+/n-={n_pos}/{n_neg}  "
              f"loss={loss:.5f}  gnorm={gnorm:.3f}  negw={effective_neg_weight:.4f}{skip_tag}  {step_duration:.1f}s")

        # accumulate
        _acc["reward_sum"] += mean_r
        _acc["reward_n"]   += 1
        _acc["steps"]      += 1
        if has_signal:
            _acc["loss_sum"]   += loss
            _acc["gnorm_sum"]  += gnorm
            _acc["grad_steps"] += 1

        # ── Eval ──────────────────────────────────────────────────────────────
        if step % args.eval_every == 0:
            if _acc["steps"] > 0:
                avg_r     = _acc["reward_sum"] / _acc["reward_n"]
                gs        = _acc["grad_steps"]
                avg_loss  = _acc["loss_sum"]  / gs if gs else float("nan")
                avg_gnorm = _acc["gnorm_sum"] / gs if gs else float("nan")
                separator()
                print(f"  last {_acc['steps']} steps:  "
                      f"r̄={avg_r:.4f}  "
                      f"loss̄={avg_loss:.5f}  "
                      f"gnorm̄={avg_gnorm:.3f}")
                separator()
            _acc.update(reward_sum=0.0, reward_n=0,
                        loss_sum=0.0, gnorm_sum=0.0, grad_steps=0, steps=0)

            print(f"\n  evaluating mean reward on {n_eval_label} examples …")
            score = eval_mean_reward(
                torch_gen, eval_examples,
                api_key=args.openai_api_key, judge_model=args.judge_model,
                max_tokens=args.max_tokens, temperature=eval_temp,
                gen_batch_size=args.eval_batch,
            )
            if prev_eval_score is not None:
                d = score - prev_eval_score
                tag = f"  Δ={d:+.4f} {'↑' if d > 0 else ('↓' if d < 0 else '=')}"
            else:
                tag = "  (baseline)"
            separator("═")
            print(f"  EVAL  step={step}/{total_steps}  mean_reward={score:.4f}"
                  f"  n={n_eval_label}{tag}")
            separator("═")
            if score > best_eval_score:
                best_eval_score = score
                best_model_state, best_optimizer_state = snapshot_training_state(model, optimizer)
                if args.refresh_kl_ref_on_best_eval and ref_model is not None:
                    refresh_reference_model(ref_model, model)
                    print(f"  KL reference updated from best eval model  score={score:.4f}")
                    print()
                if args.save_best:
                    save_checkpoint(torch_gen, config, best_save_path, step)
            else:
                if args.rollback_on_eval_drop:
                    sigma = eval_score_sigma(best_eval_score, n_eval_label)
                    rollback_threshold = best_eval_score - args.rollback_sigma * sigma
                    if score < rollback_threshold:
                        restore_training_state(model, optimizer, best_model_state, best_optimizer_state)
                        if args.refresh_kl_ref_on_best_eval and ref_model is not None:
                            refresh_reference_model(ref_model, model)
                        print(
                            f"  rollback triggered: score={score:.4f} < threshold={rollback_threshold:.4f} "
                            f"(best={best_eval_score:.4f}, σ={sigma:.4f}, z={args.rollback_sigma:.2f})"
                        )
                        score = best_eval_score
            print()
            prev_eval_score = score

            if args.save_path:
                base, ext = os.path.splitext(args.save_path)
                path = f"{base}_step{step:04d}{ext}"
                save_checkpoint(torch_gen, config, path, step)


if __name__ == "__main__":
    main()
