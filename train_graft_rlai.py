"""
train_graft_rlai.py — GRAFT on the RLAI mix with LLM-based multi-criteria scoring.

Usage:
    KERAS_BACKEND=torch python train_graft_rlai.py \
        --checkpoint checkpoints/seqcond_lin5.pt \
        --openai_api_key sk-... \
        --dataset_path data/rlai_mix.jsonl \
        --save_path checkpoints/seqcond_lin5_graft_rlai.pt
"""
import argparse, asyncio, os, random, textwrap
import numpy as np
import torch

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras import ops

from pydantic import BaseModel, Field
from seqcond.torch.generator import TorchGenerator
from convert_torch_to_keras import build_keras_model, convert_weights
from train_grpo import (
    _compute_advantages, _repetition_penalty,
    _seq_token_log_probs, sync_keras_to_torch,
)
from train_grpo_rlai import (
    load_rlai_examples, _llm_judgments, _judgment_reward,
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

def seq_avg_logprob(model, prompt_tokens, comp_tokens):
    ids = np.array([prompt_tokens + comp_tokens], dtype=np.int32)
    with torch.no_grad():
        token_lps = _seq_token_log_probs(model, ids, len(prompt_tokens))
    return float(ops.mean(token_lps))


# ── Gradient step ─────────────────────────────────────────────────────────────

def _apply_gradient(keras_model, optimizer, scored_groups, micro_batch_size=2):
    optimizer.zero_grad()

    all_items = []
    for g in scored_groups:
        pt = g["prompt_tokens"]
        for (_, _, comp, adv) in g["positives"] + g["negatives"]:
            if comp:
                all_items.append({"ids": pt + comp, "pl": len(pt), "adv": adv})

    n_terms = len(all_items)
    if n_terms == 0:
        return 0.0

    total_loss_val = 0.0
    for i in range(0, n_terms, micro_batch_size):
        batch_loss = 0
        for item in all_items[i : i + micro_batch_size]:
            ids       = np.array([item["ids"]], dtype=np.int32)
            token_lps = _seq_token_log_probs(keras_model, ids, item["pl"])
            loss      = (-item["adv"] * ops.mean(token_lps)) / n_terms
            batch_loss          += loss
            total_loss_val      += float(loss.detach())
        batch_loss.backward()

    torch.nn.utils.clip_grad_norm_(keras_model.parameters(), max_norm=1.0)
    optimizer.step()
    return total_loss_val


# ── Checkpoint ────────────────────────────────────────────────────────────────

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


# ── Eval ──────────────────────────────────────────────────────────────────────

def eval_mean_reward(torch_gen: TorchGenerator, keras_model, examples,
                     api_key, judge_model, max_tokens, temperature,
                     gen_batch_size=8):
    """Mean LLM reward on eval examples (1 completion each, parallelised).

    keras_model moved to CPU before generation and warmed up after.
    All LLM calls for a generation batch are issued concurrently.
    """
    keras_model.cpu()
    torch.cuda.empty_cache()

    all_rewards = []
    for start in range(0, len(examples), gen_batch_size):
        batch    = examples[start : start + gen_batch_size]
        prompts  = [ex["prompt"] for ex in batch]
        texts    = torch_gen.generate_batch(
            prompts, max_new_tokens=max_tokens, temperature=temperature,
            use_synth_template=False,
        )
        # Parallel LLM calls: one judgment per (instruction, completion) pair
        async def _score_batch():
            from openai import AsyncOpenAI
            from train_grpo_rlai import _score_one, _judge_messages
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

    keras_model.cuda()
    dummy = np.ones((1, 4), dtype=np.int32)
    with torch.no_grad():
        keras_model(dummy, training=False)

    return float(np.mean(all_rewards))


# ── Pretty print helpers ──────────────────────────────────────────────────────

W = 88

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
    p.add_argument("--max_tokens",     type=int,   default=1000)
    p.add_argument("--lr",             type=float, default=1e-5)
    p.add_argument("--sft_steps",      type=int,   default=1)
    p.add_argument("--steps",          type=int,   default=500)
    # Eval
    p.add_argument("--eval_every",     type=int,   default=20)
    p.add_argument("--eval_n",         type=int,   default=50,   help="eval examples (LLM calls)")
    p.add_argument("--eval_batch",     type=int,   default=8)
    p.add_argument("--eval_frac",      type=float, default=0.05, help="fraction of dataset held out for eval")
    # Save
    p.add_argument("--save_path",      default=None,
                   help="e.g. checkpoints/seqcond_lin5_graft_rlai.pt")
    p.add_argument("--seed",           type=int,   default=42)
    args = p.parse_args()

    if not args.openai_api_key:
        raise ValueError("--openai_api_key required (or set OPENAI_API_KEY env var)")

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Load models ───────────────────────────────────────────────────────────
    torch_gen  = TorchGenerator(args.checkpoint, device="cuda")
    config     = torch_gen.config
    state_dict = {k: v.detach().cpu().float().numpy()
                  for k, v in torch_gen.model.state_dict().items()}

    keras.mixed_precision.set_global_policy("mixed_float16")
    keras_model = build_keras_model(config)
    convert_weights(config, state_dict, keras_model)
    keras_model.cuda()
    optimizer = torch.optim.AdamW(
        keras_model.parameters(), lr=args.lr, betas=(0.9, 0.99),
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    train, eval_examples = load_train_eval(
        eval_frac=args.eval_frac,
        seed=args.seed,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
    )
    eval_examples = eval_examples[:args.eval_n]
    print(f"Dataset: {len(train)} train  {len(eval_examples)} eval")

    # ── Baseline eval ─────────────────────────────────────────────────────────
    print(f"\n  Baseline eval (mean LLM reward) on {len(eval_examples)} examples …")
    prev_score = eval_mean_reward(
        torch_gen, keras_model, eval_examples,
        api_key=args.openai_api_key, judge_model=args.judge_model,
        max_tokens=args.max_tokens, temperature=args.temperature,
        gen_batch_size=args.eval_batch,
    )
    separator("═")
    print(f"  BASELINE EVAL  mean_reward={prev_score:.4f}")
    separator("═")
    print()

    # ── Training loop ─────────────────────────────────────────────────────────
    last_step_time = time.perf_counter()
    for step in range(1, args.steps + 1):
        batch = random.sample(train, args.n)

        raw_groups = generate_groups(
            torch_gen, batch,
            G=args.g, max_tokens=args.max_tokens, temperature=args.temperature,
        )

        # Score (LLM) — one API batch per group (G concurrent calls)
        scored_groups = []
        for g in raw_groups:
            rewards   = score_group_llm(g, args.openai_api_key, args.judge_model)
            advantages = list(_compute_advantages(rewards, normalize_std=False))
            positives, negatives = select_pairs(g["texts"], g["comp_ids"], advantages)
            scored_groups.append({
                "example":       g["example"],
                "prompt_tokens": g["prompt_tokens"],
                "texts":         g["texts"],
                "rewards":       rewards,
                "advantages":    advantages,
                "positives":     positives,
                "negatives":     negatives,
            })

        # ── Gradient step ─────────────────────────────────────────────────────
        has_signal = any(g["positives"] or g["negatives"] for g in scored_groups)
        loss = 0.0
        if has_signal:
            for _ in range(args.sft_steps):
                loss = _apply_gradient(keras_model, optimizer, scored_groups)
            sync_keras_to_torch(keras_model, torch_gen.model, config)

        # ── Per-step summary ──────────────────────────────────────────────────
        n_comp   = sum(len(g["rewards"]) for g in scored_groups)
        mean_r   = float(np.mean([r for g in scored_groups for r in g["rewards"]]))
        skip_tag = "  (skipped)" if not has_signal else ""
        dt       = time.perf_counter() - last_step_time
        last_step_time = time.perf_counter()
        print(f"[step {step:4d}/{args.steps}]  "
              f"n={n_comp}  r̄={mean_r:.4f}  loss={loss:.5f}{skip_tag}  "
              f"step time: {dt:.1f}s")

        # ── Eval ──────────────────────────────────────────────────────────────
        if step % args.eval_every == 0:
            print(f"\n  evaluating mean reward on {len(eval_examples)} examples …")
            score = eval_mean_reward(
                torch_gen, keras_model, eval_examples,
                api_key=args.openai_api_key, judge_model=args.judge_model,
                max_tokens=args.max_tokens, temperature=args.temperature,
                gen_batch_size=args.eval_batch,
            )
            d   = score - prev_score
            tag = f"  Δ={d:+.4f} {'↑' if d > 0 else ('↓' if d < 0 else '=')}"
            separator("═")
            print(f"  EVAL  step={step}/{args.steps}  mean_reward={score:.4f}{tag}")
            separator("═")
            print()
            prev_score = score

            if args.save_path:
                base, ext = os.path.splitext(args.save_path)
                save_checkpoint(torch_gen, config, f"{base}_step{step:04d}{ext}", step)


if __name__ == "__main__":
    main()
