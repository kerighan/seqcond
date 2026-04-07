"""
RLAI fine-tuning for SeqCond with an LLM judge.

Quick start:
    KERAS_BACKEND=torch python train_grpo_rlai.py --checkpoint checkpoints/seqcond_lin5.pt

Colab:
    !pip install keras datasets openai pydantic  # once
    %env KERAS_BACKEND=torch
    !python train_grpo_rlai.py --checkpoint ... --openai_api_key sk-...

Two functions to modify:
    score_output()  — what counts as a good response  (judge design)
    grpo_step()     — how rewards drive the update     (algorithm)
"""

import argparse
import asyncio
import json
import os
import random
import re
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import ops
from pydantic import BaseModel, Field

from convert_torch_to_keras import (
    build_keras_model,
    convert_weights,
    get_config_value,
    keras_pkl_to_torch_pt,
    load_torch_checkpoint,
    save_keras_checkpoint,
)
from seqcond.dataset import Tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Fast PyTorch generation model (Triton-accelerated)
# ─────────────────────────────────────────────────────────────────────────────


def load_torch_gen_model(config, checkpoint_path):
    """Load a pure PyTorch SeqCondModel for fast Triton-accelerated generation."""
    import torch
    from seqcond.torch.model import SeqCondModel

    torch_model = SeqCondModel(**config).cpu().eval()
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    torch_model.load_state_dict(data["state_dict"], strict=False)
    n = sum(p.numel() for p in torch_model.parameters())
    print(f"Loaded PyTorch gen model ({n:,} params, Triton-accelerated)")
    return torch_model


def sync_keras_to_torch(keras_model, torch_model, config):
    """Copy weights from Keras model to PyTorch model (in-place).

    Since KERAS_BACKEND=torch, Keras weights are torch tensors.
    We reverse the mapping from convert_torch_to_keras.convert_weights.
    """
    import torch

    # Build Keras weight lookup: short path -> tensor
    keras_w = {}
    for w in keras_model.weights:
        parts = w.path.split("/")
        short = "/".join(parts[1:]) if len(parts) > 1 else w.path
        keras_w[short] = w

    # Build block map
    num_layers = get_config_value(config, "num_layers")
    seqcond_ratio = get_config_value(config, "seqcond_ratio", 3)
    transformer_idx = seqcond_idx = 0
    block_map = []
    for i in range(num_layers):
        if (i + 1) % (seqcond_ratio + 1) == 0:
            block_map.append((i, "transformer", f"transformer_block_{transformer_idx}"))
            transformer_idx += 1
        else:
            block_map.append((i, "seqcond", f"seqcond_block_{seqcond_idx}"))
            seqcond_idx += 1

    state_dict = {}

    def get(short):
        if short not in keras_w:
            return None
        return keras_w[short].value  # returns underlying torch tensor

    # Embedding
    state_dict["embedding.weight"] = get("token_embedding/embeddings")

    # Final norm
    v = get("final_norm/scale")
    if v is not None:
        state_dict["final_norm.scale"] = v

    # Blocks
    for torch_i, btype, kname in block_map:
        tp = f"blocks.{torch_i}."
        if btype == "transformer":
            state_dict[tp + "norm1.scale"] = get(f"{kname}/norm1/scale")
            state_dict[tp + "norm2.scale"] = get(f"{kname}/norm2/scale")
            state_dict[tp + "attn.q_proj.weight"] = get(f"{kname}/attn/q_proj/kernel").T
            state_dict[tp + "attn.k_proj.weight"] = get(f"{kname}/attn/k_proj/kernel").T
            state_dict[tp + "attn.v_proj.weight"] = get(f"{kname}/attn/v_proj/kernel").T
            state_dict[tp + "attn.out_proj.weight"] = get(
                f"{kname}/attn/out_proj/kernel"
            ).T
            state_dict[tp + "ff_in.weight"] = get(f"{kname}/ff_in/kernel").T
            state_dict[tp + "ff_in.bias"] = get(f"{kname}/ff_in/bias")
            state_dict[tp + "ff_out.weight"] = get(f"{kname}/ff_out/kernel").T
            state_dict[tp + "ff_out.bias"] = get(f"{kname}/ff_out/bias")
        else:  # seqcond
            state_dict[tp + "norm.scale"] = get(f"{kname}/pre_norm/scale")
            state_dict[tp + "attn.in_proj.weight"] = get(
                f"{kname}/attn/in_proj/kernel"
            ).T
            conv = get(f"{kname}/attn/conv/kernel")
            state_dict[tp + "attn.conv_weight"] = conv.permute(2, 1, 0).contiguous()
            state_dict[tp + "attn.gate_proj.weight"] = get(
                f"{kname}/attn/gate_proj/kernel"
            ).T
            state_dict[tp + "attn.out_proj.weight"] = get(
                f"{kname}/attn/out_proj/kernel"
            ).T
            for raw_key in [
                "theta_d_raw",
                "theta_raw",
                "w_int_raw",
                "decay_slopes",
                "anchor_slopes",
                "score_scale",
                "score_bias",
                "phase_scale",
            ]:
                val = get(f"{kname}/attn/{raw_key}")
                if val is not None:
                    state_dict[tp + f"attn.{raw_key}"] = val
            state_dict[tp + "attn.gated_norm.weight"] = get(
                f"{kname}/attn/gated_norm_weight"
            )
            state_dict[tp + "attn.W_readout"] = get(f"{kname}/attn/W_readout")

    # Filter None and load
    state_dict = {k: v for k, v in state_dict.items() if v is not None}
    torch_model.load_state_dict(state_dict, strict=False)


def generate_completions_torch(
    torch_model,
    tokenizer,
    prompt: str,
    num_completions: int = 4,
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    rep_penalty: float = 1.0,
    gen_batch_size: int = 4,
    return_tokens: bool = False,
    keep_on_cuda: bool = False,
):
    """Fast generation using PyTorch model with Triton kernels.

    Moves model to CUDA for generation, then back to CPU to free VRAM.
    If keep_on_cuda=True, assumes model is already on CUDA and does not move it.
    """
    import torch

    eos_id = tokenizer.encode("<|im_end|>")[0]
    prompt_toks = tokenizer([prompt])[0]
    all_texts, all_ids = [], []

    if not keep_on_cuda:
        torch_model.cuda()
    input_ids = torch.tensor([prompt_toks], device="cuda")

    with torch.no_grad():
        for start in range(0, num_completions, gen_batch_size):
            B = min(gen_batch_size, num_completions - start)

            logits, states = torch_model.prefill(input_ids)
            logits = logits.squeeze(1)

            if B > 1:
                logits = logits.repeat(B, 1)
                states = [
                    tuple(s.repeat(B, *([1] * (s.ndim - 1))) for s in state)
                    for state in states
                ]

            logits_np = logits.cpu().float().numpy()
            generated = [[] for _ in range(B)]
            finished = [False] * B
            token_buf = torch.zeros((B, 1), dtype=torch.long, device="cuda")
            current_seq_len = len(prompt_toks)
            # active_map[ci] = original sample index for compact index ci
            active_map = list(range(B))

            for _ in range(max_new_tokens):
                B_cur = len(active_map)
                if rep_penalty != 1.0:
                    for ci in range(B_cur):
                        oi = active_map[ci]
                        for tid in set(generated[oi]):
                            logits_np[ci, tid] = (
                                logits_np[ci, tid] / rep_penalty
                                if logits_np[ci, tid] > 0
                                else logits_np[ci, tid] * rep_penalty
                            )
                toks = _sample_batch(logits_np, temperature, top_k, top_p)
                newly_done = set()
                for ci in range(B_cur):
                    oi = active_map[ci]
                    tok = int(toks[ci])
                    generated[oi].append(tok)
                    if tok == eos_id:
                        finished[oi] = True
                        newly_done.add(ci)
                    else:
                        token_buf[ci, 0] = tok
                if all(finished):
                    break
                # Compact batch: remove finished samples
                if newly_done:
                    keep = [ci for ci in range(B_cur) if ci not in newly_done]
                    keep_idx = torch.tensor(keep, device="cuda")
                    token_buf = token_buf[keep_idx].contiguous()
                    states = [
                        tuple(s[keep_idx].contiguous() for s in state)
                        for state in states
                    ]
                    active_map = [active_map[ci] for ci in keep]
                current_seq_len += 1
                logits, states = torch_model.step(
                    token_buf, states, seq_len=current_seq_len, use_triton=True
                )
                logits_np = logits.cpu().float().numpy()

            for i in range(B):
                ids = generated[i]
                if ids and ids[-1] == eos_id:
                    ids = ids[:-1]
                all_ids.append(list(ids))
                try:
                    all_texts.append(tokenizer.decode(ids))
                except Exception:
                    all_texts.append("")

    # Move gen model back to CPU and free VRAM for training
    if not keep_on_cuda:
        torch_model.cpu()
        torch.cuda.empty_cache()

    return (all_texts, all_ids) if return_tokens else all_texts


# ─────────────────────────────────────────────────────────────────────────────
# Backend helpers
# ─────────────────────────────────────────────────────────────────────────────


def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.array(x)


@contextmanager
def inference_mode():
    if keras.backend.backend() == "torch":
        import torch

        with torch.no_grad():
            yield
    else:
        yield


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _extract_first(row: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        if key in row:
            text = _as_text(row.get(key))
            if text:
                return text
    return ""


def _messages_to_prompt(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    prompt_messages = messages
    reference_answer = ""
    if messages and _as_text(messages[-1].get("role")).lower() == "assistant":
        prompt_messages = messages[:-1]
        reference_answer = _as_text(messages[-1].get("content"))

    parts = []
    for msg in prompt_messages:
        role = _as_text(msg.get("role") or "user").lower()
        content = _as_text(msg.get("content"))
        if not content:
            continue
        parts.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")

    if not parts:
        return {"instruction": "", "reference_answer": reference_answer, "prompt": ""}

    prompt = "".join(parts) + "<|im_start|>assistant\n<|think_start|>"
    instruction = "\n\n".join(
        _as_text(msg.get("content"))
        for msg in prompt_messages
        if _as_text(msg.get("role") or "user").lower() == "user"
        and _as_text(msg.get("content"))
    )
    return {
        "instruction": instruction,
        "reference_answer": reference_answer,
        "prompt": prompt,
    }


def _row_to_rlai_example(
    row: Dict[str, Any],
    *,
    prompt_field: Optional[str] = None,
    response_field: Optional[str] = None,
    messages_field: Optional[str] = None,
    source_field: Optional[str] = None,
    dataset_source: str = "dataset",
):
    msg_field = messages_field
    if not msg_field:
        for candidate in ["messages", "conversation", "conversations", "dialogue"]:
            if candidate in row and isinstance(row[candidate], list):
                msg_field = candidate
                break

    if msg_field and isinstance(row.get(msg_field), list):
        msg_info = _messages_to_prompt(row[msg_field])
        if not msg_info["prompt"]:
            return None
        return {
            "instruction": msg_info["instruction"],
            "reference_answer": msg_info["reference_answer"],
            "prompt": msg_info["prompt"],
            "source": _extract_first(
                row,
                [source_field] if source_field else ["source", "dataset", "origin"],
            )
            or dataset_source,
        }

    prompt_text = _extract_first(
        row,
        [prompt_field] if prompt_field else ["prompt", "question", "query", "task"],
    )
    instruction = _extract_first(row, ["instruction"])
    extra_input = _extract_first(row, ["input", "context"])
    if instruction:
        prompt_text = (
            instruction
            if not extra_input
            else f"{instruction}\n\nContext:\n{extra_input}"
        )

    if not prompt_text:
        prompt_text = _extract_first(row, ["text"])

    if not prompt_text:
        return None

    reference_answer = _extract_first(
        row,
        (
            [response_field]
            if response_field
            else ["response", "output", "answer", "chosen", "completion"]
        ),
    )
    return {
        "instruction": prompt_text,
        "reference_answer": reference_answer,
        "prompt": f"<|im_start|>user\n{prompt_text}\n<|im_end|><|im_start|>assistant\n<|think_start|>",
        "source": _extract_first(
            row,
            [source_field] if source_field else ["source", "dataset", "origin"],
        )
        or dataset_source,
    }


def load_rlai_examples(
    *,
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
    dataset_split: str = "train",
    dataset_path: Optional[str] = None,
    prompt_field: Optional[str] = None,
    response_field: Optional[str] = None,
    messages_field: Optional[str] = None,
    source_field: Optional[str] = None,
    seed: int = 444,
    max_examples: Optional[int] = None,
):
    if dataset_path:
        dataset_source = os.path.basename(dataset_path)
        if dataset_path.endswith(".jsonl"):
            with open(dataset_path) as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            with open(dataset_path) as f:
                loaded = json.load(f)
            rows = loaded if isinstance(loaded, list) else loaded.get("data", [])
    else:
        from datasets import load_dataset

        dataset_name = dataset_name or "tatsu-lab/alpaca"
        rows = load_dataset(dataset_name, dataset_config, split=dataset_split)
        dataset_source = dataset_name

    examples = []
    for row in rows:
        ex = _row_to_rlai_example(
            dict(row),
            prompt_field=prompt_field,
            response_field=response_field,
            messages_field=messages_field,
            source_field=source_field,
            dataset_source=dataset_source,
        )
        if ex is not None:
            examples.append(ex)

    random.Random(seed).shuffle(examples)
    return examples[:max_examples] if max_examples else examples


def _filter_examples_by_prompt_length(examples, tokenizer, max_prompt_tokens=2048):
    kept = []
    skipped = 0
    for ex in examples:
        prompt = ex.get("prompt") or ""
        try:
            prompt_len = len(tokenizer([prompt])[0])
        except Exception:
            skipped += 1
            continue
        if prompt_len > max_prompt_tokens:
            skipped += 1
            continue
        kept.append(ex)
    if skipped:
        print(
            f"Filtered {skipped} overlong/invalid prompts; kept {len(kept)} examples with prompt_len <= {max_prompt_tokens}"
        )
    return kept


class RLAIJudgment(BaseModel):
    reasoning_quality: int = Field(ge=0, le=5)
    final_answer_quality: int = Field(ge=0, le=5)
    instruction_following: int = Field(ge=0, le=5)
    concision: int = Field(ge=0, le=5)
    overall_score: int = Field(ge=0, le=100)


_LLM_SYSTEM = """\
You are a strict evaluator for assistant responses.
Judge the candidate response to the user instruction on four criteria:
- reasoning_quality: soundness, coherence, and usefulness of the reasoning chain
- final_answer_quality: how helpful, correct, and complete the final answer is
- instruction_following: how well the response follows the user's request and format
- concision: whether the response is appropriately concise without omitting key content

Use integer scores from 0 to 5 for each criterion, where 5 is excellent.
Then provide an overall_score from 0 to 100 reflecting holistic response quality.

Scoring rules:
- Do not use exact-answer verification. If a reference answer is provided, use it only as soft style/context guidance, not as hard ground truth.
- For mathematical or quantitative questions: if the candidate's final answer is exactly correct, assign 5/5 for final_answer_quality. If the answer is close but not exact (e.g. off by a small amount, rounding difference, or minor arithmetic slip), assign 1/5 to 3/5 proportionally to how close it is. Only assign 0/5 if the answer is completely wrong or nonsensical.
- For mathematical or quantitative questions, prioritize the correctness and usability of the final answer over stylistic preferences.
- Reward helpfulness, clear reasoning, directness, and appropriate brevity.
- Penalize incoherence, verbosity without substance, refusal when unnecessary, or obvious fabrication.
- Strongly penalise repetitive responses: if the response repeats the same phrase, sentence, or idea more than once without adding new content, reduce reasoning_quality and final_answer_quality by at least 2 points, and reduce overall_score significantly.

Return a single JSON object with exactly these keys:
reasoning_quality, final_answer_quality, instruction_following, concision, overall_score.
"""


def _empty_judgment(reason: str) -> RLAIJudgment:
    return RLAIJudgment(
        reasoning_quality=0,
        final_answer_quality=0,
        instruction_following=0,
        concision=0,
        overall_score=0,
    )


def _parse_judgment_json(raw: str) -> RLAIJudgment:
    payload = raw.strip()
    start = payload.find("{")
    end = payload.rfind("}")
    if start != -1 and end != -1 and end > start:
        payload = payload[start : end + 1]
    try:
        return RLAIJudgment.model_validate_json(payload)
    except AttributeError:
        return RLAIJudgment.parse_raw(payload)


def _judgment_reward(judgment: RLAIJudgment) -> float:
    criterion_weights = {
        "reasoning_quality": 0.25,
        "final_answer_quality": 0.55,
        "instruction_following": 0.15,
        "concision": 0.05,
    }
    criteria_score = (
        criterion_weights["reasoning_quality"] * judgment.reasoning_quality
        + criterion_weights["final_answer_quality"] * judgment.final_answer_quality
        + criterion_weights["instruction_following"] * judgment.instruction_following
        + criterion_weights["concision"] * judgment.concision
    ) / (5.0 * sum(criterion_weights.values()))
    overall_score = judgment.overall_score / 100.0
    return 0.5 * criteria_score + 0.5 * overall_score


def _judge_messages(instruction, response, reference_answer):
    reference_block = (
        f"\n\nOptional reference answer for style/context only:\n{reference_answer}"
        if reference_answer
        else ""
    )
    return [
        {"role": "system", "content": _LLM_SYSTEM},
        {
            "role": "user",
            "content": (
                f"User instruction:\n{instruction}\n\n"
                f"Candidate response:\n{response}"
                f"{reference_block}\n\n"
                "Return the JSON judgment now."
            ),
        },
    ]


async def _score_one(
    client,
    instruction,
    response,
    reference_answer,
    semaphore,
    judge_model,
):
    async with semaphore:
        messages = _judge_messages(instruction, response, reference_answer)
        try:
            msg = await client.beta.chat.completions.parse(
                model=judge_model,
                messages=messages,
                response_format=RLAIJudgment,
                max_tokens=300,
                temperature=0.0,
            )
            parsed = msg.choices[0].message.parsed
            if parsed is not None:
                return parsed
            raw = (msg.choices[0].message.content or "").strip()
            return _parse_judgment_json(raw)
        except Exception as first_error:
            try:
                msg = await client.chat.completions.create(
                    model=judge_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=300,
                    temperature=0.0,
                )
                raw = (msg.choices[0].message.content or "").strip()
                return _parse_judgment_json(raw)
            except Exception as second_error:
                return _empty_judgment(
                    f"Judge call failed: {type(first_error).__name__}: {str(first_error)[:120]} | fallback: {type(second_error).__name__}: {str(second_error)[:120]}"
                )


def _llm_judgments(
    instruction,
    completions,
    api_key,
    *,
    reference_answer="",
    judge_model="gpt-4.1-mini",
    max_concurrent=8,
):
    from openai import AsyncOpenAI

    async def _run():
        client = AsyncOpenAI(api_key=api_key)
        sem = asyncio.Semaphore(max_concurrent)
        scores = await asyncio.gather(
            *[
                _score_one(
                    client,
                    instruction,
                    r,
                    reference_answer,
                    sem,
                    judge_model,
                )
                for r in completions
            ]
        )
        await client.close()
        return scores

    return asyncio.run(_run())


def score_output(
    instruction: str,
    completions: List[str],
    api_key: str = None,
    *,
    reference_answer: str = "",
    judge_model: str = "gpt-4.1-mini",
    judge_max_concurrent: int = 8,
    return_components: bool = False,
):
    if not api_key:
        raise ValueError("RLAI training requires an OpenAI API key for LLM judging.")

    judgments = _llm_judgments(
        instruction,
        completions,
        api_key,
        reference_answer=reference_answer,
        judge_model=judge_model,
        max_concurrent=judge_max_concurrent,
    )
    rewards = [_judgment_reward(j) for j in judgments]
    if return_components:
        return {
            "rewards": rewards,
            "overall": [j.overall_score for j in judgments],
            "reasoning_quality": [j.reasoning_quality for j in judgments],
            "final_answer_quality": [j.final_answer_quality for j in judgments],
            "instruction_following": [j.instruction_following for j in judgments],
            "concision": [j.concision for j in judgments],
        }
    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────


def _tile_states(states, n):
    return [
        tuple(ops.tile(s, (n,) + (1,) * (s.ndim - 1)) for s in state)
        for state in states
    ]


def _sample_batch(logits_np, temperature=0.7, top_k=50, top_p=0.95):
    """Sample one token per row from (B, vocab) logits."""
    B = logits_np.shape[0]
    tokens = np.empty(B, dtype=np.int64)
    for i in range(B):
        row = logits_np[i].astype(np.float64)
        if temperature <= 0.0:
            tokens[i] = np.argmax(row)
            continue
        row = row / temperature
        topk_idx = (
            np.argpartition(row, -top_k)[-top_k:]
            if 0 < top_k < len(row)
            else np.arange(len(row))
        )
        vals = row[topk_idx]
        vals -= np.max(vals)
        probs = np.exp(vals)
        probs /= probs.sum()
        if top_p < 1.0:
            order = np.argsort(probs)[::-1]
            cut = np.searchsorted(np.cumsum(probs[order]), top_p) + 1
            nucleus = order[:cut]
            p = probs[nucleus]
            p /= p.sum()
            tokens[i] = topk_idx[np.random.choice(nucleus, p=p)]
        else:
            tokens[i] = topk_idx[np.random.choice(len(probs), p=probs)]
    return tokens


def generate_completions(
    model,
    tokenizer,
    prompt: str,
    num_completions: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    rep_penalty: float = 1.0,
    gen_batch_size: int = 4,
    return_tokens: bool = False,
):
    """Generate num_completions responses for prompt. Prefills once, tiles states."""
    eos_id = tokenizer.encode("<|im_end|>")[0]
    prompt_toks = tokenizer([prompt])[0]
    all_texts, all_ids = [], []

    with inference_mode():
        for start in range(0, num_completions, gen_batch_size):
            B = min(gen_batch_size, num_completions - start)
            states = model.init_state(batch_size=1)
            for t in prompt_toks:
                logits, states = model.step(np.array([[t]], dtype=np.int32), states)
            if B > 1:
                states = _tile_states(states, B)
            logits_np = np.tile(to_numpy(logits), (B, 1))

            generated = [[] for _ in range(B)]
            finished = [False] * B
            buf = np.zeros((B, 1), dtype=np.int32)

            for _ in range(max_new_tokens):
                if rep_penalty != 1.0:
                    for i in range(B):
                        for tid in set(generated[i]):
                            logits_np[i, tid] = (
                                logits_np[i, tid] / rep_penalty
                                if logits_np[i, tid] > 0
                                else logits_np[i, tid] * rep_penalty
                            )
                toks = _sample_batch(logits_np, temperature, top_k, top_p)
                for i in range(B):
                    if not finished[i]:
                        generated[i].append(int(toks[i]))
                        finished[i] = toks[i] == eos_id
                        buf[i, 0] = toks[i]
                    else:
                        buf[i, 0] = eos_id
                if all(finished):
                    break
                logits, states = model.step(buf, states)
                logits_np = to_numpy(logits)

            for i in range(B):
                ids = generated[i]
                if ids and ids[-1] == eos_id:
                    ids = ids[:-1]
                all_ids.append(list(ids))
                try:
                    all_texts.append(tokenizer.decode(ids))
                except Exception:
                    all_texts.append("")

    return (all_texts, all_ids) if return_tokens else all_texts


# ─────────────────────────────────────────────────────────────────────────────
# GRPO internals
# ─────────────────────────────────────────────────────────────────────────────


def _seq_log_prob(model, input_ids, prompt_len):
    """Sum of per-token log probs for the completion portion of input_ids."""
    return ops.sum(_seq_token_log_probs(model, input_ids, prompt_len))


def _seq_token_log_probs(model, input_ids, prompt_len):
    """Per-token log probs for the completion portion of input_ids."""
    logits = model(input_ids)
    log_probs = ops.log_softmax(logits, axis=-1)
    shift = log_probs[0, prompt_len - 1 : -1, :]
    targets = ops.expand_dims(ops.cast(input_ids[0, prompt_len:], "int32"), -1)
    return ops.squeeze(ops.take_along_axis(shift, targets, axis=-1), axis=-1)


def _compute_entropy(model, input_ids, prompt_len: int) -> float:
    """Mean entropy (nats) of the policy's next-token distribution over the completion."""
    logits = model(input_ids)
    log_probs = ops.log_softmax(logits, axis=-1)
    shift = log_probs[0, prompt_len - 1 : -1, :]
    probs = ops.exp(shift)
    entropy_per_token = -ops.sum(probs * shift, axis=-1)
    return float(to_numpy(ops.mean(entropy_per_token)))


def _overlong_penalty(comp_len: int, max_new_tokens: int, cache_tokens: int) -> float:
    """Soft penalty for completions approaching the generation length budget.

    Returns 0.0 if comp_len < max_new_tokens - cache_tokens,
    linearly ramps to -1.0 as comp_len approaches max_new_tokens,
    and returns -1.0 if comp_len >= max_new_tokens.
    """
    if cache_tokens <= 0:
        return 0.0
    cache_tokens = min(cache_tokens, max_new_tokens)
    threshold = max_new_tokens - cache_tokens
    if comp_len < threshold:
        return 0.0
    if comp_len >= max_new_tokens:
        return -1.0
    return -(comp_len - threshold) / cache_tokens


def _repetition_penalty(
    token_ids: List[int], n: int = 4, threshold: float = 0.5,
    think_end_id: Optional[int] = None,
) -> float:
    """Penalize completions with excessive n-gram repetition (degenerate loops).

    Checks the ENTIRE completion for repetition (loops often happen during
    thinking, before <|think_end|> is ever produced).
    Returns 0.0 if unique_ratio >= threshold (healthy text),
    linearly ramps to -1.0 as unique_ratio approaches 0.
    Short completions (< 2*n tokens) are never penalized.
    """
    if len(token_ids) < 2 * n:
        return 0.0
    ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    unique_ratio = len(set(ngrams)) / len(ngrams)
    if unique_ratio >= threshold:
        return 0.0
    return -(1.0 - unique_ratio / threshold)


def _compute_advantages(rewards, normalize_std: bool = True):
    """Group-relative advantages: (r - mean) / (std + eps), or just (r - mean) with normalize_std=False (Dr. GRPO)."""
    r = np.array(rewards, dtype=np.float32)
    centered = r - r.mean()
    if not normalize_std:
        return centered
    std = r.std()
    return np.zeros_like(r) if std < 1e-8 else centered / std


def _filter_group_by_total_length(
    prompt_tokens: List[int],
    texts: List[str],
    completions_tokens: List[List[int]],
    max_total_tokens: Optional[int],
):
    if not max_total_tokens:
        return texts, completions_tokens, 0

    kept_texts = []
    kept_tokens = []
    skipped = 0
    prompt_len = len(prompt_tokens)
    for text, comp in zip(texts, completions_tokens):
        if prompt_len + len(comp) > max_total_tokens:
            skipped += 1
            continue
        kept_texts.append(text)
        kept_tokens.append(comp)
    return kept_texts, kept_tokens, skipped


def _compute_gdpo_advantages(score_info, weights, normalize_std: bool = True):
    component_advantages = {}
    active_weights = {name: weight for name, weight in weights.items() if weight != 0.0}
    scale = float(sum(abs(weight) for weight in active_weights.values()))
    combined = None
    for name, weight in active_weights.items():
        adv = _compute_advantages(score_info[name], normalize_std=normalize_std)
        component_advantages[name] = adv
        scaled_weight = weight / scale if scale > 0.0 else 0.0
        combined = (
            adv * scaled_weight if combined is None else combined + adv * scaled_weight
        )
    if combined is None:
        rewards = score_info.get("rewards", [])
        combined = np.zeros(len(rewards), dtype=np.float32)
    return combined.astype(np.float32), component_advantages


# ═════════════════════════════════════════════════════════════════════════════
# ★  GRPO FORMULA  —  modify this to change how rewards drive the update
# ═════════════════════════════════════════════════════════════════════════════


def grpo_step(
    model,
    prompt_tokens: List[int],
    completions_tokens: List[List[int]],
    advantages: np.ndarray,
    *,
    beta: float = 0.1,
    ref_model=None,
    grad_scale: float = 1.0,
) -> float:
    """Compute GRPO loss and accumulate gradients for one group of completions.

    Uses token-level loss normalization (DAPO / Dr. GRPO): the loss is divided
    by the total number of tokens across all completions in the group, giving
    each token equal weight regardless of sequence length.

    For each completion i with non-zero advantage:
        loss_i = -advantage_i * sum_log_prob_i  +  β * KL(π || π_ref) * len(comp_i)
    Total loss is divided by total_tokens (sum of all completion lengths).
    """
    prompt_len = len(prompt_tokens)
    total_tokens = sum(len(c) for c in completions_tokens if len(c) > 0)
    if total_tokens == 0:
        return 0.0

    # Reference log probs (frozen initial model, or current snapshot if no ref)
    ref = ref_model if ref_model is not None else model
    ref_avg_lps = []
    with inference_mode():
        for comp in completions_tokens:
            if len(comp) == 0:
                ref_avg_lps.append(0.0)
                continue
            ids = np.array([prompt_tokens + comp], dtype=np.int32)
            ref_avg_lps.append(float(_seq_log_prob(ref, ids, prompt_len)) / len(comp))

    # Policy gradient with token-level normalization
    total_loss = 0.0
    for i, comp in enumerate(completions_tokens):
        if len(comp) == 0 or advantages[i] == 0:
            continue
        ids = np.array([prompt_tokens + comp], dtype=np.int32)
        policy_lp = _seq_log_prob(model, ids, prompt_len)  # sum over tokens
        avg_lp = policy_lp / len(comp)
        kl = avg_lp - ref_avg_lps[i]
        # Token-level normalization: weight each completion by its token count
        token_weight = len(comp) / total_tokens
        loss_i = (-advantages[i] * avg_lp + beta * kl) * token_weight
        (loss_i * grad_scale).backward()
        total_loss += float(to_numpy(loss_i))

    return total_loss


def _compute_old_log_probs(model, prompt_tokens, completions_tokens):
    """Compute detached per-token log probs for all completions (π_old)."""
    prompt_len = len(prompt_tokens)
    old_lps = []
    with inference_mode():
        for comp in completions_tokens:
            if len(comp) == 0:
                old_lps.append(None)
                continue
            ids = np.array([prompt_tokens + comp], dtype=np.int32)
            token_lps = _seq_token_log_probs(model, ids, prompt_len)
            old_lps.append(to_numpy(token_lps).copy())
    return old_lps


def grpo_step_ppo(
    model,
    prompt_tokens: List[int],
    completions_tokens: List[List[int]],
    advantages: np.ndarray,
    old_token_log_probs: List,
    *,
    clip_eps_low: float = 0.2,
    clip_eps_high: float = 0.28,
    beta: float = 0.0,
    ref_model=None,
    grad_scale: float = 1.0,
) -> float:
    """PPO-clip GRPO: clipped surrogate objective with importance sampling.

    Uses stored π_old to compute the ratio π/π_old per token, with asymmetric
    clipping [1-ε_low, 1+ε_high] (DAPO-style).  Supports multiple epochs
    over the same batch of completions.
    """
    prompt_len = len(prompt_tokens)
    total_tokens = sum(len(c) for c in completions_tokens if len(c) > 0)
    if total_tokens == 0:
        return 0.0

    # Reference log probs for KL (optional, frozen model)
    ref_avg_lps = []
    if beta > 0:
        ref = ref_model if ref_model is not None else model
        with inference_mode():
            for comp in completions_tokens:
                if len(comp) == 0:
                    ref_avg_lps.append(0.0)
                    continue
                ids = np.array([prompt_tokens + comp], dtype=np.int32)
                ref_avg_lps.append(float(_seq_log_prob(ref, ids, prompt_len)) / len(comp))

    total_loss = 0.0
    for i, comp in enumerate(completions_tokens):
        if len(comp) == 0 or advantages[i] == 0 or old_token_log_probs[i] is None:
            continue
        ids = np.array([prompt_tokens + comp], dtype=np.int32)
        # Current per-token log probs (with gradient)
        cur_token_lps = _seq_token_log_probs(model, ids, prompt_len)
        # Old per-token log probs (detached numpy)
        old_lps_t = ops.convert_to_tensor(old_token_log_probs[i])

        # Importance sampling ratio per token
        log_ratio = cur_token_lps - old_lps_t
        ratio = ops.exp(log_ratio)

        # Clipped ratio (asymmetric: DAPO)
        clipped_ratio = ops.clip(ratio, 1.0 - clip_eps_low, 1.0 + clip_eps_high)

        # PPO surrogate: min(ratio*A, clip(ratio)*A)
        adv_i = float(advantages[i])
        surr1 = ratio * adv_i
        surr2 = clipped_ratio * adv_i
        # min handles both signs of advantage correctly
        ppo_obj = ops.minimum(surr1, surr2)

        # Token-level mean, then weight by completion length
        token_weight = len(comp) / total_tokens
        loss_i = -ops.mean(ppo_obj) * token_weight

        # KL penalty (optional)
        if beta > 0:
            avg_lp = ops.sum(cur_token_lps) / len(comp)
            kl = avg_lp - ref_avg_lps[i]
            loss_i = loss_i + beta * kl * token_weight

        (loss_i * grad_scale).backward()
        total_loss += float(to_numpy(loss_i))

    return total_loss


def gmpo_step(
    model,
    prompt_tokens: List[int],
    completions_tokens: List[List[int]],
    advantages: np.ndarray,
    *,
    clip_eps: float = 0.4,
    beta: float = 0.05,
    ref_model=None,
    grad_scale: float = 1.0,
) -> float:
    """Compute GMPO loss (true geometric-mean policy optimization).

    Each token's log importance-ratio is clipped independently to [-clip_eps, +clip_eps]
    BEFORE taking the mean (= geometric mean in linear space). This prevents outlier
    tokens from dominating the sequence ratio, which was the bug in the previous
    sequence-level clipping approach. Token-level normalization is used (same as
    grpo_step) so every token contributes equally regardless of sequence length.
    """
    prompt_len = len(prompt_tokens)
    total_tokens = sum(len(c) for c in completions_tokens if len(c) > 0)
    if total_tokens == 0:
        return 0.0

    ref = ref_model if ref_model is not None else model
    old_token_lps = []
    ref_avg_lps = []
    with inference_mode():
        for comp in completions_tokens:
            if len(comp) == 0:
                old_token_lps.append(None)
                ref_avg_lps.append(0.0)
                continue
            ids = np.array([prompt_tokens + comp], dtype=np.int32)
            old_token_lps.append(_seq_token_log_probs(model, ids, prompt_len))
            ref_avg_lps.append(float(_seq_log_prob(ref, ids, prompt_len)) / len(comp))

    total_loss = 0.0
    for i, comp in enumerate(completions_tokens):
        if len(comp) == 0 or advantages[i] == 0:
            continue

        ids = np.array([prompt_tokens + comp], dtype=np.int32)
        new_token_lps = _seq_token_log_probs(model, ids, prompt_len)
        old_lps = old_token_lps[i]

        adv = float(advantages[i])
        sign_a = 1.0 if adv > 0 else -1.0
        # GMPO (paper): per-token r_t^sgn(A), pessimistic clip (cap above only)
        # In log-space: s·ℓ_t where s=sgn(A), ℓ_t = log(π_new/π_old)
        # min(r^s, clip(r^s, e^-ε, e^+ε)) in log = min(s·ℓ_t, +ε)  (no floor)
        token_log_ratios = new_token_lps - old_lps          # ℓ_t
        signed_log_ratios = sign_a * token_log_ratios        # s·ℓ_t
        pessimistic = ops.minimum(signed_log_ratios, clip_eps)  # min(s·ℓ_t, ε)
        mean_pessimistic = ops.mean(pessimistic)
        seq_ratio = ops.exp(mean_pessimistic)                # geometric mean
        avg_policy_lp = ops.mean(new_token_lps)
        kl = avg_policy_lp - ref_avg_lps[i]

        # Token-level normalization: weight by this completion's share of total tokens
        token_weight = len(comp) / total_tokens
        loss_i = (-adv * seq_ratio + beta * kl) * token_weight
        (loss_i * grad_scale).backward()
        total_loss += float(to_numpy(loss_i))

    return total_loss


def gmpo_step_ppo(
    model,
    prompt_tokens: List[int],
    completions_tokens: List[List[int]],
    advantages: np.ndarray,
    old_token_log_probs: List,
    *,
    clip_eps: float = 0.4,
    beta: float = 0.0,
    ref_model=None,
    grad_scale: float = 1.0,
) -> float:
    """Multi-epoch GMPO: geometric-mean policy optimization with stored π_old.

    Like gmpo_step but uses pre-computed old_token_log_probs instead of
    computing π_old inline.  Each token's log importance-ratio is clipped
    symmetrically to [-clip_eps, +clip_eps] BEFORE taking the mean
    (= geometric mean in linear space).
    """
    prompt_len = len(prompt_tokens)
    total_tokens = sum(len(c) for c in completions_tokens if len(c) > 0)
    if total_tokens == 0:
        return 0.0

    ref_avg_lps = []
    if beta > 0:
        ref = ref_model if ref_model is not None else model
        with inference_mode():
            for comp in completions_tokens:
                if len(comp) == 0:
                    ref_avg_lps.append(0.0)
                    continue
                ids = np.array([prompt_tokens + comp], dtype=np.int32)
                ref_avg_lps.append(float(_seq_log_prob(ref, ids, prompt_len)) / len(comp))

    total_loss = 0.0
    for i, comp in enumerate(completions_tokens):
        if len(comp) == 0 or advantages[i] == 0 or old_token_log_probs[i] is None:
            continue
        ids = np.array([prompt_tokens + comp], dtype=np.int32)
        new_token_lps = _seq_token_log_probs(model, ids, prompt_len)
        old_lps_t = ops.convert_to_tensor(old_token_log_probs[i])

        adv = float(advantages[i])
        sign_a = 1.0 if adv > 0 else -1.0
        # GMPO (paper): s·ℓ_t, pessimistic clip = min(s·ℓ_t, +ε)
        token_log_ratios = new_token_lps - old_lps_t
        signed_log_ratios = sign_a * token_log_ratios
        pessimistic = ops.minimum(signed_log_ratios, clip_eps)
        mean_pessimistic = ops.mean(pessimistic)
        seq_ratio = ops.exp(mean_pessimistic)

        token_weight = len(comp) / total_tokens
        loss_i = -adv * seq_ratio * token_weight

        if beta > 0:
            avg_policy_lp = ops.mean(new_token_lps)
            kl = avg_policy_lp - ref_avg_lps[i]
            loss_i = loss_i + beta * kl * token_weight

        (loss_i * grad_scale).backward()
        total_loss += float(to_numpy(loss_i))

    return total_loss


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────


def load_model(checkpoint_path: str):
    """Load Keras SeqCond model + PyTorch gen model from a .pt checkpoint."""
    config, state_dict = load_torch_checkpoint(checkpoint_path)
    model = build_keras_model(config)
    convert_weights(config, state_dict, model)
    n_params = model.count_params()
    print(
        f"Loaded {checkpoint_path}  ({n_params:,} params, backend={keras.backend.backend()})"
    )
    torch_gen = load_torch_gen_model(config, checkpoint_path)
    return model, config, torch_gen


# Evaluation
# ─────────────────────────────────────────────────────────────────────────────


def _generate_eval_batch(
    torch_model,
    tokenizer,
    prompts,
    max_new_tokens=512,
    temperature=0.0,
    rep_penalty=1.0,
):
    """Generate one completion per prompt, batched across different prompts.

    Assumes model is already on CUDA. Returns list of completion strings.
    """
    import torch

    B = len(prompts)
    if B == 0:
        return []
    eos_id = tokenizer.encode("<|im_end|>")[0]

    # Tokenize all prompts
    all_toks = [tokenizer([p])[0] for p in prompts]
    prompt_lens = [len(t) for t in all_toks]

    # Prefill each prompt individually (different lengths, no padding noise)
    all_logits, all_states = [], []
    with torch.no_grad():
        for toks in all_toks:
            input_ids = torch.tensor([toks], device="cuda")
            logits_i, states_i = torch_model.prefill(input_ids)
            all_logits.append(logits_i.squeeze(1))
            all_states.append(states_i)

    # Stack into batched tensors
    logits = torch.cat(all_logits, dim=0)
    num_blocks = len(all_states[0])
    states = []
    for block_idx in range(num_blocks):
        block_state = tuple(
            torch.cat([s[block_idx][t] for s in all_states], dim=0)
            for t in range(len(all_states[0][block_idx]))
        )
        states.append(block_state)

    # Decode with batch compaction
    generated = [[] for _ in range(B)]
    finished = [False] * B
    active_map = list(range(B))
    token_buf = torch.zeros((B, 1), dtype=torch.long, device="cuda")
    current_seq_len = max(prompt_lens)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            B_cur = len(active_map)
            logits_np = logits.cpu().float().numpy()

            if rep_penalty != 1.0:
                for ci in range(B_cur):
                    oi = active_map[ci]
                    for tid in set(generated[oi]):
                        logits_np[ci, tid] = (
                            logits_np[ci, tid] / rep_penalty
                            if logits_np[ci, tid] > 0
                            else logits_np[ci, tid] * rep_penalty
                        )

            toks = _sample_batch(logits_np, temperature, 50, 0.95)
            newly_done = set()
            for ci in range(B_cur):
                oi = active_map[ci]
                tok = int(toks[ci])
                generated[oi].append(tok)
                if tok == eos_id:
                    finished[oi] = True
                    newly_done.add(ci)
                else:
                    token_buf[ci, 0] = tok
            if all(finished):
                break
            if newly_done:
                keep = [ci for ci in range(B_cur) if ci not in newly_done]
                if not keep:
                    break
                keep_idx = torch.tensor(keep, device="cuda")
                token_buf = token_buf[keep_idx].contiguous()
                states = [
                    tuple(s[keep_idx].contiguous() for s in state)
                    for state in states
                ]
                active_map = [active_map[ci] for ci in keep]
            current_seq_len += 1
            logits, states = torch_model.step(
                token_buf, states, seq_len=current_seq_len, use_triton=True
            )

    results = []
    for i in range(B):
        ids = generated[i]
        if ids and ids[-1] == eos_id:
            ids = ids[:-1]
        try:
            results.append(tokenizer.decode(ids))
        except Exception:
            results.append("")
    return results


def evaluate(
    torch_gen,
    examples,
    api_key,
    judge_model="gpt-4.1-mini",
    judge_max_concurrent=8,
    max_examples=100,
    num_completions=1,
    max_new_tokens=512,
    temperature=0.0,
    rep_penalty=1.0,
    gen_batch_size=4,
    step=None,
    log_path=None,
):
    tokenizer = Tokenizer()
    examples = _filter_examples_by_prompt_length(
        examples, tokenizer, max_prompt_tokens=2048
    )
    examples = examples[:max_examples]
    t0 = time.time()

    # ── Phase 1: Generate all completions (fast, batched, GPU) ──
    torch_gen.cuda()
    all_comps = []  # all_comps[i] = list of completions for examples[i]

    if num_completions == 1:
        # Fast path: batch different prompts together
        for batch_start in range(0, len(examples), gen_batch_size):
            prompts = [
                ex["prompt"]
                for ex in examples[batch_start : batch_start + gen_batch_size]
            ]
            batch_comps = _generate_eval_batch(
                torch_gen,
                tokenizer,
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                rep_penalty=rep_penalty,
            )
            all_comps.extend([[c] for c in batch_comps])
    else:
        for ex in examples:
            comps = generate_completions_torch(
                torch_gen,
                tokenizer,
                ex["prompt"],
                num_completions=num_completions,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                rep_penalty=rep_penalty,
                gen_batch_size=gen_batch_size,
                keep_on_cuda=True,
            )
            all_comps.append(comps)

    import torch
    torch_gen.cpu()
    torch.cuda.empty_cache()
    gen_time = time.time() - t0
    print(f"  Generated {len(examples)} eval completions in {gen_time:.1f}s")

    # ── Phase 2: Score all completions concurrently (LLM judge, API) ──
    from openai import AsyncOpenAI

    async def _score_all():
        client = AsyncOpenAI(api_key=api_key)
        sem = asyncio.Semaphore(judge_max_concurrent)
        tasks = []
        for ex, comps in zip(examples, all_comps):
            for comp in comps:
                tasks.append(
                    _score_one(
                        client,
                        ex["instruction"],
                        comp,
                        ex.get("reference_answer", ""),
                        sem,
                        judge_model,
                    )
                )
        judgments = await asyncio.gather(*tasks)
        await client.close()
        return judgments

    t_judge = time.time()
    flat_judgments = asyncio.run(_score_all())
    print(f"  Scored {len(flat_judgments)} completions in {time.time()-t_judge:.1f}s")

    # Unflatten: group judgments back per example
    total_reward = 0.0
    total_overall = 0.0
    idx = 0
    for i, (ex, comps) in enumerate(zip(examples, all_comps)):
        n = len(comps)
        judgments_i = flat_judgments[idx : idx + n]
        idx += n
        rewards = [_judgment_reward(j) for j in judgments_i]
        avg_reward = float(np.mean(rewards))
        avg_overall = float(np.mean([j.overall_score for j in judgments_i]))
        total_reward += avg_reward
        total_overall += avg_overall
        print(
            f"  [{i+1}/{len(examples)}] reward={avg_reward:.3f}  "
            f"overall={avg_overall:.1f}/100  "
            f"source={ex['source']}"
        )

    mean_reward = total_reward / max(len(examples), 1)
    mean_overall = total_overall / max(len(examples), 1)
    elapsed = time.time() - t0
    step_str = f"step={step}  " if step is not None else ""
    log_line = (
        f"{step_str}judge_reward@{num_completions}: {mean_reward:.3f}  "
        f"overall={mean_overall:.1f}/100  ({elapsed:.0f}s)"
    )
    print(f"\n  {log_line}\n")
    if log_path:
        with open(log_path, "a") as _lf:
            _lf.write(log_line + "\n")
    return mean_reward


def _save_and_upload_gcp(model, config, save_path, step, torch_gen=None):
    import subprocess, torch as _torch

    gcs_bucket = "gs://telekinesis-43/checkpoints"

    # Save directly from torch_gen state_dict to avoid lossy Keras→pkl→pt round-trip
    # (SeqCond recurrence is chaotic: even 1e-4 weight diffs change generation)
    # IMPORTANT: filter out internal cached buffers (_conv_kernel_t, _theta_cached, etc.)
    # These depend on seq_len of last forward call and corrupt generation if reloaded.
    if torch_gen is not None and save_path.endswith(".pt"):
        _CACHE_PREFIXES = ('_conv_kernel_t', '_decay_slopes_cached', '_phase_scale_b',
                           '_score_bias_b', '_score_scale_b', '_theta_cached', '_w_int_cached')
        sd = {k: v.cpu() for k, v in torch_gen.state_dict().items()
              if not any(k.endswith(sfx) for sfx in _CACHE_PREFIXES)}
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        _torch.save({"state_dict": sd, "config": config}, save_path)
        print(f"PyTorch checkpoint saved (direct): {save_path} ({len(sd)} tensors)")
    else:
        pkl_path = save_path[:-3] + ".pkl" if save_path.endswith(".pt") else save_path
        save_keras_checkpoint(model, config, pkl_path)
        if save_path.endswith(".pt"):
            keras_pkl_to_torch_pt(pkl_path, save_path)

    try:
        filename = os.path.basename(save_path)
        base, ext = os.path.splitext(filename)
        gcs_filename = f"{base}_step{step}{ext}"
        gcs_path = f"{gcs_bucket}/{gcs_filename}"

        print(f"  Uploading to {gcs_path}...", end=" ", flush=True)
        # Try gcloud storage cp first (faster, better auth on Colab),
        # fall back to gsutil cp
        for cmd in [
            ["gcloud", "storage", "cp", save_path, gcs_path],
            ["gsutil", "cp", save_path, gcs_path],
        ]:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                print(f"✓ ({cmd[0]})")
                break
            err = result.stderr.strip() or result.stdout.strip() or "unknown error"
            print(f"\n    {cmd[0]} failed: {err[:500]}")
        else:
            print(f"  ✗ upload failed for {gcs_path}")
    except Exception as e:
        print(f"✗ ({str(e)[:500]})")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────


def train_rlai(
    model,
    config,
    examples,
    *,
    eval_examples=None,
    torch_gen=None,
    use_gmpo: bool = False,
    use_gdpo: bool = False,
    num_completions: int = 6,
    max_new_tokens: int = 512,
    temperature: float = 0.5,
    rep_penalty: float = 1.0,
    gen_batch_size: int = 4,
    beta: float = 0.04,
    lr: float = 5e-5,
    optimizer_name: str = "adamw",
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    train_layers: int = 3,
    warmup_steps: int = 20,
    min_completion_tokens: int = 5,
    llm_api_key: str = None,
    judge_model: str = "gpt-4.1-mini",
    judge_max_concurrent: int = 8,
    gdpo_overall_weight: float = 0.50,
    gdpo_reasoning_weight: float = 0.15,
    gdpo_answer_weight: float = 0.15,
    gdpo_follow_weight: float = 0.10,
    gdpo_concision_weight: float = 0.05,
    num_steps: int = 250,
    log_every: int = 1,
    eval_every: int = 50,
    max_eval: int = 20,
    eval_num_completions: int = 1,
    eval_temperature: float = 0.0,
    save_gcp_every: int = 0,
    save_path: str = None,
    grad_accum_steps: int = 4,
    seed: int = 43,
    use_dr_grpo: bool = True,
    overlong_cache_tokens: int = 0,
    clip_adv: float = 0.0,
    reshuffle_layers_every: int = 1,
    lr_decay: str = "linear",
    lr_final_ratio: float = 0.0,
    ppo_epochs: int = 1,
    clip_eps_low: float = 0.2,
    clip_eps_high: float = 0.28,
    gmpo_clip_eps: float = 0.4,
):
    import torch

    tokenizer = Tokenizer()
    examples = _filter_examples_by_prompt_length(
        examples, tokenizer, max_prompt_tokens=2048
    )
    if not examples:
        raise ValueError(
            "No RLAI examples remain after filtering prompts longer than 2048 tokens"
        )
    np.random.seed(seed)
    random.seed(seed)
    max_total_tokens = int(config.get("maxlen") or 0)

    n_blocks = len(model.blocks_list)
    train_layers = min(train_layers, n_blocks)

    all_params = list(model.parameters())
    optimizer = (
        torch.optim.AdamW(all_params, lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
        if optimizer_name == "adamw"
        else torch.optim.SGD(all_params, lr=lr, momentum=0.0, weight_decay=weight_decay)
    )

    use_fast_gen = torch_gen is not None

    # Frozen reference model for KL penalty (anchors to initial policy)
    ref_model = build_keras_model(config)
    ref_model(np.zeros((1, 1), dtype=np.int32))  # ensure built
    for ref_w, src_w in zip(ref_model.weights, model.weights):
        ref_w.assign(src_w)
    for p in ref_model.parameters():
        p.requires_grad = False

    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")
    if lr_decay not in {"constant", "linear"}:
        raise ValueError(f"lr_decay must be 'constant' or 'linear', got {lr_decay}")
    if not 0.0 <= lr_final_ratio <= 1.0:
        raise ValueError(
            f"lr_final_ratio must be between 0 and 1, got {lr_final_ratio}"
        )

    effective_overlong_cache_tokens = (
        overlong_cache_tokens
        if overlong_cache_tokens > 0
        else max(64, max_new_tokens // 8)
    )

    advantage_mode = "GDPO" if use_gdpo else "GRPO"
    gdpo_weights = {
        "overall": gdpo_overall_weight,
        "reasoning_quality": gdpo_reasoning_weight,
        "final_answer_quality": gdpo_answer_weight,
        "instruction_following": gdpo_follow_weight,
        "concision": gdpo_concision_weight,
        "overlong": gdpo_concision_weight,
    }

    if use_gmpo and use_gdpo:
        objective_name = "RLAI-GMSDPO"
    elif use_gmpo:
        objective_name = "RLAI-GMPO"
    elif use_gdpo:
        objective_name = "RLAI-GDPO"
    else:
        objective_name = "RLAI-GRPO"

    print(
        f"\n── {objective_name}  {num_steps} steps  G={num_completions}  "
        f"lr={lr}  β={beta}  judge={judge_model}  train_layers={train_layers}/{n_blocks}  "
        f"warmup={warmup_steps}  decay={lr_decay}:{lr_final_ratio:.2f}  "
        f"accum={grad_accum_steps}  advantage={advantage_mode} ──\n"
    )
    if use_gdpo:
        print(
            "  GDPO weights: "
            f"overall={gdpo_overall_weight:.2f}  "
            f"reason={gdpo_reasoning_weight:.2f}  "
            f"answer={gdpo_answer_weight:.2f}  "
            f"follow={gdpo_follow_weight:.2f}  "
            f"concise={gdpo_concision_weight:.2f}  "
            f"overlong={gdpo_concision_weight:.2f}  "
            f"overlong_cache={effective_overlong_cache_tokens}\n"
        )

    t0 = time.time()
    run_reward = run_skipped = run_loss = 0.0
    run_count = run_adv_abs = 0.0
    run_entropy = 0.0
    run_gen_tokens = 0.0
    entropy_steps = 0
    pending_grad_steps = 0
    optimizer_updates = 0
    ppo_batch = []
    optimizer.zero_grad()

    def _ppo_replay_step(_pt, _ct, _adv, _olp, _grad_scale):
        """Dispatch a single PPO replay group to GMPO or GRPO objective."""
        if use_gmpo:
            return gmpo_step_ppo(
                model, _pt, _ct, _adv, _olp,
                clip_eps=gmpo_clip_eps,
                beta=beta, ref_model=ref_model,
                grad_scale=_grad_scale,
            )
        else:
            return grpo_step_ppo(
                model, _pt, _ct, _adv, _olp,
                clip_eps_low=clip_eps_low,
                clip_eps_high=clip_eps_high,
                beta=beta, ref_model=ref_model,
                grad_scale=_grad_scale,
            )

    run_skipped_mastered = 0.0
    run_overall = run_reasoning = run_answer = run_follow = 0.0
    run_concision = 0.0
    run_adv_overall = run_adv_reasoning = run_adv_answer = 0.0
    run_adv_follow = run_adv_concision = 0.0

    def _set_active_trainable_blocks(block_indices):
        for p in model.parameters():
            p.requires_grad = False
        for idx in block_indices:
            for p in model.blocks_list[idx].parameters():
                p.requires_grad = True
        # NOTE: embeddings are FROZEN — they are tied to lm_head (weight tying),
        # so training them corrupts both input and output simultaneously,
        # causing repetitive degeneration in greedy decoding.

    def _set_optimizer_lr(current_step):
        current_lr = lr
        if warmup_steps > 0 and current_step <= warmup_steps:
            current_lr = lr * (current_step / warmup_steps)
        elif lr_decay == "linear":
            decay_start = warmup_steps + 1 if warmup_steps > 0 else 1
            if num_steps > decay_start:
                progress = (current_step - decay_start) / (num_steps - decay_start)
                progress = min(max(progress, 0.0), 1.0)
                current_lr = lr * (1.0 - (1.0 - lr_final_ratio) * progress)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr
        return current_lr

    active_block_indices = sorted(random.sample(range(n_blocks), train_layers))
    _set_active_trainable_blocks(active_block_indices)

    if use_fast_gen:
        sync_keras_to_torch(model, torch_gen, config)

    for step in range(1, num_steps + 1):
        ex = random.choice(examples)
        prompt_tokens = tokenizer([ex["prompt"]])[0]

        # Generate (fast path with Triton, or fallback to Keras)
        if use_fast_gen:
            texts, ids = generate_completions_torch(
                torch_gen,
                tokenizer,
                ex["prompt"],
                num_completions=num_completions,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                rep_penalty=rep_penalty,
                gen_batch_size=gen_batch_size,
                return_tokens=True,
            )
        else:
            texts, ids = generate_completions(
                model,
                tokenizer,
                ex["prompt"],
                num_completions=num_completions,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                rep_penalty=rep_penalty,
                gen_batch_size=gen_batch_size,
                return_tokens=True,
            )

        # Filter out degenerate completions (too short to be meaningful)
        valid = [len(c) >= min_completion_tokens for c in ids]
        texts_f = [t for t, v in zip(texts, valid) if v]
        ids_f = [c for c, v in zip(ids, valid) if v]
        step_valid = bool(texts_f)
        if not step_valid:
            run_skipped += 1

        if step_valid:
            texts_f, ids_f, skipped_too_long = _filter_group_by_total_length(
                prompt_tokens,
                texts_f,
                ids_f,
                max_total_tokens,
            )
            if skipped_too_long:
                print(
                    f"  skipped {skipped_too_long} completions exceeding model maxlen={max_total_tokens}"
                )
                run_skipped += skipped_too_long
            if not texts_f:
                print("  skipped step: all sampled completions exceeded model maxlen")
                step_valid = False

        if step_valid:
            # Score
            score_info = score_output(
                ex["instruction"],
                texts_f,
                llm_api_key,
                reference_answer=ex.get("reference_answer", ""),
                judge_model=judge_model,
                judge_max_concurrent=judge_max_concurrent,
                return_components=True,
            )
            rewards = list(score_info["rewards"])
            overlong_penalties = []
            rep_penalties = []
            for _i in range(len(ids_f)):
                _pen = _overlong_penalty(
                    len(ids_f[_i]), max_new_tokens, effective_overlong_cache_tokens
                )
                overlong_penalties.append(_pen)
                if _pen != 0.0:
                    rewards[_i] = rewards[_i] + _pen
                _rep = _repetition_penalty(ids_f[_i])
                rep_penalties.append(_rep)
                if _rep != 0.0:
                    rewards[_i] = rewards[_i] + _rep
            # NOTE: rewards are NOT clipped to [0, ∞) — negative rewards from
            # repetition/overlong penalties provide contrastive signal even at
            # solve_rate=0%, pushing the model away from degenerate loops.
            score_info["rewards"] = rewards
            score_info["overlong"] = overlong_penalties
            score_info["repetition"] = rep_penalties
            overall = score_info["overall"]
            reasoning = score_info["reasoning_quality"]
            final_answer = score_info["final_answer_quality"]
            instruction_following = score_info["instruction_following"]
            concision = score_info["concision"]
            def _mmm(vals):
                """min/mean/max summary."""
                a = np.array(vals, dtype=np.float32)
                return f"{a.min():.1f}/{a.mean():.1f}/{a.max():.1f}"
            print(
                f"  reward={_mmm(rewards)}  overall={_mmm(overall)}  "
                f"reason={_mmm(reasoning)}  answer={_mmm(final_answer)}  "
                f"follow={_mmm(instruction_following)}  concise={_mmm(concision)}"
            )

            if use_gdpo:
                advantages, component_advantages = _compute_gdpo_advantages(
                    score_info, gdpo_weights, normalize_std=not use_dr_grpo
                )
                if clip_adv > 0:
                    advantages = np.clip(advantages, -clip_adv, clip_adv)
                overall_adv = component_advantages.get(
                    "overall", np.zeros_like(advantages)
                )
                reasoning_adv = component_advantages.get(
                    "reasoning_quality", np.zeros_like(advantages)
                )
                answer_adv = component_advantages.get(
                    "final_answer_quality", np.zeros_like(advantages)
                )
                follow_adv = component_advantages.get(
                    "instruction_following", np.zeros_like(advantages)
                )
                concision_adv = component_advantages.get(
                    "concision", np.zeros_like(advantages)
                )
                print(
                    f"  gdpo_adv={[f'{a:.2f}' for a in advantages]}  "
                    f"overall_adv={[f'{a:.2f}' for a in overall_adv]}  "
                    f"reason_adv={[f'{a:.2f}' for a in reasoning_adv]}  "
                    f"answer_adv={[f'{a:.2f}' for a in answer_adv]}  "
                    f"follow_adv={[f'{a:.2f}' for a in follow_adv]}"
                )
            else:
                advantages = _compute_advantages(rewards, normalize_std=not use_dr_grpo)
                if clip_adv > 0:
                    advantages = np.clip(advantages, -clip_adv, clip_adv)
                component_advantages = None
            run_reward += sum(rewards)
            run_count += len(rewards)
            run_gen_tokens += float(sum(len(c) for c in ids_f))
            run_adv_abs += float(np.mean(np.abs(advantages)))
            run_overall += sum(overall)
            run_reasoning += sum(reasoning)
            run_answer += sum(final_answer)
            run_follow += sum(instruction_following)
            run_concision += sum(concision)
            if component_advantages is not None:
                run_adv_overall += float(np.mean(np.abs(overall_adv)))
                run_adv_reasoning += float(np.mean(np.abs(reasoning_adv)))
                run_adv_answer += float(np.mean(np.abs(answer_adv)))
                run_adv_follow += float(np.mean(np.abs(follow_adv)))
                run_adv_concision += float(np.mean(np.abs(concision_adv)))

            mean_abs_adv = float(np.mean(np.abs(advantages)))
            avg_overall_group = float(np.mean(overall)) if overall else 0.0
            min_overall_group = float(min(overall)) if overall else 0.0
            skip_mastered_group = (
                avg_overall_group >= 90.0
                and min_overall_group >= 85.0
                and mean_abs_adv <= 0.25
            )
            if skip_mastered_group:
                run_skipped_mastered += 1
                print(
                    "  [skip mastered group | "
                    f"overall_avg={avg_overall_group:.1f} min_overall={min_overall_group:.1f} "
                    f"adv_abs={mean_abs_adv:.3f}]"
                )

            # Entropy health check (one forward pass on the first completion)
            with inference_mode():
                if ids_f:
                    _ent_ids = np.array([prompt_tokens + ids_f[0]], dtype=np.int32)
                    run_entropy += _compute_entropy(model, _ent_ids, len(prompt_tokens))
                    entropy_steps += 1

            # Always count toward the accumulation window (even skipped steps).
            # This ensures optimizer.step fires every grad_accum_steps steps.
            pending_grad_steps += 1

            if skip_mastered_group or np.all(advantages == 0):
                run_skipped += 1
            else:
                # Policy update — scale by actual contributing steps
                _actual_grad_scale = 1.0 / grad_accum_steps
                if ppo_epochs > 1:
                    # Multi-epoch PPO: store group + π_old for later
                    old_lps = _compute_old_log_probs(model, prompt_tokens, ids_f)
                    ppo_batch.append((prompt_tokens, ids_f, advantages, old_lps))
                    # Approximate loss for logging using π_old
                    _approx = sum(
                        -advantages[j] * float(np.mean(old_lps[j]))
                        for j in range(len(ids_f))
                        if old_lps[j] is not None and advantages[j] != 0
                    ) / max(len(ids_f), 1)
                    run_loss += _approx
                elif use_gmpo:
                    loss = gmpo_step(
                        model,
                        prompt_tokens,
                        ids_f,
                        advantages,
                        beta=beta,
                        ref_model=ref_model,
                        grad_scale=_actual_grad_scale,
                    )
                    run_loss += loss
                else:
                    loss = grpo_step(
                        model,
                        prompt_tokens,
                        ids_f,
                        advantages,
                        beta=beta,
                        ref_model=ref_model,
                        grad_scale=_actual_grad_scale,
                    )
                    run_loss += loss

            if pending_grad_steps >= grad_accum_steps:
                if ppo_epochs > 1 and ppo_batch:
                    # ── Multi-epoch PPO update ──
                    for _epoch in range(ppo_epochs):
                        optimizer.zero_grad()
                        for _pt, _ct, _adv, _olp in ppo_batch:
                            _ppo_replay_step(_pt, _ct, _adv, _olp, 1.0 / len(ppo_batch))
                        _set_optimizer_lr(step)
                        pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=max_grad_norm
                        )
                        n_grads = sum(1 for p in model.parameters() if p.grad is not None)
                        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
                        optimizer.step()
                        print(
                            f"  [optimizer.step | train_step={step} | "
                            f"ppo_epoch={_epoch+1}/{ppo_epochs} | "
                            f"grad_norm={pre_clip_norm:.4f} | "
                            f"n_grads={n_grads}/{n_trainable}]"
                        )
                    ppo_batch = []
                else:
                    # ── Single-epoch REINFORCE update ──
                    _set_optimizer_lr(step)
                    pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=max_grad_norm
                    )
                    n_grads = sum(1 for p in model.parameters() if p.grad is not None)
                    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
                    optimizer.step()
                    print(
                        f"  [optimizer.step | train_step={step} | "
                        f"grad_norm={pre_clip_norm:.4f} | "
                        f"n_grads={n_grads}/{n_trainable}]"
                    )
                if use_fast_gen:
                    sync_keras_to_torch(model, torch_gen, config)
                    print(f"  [sync weights → torch gen model at step {step}]")
                optimizer.zero_grad()
                pending_grad_steps = 0
                optimizer_updates += 1
                if optimizer_updates % reshuffle_layers_every == 0:
                    active_block_indices = sorted(
                        random.sample(range(n_blocks), train_layers)
                    )
                    _set_active_trainable_blocks(active_block_indices)
                    print(f"  [reshuffle layers → {active_block_indices}]")

        # ── Always-run section (log / eval / save) ─────────────────────

        # Log
        if step % log_every == 0:
            avg_r = run_reward / max(run_count, 1.0)
            avg_adv = run_adv_abs / max(log_every, 1)
            avg_entropy = run_entropy / max(entropy_steps, 1)
            avg_gen_len = run_gen_tokens / max(run_count, 1.0)
            avg_overall = run_overall / max(run_count, 1.0)
            avg_reasoning = run_reasoning / max(run_count, 1.0)
            avg_answer = run_answer / max(run_count, 1.0)
            avg_follow = run_follow / max(run_count, 1.0)
            avg_concision = run_concision / max(run_count, 1.0)
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            eta = elapsed / step * (num_steps - step)
            log_line = (
                f"  Step {step:4d}/{num_steps} | "
                f"lr={current_lr:.2e} | "
                f"reward_avg={avg_r:.3f} | overall_avg={avg_overall:.1f} | "
                f"reason_avg={avg_reasoning:.2f} | answer_avg={avg_answer:.2f} | "
                f"follow_avg={avg_follow:.2f} | concise_avg={avg_concision:.2f} | "
                f"adv_abs={avg_adv:.3f} | entropy={avg_entropy:.3f} | gen_len={avg_gen_len:.1f}"
            )
            if use_gdpo:
                log_line += (
                    f" | adv_overall={run_adv_overall / log_every:.3f}"
                    f" | adv_reason={run_adv_reasoning / log_every:.3f}"
                    f" | adv_answer={run_adv_answer / log_every:.3f}"
                    f" | adv_follow={run_adv_follow / log_every:.3f}"
                    f" | adv_concise={run_adv_concision / log_every:.3f}"
                )
            log_line += (
                f" | skip={int(run_skipped)} | skip_mastered={int(run_skipped_mastered)} | "
                f"ETA {int(eta//60):02d}:{int(eta%60):02d}"
            )
            print(log_line)
            run_reward = run_skipped = run_loss = 0.0
            run_count = run_adv_abs = 0.0
            run_entropy = 0.0
            run_gen_tokens = 0.0
            entropy_steps = 0
            run_skipped_mastered = 0.0
            run_overall = run_reasoning = run_answer = run_follow = 0.0
            run_concision = 0.0
            run_adv_overall = run_adv_reasoning = run_adv_answer = 0.0
            run_adv_follow = run_adv_concision = 0.0

        # Flush pending gradients before eval
        if eval_every > 0 and step % eval_every == 0 and pending_grad_steps > 0:
            if ppo_epochs > 1 and ppo_batch:
                for _epoch in range(ppo_epochs):
                    optimizer.zero_grad()
                    for _pt, _ct, _adv, _olp in ppo_batch:
                        _ppo_replay_step(_pt, _ct, _adv, _olp, 1.0 / len(ppo_batch))
                    _set_optimizer_lr(step)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()
                    print(f"  [optimizer.step | train_step={step} | ppo_epoch={_epoch+1}/{ppo_epochs} | flush=eval]")
                ppo_batch = []
            else:
                _set_optimizer_lr(step)
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                n_grads = sum(1 for p in model.parameters() if p.grad is not None)
                n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
                optimizer.step()
                print(
                    f"  [optimizer.step | train_step={step} | flush=eval | "
                    f"grad_norm={pre_clip_norm:.4f} | n_grads={n_grads}/{n_trainable}]"
                )
            if use_fast_gen:
                sync_keras_to_torch(model, torch_gen, config)
                print(f"  [sync weights → torch gen model at step {step} | flush=eval]")
            optimizer.zero_grad()
            pending_grad_steps = 0
            active_block_indices = sorted(random.sample(range(n_blocks), train_layers))
            _set_active_trainable_blocks(active_block_indices)

        # Periodic eval
        if eval_every > 0 and step % eval_every == 0:
            evaluate(
                torch_gen if use_fast_gen else model,
                eval_examples if eval_examples is not None else examples,
                api_key=llm_api_key,
                judge_model=judge_model,
                judge_max_concurrent=judge_max_concurrent,
                max_examples=max_eval,
                num_completions=eval_num_completions,
                max_new_tokens=max_new_tokens,
                temperature=eval_temperature,
                rep_penalty=rep_penalty,
                gen_batch_size=gen_batch_size,
                step=step,
                log_path=save_path.replace(".pt", "_eval.log") if save_path else None,
            )

        # Flush pending gradients before backup
        if (
            save_gcp_every > 0
            and step % save_gcp_every == 0
            and save_path
            and pending_grad_steps > 0
        ):
            if ppo_epochs > 1 and ppo_batch:
                for _epoch in range(ppo_epochs):
                    optimizer.zero_grad()
                    for _pt, _ct, _adv, _olp in ppo_batch:
                        _ppo_replay_step(_pt, _ct, _adv, _olp, 1.0 / len(ppo_batch))
                    _set_optimizer_lr(step)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()
                    print(f"  [optimizer.step | train_step={step} | ppo_epoch={_epoch+1}/{ppo_epochs} | flush=save]")
                ppo_batch = []
            else:
                _set_optimizer_lr(step)
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                n_grads = sum(1 for p in model.parameters() if p.grad is not None)
                n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
                optimizer.step()
                print(
                    f"  [optimizer.step | train_step={step} | flush=save | "
                    f"grad_norm={pre_clip_norm:.4f} | n_grads={n_grads}/{n_trainable}]"
                )
            if use_fast_gen:
                sync_keras_to_torch(model, torch_gen, config)
                print(f"  [sync weights → torch gen model at step {step} | flush=save]")
            optimizer.zero_grad()
            pending_grad_steps = 0
            active_block_indices = sorted(random.sample(range(n_blocks), train_layers))
            _set_active_trainable_blocks(active_block_indices)

        # Periodic GCP backup
        if save_gcp_every > 0 and step % save_gcp_every == 0 and save_path:
            _save_and_upload_gcp(model, config, save_path, step, torch_gen=torch_gen if use_fast_gen else None)

    # Flush any remaining accumulated gradients
    if pending_grad_steps > 0:
        if ppo_epochs > 1 and ppo_batch:
            for _epoch in range(ppo_epochs):
                optimizer.zero_grad()
                for _pt, _ct, _adv, _olp in ppo_batch:
                    _ppo_replay_step(_pt, _ct, _adv, _olp, 1.0 / len(ppo_batch))
                _set_optimizer_lr(num_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
            ppo_batch = []
        else:
            _set_optimizer_lr(num_steps)
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            n_grads = sum(1 for p in model.parameters() if p.grad is not None)
            n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
            optimizer.step()
            print(
                f"  [optimizer.step | train_step={num_steps} | flush=final | "
                f"grad_norm={pre_clip_norm:.4f} | n_grads={n_grads}/{n_trainable}]"
            )
        if use_fast_gen:
            sync_keras_to_torch(model, torch_gen, config)
            print(
                f"  [sync weights → torch gen model at step {num_steps} | flush=final]"
            )
        optimizer.zero_grad()

    print(f"\n── {objective_name} complete ({time.time()-t0:.0f}s) ──\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="RLAI fine-tuning for SeqCond")
    p.add_argument("--checkpoint", required=True, help="PyTorch .pt checkpoint")
    p.add_argument(
        "--save", default=None, help="Output .pt path (default: <base>_rlai.pt)"
    )
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--skip_baseline", action="store_true")
    p.add_argument("--dataset_name", default="tatsu-lab/alpaca")
    p.add_argument("--dataset_config", default=None)
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--dataset_path", default=None)
    p.add_argument("--prompt_field", default=None)
    p.add_argument("--response_field", default=None)
    p.add_argument("--messages_field", default=None)
    p.add_argument("--source_field", default=None)

    # Generation
    p.add_argument("--num_completions", type=int, default=6)
    p.add_argument("--max_new_tokens", type=int, default=2200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--rep_penalty", type=float, default=1.0)
    p.add_argument("--gen_batch_size", type=int, default=4)

    # Training
    p.add_argument("--num_steps", type=int, default=250)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--train_layers", type=int, default=3)
    p.add_argument("--optimizer", default="adamw", choices=["sgd", "adamw"])
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--max_eval", type=int, default=100)
    p.add_argument("--eval_num_completions", type=int, default=1)
    p.add_argument("--eval_temperature", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument(
        "--lr_decay",
        default="linear",
        choices=["constant", "linear"],
    )
    p.add_argument("--lr_final_ratio", type=float, default=0.0)
    p.add_argument(
        "--grad_accum_steps",
        type=int,
        default=4,
        help="Accumulate gradients over N steps before optimizer update",
    )
    p.add_argument(
        "--save-gcp",
        type=int,
        default=0,
        dest="save_gcp_every",
        help="Save checkpoint to GCS every N steps (0=disabled). "
        "Uploads to gs://telekinesis-43/checkpoints/",
    )

    # Reward
    p.add_argument(
        "--openai_api_key",
        default=None,
        help="OpenAI key for the RLAI judge (falls back to OPENAI_API_KEY env var)",
    )
    p.add_argument("--judge_model", default="gpt-4.1-mini")
    p.add_argument("--judge_max_concurrent", type=int, default=8)

    # GDPO
    p.add_argument(
        "--use-gdpo",
        action="store_true",
        default=True,
        help="Use decoupled per-criterion normalization before combining advantages",
    )
    p.add_argument(
        "--no-gdpo",
        action="store_false",
        dest="use_gdpo",
        help="Disable GDPO and fall back to standard scalar-reward normalization",
    )
    p.add_argument("--gdpo_overall_weight", type=float, default=0.50)
    p.add_argument("--gdpo_reasoning_weight", type=float, default=0.15)
    p.add_argument("--gdpo_answer_weight", type=float, default=0.15)
    p.add_argument("--gdpo_follow_weight", type=float, default=0.10)
    p.add_argument("--gdpo_concision_weight", type=float, default=0.05)

    # GMPO
    p.add_argument(
        "--use-gmpo",
        action="store_true",
        default=True,
        help="Use GMPO objective instead of the default GRPO objective",
    )
    p.add_argument(
        "--no-gmpo",
        action="store_false",
        dest="use_gmpo",
        help="Disable GMPO and use the original GRPO-style policy objective",
    )

    # Dr. GRPO
    p.add_argument(
        "--use-dr-grpo",
        action="store_true",
        default=True,
        dest="use_dr_grpo",
        help="Remove std from advantage denominator (Dr. GRPO) to fix difficulty bias",
    )
    p.add_argument(
        "--no-dr-grpo",
        action="store_false",
        dest="use_dr_grpo",
        help="Use standard (r - mean) / std advantage normalization",
    )

    # Advantage clipping
    p.add_argument(
        "--clip-adv",
        type=float,
        default=5.0,
        dest="clip_adv",
        help="Clip advantages to [-x, x] after normalization (0=disabled)",
    )

    # Layer rotation
    p.add_argument(
        "--reshuffle_layers_every",
        type=int,
        default=12,
        help="Reshuffle which layers are trained every N optimizer updates (default: 1 = every update).",
    )

    # Overlong penalty
    p.add_argument(
        "--overlong_cache_tokens",
        type=int,
        default=0,
        help="Soft penalty ramp length (tokens) before max_new_tokens. "
        "If 0, uses an automatic value of max(64, max_new_tokens // 8).",
    )

    # PPO multi-epoch
    p.add_argument("--ppo_epochs", type=int, default=1,
                    help="Number of PPO epochs per optimizer update (1=single-pass REINFORCE)")
    p.add_argument("--clip_eps_low", type=float, default=0.2,
                    help="PPO clip epsilon (lower bound): ratio >= 1 - eps_low")
    p.add_argument("--clip_eps_high", type=float, default=0.28,
                    help="PPO clip epsilon (upper bound): ratio <= 1 + eps_high")
    p.add_argument("--gmpo_clip_eps", type=float, default=0.4,
                    help="GMPO symmetric clip epsilon on per-token log-ratios (used when --use-gmpo + ppo_epochs>1)")

    args = p.parse_args()

    model, config, torch_gen = load_model(args.checkpoint)
    examples = load_rlai_examples(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        dataset_path=args.dataset_path,
        prompt_field=args.prompt_field,
        response_field=args.response_field,
        messages_field=args.messages_field,
        source_field=args.source_field,
        seed=4210,
        max_examples=args.max_examples,
    )
    # Split into disjoint train / eval sets
    n_eval = min(args.max_eval, len(examples))
    eval_examples = examples[:n_eval]
    train_examples = examples[n_eval:]
    print(f"Dataset: {len(train_examples)} train / {len(eval_examples)} eval (disjoint)")

    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY") or None
    if not api_key:
        raise ValueError("RLAI requires OPENAI_API_KEY or --openai_api_key.")
    print(f"LLM judge: enabled ({args.judge_model})")

    save_path = args.save or os.path.join(
        "checkpoints",
        os.path.splitext(os.path.basename(args.checkpoint))[0] + "_rlai.pt",
    )

    # ── GCP preflight check ───────────────────────────────────────────────────
    if args.save_gcp_every > 0:
        import subprocess, tempfile
        gcs_bucket = "gs://telekinesis-43/checkpoints"
        gcs_test_path = f"{gcs_bucket}/.preflight_test"
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("preflight ok\n")
            tmp_path = f.name
        print(f"  GCP preflight: testing upload to {gcs_test_path}...", end=" ", flush=True)
        uploaded = False
        for cmd in [
            ["gcloud", "storage", "cp", tmp_path, gcs_test_path],
            ["gsutil", "cp", tmp_path, gcs_test_path],
        ]:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print(f"✓ ({cmd[0]})")
                uploaded = True
                break
            err = result.stderr.strip() or result.stdout.strip() or "unknown error"
            print(f"\n    {cmd[0]}: {err[:200]}", end=" ")
        os.unlink(tmp_path)
        if not uploaded:
            raise RuntimeError(
                "GCP preflight upload failed — neither gcloud nor gsutil succeeded. "
                "Fix credentials before starting a long training run."
            )
    # ─────────────────────────────────────────────────────────────────────────

    if not args.skip_baseline:
        print("\n── Baseline eval ──")
        evaluate(
            torch_gen,
            eval_examples,
            api_key,
            judge_model=args.judge_model,
            judge_max_concurrent=args.judge_max_concurrent,
            max_examples=args.max_eval,
            num_completions=args.eval_num_completions,
            max_new_tokens=args.max_new_tokens,
            temperature=args.eval_temperature,
            rep_penalty=args.rep_penalty,
            gen_batch_size=args.gen_batch_size,
        )

    train_rlai(
        model,
        config,
        train_examples,
        eval_examples=eval_examples,
        torch_gen=torch_gen,
        use_gmpo=args.use_gmpo,
        use_gdpo=args.use_gdpo,
        num_completions=args.num_completions,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        rep_penalty=args.rep_penalty,
        gen_batch_size=args.gen_batch_size,
        beta=args.beta,
        lr=args.lr,
        optimizer_name=args.optimizer,
        train_layers=args.train_layers,
        llm_api_key=api_key,
        judge_model=args.judge_model,
        judge_max_concurrent=args.judge_max_concurrent,
        gdpo_overall_weight=args.gdpo_overall_weight,
        gdpo_reasoning_weight=args.gdpo_reasoning_weight,
        gdpo_answer_weight=args.gdpo_answer_weight,
        gdpo_follow_weight=args.gdpo_follow_weight,
        gdpo_concision_weight=args.gdpo_concision_weight,
        num_steps=args.num_steps,
        eval_every=args.eval_every,
        max_eval=args.max_eval,
        eval_num_completions=args.eval_num_completions,
        eval_temperature=args.eval_temperature,
        warmup_steps=args.warmup_steps,
        save_gcp_every=args.save_gcp_every,
        save_path=save_path,
        grad_accum_steps=args.grad_accum_steps,
        use_dr_grpo=args.use_dr_grpo,
        overlong_cache_tokens=args.overlong_cache_tokens,
        clip_adv=args.clip_adv,
        reshuffle_layers_every=args.reshuffle_layers_every,
        lr_decay=args.lr_decay,
        lr_final_ratio=args.lr_final_ratio,
        ppo_epochs=args.ppo_epochs,
        clip_eps_low=args.clip_eps_low,
        clip_eps_high=args.clip_eps_high,
        gmpo_clip_eps=args.gmpo_clip_eps,
    )

    # Save: .pkl intermediary → .pt with correct PyTorch key mapping
    try:
        pkl_path = save_path[:-3] + ".pkl" if save_path.endswith(".pt") else save_path
        save_keras_checkpoint(model, config, pkl_path)
        if save_path.endswith(".pt"):
            keras_pkl_to_torch_pt(pkl_path, save_path)
        print(f"Saved: {save_path}")
    except OSError as e:
        print(f"WARNING: could not save ({e})")

    # Upload final checkpoint to GCP
    if args.save_gcp_every > 0 and save_path:
        _save_and_upload_gcp(model, config, save_path, step="final", torch_gen=torch_gen if use_fast_gen else None)


if __name__ == "__main__":
    main()
