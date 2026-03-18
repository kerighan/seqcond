#!/usr/bin/env python3
"""Benchmark Keras3 SeqCond generation speed (tokens/sec).

Usage:
    KERAS_BACKEND=torch python benchmark_keras_gen.py --checkpoint checkpoints/seqcond_torch_100k.pt
    KERAS_BACKEND=torch python benchmark_keras_gen.py --checkpoint checkpoints/seqcond_torch_100k.pt --also-torch
"""
import argparse
import time
import numpy as np
import os

os.environ.setdefault("KERAS_BACKEND", "torch")


def benchmark_keras(checkpoint_path, num_tokens=128, warmup=16, batch_sizes=(1, 4)):
    """Benchmark Keras model step() speed."""
    from convert_torch_to_keras import build_keras_model, convert_weights, load_torch_checkpoint
    from seqcond.dataset import Tokenizer

    config, state_dict = load_torch_checkpoint(checkpoint_path)
    model = build_keras_model(config)
    convert_weights(config, state_dict, model)
    tokenizer = Tokenizer()

    prompt = "<|im_start|>user\nWhat is 2+2?\n<|im_end|><|im_start|>assistant\n<|think_start|>"
    prompt_toks = tokenizer([prompt])[0]

    print(f"\n{'='*60}")
    print(f"Keras3 (backend={os.environ.get('KERAS_BACKEND', '?')})")
    print(f"{'='*60}")

    for B in batch_sizes:
        # Init state
        states = model.init_state(batch_size=1)

        # Prefill (token-by-token, same as train_grpo.py)
        t_prefill = time.time()
        for t in prompt_toks:
            logits, states = model.step(np.array([[t]], dtype=np.int32), states)
        t_prefill = time.time() - t_prefill

        # Tile for batch
        if B > 1:
            from keras import ops
            states = [
                tuple(ops.tile(s, (B,) + (1,) * (s.ndim - 1)) for s in state)
                for state in states
            ]
            logits = np.tile(np.array(logits), (B, 1))

        # Warmup decode steps
        buf = np.zeros((B, 1), dtype=np.int32)
        for _ in range(warmup):
            tok = int(np.argmax(logits[0] if B > 1 else logits))
            buf[:, 0] = tok
            logits, states = model.step(buf, states)

        # Timed decode
        if os.environ.get("KERAS_BACKEND") == "torch":
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(num_tokens):
            tok = int(np.argmax(logits[0] if B > 1 else logits))
            buf[:, 0] = tok
            logits, states = model.step(buf, states)

        if os.environ.get("KERAS_BACKEND") == "torch":
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        elapsed = time.time() - t0
        tps = num_tokens * B / elapsed

        print(f"  B={B:2d}  prefill={t_prefill:.2f}s ({len(prompt_toks)} toks)  "
              f"decode={elapsed:.2f}s ({num_tokens} steps)  "
              f"{tps:.1f} tok/s  ({tps/B:.1f} tok/s/sample)")


def benchmark_torch(checkpoint_path, num_tokens=128, warmup=16, batch_sizes=(1, 4)):
    """Benchmark PyTorch TorchGenerator step() speed (with Triton)."""
    import torch
    from seqcond.torch.model import SeqCondModel
    from seqcond.dataset import Tokenizer

    data = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    config = data["config"]
    model = SeqCondModel(**config).cuda().eval()
    model.load_state_dict(data["state_dict"], strict=False)
    tokenizer = Tokenizer()

    prompt = "<|im_start|>user\nWhat is 2+2?\n<|im_end|><|im_start|>assistant\n<|think_start|>"
    prompt_toks = tokenizer([prompt])[0]

    print(f"\n{'='*60}")
    print(f"PyTorch (Triton)")
    print(f"{'='*60}")

    for B in batch_sizes:
        input_ids = torch.tensor([prompt_toks], device="cuda")
        with torch.no_grad():
            logits, states = model.prefill(input_ids)
            logits = logits.squeeze(1)

        # Tile for batch
        if B > 1:
            logits = logits.repeat(B, 1)
            states = [
                tuple(s.repeat(B, *([1] * (s.ndim - 1))) for s in state)
                for state in states
            ]

        # Warmup
        token_tensor = torch.zeros((B, 1), dtype=torch.long, device="cuda")
        with torch.no_grad():
            for _ in range(warmup):
                tok = torch.argmax(logits[0]).item()
                token_tensor[:, 0] = tok
                logits, states = model.step(token_tensor, states, use_triton=True)

        torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            for _ in range(num_tokens):
                tok = torch.argmax(logits[0]).item()
                token_tensor[:, 0] = tok
                logits, states = model.step(token_tensor, states, use_triton=True)

        torch.cuda.synchronize()
        elapsed = time.time() - t0
        tps = num_tokens * B / elapsed

        print(f"  B={B:2d}  decode={elapsed:.2f}s ({num_tokens} steps)  "
              f"{tps:.1f} tok/s  ({tps/B:.1f} tok/s/sample)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark SeqCond generation speed")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--num-tokens", type=int, default=128, help="Tokens to generate (default: 128)")
    parser.add_argument("--warmup", type=int, default=16, help="Warmup steps (default: 16)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8], help="Batch sizes to test")
    parser.add_argument("--also-torch", action="store_true", help="Also benchmark PyTorch+Triton for comparison")
    args = parser.parse_args()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tokens: {args.num_tokens}, Warmup: {args.warmup}")
    print(f"Batch sizes: {args.batch_sizes}")

    benchmark_keras(args.checkpoint, args.num_tokens, args.warmup, args.batch_sizes)

    if args.also_torch:
        benchmark_torch(args.checkpoint, args.num_tokens, args.warmup, args.batch_sizes)


if __name__ == "__main__":
    main()
