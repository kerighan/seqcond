#!/usr/bin/env python3
"""Compare Triton kernel output vs PyTorch path for SeqCond step.

Loads a real model, runs a few step() calls with both paths,
and reports numerical differences.
"""
import torch
from seqcond.torch.generator import TorchGenerator


def compare_step_outputs(checkpoint="checkpoints/seqcond_torch_750k.pt", num_steps=20):
    """Run N steps with both Triton and PyTorch, compare outputs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    gen = TorchGenerator(checkpoint, device=device)
    model = gen.model

    # Print model info
    block_types = model.block_types
    print(f"Block types: {block_types[:5]}... ({len(block_types)} total)")
    for i, (block, bt) in enumerate(zip(model.blocks, block_types)):
        if bt == "seqcond":
            attn = block.attn
            print(
                f"  Layer {i}: SeqCond K={attn.K}, K_q={attn.K_q}, H={attn.H}, M={attn.M}, n_rep={attn.n_rep}"
            )
            break

    # Tokenize a prompt
    prompt = "<|im_start|>user\nWhat is the capital of France?\n<|im_end|><|im_start|>assistant\n<|think_start|>"
    tokens = gen.tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], device=device)

    # Prefill (same for both)
    logits_prefill, states_base = model.prefill(input_ids)

    # Deep copy states for two parallel runs
    def clone_states(states):
        return [tuple(t.clone() for t in block_state) for block_state in states]

    states_triton = clone_states(states_base)
    states_pytorch = clone_states(states_base)

    # Get first token (greedy)
    next_token = torch.argmax(logits_prefill[:, -1, :], dim=-1, keepdim=True)  # (1, 1)
    token_triton = next_token.clone()
    token_pytorch = next_token.clone()

    print(f"\nPrompt: {len(tokens)} tokens")
    print(f"Running {num_steps} steps with both paths...\n")
    print(
        f"{'Step':>4} {'Token_T':>8} {'Token_P':>8} {'Match':>6} {'MaxDiff_logits':>15} {'MeanDiff':>12}"
    )
    print("-" * 70)

    mismatches = 0
    max_diffs = []

    for step in range(num_steps):
        # Triton path
        logits_t, states_triton = model.step(
            token_triton, states_triton, use_triton=True
        )

        # PyTorch path
        logits_p, states_pytorch = model.step(
            token_pytorch, states_pytorch, use_triton=False
        )

        # Compare logits (handle both 2D and 3D)
        lt = logits_t.view(-1) if logits_t.dim() <= 2 else logits_t[:, -1, :].view(-1)
        lp = logits_p.view(-1) if logits_p.dim() <= 2 else logits_p[:, -1, :].view(-1)
        diff = (lt.float() - lp.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        max_diffs.append(max_diff)

        # Greedy tokens
        tok_t = torch.argmax(lt)
        tok_p = torch.argmax(lp)

        match = "✓" if tok_t.item() == tok_p.item() else "✗"
        if tok_t.item() != tok_p.item():
            mismatches += 1

        print(
            f"{step:>4} {tok_t.item():>8} {tok_p.item():>8} {match:>6} {max_diff:>15.6f} {mean_diff:>12.6f}"
        )

        # Feed same token to both (use PyTorch token as reference, shape (1, 1))
        token_triton = tok_p.view(1, 1)
        token_pytorch = tok_p.view(1, 1)

        # Per-layer state comparison
        if step == 0:
            print("\n     Per-layer state diffs (step 0):")
            for bi, (st, sp) in enumerate(zip(states_triton, states_pytorch)):
                bt = block_types[bi]
                layer_diffs = []
                names = (
                    ["den_acc", "re_acc", "im_acc", "pos", "conv_buf"]
                    if bt == "seqcond"
                    else [f"t{j}" for j in range(len(st))]
                )
                for ti, (tt, tp) in enumerate(zip(st, sp)):
                    sd = (tt.float() - tp.float()).abs().max().item()
                    layer_diffs.append((names[ti] if ti < len(names) else f"t{ti}", sd))
                nonzero = [(n, d) for n, d in layer_diffs if d > 1e-6]
                if nonzero:
                    parts = ", ".join(f"{n}={d:.4f}" for n, d in nonzero)
                    print(f"       Layer {bi:>2} ({bt:>11}): {parts}")
            print()

    print("-" * 70)
    print(f"\nToken mismatches: {mismatches}/{num_steps}")
    print(f"Max logit diff across all steps: {max(max_diffs):.6f}")
    print(f"Mean of max logit diffs: {sum(max_diffs)/len(max_diffs):.6f}")

    if mismatches > 0:
        print("\n⚠️  TRITON AND PYTORCH PATHS DIVERGE!")
    else:
        print("\n✓ Triton and PyTorch paths produce identical tokens")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/seqcond_torch_750k.pt")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    compare_step_outputs(args.checkpoint, args.steps)
