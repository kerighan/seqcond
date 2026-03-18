"""
Compare Torch forward() vs prefill()+step() to find where generation diverges.

Usage:
    python debug_step_vs_forward.py
"""

import torch
import numpy as np
from seqcond.dataset import Tokenizer

TORCH_CKPT = "checkpoints/seqcond_torch_762k.pt"


def load_torch_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    from seqcond.torch.model import SeqCondModel as TorchModel

    model = TorchModel(
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        vocab_size=config["vocab_size"],
        maxlen=config["maxlen"],
        num_heads=config["num_heads"],
        num_kv_heads=config.get("num_kv_heads"),
        qk_norm=config.get("qk_norm", True),
        qk_norm_eps=config.get("qk_norm_eps", 1e-6),
        seqcond_heads=config.get("seqcond_heads", config["num_heads"]),
        num_query_heads=config.get("num_query_heads", 6),
        num_thetas=config.get("num_thetas", 4),
        conv_kernel_size=config.get("conv_kernel_size", 4),
        expand_factor=config.get("expand_factor", 1),
        out_expand_factor=config.get("out_expand_factor", 3),
        seqcond_ratio=config.get("seqcond_ratio", 3),
        num_anchor_heads=config.get("num_anchor_heads", 0),
    )
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    model.eval()
    return model, config


def main():
    tokenizer = Tokenizer()
    prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    token_ids = tokenizer.encode(prompt)
    print(f"Prompt: {len(token_ids)} tokens")

    model, config = load_torch_model(TORCH_CKPT)

    input_ids = torch.tensor([token_ids], dtype=torch.long)

    with torch.no_grad():
        # === Test 1: forward() reference ===
        fwd_logits = model(input_ids)  # (1, L, V)
        fwd_last = fwd_logits[0, -1].numpy()
        fwd_top5 = np.argsort(fwd_last)[-5:][::-1]
        print(
            f"\nforward() last-pos top5: {fwd_top5} = {[tokenizer.decode([t]) for t in fwd_top5]}"
        )

        # === Test 2: prefill() ===
        prefill_logits, states = model.prefill(input_ids, return_all_logits=True)
        pf_last = prefill_logits[0, -1].numpy()
        pf_top5 = np.argsort(pf_last)[-5:][::-1]
        print(
            f"prefill()  last-pos top5: {pf_top5} = {[tokenizer.decode([t]) for t in pf_top5]}"
        )

        pf_diff = np.abs(fwd_last - pf_last)
        print(f"  forward vs prefill max diff: {pf_diff.max():.6f}")
        print(f"  Top-1 match: {fwd_top5[0] == pf_top5[0]}")

        # === Test 3: step-by-step from init_state ===
        print("\n--- Step-by-step from init_state ---")
        step_states = model.init_state(1, device=torch.device("cpu"))

        for t in range(len(token_ids)):
            tok_tensor = torch.tensor([[token_ids[t]]], dtype=torch.long)
            pos_tensor = torch.tensor([t], dtype=torch.long)
            step_logits, step_states = model.step(
                tok_tensor, step_states, pos=pos_tensor
            )

        step_last = step_logits[0].numpy()
        step_top5 = np.argsort(step_last)[-5:][::-1]
        print(
            f"step()     last-pos top5: {step_top5} = {[tokenizer.decode([t]) for t in step_top5]}"
        )

        step_diff = np.abs(fwd_last - step_last)
        print(f"  forward vs step max diff: {step_diff.max():.6f}")
        print(f"  Top-1 match: {fwd_top5[0] == step_top5[0]}")

        # === Test 4: Side-by-side generation — step vs forward ===
        N_GEN = 30
        print(f"\n--- Side-by-side generation: {N_GEN} tokens ---")

        # Prefill for step path
        prefill_logits_gen, gen_states = model.prefill(input_ids)
        first_tok = torch.argmax(prefill_logits_gen[0, 0]).item()

        # Forward path
        fwd_ids = list(token_ids)

        step_gen = [first_tok]
        fwd_gen = [first_tok]
        fwd_ids.append(first_tok)

        tok_tensor = torch.tensor([[first_tok]], dtype=torch.long)

        print(
            f"{'Pos':>4} | {'Step tok':>10} {'Step text':>15} | {'Fwd tok':>10} {'Fwd text':>15} | {'Logit MaxDiff':>13} | {'Match':>5}"
        )
        print("-" * 90)

        for i in range(N_GEN - 1):
            # Step path
            step_logits, gen_states = model.step(tok_tensor, gen_states)
            step_next = torch.argmax(step_logits[0]).item()

            # Forward path
            fwd_input = torch.tensor([fwd_ids], dtype=torch.long)
            fwd_logits = model(fwd_input)
            fwd_next = torch.argmax(fwd_logits[0, -1]).item()

            # Compare logits
            step_l = step_logits[0].numpy()
            fwd_l = fwd_logits[0, -1].numpy()
            max_diff = np.abs(step_l - fwd_l).max()

            match = step_next == fwd_next
            step_text = tokenizer.decode([step_next])[:12]
            fwd_text = tokenizer.decode([fwd_next])[:12]

            flag = "" if match else " <<< DIVERGE"
            print(
                f"{i+1:4d} | {step_next:10d} {step_text:>15} | {fwd_next:10d} {fwd_text:>15} | {max_diff:13.4f} | {match!s:>5}{flag}"
            )

            # Both paths use STEP's token to continue (to track cumulative drift)
            step_gen.append(step_next)
            fwd_gen.append(fwd_next)
            fwd_ids.append(step_next)  # Use step's token so both paths see same input
            tok_tensor = torch.tensor([[step_next]], dtype=torch.long)

            if step_next == tokenizer.encode("<|im_end|>")[0]:
                print("  [EOS reached]")
                break

        print(f"\nStep generated: '{tokenizer.decode(step_gen[:50])}'")
        print(f"Fwd  generated: '{tokenizer.decode(fwd_gen[:50])}'")
        print(f"All matched: {step_gen == fwd_gen}")


if __name__ == "__main__":
    main()
