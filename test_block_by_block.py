"""Compare block-by-block outputs between JAX and PyTorch."""

import subprocess
import sys
import numpy as np

CHECKPOINT_JAX = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step60000.pkl"
CHECKPOINT_TORCH = "checkpoints/seqcond_torch_60k.pt"
TOKEN_ID = 792


def run_jax():
    import jax
    import jax.numpy as jnp
    from seqcond.jax.generator import Generator as JAXGenerator

    print("[JAX] Loading model...")
    jax_gen = JAXGenerator(CHECKPOINT_JAX)

    states = jax_gen.model.apply(
        {"params": jax_gen.params},
        batch_size=1,
        method=jax_gen.model.init_state,
    )

    def step_with_intermediates(
        self, token_id, states, pos, deterministic: bool = True
    ):
        if token_id.ndim == 1:
            token_id = token_id[:, None]

        b = token_id.shape[0]

        x = self.embedding(token_id)[:, 0, :]  # (B, D)

        if self.use_positional_embedding:
            pos_emb = self.position_embedding(jnp.array([[pos]]))[:, 0, :]
            x = x + pos_emb

        head_dim_half = self.cos_emb.shape[1]
        cos_t = jax.lax.dynamic_slice(self.cos_emb, (pos, 0), (1, head_dim_half))
        sin_t = jax.lax.dynamic_slice(self.sin_emb, (pos, 0), (1, head_dim_half))
        cos_t = cos_t[None, :, None, :]
        sin_t = sin_t[None, :, None, :]
        cos_t = jnp.broadcast_to(cos_t, (b, 1, self.num_heads, head_dim_half))
        sin_t = jnp.broadcast_to(sin_t, (b, 1, self.num_heads, head_dim_half))

        xs = []
        new_states = []
        for i, (block_type, block) in enumerate(self.blocks):
            if block_type == "transformer":
                x, new_state = block.step(
                    x, states[i], pos, cos_t, sin_t, deterministic=deterministic
                )
            else:
                x, new_state = block.step(x, states[i], deterministic=deterministic)
            new_states.append(new_state)
            xs.append(x)

        if self.tie_weights:
            logits = self.output_projection(x[:, None, :], self.embedding.embedding)[
                :, 0, :
            ]
        else:
            logits = self.output_projection(x)

        return logits, new_states, xs

    token_array = jnp.array([[TOKEN_ID]], dtype=jnp.int32)
    pos = jnp.array(0, dtype=jnp.int32)
    logits, _, xs = jax_gen.model.apply(
        {"params": jax_gen.params},
        token_array,
        states,
        pos,
        deterministic=True,
        method=step_with_intermediates,
    )

    emb = jax_gen.params["token_embedding"]["embedding"][TOKEN_ID]
    results = {"embedding": np.array(emb[None, :]).astype(np.float32)}
    for i, x in enumerate(xs):
        results[f"block_{i}_output"] = np.array(x).astype(np.float32)
        if i < 5 or i >= 35 or (i + 1) % 4 == 0:
            x_np = results[f"block_{i}_output"]
            print(f"[JAX] Block {i}: mean={x_np.mean():.6f}, std={x_np.std():.6f}")

    results["logits"] = np.array(logits[0]).astype(np.float32)

    np.savez("/tmp/jax_results.npz", **results)
    print(f"[JAX] Embedding mean: {results['embedding'].mean():.6f}")
    print(
        f"[JAX] Logits mean: {results['logits'].mean():.6f}, argmax: {results['logits'].argmax()}"
    )


def run_torch():
    import torch
    from seqcond.torch.generator import TorchGenerator

    print("[PyTorch] Loading model...")
    torch_gen = TorchGenerator(CHECKPOINT_TORCH)
    model = torch_gen.model

    states = model.init_state(batch_size=1, device="cuda")

    results = {}

    with torch.no_grad():
        token_id = torch.tensor([[TOKEN_ID]], dtype=torch.long, device="cuda")

        # Match SeqCondModel.step exactly: embed token, pick global pos from first block,
        # compute RoPE once, then run each block step.
        x = model.embedding(token_id).squeeze(1)  # (1, D)
        results["embedding"] = x.cpu().numpy()

        if isinstance(states[0], tuple) and len(states[0]) == 5:  # SeqCond
            pos = states[0][3]
        else:  # Transformer
            pos = states[0][2]

        cos_all, sin_all = model.rotary_emb(
            x, seq_len=model.maxlen
        )  # (maxlen, half_dim)
        cos_t = cos_all[pos].unsqueeze(1)  # (B, 1, half_dim)
        sin_t = sin_all[pos].unsqueeze(1)
        cos_t = cos_t.unsqueeze(2).expand(1, 1, model.num_heads, -1)
        sin_t = sin_t.unsqueeze(2).expand(1, 1, model.num_heads, -1)

        for i, block in enumerate(model.blocks):
            if isinstance(states[i], (list, tuple)):
                if len(states[i]) == 5:  # SeqCond
                    p_val = states[i][3].item()
                else:  # Transformer
                    p_val = states[i][2][0].item()
            else:
                p_val = "N/A"

            # Diagnostic for Block 39
            if i == 39:
                print(f"\n--- Diagnostic Block 39 ---")
                print(f"Input x mean={x.mean().item():.6f}, std={x.std().item():.6f}")
                print(f"Pos={p_val}")
                # Check cos_t/sin_t
                print(f"cos_t mean={cos_t.mean().item():.6f}")

            if hasattr(block, "q_proj"):  # Transformer block
                x, new_state = block.step(x, states[i], cos_t, sin_t)
                if i == 39:
                    print(
                        f"Output x mean={x.mean().item():.6f}, std={x.std().item():.6f}"
                    )
            else:  # SeqCond block
                x, new_state = block.step(x, states[i])

            states[i] = new_state
            results[f"block_{i}_output"] = x.cpu().numpy()

            if i < 10 or i >= 35 or (i + 1) % 4 == 0:
                print(
                    f"[PyTorch] Block {i} (pos={p_val}): mean={x.mean().item():.6f}, std={x.std().item():.6f}"
                )

        # Final logits
        logits = model.output_projection(x)
        results["logits"] = logits[0].cpu().numpy()

    np.savez("/tmp/torch_results.npz", **results)
    print(
        f"[PyTorch] Logits mean: {results['logits'].mean():.6f}, argmax: {results['logits'].argmax()}"
    )


def compare():
    jax = np.load("/tmp/jax_results.npz")
    torch_r = np.load("/tmp/torch_results.npz")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    print(
        f"\nEmbedding diff: {np.abs(jax['embedding'] - torch_r['embedding']).max():.8f}"
    )
    print(f"Logits diff: max={np.abs(jax['logits'] - torch_r['logits']).max():.4f}")

    a = jax["logits"].reshape(-1).astype(np.float64)
    b = torch_r["logits"].reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    logits_cos = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    print(f"Logits cosine similarity: {logits_cos:.6f}")
    print(f"\nJAX argmax: {jax['logits'].argmax()}")
    print(f"PyTorch argmax: {torch_r['logits'].argmax()}")

    # Find first diverging block
    first_bad = None
    biggest_cos_drop = None
    prev_cos = None
    for i in range(40):
        key = f"block_{i}_output"
        if key not in jax or key not in torch_r:
            continue
        diff = np.abs(jax[key] - torch_r[key])
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        a = jax[key].reshape(-1).astype(np.float64)
        b = torch_r[key].reshape(-1).astype(np.float64)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cos_sim = float(np.dot(a, b) / denom) if denom > 0 else 0.0

        if prev_cos is not None:
            drop = prev_cos - cos_sim
            if biggest_cos_drop is None or drop > biggest_cos_drop[0]:
                biggest_cos_drop = (drop, i - 1, i, prev_cos, cos_sim)
        prev_cos = cos_sim

        in_key = "embedding" if i == 0 else f"block_{i-1}_output"
        da = (jax[key] - jax[in_key]).reshape(-1).astype(np.float64)
        db = (torch_r[key] - torch_r[in_key]).reshape(-1).astype(np.float64)
        d_denom = np.linalg.norm(da) * np.linalg.norm(db)
        delta_cos = float(np.dot(da, db) / d_denom) if d_denom > 0 else 0.0

        block_type = "transformer" if (i + 1) % 4 == 0 else "seqcond"
        if (first_bad is None) and (max_diff > 1e-3):
            first_bad = (i, max_diff, mean_diff)
        if i < 10 or i >= 35 or (i + 1) % 4 == 0:
            print(
                f"\nBlock {i} ({block_type}) diff: max={max_diff:.6f}, mean={mean_diff:.6f}, cos={cos_sim:.6f}, delta_cos={delta_cos:.6f}"
            )

    if first_bad is not None:
        i, max_d, mean_d = first_bad
        print(f"\nFIRST DIVERGENCE: block {i} max={max_d:.6f} mean={mean_d:.6f}")
    else:
        print("\nNo divergence detected above threshold")

    if biggest_cos_drop is not None:
        drop, i0, i1, c0, c1 = biggest_cos_drop
        print(
            f"\nBIGGEST COS DROP: block {i0}->{i1} drop={drop:.6f} ({c0:.6f} -> {c1:.6f})"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "jax":
            run_jax()
        elif sys.argv[1] == "torch":
            run_torch()
        elif sys.argv[1] == "compare":
            compare()
    else:
        subprocess.run([sys.executable, __file__, "jax"], check=True)
        print()
        subprocess.run([sys.executable, __file__, "torch"], check=True)
        compare()
