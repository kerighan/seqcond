"""Compare first SeqCond block output between JAX and PyTorch."""

import subprocess
import sys
import numpy as np

CHECKPOINT_JAX = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step50000.pkl"
CHECKPOINT_TORCH = "checkpoints/seqcond_torch_50k.pt"
TOKEN_ID = 792


def run_jax():
    import jax
    import jax.numpy as jnp
    from seqcond.jax.generator import Generator as JAXGenerator, _make_step_fn

    print("[JAX] Loading model...")
    jax_gen = JAXGenerator(CHECKPOINT_JAX)

    # Initialize state
    states = jax_gen.model.apply(
        {"params": jax_gen.params},
        batch_size=1,
        method=jax_gen.model.init_state,
    )

    # Run step
    step_fn = _make_step_fn(jax_gen.model, jax_gen.params)
    token_array = jnp.array([[TOKEN_ID]], dtype=jnp.int32)
    pos = jnp.array(0, dtype=jnp.int32)
    logits, new_states = step_fn(token_array, states, pos)

    # Save states for multiple blocks
    for i in [0, 10, 20, 30, 39]:
        if i < len(new_states):
            state = new_states[i]
            if len(state) == 5:  # SeqCond block
                np.save(f"/tmp/jax_block{i}_den.npy", np.array(state[0]))
                np.save(f"/tmp/jax_block{i}_re.npy", np.array(state[1]))
                print(f"[JAX] Block {i}: den_acc max={np.abs(state[0]).max():.6f}")

    np.save("/tmp/jax_logits.npy", np.array(logits[0]))


def run_torch():
    import torch
    from seqcond.torch.generator import TorchGenerator

    print("[PyTorch] Loading model...")
    torch_gen = TorchGenerator(CHECKPOINT_TORCH)

    # Initialize state
    state = torch_gen.model.init_state(batch_size=1, device="cuda")

    # Run step
    with torch.no_grad():
        token = torch.tensor([[TOKEN_ID]], device="cuda")
        logits, new_state = torch_gen.model.step(token, state)

    # Save states for multiple blocks
    for i in [0, 10, 20, 30, 39]:
        if i < len(new_state):
            s = new_state[i]
            if len(s) == 5:  # SeqCond block
                np.save(f"/tmp/torch_block{i}_den.npy", s[0].cpu().numpy())
                np.save(f"/tmp/torch_block{i}_re.npy", s[1].cpu().numpy())
                print(f"[PyTorch] Block {i}: den_acc max={s[0].abs().max().item():.6f}")

    np.save("/tmp/torch_logits.npy", logits[0].cpu().numpy())


def compare():
    print("\n" + "=" * 60)
    print("MULTI-BLOCK STATE COMPARISON")
    print("=" * 60)

    for i in [0, 10, 20, 30, 39]:
        try:
            jax_den = np.load(f"/tmp/jax_block{i}_den.npy")
            torch_den = np.load(f"/tmp/torch_block{i}_den.npy")
            jax_re = np.load(f"/tmp/jax_block{i}_re.npy")
            torch_re = np.load(f"/tmp/torch_block{i}_re.npy")

            den_diff = np.abs(jax_den - torch_den).max()
            re_diff = np.abs(jax_re - torch_re).max()
            print(f"Block {i:2d}: den_diff={den_diff:.6f}, re_diff={re_diff:.6f}")
        except FileNotFoundError:
            print(f"Block {i:2d}: (Transformer block, skipped)")

    # Compare logits
    jax_logits = np.load("/tmp/jax_logits.npy")
    torch_logits = np.load("/tmp/torch_logits.npy")
    print(f"\nLogits: max_diff={np.abs(jax_logits - torch_logits).max():.6f}")

    jax_argmax = np.argmax(jax_logits)
    torch_argmax = np.argmax(torch_logits)
    if jax_argmax == torch_argmax:
        print(f"✅ Argmax MATCH: {jax_argmax}")
    else:
        print(f"❌ Argmax DIFFER: JAX={jax_argmax}, PyTorch={torch_argmax}")


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
