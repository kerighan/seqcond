#!/usr/bin/env python3
"""Sanity check: load → save → reload → compare weights, buffers, logits, generation."""

import os
import sys
import time
import tempfile

import numpy as np
import torch

CHECKPOINT = "./checkpoints/seqcond_lin5.pt"
if len(sys.argv) > 1:
    CHECKPOINT = sys.argv[1]

CACHE_PREFIXES = (
    '_conv_kernel_t', '_decay_slopes_cached', '_phase_scale_b',
    '_score_bias_b', '_score_scale_b', '_theta_cached',
    '_w_int_cached', '_anchor_slopes_cached',
)


def load_model(path):
    from seqcond.torch.model import SeqCondModel
    data = torch.load(path, map_location="cpu", weights_only=False)
    config = data["config"]
    model = SeqCondModel(**config).cpu().eval()
    model.load_state_dict(data["state_dict"], strict=False)
    n = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {path}  ({n:,} params, {len(data['state_dict'])} keys in checkpoint)")
    return model, config


def save_model(model, config, path):
    sd = {k: v.cpu() for k, v in model.state_dict().items()
          if not any(k.endswith(sfx) for sfx in CACHE_PREFIXES)}
    torch.save({"state_dict": sd, "config": config}, path)
    print(f"  Saved {path}  ({len(sd)} tensors)")


def compare_params(m1, m2, label=""):
    max_diff, max_name = 0.0, "(none)"
    n_diff = 0
    for (n1, p1), (n2, p2) in zip(sorted(m1.named_parameters()), sorted(m2.named_parameters())):
        assert n1 == n2, f"Name mismatch: {n1} vs {n2}"
        d = (p1.float() - p2.float()).abs().max().item()
        if d > 0:
            n_diff += 1
        if d > max_diff:
            max_diff = d
            max_name = n1
    total = sum(1 for _ in m1.named_parameters())
    status = "✓ IDENTICAL" if max_diff == 0 else f"✗ DIFFERS"
    print(f"  [{label}] Parameters: {status}  max_diff={max_diff:.10f} at {max_name}  ({n_diff}/{total} differ)")
    return max_diff


def compare_buffers(m1, m2, label=""):
    max_diff, max_name = 0.0, "(none)"
    n_compared = 0
    n_diff = 0
    for (n1, b1), (n2, b2) in zip(sorted(m1.named_buffers()), sorted(m2.named_buffers())):
        if b1 is None or b2 is None:
            continue
        if b1.shape != b2.shape:
            print(f"    Buffer shape mismatch: {n1}: {b1.shape} vs {b2.shape}")
            continue
        n_compared += 1
        d = (b1.float() - b2.float()).abs().max().item()
        if d > 0:
            n_diff += 1
        if d > max_diff:
            max_diff = d
            max_name = n1
    status = "✓ IDENTICAL" if max_diff == 0 else f"✗ DIFFERS"
    print(f"  [{label}] Buffers: {status}  max_diff={max_diff:.10f} at {max_name}  ({n_diff}/{n_compared} differ)")
    return max_diff


def compare_logits(m1, m2, device="cuda"):
    """Compare prefill logits on several fixed inputs."""
    m1 = m1.to(device)
    m2 = m2.to(device)

    test_inputs = [
        [1, 2, 3, 4, 5],
        [100, 200, 300, 400, 500],
        list(range(1, 51)),  # 50 tokens
    ]

    for ids in test_inputs:
        inp = torch.tensor([ids], device=device)
        with torch.no_grad():
            l1 = m1.prefill(inp)[0]
            l2 = m2.prefill(inp)[0]
        diff = (l1 - l2).abs()
        print(f"  Logits [{len(ids)} tokens]: max_diff={diff.max().item():.8f}  "
              f"mean_diff={diff.mean().item():.8f}  "
              f"argmax_match={l1.argmax(-1).equal(l2.argmax(-1))}")

    m1.cpu()
    m2.cpu()


def greedy_generate(model, tokenizer, prompt, max_tokens=50, device="cuda"):
    """Greedy decode from a model. Returns list of token ids."""
    toks = tokenizer([prompt])[0]
    inp = torch.tensor([toks], device=device)
    model = model.to(device).eval()
    with torch.no_grad():
        logits, states = model.prefill(inp)
        generated = []
        eos = tokenizer.encode("<|im_end|>")[0]
        for i in range(max_tokens):
            # prefill returns (B,1,V), step returns (B,V)
            last_logits = logits[:, -1, :] if logits.dim() == 3 else logits
            tok_id = last_logits.argmax(-1)  # (B,)
            generated.append(tok_id.item())
            if tok_id.item() == eos:
                break
            logits, states = model.step(tok_id, states, seq_len=len(toks) + i + 1)
    return generated


def compare_generation(m1, m2, device="cuda"):
    """Generate with greedy decoding from both models and compare."""
    sys.path.insert(0, ".")
    from train_grpo import Tokenizer

    tokenizer = Tokenizer()
    prompt = "What is 2 + 3?\n"

    for label, model in [("model_A", m1), ("model_B", m2)]:
        gen = greedy_generate(model, tokenizer, prompt, device=device)
        text = tokenizer.decode(gen)
        print(f"  [{label}] {repr(text[:100])}")
        model.cpu()


def test_post_generation_save_reload(m_orig, config, device="cuda"):
    """KEY TEST: generate with model, save, reload, compare logits.
    This tests whether generation leaves dirty state that gets saved."""
    sys.path.insert(0, ".")
    from train_grpo import Tokenizer

    tokenizer = Tokenizer()
    prompt = "Solve step by step: What is 15 * 23?\n"

    # Generate 200 tokens to really exercise all the recurrent state
    print("  Generating 200 tokens to warm up model state...")
    gen = greedy_generate(m_orig, tokenizer, prompt, max_tokens=200, device=device)
    print(f"  Generated {len(gen)} tokens")

    # Now save (model still on GPU with all caches populated)
    m_orig_gpu = m_orig.to(device)
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp.close()
    save_model(m_orig_gpu, config, tmp.name)

    # Reload fresh
    m_fresh, _ = load_model(tmp.name)
    m_fresh = m_fresh.to(device)

    # Compare params
    print("  Params after gen+save+reload:")
    compare_params(m_orig_gpu, m_fresh, "post-gen")

    # Compare ALL state_dict keys (including buffers/caches)
    sd1 = {k: v for k, v in m_orig_gpu.state_dict().items()}
    sd2 = {k: v for k, v in m_fresh.state_dict().items()}
    keys_only_1 = set(sd1.keys()) - set(sd2.keys())
    keys_only_2 = set(sd2.keys()) - set(sd1.keys())
    if keys_only_1:
        print(f"  Keys only in original (not saved): {keys_only_1}")
    if keys_only_2:
        print(f"  Keys only in reloaded: {keys_only_2}")

    # Compare shared keys
    max_diff, max_name = 0.0, "(none)"
    for k in sorted(set(sd1.keys()) & set(sd2.keys())):
        if sd1[k].shape != sd2[k].shape:
            print(f"  Shape mismatch: {k}: {sd1[k].shape} vs {sd2[k].shape}")
            continue
        d = (sd1[k].float() - sd2[k].float()).abs().max().item()
        if d > max_diff:
            max_diff = d
            max_name = k
    print(f"  All state_dict diff: max={max_diff:.10f} at {max_name}")

    # CRITICAL: compare logits on the SAME input
    test_inputs = [
        torch.tensor([[1, 2, 3, 4, 5]], device=device),
        torch.tensor([tokenizer(["What is 2+3?\n"])[0]], device=device),
    ]
    for inp in test_inputs:
        with torch.no_grad():
            l1 = m_orig_gpu.prefill(inp)[0]
            l2 = m_fresh.prefill(inp)[0]
        diff = (l1 - l2).abs()
        print(f"  Logits [{inp.shape[1]} tok]: max={diff.max().item():.8f}  "
              f"mean={diff.mean().item():.8f}  argmax_match={l1.argmax(-1).equal(l2.argmax(-1))}")

    # Compare step-by-step generation
    print("  Greedy generation comparison after save/reload:")
    gen1 = greedy_generate(m_orig_gpu, tokenizer, "What is 7*8?\n", max_tokens=50, device=device)
    gen2 = greedy_generate(m_fresh, tokenizer, "What is 7*8?\n", max_tokens=50, device=device)
    match = gen1 == gen2
    print(f"  Tokens match: {match}  (len={len(gen1)} vs {len(gen2)})")
    if not match:
        # Find first divergence
        for i, (a, b) in enumerate(zip(gen1, gen2)):
            if a != b:
                print(f"  First divergence at token {i}: {a} vs {b}")
                break

    m_orig_gpu.cpu()
    m_fresh.cpu()
    os.unlink(tmp.name)


def main():
    print(f"\n{'='*60}")
    print(f"SANITY CHECK: Save/Reload Round-Trip")
    print(f"{'='*60}\n")

    # 1. Load original
    print("1. Load original checkpoint")
    m_orig, config = load_model(CHECKPOINT)

    # 2. Save to temp file
    print("\n2. Save to temp file")
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp.close()
    save_model(m_orig, config, tmp.name)

    # 3. Reload from temp
    print("\n3. Reload from temp file")
    m_reload, _ = load_model(tmp.name)

    # 4. Compare params
    print("\n4. Compare parameters (orig vs reload)")
    compare_params(m_orig, m_reload, "save/reload")

    # 5. Compare buffers
    print("\n5. Compare buffers (orig vs reload)")
    compare_buffers(m_orig, m_reload, "save/reload")

    # 6. Warm up caches by running prefill, then compare again
    print("\n6. Warm up caches (run prefill on both)")
    with torch.no_grad():
        m_orig.cuda().prefill(torch.tensor([[1, 2, 3]], device="cuda"))
        m_reload.cuda().prefill(torch.tensor([[1, 2, 3]], device="cuda"))
    m_orig.cpu()
    m_reload.cpu()
    compare_buffers(m_orig, m_reload, "post-warmup")

    # 7. Compare logits
    print("\n7. Compare logits")
    compare_logits(m_orig, m_reload)

    # 8. Compare generation
    print("\n8. Compare generation (greedy)")
    compare_generation(m_orig, m_reload)

    # 9. KEY TEST: generate → save → reload → compare
    print("\n9. POST-GENERATION save/reload (does generation leave dirty state?)")
    # Reload a fresh copy for this test (m_orig may have been warmed up)
    m_for_gen, config_gen = load_model(CHECKPOINT)
    test_post_generation_save_reload(m_for_gen, config_gen)

    # 10. Double round-trip: save reload → save again → reload again
    print("\n10. Double round-trip (save → reload → save → reload)")
    tmp2 = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp2.close()
    save_model(m_reload, config, tmp2.name)
    m_reload2, _ = load_model(tmp2.name)
    compare_params(m_reload, m_reload2, "double round-trip")

    # 11. Test: does eval() matter?
    print("\n11. train() vs eval() logit diff")
    m_train = m_orig
    m_train.train()
    m_eval = m_reload
    m_eval.eval()
    compare_logits(m_train, m_eval)

    # Cleanup
    os.unlink(tmp.name)
    os.unlink(tmp2.name)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
