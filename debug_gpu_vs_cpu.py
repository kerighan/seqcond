"""
Compare generation on GPU vs CPU to check for precision divergence.
Uses the EXACT same code path as generate.py (TorchGenerator).
"""

import torch
from seqcond.torch.model import SeqCondModel
from seqcond.dataset import Tokenizer

CKPT = "checkpoints/seqcond_torch_762k.pt"
# Use the exact same template as TorchGenerator
PROMPT_RAW = "What is 2+2?"
PROMPT = (
    "<|im_start|>user\n"
    + PROMPT_RAW
    + "\n<|im_end|><|im_start|>assistant\n<|think_start|>"
)


def load_model(device):
    checkpoint = torch.load(CKPT, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = SeqCondModel(**config).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model, config


@torch.no_grad()
def generate_greedy(model, tokenizer, prompt, device, n_tokens=20):
    tokens = tokenizer([prompt])[0]
    input_ids = torch.tensor([tokens], device=device)

    logits, states = model.prefill(input_ids)
    logits = logits.squeeze(1)

    generated = []
    token_tensor = torch.zeros((1, 1), dtype=torch.long, device=device)
    eos = tokenizer.encode("<|im_end|>")[0]

    for i in range(n_tokens):
        next_token = torch.argmax(logits[0]).item()
        generated.append(next_token)

        if next_token == eos:
            break

        token_tensor[0, 0] = next_token
        logits, states = model.step(token_tensor, states)

        if i < 5:
            top5 = torch.topk(logits[0], 5)
            print(
                f"  Token {i}: {next_token:6d} = '{tokenizer.decode([next_token])}' | "
                f"top5_vals={[f'{v:.4f}' for v in top5.values.tolist()]} "
                f"top5_ids={top5.indices.tolist()}"
            )

    return generated


def main():
    tokenizer = Tokenizer()
    print(f"Prompt: {PROMPT}")

    if not torch.cuda.is_available():
        print("No GPU available")
        return

    # GPU with CUDA graph (via TorchGenerator)
    print("\n=== GPU (CUDA graph via TorchGenerator) ===")
    from seqcond.torch.generator import TorchGenerator

    gen_obj = TorchGenerator(CKPT, device="cuda")
    gen_obj.precompute(max_seq_len=4096)

    output_cg = []
    for tok_text in gen_obj.generate(
        PROMPT_RAW,
        max_new_tokens=30,
        temperature=0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        use_cuda_graph=True,
        use_synth_template=True,
    ):
        output_cg.append(tok_text)
    text_cg = "".join(output_cg)
    print(f"Generated ({len(output_cg)} toks): {text_cg}")

    # No CUDA graph via TorchGenerator
    print("\n=== GPU (no CUDA graph via TorchGenerator) ===")
    output_nocg = []
    for tok_text in gen_obj.generate(
        PROMPT_RAW,
        max_new_tokens=30,
        temperature=0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        use_cuda_graph=False,
        use_synth_template=True,
    ):
        output_nocg.append(tok_text)
    text_nocg = "".join(output_nocg)
    print(f"Generated ({len(output_nocg)} toks): {text_nocg}")

    # Compare
    print("\n=== COMPARISON ===")
    print(f"CUDA graph:    '{text_cg[:120]}'")
    print(f"No CUDA graph: '{text_nocg[:120]}'")
    print(f"Match: {text_cg == text_nocg}")


if __name__ == "__main__":
    main()
