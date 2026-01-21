import torch
import time
import numpy as np
from seqcond.torch.model import SeqCondModel
from seqcond.dataset import Tokenizer
from seqcond.config import ModelConfig


class TorchGenerator:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = Tokenizer()

        print(f"Loading Torch checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config_dict = checkpoint["config"]
        state_dict = checkpoint["state_dict"]

        # Reconstruct model config
        model_config = ModelConfig(**config_dict["model"])
        self.model_config = model_config

        # Create model
        self.model = SeqCondModel(model_config).to(device).eval()

        # Load weights
        # We need to handle potential float/bfloat16 conversion
        self.model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})

        print(
            f"Generator initialized. Model: {model_config.num_layers} layers, d_model={model_config.d_model}"
        )

        self.graph = None
        self.static_token_id = None
        self.static_states = None
        self.static_logits = None

    def _copy_states(self, src_states, dst_states):
        """Copy values from src_states to dst_states in-place."""
        for i, (src, dst) in enumerate(zip(src_states, dst_states)):
            # src/dst are tuples of tensors
            for s_tensor, d_tensor in zip(src, dst):
                d_tensor.copy_(s_tensor)

    def capture_graph(self, batch_size=1):
        """Capture CUDA graph for a single decoding step."""
        print("Capturing CUDA graph...")
        self.static_token_id = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=self.device
        )
        self.static_states = self.model.init_state(batch_size, device=self.device)

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.model.step(self.static_token_id, self.static_states)
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_logits, _ = self.model.step(
                self.static_token_id, self.static_states
            )

        print("CUDA graph captured.")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        verbose: bool = True,
        use_cuda_graph: bool = True,
    ):
        tokens = self.tokenizer([prompt])[0]
        input_ids = torch.tensor([tokens], device=self.device)

        batch_size = 1
        states = self.model.init_state(batch_size, device=self.device)

        # Prefill
        t0 = time.time()
        logits = None
        for i in range(len(tokens)):
            token_id = input_ids[:, i : i + 1]
            logits, states = self.model.step(token_id, states)

        torch.cuda.synchronize()
        prefill_time = time.time() - t0
        if verbose:
            print(
                f"Prefill done in {prefill_time:.3f}s ({prefill_time/len(tokens)*1000:.1f}ms/token)"
            )
            print(prompt, end="", flush=True)

        # Generation
        generated_tokens = []
        gen_start_time = time.time()

        # Current token is the last one from prefill
        curr_token_id = torch.argmax(logits, dim=-1, keepdim=True)

        # Initialize graph if needed and not present
        if use_cuda_graph and self.graph is None:
            self.capture_graph(batch_size)

        # If using graph, copy current state to static state
        if use_cuda_graph:
            self._copy_states(states, self.static_states)

        for i in range(max_new_tokens):
            token_str = self.tokenizer.decode([curr_token_id.item()])
            generated_tokens.append(curr_token_id.item())
            if verbose:
                print(token_str, end="", flush=True)

            if use_cuda_graph:
                self.static_token_id.copy_(curr_token_id)
                self.graph.replay()
                logits = self.static_logits
                # static_states are updated in-place by the graph replay
            else:
                logits, states = self.model.step(curr_token_id, states)

            if temperature == 0:
                curr_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                curr_token_id = torch.multinomial(probs, num_samples=1)

        torch.cuda.synchronize()
        gen_time = time.time() - gen_start_time
        if verbose:
            print(
                f"\nGeneration done in {gen_time:.3f}s ({gen_time/max_new_tokens*1000:.1f}ms/token)"
            )

        return prompt + self.tokenizer.decode(generated_tokens)


if __name__ == "__main__":
    import sys

    checkpoint_path = "checkpoints/seqcond_torch.pt"
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]

    gen = TorchGenerator(checkpoint_path)
    prompt = "The quick brown fox"
    print(f"\nPrompt: {prompt}\n")
    gen.generate(prompt, max_new_tokens=32, temperature=0.0)
