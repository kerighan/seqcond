"""
Generator for SeqCond PyTorch model.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List
from .model import SeqCondModel
from seqcond.dataset import Tokenizer


class TorchGenerator:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.device = device
        self.config = checkpoint["config"]
        self.tokenizer = Tokenizer()

        self.model = SeqCondModel(**self.config).to(device).eval()
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)

        print(
            f"Model loaded: {self.config['num_layers']} layers, d_model={self.config['d_model']}"
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        no_repeat_ngram_size: int = 0,
        verbose: bool = True,
        use_cuda_graph: bool = False,  # Not implemented yet
    ) -> str:
        tokens = self.tokenizer([prompt])[0]

        # Initialize state
        states = self.model.init_state(batch_size=1, device=self.device)

        # Process prompt
        for token_id in tokens:
            token_tensor = torch.tensor([[token_id]], device=self.device)
            logits, states = self.model.step(token_tensor, states)

        # Track generated tokens for penalties
        generated = list(tokens)
        token_counts = {}
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1

        # Generate
        for _ in range(max_new_tokens):
            # Apply temperature
            if temperature > 0:
                logits_scaled = logits[0] / temperature
            else:
                logits_scaled = logits[0]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated):
                    logits_scaled[token_id] /= repetition_penalty

            # Apply frequency penalty
            if frequency_penalty > 0:
                for token_id, count in token_counts.items():
                    logits_scaled[token_id] -= frequency_penalty * count

            # Apply no_repeat_ngram
            if no_repeat_ngram_size > 0 and len(generated) >= no_repeat_ngram_size:
                ngram = tuple(generated[-(no_repeat_ngram_size - 1) :])
                for i in range(len(generated) - no_repeat_ngram_size + 1):
                    if tuple(generated[i : i + no_repeat_ngram_size - 1]) == ngram:
                        blocked_token = generated[i + no_repeat_ngram_size - 1]
                        logits_scaled[blocked_token] = float("-inf")

            # Sample
            if temperature == 0:
                next_token = torch.argmax(logits_scaled).item()
            else:
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = (
                        logits_scaled < torch.topk(logits_scaled, top_k)[0][-1]
                    )
                    logits_scaled[indices_to_remove] = float("-inf")

                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        logits_scaled, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits_scaled[indices_to_remove] = float("-inf")

                probs = F.softmax(logits_scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)
            token_counts[next_token] = token_counts.get(next_token, 0) + 1

            if verbose:
                print(self.tokenizer.decode([next_token]), end="", flush=True)

            # Next step
            token_tensor = torch.tensor([[next_token]], device=self.device)
            logits, states = self.model.step(token_tensor, states)

        return self.tokenizer.decode(generated)
