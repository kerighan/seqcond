"""
Generator for SeqCond PyTorch model.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List
from .model import SeqCondModel
from seqcond.dataset import Tokenizer


class TorchGenerator:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        compile: bool = False,  # Disabled by default - dynamic shapes make it slower
        dtype: str = "float32",
    ):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.device = device
        self.config = checkpoint["config"]
        self.tokenizer = Tokenizer()
        self.compile = compile

        # Set dtype
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        # Load model (FP32 only for now - FP16 requires deeper changes)
        self.model = SeqCondModel(**self.config).to(device).eval()
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)

        if compile and hasattr(torch, "compile"):
            print("Compiling model with torch.compile()...")
            # Use default mode - reduce-overhead requires static shapes
            self.model.step = torch.compile(self.model.step, dynamic=True)

        print(
            f"Model loaded: {self.config['num_layers']} layers, d_model={self.config['d_model']}, dtype={self.dtype}"
        )

        # CUDA graph state - power-of-2 multi-graph system
        self._graphs = {}  # seq_len -> CUDAGraph
        self._static_logits_dict = {}  # seq_len -> logits tensor
        self._static_token = None
        self._static_states = None
        self._seq_lens = [8, 16, 32, 64, 128, 256, 512, 1024]  # Power-of-2 lengths
        self._use_triton = False  # Whether to use Triton kernels in CUDA graphs

    def _get_quantized_seq_len(self, pos: int) -> int:
        """Get the smallest power-of-2 seq_len that covers pos+1."""
        needed = pos + 1  # Need to cover positions 0..pos
        for seq_len in self._seq_lens:
            if seq_len >= needed:
                return seq_len
        return self._seq_lens[-1]  # Fallback to max

    def _capture_graph_for_seq_len(self, seq_len: int):
        """Capture CUDA graph for a specific seq_len, preserving states."""
        print(f"Capturing graph for seq_len={seq_len}...")

        # Save current states (deep copy of tensor values)
        saved = self.model.init_state(1, device=self.device)
        self._copy_states(self._static_states, saved)

        # Warmup runs (will corrupt states)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                logits, _ = self.model.step(
                    self._static_token,
                    self._static_states,
                    seq_len=seq_len,
                    use_triton=self._use_triton,
                )
        torch.cuda.current_stream().wait_stream(s)

        # Restore states before capture
        self._copy_states(saved, self._static_states)

        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits, _ = self.model.step(
                self._static_token,
                self._static_states,
                seq_len=seq_len,
                use_triton=self._use_triton,
            )

        # Restore states after capture
        self._copy_states(saved, self._static_states)

        self._graphs[seq_len] = graph
        self._static_logits_dict[seq_len] = logits

    def _copy_states(self, src, dst):
        """Copy state tensors from src to dst."""
        for s, d in zip(src, dst):
            for st, dt in zip(s, d):
                dt.copy_(st)

    @torch.no_grad()
    def precompute(self, max_seq_len: int = 1024, use_triton: bool = False):
        """Pre-capture all CUDA graphs up to max_seq_len.

        Call this after loading the model to eliminate graph capture
        overhead from the first real generation.
        """
        self._use_triton = use_triton
        backend = "Triton" if use_triton else "PyTorch"
        print(
            f"Pre-capturing CUDA graphs up to seq_len={max_seq_len} (backend: {backend})..."
        )

        # Initialize static tensors
        self._static_token = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self._static_states = self.model.init_state(1, device=self.device)

        # Capture graphs for all relevant seq_lens
        for seq_len in self._seq_lens:
            if seq_len > max_seq_len:
                break
            self._capture_graph_for_seq_len(seq_len)

        print(
            f"Pre-captured {len(self._graphs)} CUDA graphs. Ready for fast generation!"
        )

    @torch.no_grad()
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
        use_cuda_graph: bool = False,
        use_triton: bool = False,
    ) -> str:
        tokens = self.tokenizer([prompt])[0]

        # Flag to indicate new generation (need to copy prefill states)
        self._new_generation = True

        # Fast prefill
        input_ids = torch.tensor([tokens], device=self.device)
        logits, states = self.model.prefill(input_ids)
        logits = logits.squeeze(1)  # (B, vocab_size)

        # Pre-allocate token tensor for generation loop
        token_tensor = torch.zeros((1, 1), dtype=torch.long, device=self.device)

        # Track generated tokens for penalties
        generated = list(tokens)
        token_counts = {}
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1

        # Generate
        if verbose:
            print(prompt, end="", flush=True)
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
                try:
                    print(self.tokenizer.decode([next_token]), end="", flush=True)
                except Exception as e:
                    # print(f"Error decoding token: {e}")
                    next_token = 12

            # Next step
            token_tensor[0, 0] = next_token

            if use_cuda_graph:
                # Initialize static tensors on first use
                if self._static_token is None:
                    self._static_token = token_tensor.clone()
                    self._static_states = self.model.init_state(1, device=self.device)

                # Copy prefill states at start of new generation
                if self._new_generation:
                    self._copy_states(states, self._static_states)
                    self._new_generation = False

                # Get current position and quantized seq_len
                current_pos = len(generated) - 1  # Position we're generating for
                seq_len = self._get_quantized_seq_len(current_pos)

                # Capture graph for this seq_len if needed
                if seq_len not in self._graphs:
                    self._capture_graph_for_seq_len(seq_len)

                # Copy token to static tensor and replay appropriate graph
                self._static_token.copy_(token_tensor)
                self._graphs[seq_len].replay()
                logits = self._static_logits_dict[seq_len]
            else:
                logits, states = self.model.step(
                    token_tensor, states, use_triton=use_triton
                )

        return self.tokenizer.decode(generated)
