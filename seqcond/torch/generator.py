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
        self.eos_token_id = self.tokenizer.encode("<|im_end|>")[0]
        self.think_start_id = self.tokenizer.encode("<|think_start|>")[0]
        self.think_end_id = self.tokenizer.encode("<|think_end|>")[0]
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
        # self._seq_lens = [16, 64, 256, 1024]  # Power-of-4 lengths
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
        # print(f"Capturing graph for seq_len={seq_len}...")

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
        use_synth_template: bool = True,
        max_thinking_tokens: Optional[int] = None,
    ) -> str:

        if use_synth_template:
            prompt = "<|im_start|>user\n" + prompt + "\n<|im_end|>"
            prompt += "<|im_start|>assistant\n<|think_start|>"
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

        # Track thinking tokens for budget
        in_thinking = use_synth_template  # We start inside <|think_start|>
        thinking_tokens = 0
        think_end_injected = False

        # Generate
        # if verbose:
        #     print(prompt, end="", flush=True)
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

            # Track thinking state
            if next_token == self.think_end_id:
                in_thinking = False
            elif next_token == self.think_start_id:
                in_thinking = True
                thinking_tokens = 0

            if in_thinking:
                thinking_tokens += 1

            # Inject <|think_end|> if thinking budget exhausted
            if (
                max_thinking_tokens is not None
                and in_thinking
                and thinking_tokens >= max_thinking_tokens
                and not think_end_injected
            ):
                next_token = self.think_end_id
                in_thinking = False
                think_end_injected = True

            generated.append(next_token)
            token_counts[next_token] = token_counts.get(next_token, 0) + 1

            # if verbose:
            try:
                # print(self.tokenizer.decode([next_token]), end="", flush=True)
                yield self.tokenizer.decode([next_token])
            except Exception as e:
                print(f"Error decoding token: {e}")

            # Stop if we hit the end token
            if next_token == self.eos_token_id:
                break

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

        try:
            return self.tokenizer.decode(generated)
        except Exception as e:
            print(f"Error decoding token: {e}")

    @staticmethod
    def _stack_states(all_states):
        """Stack states from multiple individual prefills into a single batched state.

        Each element of all_states is a list of block states from one prefill (B=1).
        Returns a single list of block states with B = len(all_states).
        """
        num_blocks = len(all_states[0])
        batched = []
        for block_idx in range(num_blocks):
            block_state_tuple = tuple(
                torch.cat([s[block_idx][tensor_idx] for s in all_states], dim=0)
                for tensor_idx in range(len(all_states[0][block_idx]))
            )
            batched.append(block_state_tuple)
        return batched

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        use_synth_template: bool = True,
        max_thinking_tokens: Optional[int] = None,
        output_constraints: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate completions for a batch of prompts in parallel.

        Prefills each prompt individually (no padding noise in states),
        stacks the states, then decodes in lockstep with per-sample positions.

        Args:
            output_constraints: If provided, after <|think_end|> the model is
                forced to emit one of these strings (e.g. ["A.", "B.", "C.", "D."])
                by masking logits at each position. Free generation resumes after.

        Returns a list of generated strings (completion only, no prompt).
        """
        B = len(prompts)
        if B == 0:
            return []

        # Tokenize all prompts
        if use_synth_template:
            prompts = [
                "<|im_start|>user\n"
                + p
                + "\n<|im_end|><|im_start|>assistant\n<|think_start|>"
                for p in prompts
            ]
        all_tokens = self.tokenizer(prompts)
        prompt_lens = [len(t) for t in all_tokens]

        model_maxlen = getattr(self.model, "maxlen", 2048)

        # Skip samples whose prompts exceed maxlen (can't prefill)
        skipped = [False] * B
        for i in range(B):
            if prompt_lens[i] >= model_maxlen:
                skipped[i] = True

        active_indices = [i for i in range(B) if not skipped[i]]
        if not active_indices:
            return [""] * B

        max_prompt_len = max(prompt_lens[i] for i in active_indices)

        # Clamp max_new_tokens so longest active prompt doesn't exceed maxlen
        max_new_tokens = min(max_new_tokens, model_maxlen - max_prompt_len - 1)
        if max_new_tokens <= 0:
            return [""] * B

        # Per-sample thinking budget (absolute position semantics):
        # max_thinking_tokens = total position limit (prompt + thinking).
        # Thinking budget for sample i = max(0, max_thinking_tokens - prompt_lens[i])
        # Also clamped by (max_new_tokens - margin) so think_end fires before loop ends.
        ANSWER_MARGIN = 20
        per_sample_max_thinking = [None] * B
        if max_thinking_tokens is not None:
            for i in range(B):
                absolute_budget = max(0, max_thinking_tokens - prompt_lens[i])
                loop_budget = max(0, max_new_tokens - ANSWER_MARGIN)
                per_sample_max_thinking[i] = min(absolute_budget, loop_budget)

        # Individual prefill for each prompt (no padding, clean states)
        all_logits = []
        all_states = []
        for i, tokens in enumerate(all_tokens):
            if skipped[i]:
                dummy = torch.tensor([[self.eos_token_id]], device=self.device)
                logits_i, states_i = self.model.prefill(dummy)
                all_logits.append(logits_i.squeeze(1))
                all_states.append(states_i)
            else:
                input_ids = torch.tensor([tokens], device=self.device)
                logits_i, states_i = self.model.prefill(input_ids)
                all_logits.append(logits_i.squeeze(1))
                all_states.append(states_i)

        # Stack into batched tensors
        logits = torch.cat(all_logits, dim=0)  # (B, vocab_size)
        states = self._stack_states(all_states)

        # Pre-tokenize output constraints (once)
        constraint_token_seqs = None
        if output_constraints:
            constraint_token_seqs = [
                self.tokenizer.encode("\n" + c) for c in output_constraints
            ]

        # Track generated tokens per sample and finished status
        generated = [[] for _ in range(B)]
        finished = [skipped[i] for i in range(B)]
        thinking_counts = [0] * B
        in_thinking = [use_synth_template] * B  # start inside <|think_start|>
        think_end_injected = [False] * B
        # Constraint state: position within constraint prefix (-1 = not active, >=0 = forcing)
        constraint_pos = [-1] * B
        # Which constraints are still viable for each sample
        constraint_viable = [None] * B
        token_tensor = torch.zeros((B, 1), dtype=torch.long, device=self.device)

        for step_idx in range(max_new_tokens):
            # Apply constraint masking before sampling
            if constraint_token_seqs is not None:
                for i in range(B):
                    pos = constraint_pos[i]
                    if pos >= 0 and not finished[i]:
                        # Mask logits: only allow tokens that are valid at this position
                        viable = constraint_viable[i]
                        allowed_tokens = set()
                        for ci in viable:
                            seq = constraint_token_seqs[ci]
                            if pos < len(seq):
                                allowed_tokens.add(seq[pos])
                        if allowed_tokens:
                            mask = torch.full_like(logits[i], float("-inf"))
                            for t in allowed_tokens:
                                mask[t] = 0.0
                            logits[i] = logits[i] + mask

            # Greedy or temperature sampling
            if temperature == 0:
                next_tokens = torch.argmax(logits, dim=-1)  # (B,)
            else:
                logits_scaled = logits / temperature
                probs = F.softmax(logits_scaled, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Record tokens and check EOS
            for i in range(B):
                if not finished[i]:
                    tok = next_tokens[i].item()

                    # Track thinking state
                    if tok == self.think_end_id:
                        in_thinking[i] = False
                        # Activate constraint forcing on the next token
                        if constraint_token_seqs is not None and constraint_pos[i] < 0:
                            constraint_pos[i] = 0
                            constraint_viable[i] = list(
                                range(len(constraint_token_seqs))
                            )
                    elif tok == self.think_start_id:
                        in_thinking[i] = True
                        thinking_counts[i] = 0

                    if in_thinking[i]:
                        thinking_counts[i] += 1

                    # Inject <|think_end|> if thinking budget exhausted
                    if (
                        per_sample_max_thinking[i] is not None
                        and in_thinking[i]
                        and thinking_counts[i] >= per_sample_max_thinking[i]
                        and not think_end_injected[i]
                    ):
                        tok = self.think_end_id
                        next_tokens[i] = tok
                        in_thinking[i] = False
                        think_end_injected[i] = True
                        # Activate constraint forcing on the next token
                        if constraint_token_seqs is not None and constraint_pos[i] < 0:
                            constraint_pos[i] = 0
                            constraint_viable[i] = list(
                                range(len(constraint_token_seqs))
                            )

                    # Advance constraint state
                    if constraint_pos[i] >= 0 and tok != self.think_end_id:
                        # Narrow viable set based on emitted token
                        pos = constraint_pos[i]
                        constraint_viable[i] = [
                            ci
                            for ci in constraint_viable[i]
                            if pos < len(constraint_token_seqs[ci])
                            and constraint_token_seqs[ci][pos] == tok
                        ]
                        constraint_pos[i] = pos + 1
                        # Check if any constraint is fully emitted
                        if not constraint_viable[i] or all(
                            constraint_pos[i] >= len(constraint_token_seqs[ci])
                            for ci in constraint_viable[i]
                        ):
                            constraint_pos[i] = -1  # done, resume free generation

                    generated[i].append(tok)
                    if tok == self.eos_token_id:
                        finished[i] = True

            if all(finished):
                break

            # For finished samples, feed EOS to keep states in sync
            next_tokens[torch.tensor(finished, device=self.device)] = self.eos_token_id
            token_tensor[:, 0] = next_tokens

            logits, states = self.model.step(token_tensor, states)

        # Decode each sample's generated tokens (strip trailing EOS)
        results = []
        for i in range(B):
            toks = generated[i]
            if toks and toks[-1] == self.eos_token_id:
                toks = toks[:-1]
            try:
                results.append(self.tokenizer.decode(toks))
            except Exception:
                results.append("")
        return results
