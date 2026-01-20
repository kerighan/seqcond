"""
Efficient text generation using step-by-step decoding.

This module provides a Generator class that uses O(1) step decoding for SeqCond blocks
and O(L) step decoding with KV cache for Transformer blocks, instead of recomputing
the full forward pass at each token.
"""

import pickle
from typing import Optional, Any, Tuple, List
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .model import SeqCondModel
from .rope import precompute_freqs, get_rope_embeddings
from ..dataset import Tokenizer
from ..config import ModelConfig


@dataclass
class GeneratorConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    seed: Optional[int] = None


class Generator:
    """
    Efficient text generator using step-by-step decoding.

    Uses O(1) step decoding for SeqCond blocks and O(L) KV cache for Transformer blocks.
    """

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer: Optional[Tokenizer] = None,
    ):
        """
        Initialize generator from a checkpoint.

        Args:
            checkpoint_path: Path to the .pkl checkpoint file
            tokenizer: Optional tokenizer (uses default if not provided)
        """
        self.tokenizer = tokenizer or Tokenizer()

        # Load checkpoint
        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)

        self.params = data["params"]
        self.config_dict = data["config"]
        self.step = data.get("step", 0)

        # Reconstruct model config
        model_config_dict = self.config_dict["model"]
        self.model_config = ModelConfig(**model_config_dict)

        # Create model
        self.model = SeqCondModel(
            vocab_size=self.model_config.vocab_size,
            d_model=self.model_config.d_model,
            num_layers=self.model_config.num_layers,
            num_heads=self.model_config.num_heads,
            num_kv_heads=self.model_config.num_kv_heads,
            d_ff=self.model_config.d_ff,
            maxlen=self.model_config.maxlen,
            dropout=0.0,  # No dropout during inference
            tie_weights=self.model_config.tie_weights,
            qk_norm=self.model_config.qk_norm,
            qk_norm_eps=self.model_config.qk_norm_eps,
            seqcond_heads=self.model_config.seqcond_heads,
            num_query_heads=self.model_config.num_query_heads,
            num_thetas=self.model_config.num_thetas,
            derivative_order=self.model_config.derivative_order,
            num_anchor_heads=self.model_config.num_anchor_heads,
            conv_kernel_size=self.model_config.conv_kernel_size,
            expand_factor=self.model_config.expand_factor,
            out_expand_factor=self.model_config.out_expand_factor,
            seqcond_ratio=self.model_config.seqcond_ratio,
            use_square_matrix=self.model_config.use_square_matrix,
            remat=False,  # No remat during inference
        )

        # Precompute RoPE embeddings
        self.cos_emb, self.sin_emb = precompute_freqs(
            self.model_config.maxlen,
            self.model_config.d_model // self.model_config.num_heads,
        )

        # Cache block info for step generation
        self._analyze_blocks()

        print(f"Generator initialized from checkpoint step {self.step}")
        print(
            f"Model: {self.model_config.num_layers} layers, d_model={self.model_config.d_model}"
        )
        print(f"Block pattern: {self._block_pattern}")

    def _analyze_blocks(self):
        """Analyze the block structure for state initialization."""
        self._block_pattern = []
        self._num_seqcond_blocks = 0
        self._num_transformer_blocks = 0

        for i in range(self.model_config.num_layers):
            if (i + 1) % (self.model_config.seqcond_ratio + 1) == 0:
                self._block_pattern.append("transformer")
                self._num_transformer_blocks += 1
            else:
                self._block_pattern.append("seqcond")
                self._num_seqcond_blocks += 1

    def _init_seqcond_state(self, batch_size: int) -> Tuple:
        """Initialize state for a SeqCond block."""
        mc = self.model_config
        num_heads = mc.seqcond_heads or mc.num_heads
        num_query_heads = mc.num_query_heads or num_heads
        num_thetas = mc.num_thetas

        d_inner = int(mc.d_model * mc.expand_factor)
        H = max(1, d_inner // (num_heads * num_thetas))
        conv_kernel_size = mc.conv_kernel_size

        dim_memory = num_heads * H
        dim_query_head = H * num_thetas * 2
        dim_query_total = num_query_heads * dim_query_head
        dim_mem_total = dim_memory + num_heads
        dim_conv_total = dim_mem_total + dim_query_total

        den_acc = jnp.zeros((batch_size, num_heads), dtype=jnp.float32)
        re_acc = jnp.zeros((batch_size, num_heads, H, num_thetas), dtype=jnp.float32)
        im_acc = jnp.zeros((batch_size, num_heads, H, num_thetas), dtype=jnp.float32)
        pos = jnp.zeros((batch_size,), dtype=jnp.int32)
        conv_buffer = jnp.zeros(
            (batch_size, conv_kernel_size - 1, dim_conv_total), dtype=jnp.float32
        )

        return (den_acc, re_acc, im_acc, pos, conv_buffer)

    def _init_transformer_state(self, batch_size: int) -> Tuple:
        """Initialize KV cache for a Transformer block."""
        mc = self.model_config
        num_kv_heads = mc.num_kv_heads or mc.num_heads
        head_dim = mc.d_model // mc.num_heads

        # Pre-allocate cache to maxlen
        k_cache = jnp.zeros(
            (batch_size, mc.maxlen, num_kv_heads, head_dim), dtype=jnp.bfloat16
        )
        v_cache = jnp.zeros(
            (batch_size, mc.maxlen, num_kv_heads, head_dim), dtype=jnp.bfloat16
        )

        return (k_cache, v_cache)

    def _init_all_states(self, batch_size: int) -> List[Tuple]:
        """Initialize states for all blocks."""
        states = []
        for block_type in self._block_pattern:
            if block_type == "seqcond":
                states.append(self._init_seqcond_state(batch_size))
            else:
                states.append(self._init_transformer_state(batch_size))
        return states

    def _prefill(self, tokens: List[int]) -> Tuple[jnp.ndarray, List[Tuple], int]:
        """
        Prefill: run full forward pass on prompt to get initial states.

        Returns:
            logits: Logits for the last token
            states: Updated states for all blocks
            pos: Current position after prefill
        """
        batch_size = 1
        seq_len = len(tokens)

        # Pad to power of 2 for efficiency
        padded_len = 1 << (seq_len - 1).bit_length()
        padded_len = min(padded_len, self.model_config.maxlen)
        padded = tokens + [0] * (padded_len - seq_len)

        input_array = jnp.array([padded], dtype=jnp.int32)

        # Full forward pass
        logits = self.model.apply(
            {"params": self.params}, input_array, deterministic=True
        )

        # For now, we need to also run step-by-step to build up the states
        # This is a limitation - ideally we'd extract states from the forward pass
        states = self._init_all_states(batch_size)

        # Run step-by-step on the prompt to build states
        x = self.model.embedding.apply(
            {"params": self.params["token_embedding"]},
            jnp.array([tokens], dtype=jnp.int32),
        )

        # Process each token through each block
        for t in range(seq_len):
            x_t = x[:, t, :]  # (1, d_model)

            # Get RoPE for this position
            cos_t, sin_t = get_rope_embeddings(
                1, self.cos_emb, self.sin_emb, 1, self.model_config.num_heads
            )
            # Shift to correct position
            cos_t = self.cos_emb[t : t + 1, :][None, :, None, :]
            sin_t = self.sin_emb[t : t + 1, :][None, :, None, :]
            cos_t = jnp.broadcast_to(
                cos_t, (1, 1, self.model_config.num_heads, cos_t.shape[-1])
            )
            sin_t = jnp.broadcast_to(
                sin_t, (1, 1, self.model_config.num_heads, sin_t.shape[-1])
            )

            new_states = []
            seqcond_idx = 0
            transformer_idx = 0

            for i, block_type in enumerate(self._block_pattern):
                if block_type == "seqcond":
                    block_name = f"seqcond_block_{seqcond_idx}"
                    block_params = self.params[block_name]

                    # Apply RMSNorm first (matching SeqCondBlock.__call__)
                    h = self._apply_rms_norm(x_t, block_params["RMSNorm_0"])

                    # Step through SeqCondAttention
                    h, new_state = self._seqcond_step(
                        h, states[i], block_params["SeqCondAttention_0"]
                    )
                    x_t = x_t + h  # Residual
                    new_states.append(new_state)
                    seqcond_idx += 1
                else:
                    block_name = f"transformer_block_{transformer_idx}"
                    block_params = self.params[block_name]

                    x_t, new_state = self._transformer_step(
                        x_t, states[i], t, cos_t, sin_t, block_params
                    )
                    new_states.append(new_state)
                    transformer_idx += 1

            states = new_states

        return logits[0, seq_len - 1, :], states, seq_len

    def _apply_rms_norm(self, x: jnp.ndarray, params: dict) -> jnp.ndarray:
        """Apply RMSNorm manually."""
        scale = params["scale"]
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(x_f32**2, axis=-1, keepdims=True)
        x_normed = x_f32 * jax.lax.rsqrt(variance + 1e-6)
        return (x_normed * scale).astype(x.dtype)

    def _seqcond_step(
        self, x_t: jnp.ndarray, state: Tuple, params: dict
    ) -> Tuple[jnp.ndarray, Tuple]:
        """Run one step of SeqCondAttention."""
        # This would need to call the actual step method
        # For now, this is a placeholder - the actual implementation
        # would need to match seqcond_fast.py step() exactly
        raise NotImplementedError("SeqCond step not yet integrated")

    def _transformer_step(
        self,
        x_t: jnp.ndarray,
        kv_cache: Tuple,
        pos: int,
        cos_t: jnp.ndarray,
        sin_t: jnp.ndarray,
        params: dict,
    ) -> Tuple[jnp.ndarray, Tuple]:
        """Run one step of TransformerDecoderBlock."""
        # This would need to call the actual step method
        # For now, this is a placeholder
        raise NotImplementedError("Transformer step not yet integrated")

    def generate(
        self,
        prompt: str,
        config: Optional[GeneratorConfig] = None,
        verbose: bool = True,
    ) -> str:
        """
        Generate text from a prompt using efficient step-by-step decoding.

        Args:
            prompt: Text prompt to continue
            config: Generation configuration
            verbose: Print tokens as they are generated

        Returns:
            Generated text including the prompt
        """
        config = config or GeneratorConfig()

        if config.seed is not None:
            np.random.seed(config.seed)

        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt)

        # Prefill
        if verbose:
            print(f"Prefilling {len(tokens)} tokens...")

        logits, states, pos = self._prefill(tokens)

        # Generate tokens
        generated_tokens = []
        text = prompt

        for _ in range(config.max_new_tokens):
            # Sample next token
            logits_np = np.array(logits, dtype=np.float64)

            # Apply repetition penalty
            if config.repetition_penalty != 1.0 and generated_tokens:
                for token_id in set(generated_tokens):
                    if logits_np[token_id] > 0:
                        logits_np[token_id] /= config.repetition_penalty
                    else:
                        logits_np[token_id] *= config.repetition_penalty

            # Apply temperature
            logits_np = logits_np / config.temperature

            # Convert to probabilities
            probs = np.exp(logits_np - logits_np.max())
            probs = probs / probs.sum()

            # Top-k filtering
            if config.top_k > 0:
                top_k_idx = np.argpartition(probs, -config.top_k)[-config.top_k :]
                mask = np.zeros_like(probs)
                mask[top_k_idx] = 1
                probs = probs * mask
                probs = probs / probs.sum()

            # Top-p (nucleus) filtering
            if config.top_p < 1.0:
                sorted_idx = np.argsort(probs)[::-1]
                cumsum = np.cumsum(probs[sorted_idx])
                cutoff_idx = np.searchsorted(cumsum, config.top_p) + 1
                mask = np.zeros_like(probs)
                mask[sorted_idx[:cutoff_idx]] = 1
                probs = probs * mask
                probs = probs / probs.sum()

            # Sample
            token_id = int(np.random.choice(len(probs), p=probs))
            token = self.tokenizer.decode([token_id])

            tokens.append(token_id)
            generated_tokens.append(token_id)
            text += token

            if verbose:
                print(token, end="", flush=True)

            # Check max length
            if pos >= self.model_config.maxlen - 1:
                break

            # Step through model for next token
            # TODO: Implement actual step-by-step generation
            # For now, fall back to full forward pass
            logits, states, pos = self._step_all_blocks(token_id, states, pos)

        if verbose:
            print()

        return text

    def _step_all_blocks(
        self, token_id: int, states: List[Tuple], pos: int
    ) -> Tuple[jnp.ndarray, List[Tuple], int]:
        """
        Step through all blocks for a single token.

        This is the core of efficient generation - O(1) for SeqCond, O(L) for Transformer.
        """
        # For now, fall back to full forward pass
        # TODO: Implement actual step-by-step generation
        raise NotImplementedError(
            "Step-by-step generation not yet fully implemented. "
            "Use callback.generate_text() for now."
        )
