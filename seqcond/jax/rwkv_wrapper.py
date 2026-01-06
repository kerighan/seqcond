"""
Flax wrapper for RWKV to make it compatible with the training pipeline.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional

from .rwkv import AssociativeScanRWKV, ScanRWKV


class RWKVConfig:
    """Configuration for RWKV model."""

    def __init__(
        self,
        vocab_size: int = 100300,
        n_embd: int = 768,
        n_layer: int = 12,
        use_scan: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.use_scan = use_scan
        self.dtype = dtype


class RWKVModel(nn.Module):
    """
    Flax wrapper for RWKV that makes it compatible with the training pipeline.

    This wrapper:
    - Converts RWKV's functional API to Flax Module API
    - Handles state initialization and management
    - Provides a standard forward pass interface
    """

    vocab_size: int = 100300
    n_embd: int = 768
    n_layer: int = 12
    use_scan: bool = True
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        # RWKV implementation to use
        self.rwkv_impl = AssociativeScanRWKV if self.use_scan else ScanRWKV

        # Create RWKV config
        self.rwkv_config = RWKVConfig(
            vocab_size=self.vocab_size,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            use_scan=self.use_scan,
            dtype=self.dtype,
        )

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Forward pass compatible with the training pipeline.

        Args:
            inputs: Token IDs of shape (batch, seq_len)
            deterministic: Ignored (RWKV doesn't use dropout in this implementation)

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = inputs.shape

        # Get or initialize RWKV parameters
        # In Flax, we use self.param() to declare parameters
        # But RWKV has a custom initialization, so we need to handle this carefully

        # Check if this is initialization or forward pass
        is_initializing = not self.has_variable("params", "rwkv_params")

        if is_initializing:
            # Initialize RWKV parameters
            key = self.make_rng("params")
            rwkv_params, _ = self.rwkv_impl.randomize_weights(
                key=key,
                n_layer=self.n_layer,
                n_embd=self.n_embd,
                vocab_size=self.vocab_size,
                config=self.rwkv_config,
                dtype=self.dtype,
            )
            self.put_variable("params", "rwkv_params", rwkv_params)
        else:
            rwkv_params = self.get_variable("params", "rwkv_params")

        # Initialize state for each sequence in the batch
        state = self.rwkv_impl.default_state(rwkv_params, self.rwkv_config)
        # Expand state for batch
        state = jnp.tile(state[None], (batch_size, 1, 1, 1))

        # Process each sequence in the batch
        def process_sequence(tokens):
            seq_state = self.rwkv_impl.default_state(rwkv_params, self.rwkv_config)
            logits, _ = self.rwkv_impl.forward(
                params=rwkv_params,
                tokens=tokens,
                state=seq_state,
                length=seq_len,
                new_starts=None,
                config=self.rwkv_config,
            )
            return logits

        # Vectorize over batch dimension
        logits = jax.vmap(process_sequence)(inputs)

        return logits


def create_rwkv_model(
    vocab_size: int = 100300,
    n_embd: int = 768,
    n_layer: int = 12,
    use_scan: bool = True,
    dtype: jnp.dtype = jnp.bfloat16,
) -> RWKVModel:
    """
    Create a RWKV model.

    Args:
        vocab_size: Vocabulary size
        n_embd: Embedding dimension
        n_layer: Number of layers
        use_scan: Use AssociativeScanRWKV (True) or ScanRWKV (False)
        dtype: Data type for parameters

    Returns:
        RWKVModel instance
    """
    return RWKVModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_layer=n_layer,
        use_scan=use_scan,
        dtype=dtype,
    )
