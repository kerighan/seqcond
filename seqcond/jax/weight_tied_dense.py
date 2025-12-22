import jax.numpy as jnp
import flax.linen as nn


class WeightTiedDense(nn.Module):
    """Dense layer that shares weights with an embedding layer."""

    vocab_size: int
    use_bias: bool = False

    @nn.compact
    def __call__(
        self, inputs: jnp.ndarray, embedding_weights: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Args:
            inputs: Input tensor of shape [batch, seq_len, d_model]
            embedding_weights: Embedding weights of shape [vocab_size, d_model]
        Returns:
            Logits of shape [batch, seq_len, vocab_size]
        """
        logits = jnp.matmul(inputs, embedding_weights.T)
        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros, (self.vocab_size,))
            logits = logits + bias
        return logits
