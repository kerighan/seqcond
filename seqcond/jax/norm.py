import jax
import jax.numpy as jnp
import flax.linen as nn


class RMSNorm(nn.Module):
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        x_f32 = x.astype(jnp.float32)
        mean_sq = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        y = x_f32 * jax.lax.rsqrt(mean_sq + self.epsilon)
        y = y * scale.astype(y.dtype)
        return y.astype(x.dtype)
