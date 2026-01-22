import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


class ConvTest(nn.Module):
    features: int
    kernel_size: int

    @nn.compact
    def __call__(self, x):
        # x: (B, L, C)
        # Causal convolution using nn.Conv
        out = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            padding=((self.kernel_size - 1, 0),),
            feature_group_count=self.features,
            use_bias=False,
            name="conv",
        )(x)
        return out

    @nn.compact
    def step(self, buffer, x_t):
        # x_t: (B, C)
        # buffer: (B, K-1, C)

        # Get kernel parameters exactly as nn.Conv would have them
        # (K, 1, C)
        kernel = self.scope.push("conv").param(
            "kernel",
            nn.initializers.lecun_normal(),
            (self.kernel_size, 1, self.features),
        )

        # Prepare input window (B, K, C)
        # Concatenate buffer and current input
        # buffer is [x_{t-K+1}, ..., x_{t-1}]
        # x_t is x_t
        # window is [x_{t-K+1}, ..., x_t]
        x_expanded = x_t[:, None, :]  # (B, 1, C)
        window = jnp.concatenate([buffer, x_expanded], axis=1)

        # Manual computation
        # einsum "bkc,kdc->bc" where d=1 (input_dim // groups)
        # kernel is (K, 1, C)
        # We want to map window (B, K, C) to (B, C)
        # Since group count = features, it's depthwise.
        # Each channel c interacts only with channel c.
        # sum_k window[b, k, c] * kernel[k, 0, c]

        out = jnp.einsum("bkc,kc->bc", window, kernel[:, 0, :])

        # Update buffer
        new_buffer = jnp.concatenate([buffer[:, 1:, :], x_expanded], axis=1)

        return out, new_buffer


def run_test():
    B = 1
    L = 5
    C = 4
    K = 3

    model = ConvTest(features=C, kernel_size=K)

    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (B, L, C))

    # Init
    variables = model.init(rng, x)
    params = variables["params"]

    print("Kernel shape:", params["conv"]["kernel"].shape)
    # (3, 1, 4) -> (K, 1, C)

    # Run __call__
    out_call = model.apply(variables, x)
    print("Call output shape:", out_call.shape)

    # Run step loop
    buffer = jnp.zeros((B, K - 1, C))
    out_step_list = []

    for t in range(L):
        x_t = x[:, t, :]
        out_t, buffer = model.apply(variables, buffer, x_t, method=model.step)
        out_step_list.append(out_t)

    out_step = jnp.stack(out_step_list, axis=1)

    print("\n--- Comparison ---")
    for t in range(L):
        d_t = jnp.max(jnp.abs(out_call[:, t, :] - out_step[:, t, :]))
        print(f"t={t}, Max Diff: {d_t:.6e}")
        if d_t > 1e-5:
            print(f"  Call: {out_call[0, t, :]}")
            print(f"  Step: {out_step[0, t, :]}")


if __name__ == "__main__":
    run_test()
