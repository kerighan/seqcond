"""
Normalization layers - mirrors JAX implementation exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm - matches JAX norm.py"""

    def __init__(self, hidden_size: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match JAX: cast to float32, compute, cast back
        orig_dtype = x.dtype
        x_f32 = x.float()
        mean_sq = x_f32.pow(2).mean(dim=-1, keepdim=True)
        y = x_f32 * torch.rsqrt(mean_sq + self.epsilon)
        y = y * self.scale.float()
        return y.to(orig_dtype)


class GatedRMSNorm(nn.Module):
    """RMSNorm with gating on residual (Mamba2 style).

    Applies: x = rmsnorm(x * silu(residual))
    Matches JAX seqcond_fast.py GatedRMSNorm
    """

    def __init__(self, hidden_size: int, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # Match JAX exactly
        x = x.float()
        res = residual.float()

        # Gate with silu (JAX uses silu here)
        x = x * F.silu(res)

        # RMSNorm
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.epsilon)
        x = x * self.weight.float()

        return x  # Keep as float32, caller will cast if needed
