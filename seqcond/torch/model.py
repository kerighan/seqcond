"""
SeqCondModel - mirrors JAX model.py exactly.
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import numpy as np

from .norm import RMSNorm
from .rope import TransformerDecoderBlock, precompute_freqs
from .seqcond import SeqCondBlock


class SeqCondModel(nn.Module):
    """SeqCond model - matches JAX SeqCondModel exactly."""

    def __init__(
        self,
        d_model: int = 768,
        d_ff: int = 2304,
        num_layers: int = 12,
        vocab_size: int = 100300,
        maxlen: int = 768,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        seqcond_heads: int = 32,
        num_query_heads: int = 6,
        num_thetas: int = 4,
        conv_kernel_size: int = 4,
        expand_factor: int = 1,
        out_expand_factor: int = 3,
        seqcond_ratio: int = 3,
        dropout: float = 0.0,
        use_positional_embedding: bool = False,
        skip_low_rank: bool = True,
        num_anchor_heads: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.seqcond_ratio = seqcond_ratio
        self.conv_kernel_size = conv_kernel_size

        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional embedding (optional)
        self.use_positional_embedding = use_positional_embedding
        if use_positional_embedding:
            self.position_embedding = nn.Embedding(maxlen, d_model)

        # Precompute RoPE
        head_dim = d_model // num_heads
        cos, sin = precompute_freqs(maxlen, head_dim)
        self.register_buffer("cos_emb", cos)
        self.register_buffer("sin_emb", sin)

        # Build blocks
        self.blocks = nn.ModuleList()
        self.block_types = []

        for i in range(num_layers):
            if (i + 1) % (seqcond_ratio + 1) == 0:
                # Transformer block
                block = TransformerDecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    num_kv_heads=self.num_kv_heads,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    qk_norm_eps=qk_norm_eps,
                )
                self.block_types.append("transformer")
            else:
                # SeqCond block
                block = SeqCondBlock(
                    d_model=d_model,
                    num_heads=seqcond_heads,
                    num_query_heads=num_query_heads,
                    num_anchor_heads=num_anchor_heads,
                    num_thetas=num_thetas,
                    conv_kernel_size=conv_kernel_size,
                    expand_factor=expand_factor,
                    out_expand_factor=out_expand_factor,
                    skip_low_rank=skip_low_rank,
                    dropout=dropout,
                    maxlen=maxlen,
                )
                self.block_types.append("seqcond")
            self.blocks.append(block)

        # Final norm and head
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass - matches JAX __call__."""
        B, L = input_ids.shape

        x = self.embedding(input_ids)

        if self.use_positional_embedding:
            positions = torch.arange(L, device=input_ids.device)
            x = x + self.position_embedding(positions)

        # Get RoPE embeddings
        cos = self.cos_emb[:L].unsqueeze(0).unsqueeze(2)  # (1, L, 1, D/2)
        sin = self.sin_emb[:L].unsqueeze(0).unsqueeze(2)
        cos = cos.expand(B, L, self.num_heads, -1)
        sin = sin.expand(B, L, self.num_heads, -1)

        for block, block_type in zip(self.blocks, self.block_types):
            if block_type == "transformer":
                x = block(x, cos, sin)
            else:
                x = block(x)

        x = self.final_norm(x)
        return self.lm_head(x)

    def prefill(
        self, input_ids: torch.Tensor, return_all_logits: bool = False
    ) -> Tuple[torch.Tensor, List[Tuple]]:
        """
        Process prompt in parallel and return logits + states for step() continuation.
        Much faster than processing token-by-token with step().

        Args:
            return_all_logits: If True, return logits for all positions (for evaluation).
                               If False (default), return only last token logits (for generation).
        """
        B, L = input_ids.shape
        device = input_ids.device

        x = self.embedding(input_ids)

        if self.use_positional_embedding:
            positions = torch.arange(L, device=device)
            x = x + self.position_embedding(positions)

        # Get RoPE embeddings
        cos = self.cos_emb[:L].unsqueeze(0).unsqueeze(2)
        sin = self.sin_emb[:L].unsqueeze(0).unsqueeze(2)
        cos = cos.expand(B, L, self.num_heads, -1)
        sin = sin.expand(B, L, self.num_heads, -1)

        states = []
        for block, block_type in zip(self.blocks, self.block_types):
            if block_type == "transformer":
                x, kv_state = block(x, cos, sin, return_state=True)
                # kv_state is (k, v) with shape (B, L, num_kv_heads, head_dim)
                # Pad to maxlen for KV cache
                k, v = kv_state
                k_cache = torch.zeros(
                    B,
                    self.maxlen,
                    self.num_kv_heads,
                    self.d_model // self.num_heads,
                    device=device,
                    dtype=k.dtype,
                )
                v_cache = torch.zeros_like(k_cache)
                k_cache[:, :L] = k
                v_cache[:, :L] = v
                states.append((k_cache, v_cache))
            else:
                x, state = block(x, return_state=True)
                states.append(state)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        # Return only last token logits for generation (default)
        # Use return_all_logits=True for evaluation
        if return_all_logits:
            return logits, states
        return logits[:, -1:, :], states

    def init_state(self, batch_size: int, device: torch.device) -> List[Tuple]:
        """Initialize states for all blocks - matches JAX init_state."""
        states = []

        for block, block_type in zip(self.blocks, self.block_types):
            if block_type == "transformer":
                # Transformer: (k_cache, v_cache)
                k_cache = torch.zeros(
                    batch_size,
                    self.maxlen,
                    self.num_kv_heads,
                    self.d_model // self.num_heads,
                    device=device,
                )
                v_cache = torch.zeros_like(k_cache)
                states.append((k_cache, v_cache))
            else:
                # SeqCond: (den_acc, re_acc, im_acc, pos, conv_buffer)
                attn = block.attn
                den_acc = torch.zeros(batch_size, attn.K, device=device)
                re_acc = torch.zeros(batch_size, attn.K, attn.H, attn.M, device=device)
                im_acc = torch.zeros_like(re_acc)
                pos = torch.zeros(batch_size, device=device)
                conv_buffer = torch.zeros(
                    batch_size,
                    attn.conv_kernel_size - 1,
                    attn.dim_conv_total,
                    device=device,
                )
                states.append((den_acc, re_acc, im_acc, pos, conv_buffer))

        return states

    def step(
        self,
        token_id: torch.Tensor,
        states: List[Tuple],
        pos: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
        use_triton: bool = False,
    ) -> Tuple[torch.Tensor, List[Tuple]]:
        """Single step for generation - matches JAX step.

        Args:
            seq_len: Optional fixed sequence length for transformer attention.
                     If provided, uses k_cache[:, :seq_len] instead of full cache.
                     Used for power-of-2 CUDA graph optimization.
            use_triton: If True, use Triton kernels for SeqCond blocks.
        """
        B = token_id.size(0)

        # Get position from first SeqCond state if not provided
        if pos is None:
            for state, block_type in zip(states, self.block_types):
                if block_type == "seqcond":
                    pos = state[3]  # pos is 4th element in SeqCond state
                    break
            # If still None (pure transformer model), initialize to 0
            if pos is None:
                pos = torch.zeros(B, device=token_id.device, dtype=torch.long)

        x = self.embedding(token_id).squeeze(1)  # (B, D)

        if self.use_positional_embedding:
            pos_idx = pos.long()  # (B,) — per-sample positions
            x = x + torch.index_select(self.position_embedding.weight, 0, pos_idx)

        # Get RoPE for current position (use index_select for CUDA graph compatibility)
        pos_idx = pos.long()  # (B,) — per-sample positions
        cos_t = torch.index_select(self.cos_emb, 0, pos_idx).unsqueeze(1).unsqueeze(1)
        sin_t = torch.index_select(self.sin_emb, 0, pos_idx).unsqueeze(1).unsqueeze(1)
        cos_t = cos_t.expand(B, 1, self.num_heads, -1)
        sin_t = sin_t.expand(B, 1, self.num_heads, -1)

        new_states = []
        for i, (block, block_type, state) in enumerate(
            zip(self.blocks, self.block_types, states)
        ):
            if block_type == "transformer":
                x, new_state = block.step(x, state, pos, cos_t, sin_t, seq_len=seq_len)
                # Update pos for next iteration (transformer doesn't update it)
                if pos is not None:
                    pos = pos + 1
            else:
                x, new_state = block.step(x, state, use_triton=use_triton)
                # SeqCond updates pos in its state, extract it
                pos = new_state[3]
            new_states.append(new_state)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, new_states
