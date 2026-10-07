import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x, cos, sin):
    return x * cos + _rotate_half(x) * sin


class RotaryEmbedding2D(nn.Module):
    """Axial 2D RoPE: half the head dim is rotated by row position, the
    other half by column position. Fixed (no learned parameters) — unlike
    the additive absolute embedding, position is injected inside every
    attention layer's Q/K, not once at the input."""

    def __init__(self, head_dim, grid_h, grid_w, base=10000.0):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for axial 2D RoPE"
        axis_dim = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, axis_dim, 2).float() / axis_dim))

        row_ids = torch.arange(grid_h).repeat_interleave(grid_w).float()
        col_ids = torch.arange(grid_w).repeat(grid_h).float()

        row_angles = torch.cat([row_ids[:, None] * inv_freq, row_ids[:, None] * inv_freq], dim=-1)
        col_angles = torch.cat([col_ids[:, None] * inv_freq, col_ids[:, None] * inv_freq], dim=-1)

        self.axis_dim = axis_dim
        self.register_buffer("row_cos", row_angles.cos(), persistent=False)
        self.register_buffer("row_sin", row_angles.sin(), persistent=False)
        self.register_buffer("col_cos", col_angles.cos(), persistent=False)
        self.register_buffer("col_sin", col_angles.sin(), persistent=False)

    def _rotate(self, x):
        # x: [B, n_heads, L, head_dim]
        x_row, x_col = x[..., : self.axis_dim], x[..., self.axis_dim :]
        x_row = _apply_rope(x_row, self.row_cos, self.row_sin)
        x_col = _apply_rope(x_col, self.col_cos, self.col_sin)
        return torch.cat([x_row, x_col], dim=-1)

    def forward(self, q, k):
        return self._rotate(q), self._rotate(k)


class MultiHeadSelfAttention(nn.Module):
    """Manual bidirectional MHSA (no causal mask) — written by hand rather
    than nn.MultiheadAttention so RoPE can rotate Q/K before the attention
    dot product. Equivalent parameter count to nn.MultiheadAttention when
    rope=None."""

    def __init__(self, d_model, n_heads, dropout=0.0, rope=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.rope = rope

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, l, d = x.shape
        qkv = self.qkv(x).view(b, l, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, n_heads, L, head_dim]

        if self.rope is not None:
            q, k = self.rope(q, k)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, l, d)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm bidirectional block: MHSA (no causal mask) + MLP, residual."""

    def __init__(self, d_model, n_heads, mlp_ratio=4, dropout=0.0, rope=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout, rope=rope)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MaskedTokenTransformer(nn.Module):
    """Bidirectional Transformer over a 2D grid of VQ-VAE token ids.

    No timestep conditioning: for absorbing/masked diffusion under the
    linear noise schedule, the optimal denoiser doesn't need t as input
    (confirmed against ~/Omni's Dream model and the local MDLM reference,
    both run with time_conditioning effectively off).

    pos_embed_type:
      "absolute" — learned row/col embeddings, added once at the input.
      "rope"     — fixed axial 2D rotary embedding, applied inside every
                   attention layer's Q/K instead of an additive input embedding.
    """

    def __init__(
        self,
        vocab_size=512,
        grid_size=(16, 24),
        d_model=384,
        n_layers=8,
        n_heads=6,
        mlp_ratio=4,
        dropout=0.1,
        pos_embed_type="absolute",
    ):
        super().__init__()
        assert pos_embed_type in ("absolute", "rope"), pos_embed_type
        self.pos_embed_type = pos_embed_type
        self.vocab_size = vocab_size
        self.mask_id = vocab_size  # extra embedding row reserved for MASK
        self.grid_h, self.grid_w = grid_size
        self.seq_len = self.grid_h * self.grid_w

        self.token_embed = nn.Embedding(vocab_size + 1, d_model)
        self.dropout = nn.Dropout(dropout)

        rope = None
        if pos_embed_type == "absolute":
            self.row_embed = nn.Embedding(self.grid_h, d_model)
            self.col_embed = nn.Embedding(self.grid_w, d_model)
            row_ids = torch.arange(self.grid_h).repeat_interleave(self.grid_w)
            col_ids = torch.arange(self.grid_w).repeat(self.grid_h)
            self.register_buffer("row_ids", row_ids, persistent=False)
            self.register_buffer("col_ids", col_ids, persistent=False)
        else:
            head_dim = d_model // n_heads
            rope = RotaryEmbedding2D(head_dim, self.grid_h, self.grid_w)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, mlp_ratio, dropout, rope=rope)
                for _ in range(n_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        """token_ids: [B, seq_len] with values in [0, vocab_size] (vocab_size == mask_id).
        Returns logits [B, seq_len, vocab_size] over the real codebook only."""
        x = self.token_embed(token_ids)
        if self.pos_embed_type == "absolute":
            x = x + self.row_embed(self.row_ids) + self.col_embed(self.col_ids)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        return self.head(x)
