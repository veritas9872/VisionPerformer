import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

from src.module import quant_noise, FairseqDropout, softmax


def generalized_kernel_feature_creator(
        x: Tensor,
        projection_matrix: Tensor,
        kernel_fn: callable,
        eps: float,
        normalize_data: bool
):
    normalizer = x.size(-1) ** -0.25 if normalize_data else 1.0
    if projection_matrix is None:
        x_prime = kernel_fn(normalizer * x) + eps
    else:
        x_prime = kernel_fn(normalizer * x @ projection_matrix) + eps
    return x_prime


def create_gaussian_projection(*size):
    """
    Orthogonalization based on PyTorch `nn.init.orthogonal_`

    While different from original Performer implementation,
    it appears close enough.

    Differences are that this version uses batch-level qr decomposition
    and that transpose is used instead of stacking for fat matrices.
    """
    if len(size) < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")
    m = torch.normal(mean=0, std=1, size=size)
    rows = size[-2]
    cols = size[-1]

    if rows < cols:
        m = torch.transpose(m, -2, -1)

    q, r = torch.linalg.qr(m)
    ph = torch.diag(r, diagonal=0).sign()
    q *= ph

    if rows < cols:
        q = torch.transpose(q, -2, -1)

    assert q.size() == size, f'Sanity check failed. {q.size()} != {size}.'

    return q


class PerformerBasicMHA(nn.Module):
    """
    Basic Performer MHA with very limited capability.
    Mostly for practice purposes.
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        k_dim=None,
        v_dim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.redraw_projection = True
        self.embed_dim = embed_dim
        self.k_dim = k_dim if k_dim is not None else embed_dim
        self.v_dim = v_dim if v_dim is not None else embed_dim
        self.qkv_same_dim = self.k_dim == embed_dim and self.v_dim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.k_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.v_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()


    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, x):
        """Input shape: (N, S, E) batch-first order."""
        # Only encoder self-attention for now.
        n, s, _ = x.shape
        # Multi-head attention
        q = self.q_proj(x).view(n, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(n, s, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(n, s, self.num_heads, self.head_dim)

        # Convert to contiguous reshaped arrays for efficient matrix multiplication.
        q = torch.permute(q, dims=(0, 2, 1, 3)).contiguous()  # N, H, S, D
        k = torch.permute(k, dims=(0, 2, 3, 1)).contiguous()  # N, H, D, S
        v = torch.permute(v, dims=(0, 2, 1, 3)).contiguous()  # N, H, S, D

        # Not using FAVOR+ random projection for now.
        eps = 1e-3
        q: Tensor = F.relu(q) + eps
        k: Tensor = F.relu(k) + eps

        # Sum along sequence axis for normalization.
        # Necessary to make each hypothetical SA row sum to 1.
        k_sum = torch.sum(k, dim=-1, keepdim=True)
        diag = q @ k_sum

        return (q @ (k @ v)) / diag
