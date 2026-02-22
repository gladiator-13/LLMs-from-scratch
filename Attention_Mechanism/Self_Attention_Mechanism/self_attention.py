"""
Self-Attention Mechanism (Scaled Dot-Product Attention)

Implements:
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
"""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Single-head self-attention layer.
    """

    def __init__(self, d_model: int):
        super().__init__()

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, seq_len, d_model)
        """

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.matmul(Q, K.transpose(-2, -1))

        d_k = Q.size(-1)
        scaled_scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        attention_weights = torch.softmax(scaled_scores, dim=-1)

        context = torch.matmul(attention_weights, V)

        return context
