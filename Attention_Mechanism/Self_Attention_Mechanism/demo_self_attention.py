import torch
from self_attention import SelfAttention

if __name__ == "__main__":

    torch.manual_seed(42)

    batch_size = 1
    seq_len = 8
    d_model = 64

    x = torch.rand(batch_size, seq_len, d_model)

    attention = SelfAttention(d_model)

    output = attention(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)
