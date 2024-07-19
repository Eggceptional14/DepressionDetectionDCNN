import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(CNNAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, cnn_output):
        attn_output, _ = self.attention(cnn_output, cnn_output, cnn_output)
        return attn_output.mean(dim=1)