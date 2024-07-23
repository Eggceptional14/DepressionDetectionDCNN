import torch
import torch.nn as nn
import torch.nn.functional as F

    
class Attention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(Attention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        output, attention_weights = self.attention(x, x, x)
        return output, attention_weights