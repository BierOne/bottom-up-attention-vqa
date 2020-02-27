import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from model.fc import FCNet

class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, hid_dim, glimpses=1, dropout=0.2):
        super(Attention, self).__init__()

        self.v_proj = FCNet([v_dim, hid_dim], dropout)
        self.q_proj = FCNet([q_dim, hid_dim], dropout)
        self.drop = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(hid_dim, glimpses), dim=None)
    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v_proj = self.v_proj(v)  # [batch, k, vdim]
        q_proj = self.q_proj(q).unsqueeze(1) # [batch, 1, qdim]
        logits = self.linear(self.drop(v_proj * q_proj))
        return nn.functional.softmax(logits, 1), logits
