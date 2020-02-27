import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from model.fc import FCNet


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.0):
        super(SimpleClassifier, self).__init__()
        self.q_net = FCNet([in_dim[0], hid_dim[0]], dropout)
        self.v_net = FCNet([in_dim[1], hid_dim[0]], dropout)
        self.main = nn.Sequential(
            nn.Linear(hid_dim[0], hid_dim[1]),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(hid_dim[1], out_dim)
        )

    def forward(self, q_emb, v_emb):
        joint_repr = self.q_net(q_emb) * self.v_net(v_emb)
        logits = self.main(joint_repr)
        return logits

