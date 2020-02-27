import torch
import torch.nn as nn


class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, init_emb, freeze=False, dropout=0.0):
        super(WordEmbedding, self).__init__()
        weights = torch.from_numpy(init_emb)
        # self.emb = nn.Embedding.from_pretrained(weights, freeze=freeze) # padding_idx= ntoken
        ntoken, emb_dim = weights.shape
        self.emb = nn.Embedding(ntoken, emb_dim, padding_idx=-1)
        self.emb.weight.data = weights
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.emb(self.drop(x))


class QuestionEmbedding(nn.Module):
    def __init__(self, w_dim, hid_dim, nlayers, bidirect, dropout=0.0, rnn_type='GRU'):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type in ['LSTM', 'GRU']
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU

        self.ndirections = 1 + int(bidirect)
        self.rnn = rnn_cls(
            w_dim, hid_dim, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

    def forward(self, x):
        """
        x: [batch, sequence, in_dim]
        return: if ndirections==1: [batch, num_hid] else [batch, 2*num_hid]
        """
        output, hidden = self.rnn(x) # output->[b, s, h], hidden->[1, b, h]
        if self.ndirections == 1:
            return hidden.squeeze(0)
        # get the last hidden unit in forward direction
        forward_ = output[:, -1, :self.num_hid]
        # get the first hidden unit in backward direction
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)
