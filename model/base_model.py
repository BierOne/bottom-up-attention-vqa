import torch
import torch.nn as nn
from model.attention import Attention
from model.language_model import WordEmbedding, QuestionEmbedding
from model.classifier import SimpleClassifier
from utilities import config


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.classifier = classifier

    def forward(self, v, b, q, a, m=config.masks):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        return: logits, attention_weights
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]
        att, att_logits = self.v_att(v, q_emb)  # [batch, objs, 1]
        v_emb = (att * v).sum(1)  # [batch, v_dim]
        logits = self.classifier(q_emb, v_emb)
        return logits, att


def build_baseline(embeddings, num_ans_candidates):
    vision_features = config.output_features
    visual_glimpses = config.visual_glimpses
    question_features = hidden_features = config.hid_dim
    w_emb = WordEmbedding(
        embeddings,
        dropout=0.0
    )

    q_emb = QuestionEmbedding(
        w_dim=300,
        hid_dim=question_features,
        nlayers=1,
        bidirect=False,
        dropout=0.0
    )

    v_att = Attention(
        v_dim=vision_features,
        q_dim=question_features*q_emb.ndirections,
        hid_dim=hidden_features,
        glimpses=visual_glimpses,
    )

    classifier = SimpleClassifier(
        in_dim=(question_features*q_emb.ndirections, vision_features),
        hid_dim=(hidden_features, hidden_features * 2),
        out_dim=num_ans_candidates,
        dropout=0.5
    )
    return BaseModel(w_emb, q_emb, v_att, classifier)
