import copy
import itertools
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# from model_utils import *
from timm.models.vision_transformer import PatchEmbed, Block
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import BertModel, BertConfig


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_len.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]


def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=18):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''

        return self.pe[:x.size(0) + 1, :]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k)  # scores : [batch_size, n_heads, seq_len, seq_len]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.fcc = nn.Linear(n_heads * d_v, d_model)
        self.fccd = nn.LayerNorm(d_model)

    def forward(self, Q, K, V):
        # q: [batch_size, seq_len, d_model],
        # k: [batch_size, seq_len, d_model],
        # v: [batch_size, seq_len, d_model]

        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                 2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                 2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,
                                                                                 2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            self.n_heads * self.d_v)  # context: [batch_size, seq_len, n_heads, d_v]

        output = self.fcc(context)
        return self.fccd(output + residual)  # output: [batch_size, seq_len, d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


class BERT_main(nn.Module):
    def __init__(self, args, num_class, dropout):
        super().__init__()
        self.n_layers = args.BERT_layers
        self.d_model = 1024
        self.d_k = 64
        self.d_v = 64
        self.d_ff = 2024
        self.n_heads = self.d_model // self.d_k
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, self.d_k, self.d_v, self.n_heads, self.d_ff) for _ in
                                     range(self.n_layers)])  # 完成了self.n_layers次的多头注意力机制
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pos_emb = PositionalEncoding(self.d_model)

        self.fc = nn.Sequential(nn.Linear(self.d_model * 2, self.d_model), nn.Dropout(dropout), nn.Tanh(), )
        self.classifier = nn.Linear(self.d_model, num_class)

        self.fc11 = nn.Linear(num_class, 300)
        self.fc12 = nn.Linear(300, self.d_model)

    def forward(self, x, label_1):
        label_1 = self.fc12(self.fc11(label_1))  # (64, 1024)

        postion = self.pos_emb(x.transpose(0, 1))
        cls_token = self.cls_token + postion[:1, :, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = (x.transpose(0, 1) + postion[1:, :, :]).transpose(0, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            x = layer(x)

        h_pooled = self.fc(torch.cat((label_1, x[:, 0]), dim=1))  # [batch_size, d_model*2]
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2] predict isNext

        return logits_clsf
