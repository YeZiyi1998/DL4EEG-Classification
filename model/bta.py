from typing import Optional, Any
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d
import mne
import numpy as np
import json
from layers import PolarPositionalEncoding

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

def get_pos_encoder(pos_encoding):
    return PolarPositionalEncoding

class BatchNorm(nn.modules.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(BatchNorm, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5) 
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]                     
        src = src + self.dropout1(src2)  # (channels, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, channels)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (channels, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (channels, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, channels)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (channels, batch_size, d_model)
        return src

class BTANet(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1, pos_encoding='polar', activation='gelu', norm='BatchNorm', freeze=False, args = None):
        super(BTANet, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.input_layer = nn.Linear(feat_dim, d_model)
        self.pos_enc = pos_encoding
        encoder_layer = BatchNorm(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        self.spatial_attention_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.feat_dim = feat_dim
        self.output_layer = nn.Linear(d_model, feat_dim)
        self.mode = args.mode

    def forward(self, X, padding_masks):
        inp = X.permute(1, 0, 2)
        inp = self.input_layer(inp) * math.sqrt(self.d_model)  # (channels, batch_size, d_model) 
        inp = self.pos_enc(inp)  # add positional encoding: (channels, batch_size, d_model)  
        output = self.spatial_attention_encoder(inp, src_key_padding_mask=~padding_masks)  # (channels, batch_size, d_model)
        output = self.act(output)  # (batch_size, channels, d_model)
        output = output.permute(1, 0, 2)  # (batch_size, channels, d_model)
        output = self.dropout1(output)
        if self.mode == 'unsupervised':
            output = self.output_layer(output) # (batch_size, channels, feat_dim)
        elif self.mode == 'supervised':
            output = output.reshape(output.shape[0], -1)  # (batch_size, channels * d_model)
        return output

class SupervisedBTA(nn.Module):
    def __init__(self, model1, model2, d_model, max_len, num_classes, mask):
        super(SupervisedBTA, self).__init__()
        self.output_layer = self.build_output_module(d_model, max_len, num_classes, mask)
        self.model = [None, None]
        self.model1 = model1
        self.model2 = model2
        self.mask = mask
        
    def build_output_module(self, d_model, max_len, num_classes, mask):
        if mask == 'frequency' or mask == 'temporal':
            output_layer = nn.Linear(d_model * max_len, num_classes)
        else:
            output_layer = nn.Linear(d_model * 2 * max_len, num_classes)
        return output_layer

    def forward(self, X1, X2, padding_masks):
        X1 = self.model1(X1, padding_masks)
        X2 = self.model2(X2, padding_masks)
        if self.mask == 'frequency':
            output = self.output_layer(X2)
        elif self.mask == 'temporal':
            output = self.output_layer(X1)
        else:
            output = self.output_layer(torch.cat((X1, X2), 1))
        return output

class UnsupervisedBTA(nn.Module):
    def __init__(self, model1, model2):
        super(UnsupervisedBTA, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, X1, X2, padding_masks):
        X1 = self.model1(X1, padding_masks)
        X2 = self.model2(X2, padding_masks)
        return X1, X2
