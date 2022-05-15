from typing import Optional, Any
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
import mne
import numpy as np
import json
from utils import get_dataset_dict

class PolarPositionalEncoding(nn.Module):
    def coor2polar(self, x, y, z):
        r = math.sqrt(x**2+y**2+z**2)
        if abs(z) > 1e-12:
            theta = math.atan(math.sqrt(x**2+y**2) / z )
        else:
            theta = math.pi/2
        if abs(x) > 1e-12:
            phi = math.atan(y/x)
        else:
            phi = int(np.sign(y)) * math.pi / 2
        return r, theta, phi

    def __init__(self, d_model, dropout=0.1, max_len=1024, args = None, ):
        super(PolarPositionalEncoding, self).__init__()
        yuan_dian_len = args.polar_len
        dataset_info = get_dataset_dict(args.dataset, args.model)
        self.dropout = nn.Dropout(p=dropout)
        self.polar = []
        montage = mne.channels.read_dig_fif('mode/montage.fif')
        montage.ch_names = json.load(open("mode/montage_ch_names.json"))
        if args.dataset == 'AMIGOS':
            tmp_ch_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4", "ECG Right", "ECG Left", "GSR"]
        else:
            tmp_ch_names = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"]
        for idx in range(len(montage.dig)):
            montage.dig[idx]['r'] = [xyz / 1000000.0 for xyz in montage.dig[idx]['r']]
        yuan_dian_list = []
        yuan_dian_list.append((np.array(montage.dig[1]['r']) + np.array(montage.dig[-8]['r'])) / 2)
        yuan_dian_list.append(np.array(montage.dig[montage.ch_names.index('M1')]['r']))
        yuan_dian_list.append(np.array(montage.dig[montage.ch_names.index('M2')]['r']))
        for idx in range(dataset_info['max_len']):
            montage_idx = montage.ch_names.index(tmp_ch_names[idx])
            self.polar.append([])
            for i in range(yuan_dian_len):
                x = montage.dig[montage_idx]['r'][0] - yuan_dian_list[i][0]
                y = montage.dig[montage_idx]['r'][1] - yuan_dian_list[i][1]
                z = montage.dig[montage_idx]['r'][2] - yuan_dian_list[i][2]
                r, theta, phi =  self.coor2polar(x, y, z)
                self.polar[-1] += [r, theta, phi]
        self.count = 0
        self.polar = nn.Parameter(torch.tensor(self.polar), requires_grad = False)
        self.pe = nn.Parameter(torch.empty(3 * yuan_dian_len, d_model))  # requires_grad automatically set to True
        self.linear1 = Linear(d_model * 2, d_model)
        self.para = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        tmp_pe = torch.matmul(self.polar, self.pe)
        tmp_pe = tmp_pe.unsqueeze(1)
        x1 = x + tmp_pe[:x.size(0), :]
        return self.dropout(x1)
