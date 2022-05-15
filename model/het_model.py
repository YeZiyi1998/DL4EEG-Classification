import torch as t
from model.het.GTblock import GTN
from model.het.STDCN import *


class permute(nn.Module):
    def __init__(self):
        super(permute, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)


class Het(nn.Module):
    def __init__(self, args, input_dim, num_nodes, device, input_dim2 = None):
        super(Het, self).__init__()
        # Load model configuration
        self.num_node = num_nodes
        self.num_edge = 3
        self.final_out_node = num_nodes
        self.dropout = 0.2
        self.sample_feature_num = input_dim2
        self.per1 = permute()
        # Registration layer
        self.GT_F = GTN(3, 3, 3, 0, device)
        self.GT_T = GTN(3, 3, 3, 0, device)

        self.STDCN1_F = STDCN_with_GRU(input_dim, self.num_node, self.final_out_node, 3, device)
        self.linF = nn.Linear(input_dim, self.sample_feature_num // 2)
        # Note that the feature number is multiplied by two after the bi-GRU comes out
        self.STDCN1_T = STDCN_with_GRU(self.sample_feature_num, self.num_node, self.final_out_node, 3, device)
        self.flatten = nn.Flatten()
        self.flatten = nn.Flatten()
        self.linT = nn.Linear(self.sample_feature_num, self.sample_feature_num // 2)
        self.all = self.sample_feature_num * self.final_out_node * 2
        self.Dropout = nn.Dropout(0.2)
        self.lin1 = nn.Linear(self.all, 64)
        self.act1 = nn.LeakyReLU()
        self.lin2 = nn.Linear(64, 2)

    def forward(self, X1, X2, padding_masks):
        x_F, x_T, A = X1, X2, padding_masks
        x_F = self.Dropout(x_F)
        x_T = self.Dropout(x_T)
        A = A.view((-1, self.num_node, self.num_node, self.num_edge))
        x_F = x_F.view((-1, self.num_node, x_F.shape[-1]))
        x_T = x_T.view((-1, self.num_node, x_T.shape[-1]))
        # GTN
        H_F = self.GT_F(A)
        H_T = self.GT_T(A)
        # GCN and GRU
        out_F = self.STDCN1_F(x_F, H_F)
        out_F = self.linF(out_F)
        out_F = self.flatten(out_F)
        out_T = self.STDCN1_T(x_T, H_T)
        out_T = self.linT(out_T)
        out_T = self.flatten(out_T)
        # Fusion and classification
        out = t.cat([out_F, out_T], dim=-1)
        out = self.lin1(out)
        out = self.act1(out)
        out = self.lin2(out)
        return out
