import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GTLayer(nn.Module):

    def __init__(self, in_channels, out_channels, device, first=True):
        self.device = device
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first  # Mark whether this layer is the first GT layer
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels, device)
            self.conv2 = GTConv(in_channels, out_channels, device)
        else:
            self.conv1 = GTConv(in_channels, out_channels, device)

    def forward(self, A, H_=None):
        # A shape (b,1,c,n,n)
        # Compute the new graph structure
        if self.first == True:
            a = self.conv1(A)
            batch = a.shape[0]
            num_node = a.shape[-1]
            b = self.conv2(A)
            a = a.view((-1, a.shape[-2], a.shape[-1]))
            b = b.view((-1, b.shape[-2], b.shape[-1]))
            H = torch.bmm(a, b)
            H = H.view((batch, -1, num_node, num_node))
            W = [(F.softmax(self.conv1.weight, dim=2)).detach(), (F.softmax(self.conv2.weight, dim=2)).detach()]
        else:
            a = self.conv1(A)
            batch = a.shape[0]
            num_node = a.shape[-1]
            a = a.view((-1, a.shape[-2], a.shape[-1]))
            H_ = H_.view(-1, H_.shape[-2], H_.shape[-1])
            H = torch.bmm(H_, a)
            H = H.view((batch, -1, num_node, num_node))
            W = [(F.softmax(self.conv1.weight, dim=2)).detach()]
        return H, W


# GT layer
class GTConv(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(1, out_channels, in_channels, 1, 1))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()
        self.device = device

    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        nn.init.uniform_(self.weight, 0, 1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        # A shape (b,1,c,n,n)
        # self.weight = self.weight.to(self.device)
        A = torch.sum(A * F.softmax(self.weight, dim=2), dim=2)
        # A shape (b,out,n,n)
        return A
