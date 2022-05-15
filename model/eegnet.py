import torch.nn as nn
import torch.nn.functional as F
import torch

class EEGNet(nn.Module):
    def __init__(self, args, input_dim, num_nodes, device = None):
        super(EEGNet, self).__init__()
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        # 62 * 1251        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        kernel_size2 = 32 if self.num_nodes > 32 else int(self.num_nodes / 2) #amigos
        # Layer 2
        self.conv2 = nn.Conv2d(self.input_dim - 64 + 1, 4, (2, kernel_size2))
        # 4, 15, self.input_dim - kernel_size2 + 1
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        # 4, 4, int((self.num_nodes -31 -2) / 4) + 1
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        # 4, 
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        # 4, -8 + 2, -4 + 2
        # 4, 4, int((self.num_nodes -31 -2) / 4) + 1
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        if int((self.num_nodes - kernel_size2 + 1 -2) / 4) > 3:
            self.pooling3 = nn.MaxPool2d((2, 4))
            self.fc1 = nn.Linear(8* (int((int((self.num_nodes - kernel_size2 + 1 -2) / 4) - 3)/4)+1), 2) 
        else:
            self.pooling3 = nn.MaxPool2d((2, 1)) # amigos
            self.fc1 = nn.Linear(8 * (int((int((self.num_nodes - kernel_size2 + 1 -2) / 4))/1)+1), 2) 
            # self.fc1 = nn.Linear(4 * (int((self.num_nodes - kernel_size2 + 1 -2) / 4)), 2) 
        # 4, 2, int((int((self.num_nodes -31 -2) / 4) - 1)/4) + 1
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints. 
           
    def forward(self, X1, X2, padding_masks):
        batch_size = len(X1)
        bn = batch_size > 1 and torch.norm(X2, p=2) > 0
        x = X2
        x = x.unsqueeze(1)
        x = F.elu(self.conv1(x))
        if bn:
            x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = F.elu(self.conv2(x))
        if bn:
            x = self.batchnorm2(x)
            
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        if bn:
            x = self.batchnorm3(x)
            
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # FC Layer
        x = x.reshape(-1, 16)
        x = torch.sigmoid(self.fc1(x))
        return x

