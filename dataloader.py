import json
from torch.utils.data import Dataset
import torch
import numpy as np
import os
from  sklearn import preprocessing
from model.het.utils import getMI, getmatrix
from utils import get_dataset_dict

# Todo: normalize

def value2label(v, mode, split_mode):
    if mode == 'AMIGOS':
        if v[1] >= 5:
            return 1
        return 0
    elif mode == 'Search-Brainwave':
        split_mode_list = split_mode.split('_')
        upperbound = int(split_mode_list[1][0])
        lowerbound = int(split_mode_list[0][-1])
        if v >= upperbound:
            return 1
        elif v <= lowerbound:
            return 0
        return -1

class MyDataloader():
    def __init__(self, args):
        self.normalized = args.normalized
        self.base_path = args.base_path
        self.data_info = get_dataset_dict(args.dataset, args.model)
        self.split_mode = args.split_mode
        self.model = args.model
        if self.model == 'Het':
            self.A = {}

    def get_normalizer(self, train, valid):
        data = [item[1]['eeg'] for item in train] + [item[1]['eeg'] for item in valid]
        data = np.array(data)
        data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
        self.my_std = preprocessing.StandardScaler()
        self.my_std.fit(data)

    def make_graph(self, data):
        if f'{data["info"]}' in self.A.keys():
            data['A'] = self.A[f'{data["info"]}']
            return
        if os.path.exists(f'cache_data/het_A_{self.model}_{data["info"]}.npy'):
            data['A'] = torch.tensor(np.load(f'cache_data/het_A_{self.model}_{data["info"]}.npy'), dtype=torch.long)
            self.A[f'{data["info"]}'] = data['A']
            return
        corr_threshold = 0.7
        adj_matrix = getMI(data['data'], corr_threshold)
        edge_index, A = getmatrix(adj_matrix)
        data['A'] = A
        self.A[f'{data["info"]}'] = data['A']
        np.save(f'cache_data/het_A_{self.model}_{data["info"]}.npy', A.numpy())

    def processed(self, train, ):
        data = []
        for item in train:
            item[0]['user_name'] = str(item[0]['user_name'])
            new_score = value2label(item[1]['score'], self.data_info['name'], self.split_mode)
            if new_score != -1:
                data.append({'data':np.array(item[1]['eeg'])[:self.data_info['max_len'],:self.data_info['freq_len']], 'temp_data':np.array(item[1]['eeg'])[:self.data_info['max_len'],self.data_info['freq_len']:], 'score': new_score, 'info':item[0]})
                if self.model == 'Het':
                    # pre caculated the graph for HetEmotionNet
                    self.make_graph(data[-1])
        return data

    def load_data(self, valid_id, strategy, normalized):
        train = json.load(open(os.path.join(self.base_path + strategy, f'train_{valid_id}.json')))
        valid = json.load(open(os.path.join(self.base_path + strategy, f'valid_{valid_id}.json')))
        if normalized:
            self.get_normalizer(train, valid)
            for data in [train, valid]:
                for item in data:
                    # note that the normalization is not channel dependent
                    item[1]['eeg'] = self.my_std.transform(item[1]['eeg'])
        return self.processed(train, ), self.processed(valid, )

class MyDataset(Dataset):
    def __init__(self, train, device, args):
        super(MyDataset, self).__init__()
        self.train = train
        self.device = device
        self.args = args
        self.model = args.model

    def __getitem__(self, ind):
        X1 = np.array(self.train[ind]['data']) # (seq_length, feat_dim) array
        X2 = np.array(self.train[ind]['temp_data']) # (seq_length, feat_dim) arrays
        Y = np.array(self.train[ind]['score'])
        mask = np.ones(X1.shape[0], bool)  # (seq_length,) boolean array
        if self.model == 'Het':
            mask = np.array(self.train[ind]['A'])
            return torch.from_numpy(X1).to(self.device, dtype=torch.float32), torch.from_numpy(X2).to(self.device, dtype=torch.float32), torch.from_numpy(Y).to(self.device, dtype=torch.long), torch.from_numpy(mask).to(self.device, dtype=torch.float)
        return torch.from_numpy(X1).to(self.device, dtype=torch.float32), torch.from_numpy(X2).to(self.device, dtype=torch.float32), torch.from_numpy(Y).to(self.device, dtype=torch.long), torch.from_numpy(mask).to(self.device, dtype=torch.bool)
    
    def __len__(self,):
        return len(self.train)

class MaskDataset(Dataset):
    """Dynamically computes missingness (noise) mask for each sample"""
    def __init__(self, train, device, mean_mask_length=3, masking_ratio=0.15,
                 mode='separate', distribution='geometric', exclude_feats=None):
        super(MaskDataset, self).__init__()

        self.train = train  # this is a subclass of the BaseData class in data.py
        self.device = device
        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length
        self.mode = mode
        self.distribution = distribution
        self.exclude_feats = exclude_feats

    def __getitem__(self, ind):
        X1 = np.array(self.train[ind]['data']) # (seq_length, feat_dim) array
        X1_mask = noise_mask(X1, self.masking_ratio, self.mean_mask_length, self.mode, self.distribution, self.exclude_feats)  # (seq_length, feat_dim) boolean array
        X2 = np.array(self.train[ind]['temp_data']) # (seq_length, feat_dim) array
        X2_mask = noise_mask(X2, self.masking_ratio, self.mean_mask_length, self.mode, self.distribution, self.exclude_feats)  # (seq_length, feat_dim) boolean array
        Y1 = torch.from_numpy(X1).to(self.device, dtype=torch.float32).clone()
        Y2 = torch.from_numpy(X2).to(self.device, dtype=torch.float32).clone()
        X1 = X1 * X1_mask
        X2 = X2 * X2_mask
        mask = np.ones(X1.shape[0], bool)  # (seq_length,) boolean array

        return torch.from_numpy(X1).to(self.device, dtype=torch.float32), torch.from_numpy(X2).to(self.device, dtype=torch.float32), Y1, Y2, torch.from_numpy(X1_mask).to(self.device, dtype=torch.bool), torch.from_numpy(X2_mask).to(self.device, dtype=torch.bool), torch.from_numpy(mask).to(self.device, dtype=torch.bool)

    def update(self):
        self.mean_mask_length = min(20, self.mean_mask_length + 1)
        self.masking_ratio = min(1, self.masking_ratio + 0.05)

    def __len__(self):
        return len(self.train)

# from https://github.com/gzerveas/mvts_transformer
def noise_mask(X, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

    return mask

# from https://github.com/gzerveas/mvts_transformer
def geom_noise_mask_single(L, lm, masking_ratio):
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state
    return keep_mask
