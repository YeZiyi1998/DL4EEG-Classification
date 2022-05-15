import logging
from time import time
import sklearn.metrics as metrics
import numpy as np
import torch as t


def label_process_deap(x, *y):
    '''
        greater than 5 is 1, otherwise 0
        :param x: Label tension
        :param y: unused
        :return:
    '''
    if x > 5:
        return 1
    else:
        return 0


def label_process_hci(x, *y):
    '''
        greater than 5 is 1, otherwise 0
        :param x: Label tension
        :param y: unused
        :return:
    '''
    if x > 5:
        return 1
    else:
        return 0


def getMI(sample, threshold, bins=10):
    '''
        Calculate mutual information among channels and get adjacent matrix
        :param sample: temporal domain data
        :param threshold: threshold
        :param bins:
        :return:
    '''
    # Number of channels
    num_of_v = sample.shape[0]
    # Tensor to numpy array
    # ziyi
    # sample = sample.numpy()
    adj_matrix = np.zeros((num_of_v, num_of_v))

    for i in range(num_of_v):
        for j in range(i, num_of_v):
            y = np.histogram2d(sample[i], sample[j], bins=bins)[0]
            adj_matrix[i][j] = metrics.mutual_info_score(None, None, contingency=y)
            adj_matrix[j][i] = adj_matrix[i][j]
    # Numpy array to tensor
    adj_matrix = t.from_numpy(adj_matrix)
    # Replace inf and nan
    adj_matrix = t.where(t.isinf(adj_matrix), t.full_like(adj_matrix, 0), adj_matrix)
    adj_matrix = t.where(t.isnan(adj_matrix), t.full_like(adj_matrix, 0), adj_matrix)
    # Threshold
    adj_matrix[adj_matrix < threshold] = 0
    return adj_matrix


def getmatrix(adj_matrix):
    # number of channels
    channels = adj_matrix.shape[0]
    enter = []
    out = []
    A = t.zeros((channels, channels, 3))
    for i in range(channels):
        for j in range(channels):
            if adj_matrix[i, j] != 0:
                enter.append(i)
                out.append(j)
                type = 0
                A[i, j, type] = abs(adj_matrix[i, j])
                A[j, i, type] = abs(adj_matrix[i, j])
    edge_index = [enter, out]
    edge_index = t.tensor(edge_index, dtype=t.long)
    return edge_index, A


def z_score_norm(data):
    # Standardized based on z_score
    feature_dim = data.size(-1)
    result = (data - t.mean(data, dim=2)[..., None].repeat(1, 1, feature_dim)) / t.std(data, dim=2)[..., None].repeat(1,
                                                                                                                      1,
                                                                                                                      feature_dim)
    result = t.where(t.isnan(result), t.full_like(result, 0), result)
    result = t.where(t.isinf(result), t.full_like(result, 0), result)
    return result


def min_max_norm(data):
    # Standardized based on min_max
    feature_dim = data.size(-1)
    result = t.div(data - t.min(data, dim=2)[0][..., None].repeat(1, 1, feature_dim),
                   (t.max(data, dim=2)[0] - t.min(data, dim=2)[0])[..., None].repeat(1, 1, feature_dim))
    return result


def get_window_function(type, window_points):
    '''
        Gets the window function
        :param type: window function type
            e.g. 'hanning'
        :param window_points: the total number of window function samples
        :return: The window function samples the
    '''
    pi = t.acos(t.zeros(1)).item() * 2
    if type == 'hanning':
        window_tensor = t.tensor(
            [0.5 - t.cos(t.tensor(2 * pi * x / (window_points + 1))) for x in range(1, window_points + 1)])
    else:
        window_tensor = None
    return window_tensor


def log_run_time(func):
    # timing tool
    def wrapper(*args, **kw):
        local_time = time()
        func(*args, **kw)
        logging.debug('current Function [%s] run time is %.2f' % (func.__name__, time() - local_time))

    return wrapper


def getwhichpart(dataset, i, label_num):
    '''
        In ten-fold cross-validation, to divide the test set and the training set of the current fold
        :param i: the idx of the fold
        :param label_num: the idx of the label used
        :return: training set and testing set
    '''
    sample_num = len(dataset)
    low = []
    high = []
    for k in range(len(dataset)):
        if dataset[k].Y[label_num] == 0:
            low.append(k)
        else:
            high.append(k)
    # low
    test_low_idx = t.zeros(sample_num, dtype=t.bool)
    train_low_idx = t.ones(sample_num, dtype=t.int)
    length = int(len(low) // 10)
    for k in range(i * length, (i + 1) * length):
        test_low_idx[low[k]] = True
        train_low_idx[low[k]] = 0
    # high
    test_high_idx = t.zeros(sample_num, dtype=t.bool)
    train_high_idx = t.ones(sample_num, dtype=t.int)
    length = int(len(high) // 10)
    for k in range(i * length, (i + 1) * length):
        test_high_idx[high[k]] = True
        train_high_idx[high[k]] = 0
    train_data = dataset[train_low_idx + train_high_idx == 2]
    test_data = dataset[test_low_idx + test_high_idx]
    return train_data, test_data
