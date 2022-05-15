# official preprocessing code of search-brainwave
import argparse
import torch
import json
from utils import *
import os
import numpy as np
import scipy.io as sio
from multiprocessing.pool import Pool
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def print_error(value):
    print("error: ", value)


def process_all(base_path, user_id_list, mode, out_path):
    cpu_num = 4
    pool = Pool(cpu_num)
    for user_id in user_id_list:
        file_name = base_path + '/features_info/' + user_id+mode
        os.makedirs(out_path + '/DE5T1251/', exist_ok=True)
        out_file_name = out_path + '/DE5T1251/' + user_id + '.json'
        pool.apply_async(process, args=(
            file_name, out_file_name), error_callback=print_error)
    pool.close()
    pool.join()
    print("extracted completed")


def down_sample(data, rate=1):
    return [np.mean(data[i*rate:(i+1)*rate]) for i in range(int(len(data)/rate))]


def get_de_features(raw_X, time_sample_rate, frequency_sample_rate, window_type, bands):
    overlap = 0.0
    window_length = 1.8
    all_de_features = []
    for i in range(1):
        de_features = DE(raw_X[:, int(i*time_sample_rate*overlap):int(i*time_sample_rate*overlap+window_length *
                         time_sample_rate)+1], window_length, window_type, time_sample_rate, frequency_sample_rate, bands).numpy()
        de_features = np.squeeze(de_features, axis=0)
        all_de_features.append(de_features)
    all_de_features = np.concatenate(all_de_features, axis=1)
    return all_de_features


def process(file_name, out_file_name):
    feature_type = 'de'
    window_type = 'hanning'
    time_sample_rate = 500
    frequency_sample_rate = 5120
    bands = {'eeg': [0.5, 4, 8, 13, 30, 50]}
    raw_json = json.load(open(file_name))
    for qid in raw_json.keys():
        for did in raw_json[qid].keys():
            raw_X = torch.Tensor(raw_json[qid][did]['eeg'])
            features = get_de_features(
                raw_X, time_sample_rate, frequency_sample_rate, window_type, bands)
            raw_json[qid][did]['eeg'] = np.concatenate([features, [down_sample(
                item) for item in raw_json[qid][did]['eeg']]], axis=1).tolist()
    json.dump(raw_json, open(out_file_name, 'w'))
    print(out_file_name, 'saved')


def load_info(args):
    with open(args.base_path+'/info.txt') as f:
        lines = f.readlines()
        return json.loads(lines[1].split('=')[1].strip())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-base_path', type=str, help='data path',
                        default='../../../dataset/Non-click')
    args = parser.parse_args()
    args.out_path = '../data/search_brainwave_preprocessed'
    os.makedirs(args.out_path, exist_ok=True)

    user_id_list = load_info(args)
    process_all(args.base_path, user_id_list, '_raw.json', args.out_path)
