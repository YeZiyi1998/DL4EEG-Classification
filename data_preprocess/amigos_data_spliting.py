import argparse
import configparser
import os
import numpy as np
import json
import tqdm
import random
random.seed(2021)


def shuffle(train_data):
    index = np.arange(train_data[0].shape[0])
    np.random.shuffle(index)
    for i in range(3):
        train_data[i] = train_data[i][index]


def write_json(json_file, path):
    json.dump(json_file, open(path, 'w'))


def print_error(value):
    print("error:", value)


def run(args, ):
    print("strategy:", args.strategy)
    data = []
    base_path = args.out_path + 'DE4T32/'
    for user_name in tqdm.tqdm(range(1, 41)):
        q2d2f = json.load(open(base_path + str(user_name) + '.json'))
        for q in q2d2f.keys():
            for d in q2d2f[q].keys():
                frequency = np.array(q2d2f[q][d]['eeg'])[:, :4]
                temporal = np.array(
                    [item for item in np.array(q2d2f[q][d]['eeg'])[:, 4:]])
                q2d2f[q][d]['eeg'] = np.concatenate(
                    [frequency, temporal], axis=1).tolist()
                data.append(
                    [{'user_name': user_name, 'q': q, 'd': d}, q2d2f[q][d]])
    q_set_list = list(set([data[i][0]['q'] for i in range(len(data))]))
    q_set_list = sorted(q_set_list)

    from multiprocessing import Pool
    if args.strategy == "PALL":
        os.makedirs(base_path + args.strategy, exist_ok=True)
        for uid in tqdm.tqdm(range(1, 41)):
            pool = Pool(20)
            all_data = [data[item] for item in range(
                len(data)) if data[item][0]['user_name'] == uid]
            random.shuffle(all_data)
            for i in range(10):
                train = [all_data[idx]
                         for idx in range(len(all_data)) if idx % 10 != i]
                valid = [all_data[idx]
                         for idx in range(len(all_data)) if idx % 10 == i]
                pool.apply_async(write_json, args=(train, base_path + args.strategy +
                                 '/train_' + str(uid * 10 + i)+'.json'), error_callback=print_error)
                pool.apply_async(write_json, args=(valid, base_path + args.strategy +
                                 '/valid_' + str(uid * 10 + i)+'.json'), error_callback=print_error)
            pool.close()
            pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Argument of running data spliting.')
    parser.add_argument('-strategy', type=str,
                        help='Training strategy.', default='PALL', choices=['PALL'])
    args = parser.parse_args()
    args.out_path = '../data/amigos_preprocessed/'
    run(args)
