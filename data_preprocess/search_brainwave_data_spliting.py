
import argparse
import configparser
import os
import numpy as np
import json
import tqdm


def load_info(args):
    with open(args.info_path) as f:
        lines = f.readlines()
        return json.loads(lines[1].split('=')[1].strip())


def shuffle(train_data):
    index = np.arange(train_data[0].shape[0])
    np.random.shuffle(index)
    for i in range(3):
        train_data[i] = train_data[i][index]


def run(args):
    uid_list = load_info(args)
    data = []
    base_path = args.base_path + 'DE5T62/'
    os.makedirs(base_path + args.strategy, exist_ok=True)
    for user_name in tqdm.tqdm(uid_list):
        q2d2f = json.load(open(base_path + str(user_name) + '.json'))
        for q in q2d2f.keys():
            for d in q2d2f[q].keys():
                frequency = np.array(q2d2f[q][d]['eeg'])[:, :5]
                temporal = np.array(
                    [item for item in np.array(q2d2f[q][d]['eeg'])[:, 5:]])
                q2d2f[q][d]['eeg'] = np.concatenate(
                    [frequency, temporal], axis=1).tolist()
                data.append(
                    [{'user_name': user_name, 'q': q, 'd': d}, q2d2f[q][d]])
    q_set_list = list(set([data[i][0]['q'] for i in range(len(data))]))
    q_set_list = sorted(q_set_list)

    if args.strategy == "CVOQ":
        valid_number = 10
        for valid_id in tqdm.tqdm(range(valid_number)):
            train = [data[item] for item in range(len(data)) if q_set_list.index(
                data[item][0]['q']) % valid_number != valid_id]
            valid = [data[item] for item in range(len(data)) if q_set_list.index(
                data[item][0]['q']) % valid_number == valid_id]
            json.dump(train, open(base_path + args.strategy +
                      '/train_' + str(valid_id)+'.json', 'w'))
            json.dump(valid, open(base_path + args.strategy +
                      '/valid_' + str(valid_id)+'.json', 'w'))
    elif args.strategy == "LOPO":
        valid_number = 18
        for valid_id in tqdm.tqdm(range(valid_number)):
            user_name = uid_list[valid_id]
            train = [data[item] for item in range(
                len(data)) if data[item][0]['user_name'] != user_name]
            valid = [data[item] for item in range(
                len(data)) if data[item][0]['user_name'] == user_name]
            json.dump(train, open(base_path + args.strategy +
                      '/train_' + str(valid_id)+'.json', 'w'))
            json.dump(valid, open(base_path + args.strategy +
                      '/valid_' + str(valid_id)+'.json', 'w'))
    elif args.strategy == 'PCVOQ':
        valid_id = 0
        for u in tqdm.tqdm(range(18)):
            user_name = uid_list[u]
            valid_number = 10
            for q in range(10):
                train = [data[item] for item in range(len(data)) if data[item][0]['user_name'] == user_name and q_set_list.index(
                    data[item][0]['q']) % valid_number != q]
                valid = [data[item] for item in range(len(data)) if data[item][0]['user_name'] == user_name and q_set_list.index(
                    data[item][0]['q']) % valid_number == q]
                json.dump(train, open(base_path + args.strategy +
                          '/train_' + str(valid_id)+'.json', 'w'))
                json.dump(valid, open(base_path + args.strategy +
                          '/valid_' + str(valid_id)+'.json', 'w'))
                valid_id += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Argument of running data spliting.')
    ''' spliting choices:
        LOPO: leave one participant out
        CVOQ: cross validation on questions/tasks
        PCVOQ: cross validation on questions/tasks for each subject/participants
    '''
    parser.add_argument('-strategy', type=str, help='spliting strategy.',
                        default='PCVOQ', choices=['LOPO', 'CVOQ', 'PCVOQ', ])
    parser.add_argument('-info_path', type=str, help='data path',
                        default='../../../../../dataset/Non-click/info.txt')
    parser.add_argument('-base_path', type=str,
                        default='../data/search_brainwave_preprocessed/')
    args = parser.parse_args()
    run(args)
