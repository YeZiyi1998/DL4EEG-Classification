import pandas as pd
import os
import json
import argparse
from utils import *
import scipy.io as scio
import tqdm
import math
import numpy as np


def part1_csv_preprocesss(args):
    def user_id_info_preprocess(dic1, u2info, info_keys):
        for key in dic1['UserID'].keys():
            if str(int(float(dic1['UserID'][key]))) not in u2info.keys():
                u2info[str(int(float(dic1['UserID'][key])))] = []
            for info_key in info_keys:
                if type(info_key) == list:
                    u2info[str(int(float(dic1['UserID'][key])))].append(
                        float(info_key[1][dic1[info_key[0]][key]]))
                else:
                    u2info[str(int(float(dic1['UserID'][key])))].append(
                        float(dic1[info_key][key]))

    base_path = f'{args.out_path}/Metadata/'
    os.makedirs(base_path, exist_ok=True)
    u2v2info = {}
    u2info = {}
    Participant_Questionnaire = pd.read_csv(
        base_path + "Participant_Questionnaires.0.csv").to_dict()
    Participant_Panas_1 = pd.read_csv(base_path + "Participants_Panas.1.csv")
    Participant_Panas_2 = pd.read_csv(base_path + "Participants_Panas.2.csv")
    user_id_info_preprocess(Participant_Questionnaire, u2info, [
                            ['Gender', {'m': 1, 'f': 0}], 'Age'])
    user_id_info_preprocess(Participant_Panas_1, u2info, [
                            'HIGH PA', 'HIGH NA'])
    user_id_info_preprocess(Participant_Panas_2, u2info, ['Interested', 'Excited', 'Strong', 'Distressed', 'Upset', 'Guilty', 'Enthusiastic', 'Proud',
                            'Alert', 'Inspired', 'Scared', 'Hostile', 'Irritable', 'Ashamed', 'Determined', 'Attentive', 'Active', 'Nervous', 'Jittery', 'Afraid'])
    Participant_Personality = pd.read_csv(
        base_path + "Participants_Personality.5.csv")
    Participant_Personality = pd.DataFrame(
        Participant_Personality.values.T, index=Participant_Personality.columns, columns=Participant_Personality.index)

    for key in u2info.keys():
        if str(key) in Participant_Personality._stat_axis.values.tolist():
            u2info[key] += [float(item)
                            for item in Participant_Personality.loc[str(key), :].tolist()]

    Self_Assessment = open(base_path + 'SelfAsessment.0.csv').readlines()
    line_split = Self_Assessment[1].strip().split(',')
    info_keys = ['arousal', 'valence', 'dominance', 'liking', 'familiarity',
                 'neutral', 'disgust', 'happiness', 'surprise', 'anger', 'fear', 'sadness']

    start_index = line_split.index('arousal', len(info_keys))
    end_index = start_index + len(info_keys)
    for line in Self_Assessment[2:]:
        line_split = line.strip().split(',')
        UserID = str(int(float(line_split[0])))
        VedioID = str(int(line_split[1]))
        if UserID not in u2v2info.keys():
            u2v2info[UserID] = {}
        u2v2info[UserID][VedioID] = [
            float(item) for item in line_split[start_index:end_index]]
        u2v2info[UserID][VedioID] = dict(
            (info_keys[i], u2v2info[UserID][VedioID][i]) for i in range(len(info_keys)))
    out_path = f'{args.out_path}/Metadata/'

    json.dump(u2v2info, open(out_path + 'u2v2info' + '.json', 'w'))
    json.dump(u2info, open(out_path + 'u2info' + '.json', 'w'))


def down_sample(data, rate=4):
    return [np.mean(data[i*rate:(i+1)*rate]) for i in range(int(len(data)/rate))]


def FFT_Processing(in_path, out_path, filter, FREQ_BANDS):
    err_num = 0
    all_num = 0
    q2d2info = {}
    with open(in_path, 'rb') as file:
        # resolve the python 2 data problem by encoding : latin1
        subject = scio.loadmat(file)
        subject["joined_data"] = subject["joined_data"][0]
        subject["labels_selfassessment"] = subject["labels_selfassessment"][0]
        subject["labels_ext_annotation"] = subject["labels_ext_annotation"][0]

        for i in tqdm.tqdm(range(0, 16)):
            q2d2info[i] = {}
            # loop over 0-39 trails
            data = subject["joined_data"][i]
            rate = np.sum(np.isnan(data)) / data.size
            if rate > 0.5:
                continue
            labels = subject["labels_selfassessment"][i][0]
            extlabels = subject["labels_ext_annotation"][i][0]
            start = 0
            fs = 128
            window_size = fs
            num = 0
            while start + window_size < data.shape[0]:
                q2d2info[i][num] = {}
                tmp_preprocessed = []
                # 17 * 4
                tmp_temporal = []
                # 17 * 128
                for channel_id in range(data.shape[1]):
                    data[start: start+window_size, channel_id][np.isnan(
                        data[start: start+window_size, channel_id])] = 0
                    tmp_preprocessed.append([])
                    tmp_temporal.append(down_sample(
                        data[start: start+window_size, channel_id]))
                    for band in FREQ_BANDS.values():
                        all_num += 1
                        try:
                            v = math.log(
                                bandpower(data[start: start+window_size, channel_id], fs, band, window_sec=1))
                            if np.isnan(math.log(bandpower(data[start: start+window_size, channel_id], fs, band, window_sec=1))):
                                err_num += 1
                                tmp_preprocessed[-1].append(0)
                            else:
                                tmp_preprocessed[-1].append(v)
                        except Exception as e:
                            err_num += 1
                            tmp_preprocessed[-1].append(0)

                q2d2info[i][num]['eeg'] = np.concatenate(
                    [tmp_preprocessed, tmp_temporal], axis=1)
                q2d2info[i][num]['eeg'] = q2d2info[i][num]['eeg'].tolist()
                q2d2info[i][num]['score'] = labels.tolist()
                q2d2info[i][num]['annotation'] = extlabels.tolist()
                start += window_size
                num += 1
    print('err rate', err_num / all_num)
    json.dump(q2d2info, open(out_path, 'w'))


def print_error(value):
    print('multi_thread error:', value)


def part2_eeg_preprocesss(args):
    FREQ_BANDS = {
        "theta": [4, 8],     # 4-7
        "alpha": [8, 13],    # 8-12
        "beta": [13, 30],    # 13-30
        "gamma": [30, 50]
    }
    from multiprocessing.pool import Pool
    base_path = args.base_path
    pool = Pool(40)
    filter = True
    multi_thread = False
    subjectList = [item for item in os.listdir(base_path) if 'Data' in item]
    os.makedirs(args.out_path + 'DE4T32/', exist_ok=True)
    if multi_thread:
        for subjects in subjectList:
            out_path = args.out_path + 'DE4T32/' + \
                str(int(subjects.split('P')[-1])) + '.json'
            pool.apply_async(FFT_Processing, args=(os.path.join(
                base_path + subjects, f'{subjects}.mat'), out_path, filter, FREQ_BANDS), error_callback=print_error)
        pool.close()
        pool.join()
    else:
        for subjects in subjectList:
            out_path = args.out_path + 'DE4T32/' + \
                str(int(subjects.split('P')[-1])) + '.json'
            FFT_Processing(os.path.join(base_path + subjects,
                           f'{subjects}.mat'), out_path, filter, FREQ_BANDS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-base_path', type=str, help='data path',
                        default='../../../../../dataset/AMIGOS/')
    parser.add_argument('-meta_data_path', type=str,
                        help='metadata path: transform AMIGOS xlsx metadata into csv for further analyze', default='../data/amigos_preprocessed/')
    args = parser.parse_args()
    args.out_path = '../data/amigos_preprocessed/'
    os.makedirs(args.out_path, exist_ok=True)

    # part1 csv preprocess
    part1_csv_preprocesss(args)
    # part2 eeg process
    part2_eeg_preprocesss(args)
