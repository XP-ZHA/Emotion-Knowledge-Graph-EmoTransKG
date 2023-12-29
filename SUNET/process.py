# Steps for this work

import pandas as pd
import config
import json
import numpy as np
from tqdm import tqdm
import pickle
import config

# def get_speaker_feas():
#     dev_frame = pd.read_csv('../data/datasets/{}/dev_sent_emo.csv'.format(config.dataset))
#     test_frame = pd.read_csv('../data/datasets/{}/test_sent_emo.csv'.format(config.dataset))
#     train_frame = pd.read_csv('../data/datasets/{}/train_sent_emo.csv'.format(config.dataset))
#
#     frame = dev_frame.append(test_frame).append(train_frame)
#
#     speakers = list(frame['Speaker'].values)
#     utterances = list(frame['Utterance'].values)
#
#     assert len(speakers) == len(utterances)

def get_speaker_feas():
    with open('../data/roberta_feas/{}/dev_data_roberta.json.feature'.format(config.dataset), 'r') as rf:
        dev = json.load(rf)
    with open('../data/roberta_feas/{}/test_data_roberta.json.feature'.format(config.dataset), 'r') as rf:
        test = json.load(rf)
    with open('../data/roberta_feas/{}/train_data_roberta.json.feature'.format(config.dataset), 'r') as rf:
        train = json.load(rf)
    print(len(dev), len(test), len(train))
    # datas = dev + test + train
    datas = train
    Speaker2fea = {}
    i = 0
    for conversation in tqdm(datas):
        i += 1
        for utterance in conversation:
            speaker = utterance['speaker']
            # print(i, speaker)
            # print(len(utterance['feature']))
            cls = utterance['cls']
            if speaker not in Speaker2fea.keys():
                Speaker2fea[speaker] = [np.array(cls)]
            else:
                Speaker2fea[speaker].append(np.array(cls))
    speaker2ave_fea = {}
    for speaker in Speaker2fea.keys():
        # print(len(Speaker2fea[speaker]))
        feas = np.array(Speaker2fea[speaker])
        feas = np.mean(feas, axis=0)
        speaker2ave_fea[speaker] = feas
    with open('../data/roberta_feas/{}/s2f.pkl'.format(config.dataset), 'wb') as wf:
        pickle.dump(speaker2ave_fea, wf)


if __name__ == '__main__':
    get_speaker_feas()
    with open('../data/roberta_feas/{}/label_vocab.pkl'.format(config.dataset), 'rb') as rf:
        label = pickle.load(rf)
        print(label)