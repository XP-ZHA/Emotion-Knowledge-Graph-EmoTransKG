import json
import numpy as np
import numpy as np
import pandas as pd
import pandas as pd
import pickle
import pickle
import random
import random
import torch
import torch
import torch.nn.functional as F
from pandas import DataFrame
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import Dataset


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class IEMOCAPRobertaCometDataset(Dataset):
    def __init__(self, split):
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.speakers, self.labels, self.roberta1, self.roberta2, self.roberta3, self.roberta4, self.sentences, self.trainId, self.testId, self.validId = pickle.load(
            open('../iemocap_features_roberta.pkl', 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainId]
            instances = []
            for x in self.trainId:
                r = torch.FloatTensor(self.roberta1[x])
                label = self.labels[x]
                speaker = self.speakers[x]
                j = 0
                while j <= len(label) - 2:
                    k = j + 1
                    while k <= len(label) - 1:
                        if speaker[j] == speaker[k]:
                            features = []
                            label_1 = []
                            labal_2 = []
                            label_1.append(list(np.squeeze(np.array(
                                F.one_hot(torch.tensor(label[j]), num_classes=6).type(torch.FloatTensor))).astype(
                                'float32')))
                            labal_2.append(label[k])
                            for s in range(j, k + 1):
                                features.append(r[s].tolist())
                            lengths = len(features)
                            dd = list(np.array(torch.zeros(np.array(r[j]).shape)))
                            while lengths < 17:
                                features.append(dd)
                                lengths = lengths + 1
                            instances.append({'features': features, 'label1': labal_1, 'label2': labal_2})
                            break
                        k = k + 1
                    j = j + 1

        elif split == 'valid':
            self.keys = [x for x in self.validId]
            instances = []
            for x in self.validId:
                r = torch.FloatTensor(self.roberta1[x])
                label = self.labels[x]
                speaker = self.speakers[x]
                j = 0
                while j <= len(label) - 2:
                    k = j + 1
                    while k <= len(label) - 1:
                        if speaker[j] == speaker[k]:
                            features = []
                            label_1 = []
                            label_2 = []
                            label_1.append(list(np.squeeze(np.array(
                                F.one_hot(torch.tensor(label[j]), num_classes=6).type(torch.FloatTensor))).astype(
                                'float32')))
                            label_2.append(label[k])
                            for s in range(j, k + 1):
                                features.append(r[s].tolist())
                            lengths = len(features)
                            dd = list(np.array(torch.zeros(np.array(r[j]).shape)))
                            while lengths < 17:
                                features.append(dd)
                                lengths = lengths + 1
                            instances.append({'features': features, 'label1': label_1, 'label2': label_2})
                            break
                        k = k + 1
                    j = j + 1

        random.shuffle(raw_data)
        self.data = raw_data
        self.len = len(self.data)

    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]['features']), torch.FloatTensor(
            self.data[index]['label1']), torch.LongTensor(self.data[index]['label2'])

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        feaures = pad_sequence([d[0] for d in data], batch_first=True)  # (B, N, D)
        labels_1 = pad_sequence([d[1] for d in data], batch_first=True, padding_value=-1)  # (B, N )
        labels_2 = pad_sequence([d[2] for d in data], batch_first=True, padding_value=-1)  # (B, N )
        return feaures, labels_1, labels_2
