'''
My Dataset Class to build dataloader
'''
import pickle

import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import torch
import config
from torch.utils.data import DataLoader

class MyDataset(Dataset):

    def __init__(self, dataset_name, split):
        self.dataset_name = dataset_name
        self.load_vocab()
        self.data = self.read(dataset_name, split)
        self.len = len(self.data)
        # How many speakers in each conversation? 9 for MELD
        self.max_speaker_len = [len(set(each['utterances'])) for each in self.data]
        # print(sum(self.max_speaker_len)/len(self.max_speaker_len))

    def load_vocab(self):
        with open('../data/roberta_feas/%s/speaker_vocab.pkl'%(self.dataset_name), 'rb') as f:
            self.speaker_vocab = pickle.load(f)
            # print(self.speaker_vocab)
        with open('../data/roberta_feas/%s/label_vocab.pkl'%(self.dataset_name), 'rb') as f:
            self.label_vocab = pickle.load(f)
            print(self.label_vocab)

    def read(self, dataset_name, split):
        with open('../data/roberta_feas/%s/%s_data_roberta.json.feature'%(dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)
        with open('../data/roberta_feas/%s/s2f.pkl'%(dataset_name), 'rb') as f:
            self.speaker2fea = pickle.load(f)
        # process dialogue
        dialogs = []
        for d in raw_data:
            utterances = []
            labels = []
            speakers = []
            names = []
            features = []
            for i, u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)
                id = self.speaker_vocab['stoi'][u['speaker']]
                speakers.append(id)
                features.append(u['cls'])
                names.append(u['speaker'])
                # features.append(u['feature'])
            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'features': features,
                'names':names
            })
            # print(speakers, len(features))
        print(len(self.speaker_vocab['stoi'].keys()))
        return dialogs

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''
        return torch.FloatTensor(self.data[index]['features']), \
               torch.LongTensor(self.data[index]['labels']),\
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['utterances'], self.data[index]['names']

    def __len__(self):
        return self.len

    def get_adj(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for speaker in speakers:   # for each batch
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i in range(len(speaker)):
                w1, w2 = config.w1, config.w2
                j = i-1
                while w1 > 0 and j >= 0:
                    a[i, j] = 1
                    # a[j, i] = 1
                    w1 = w1 - 1
                    j = j - 1
            adj.append(a)
        return torch.stack(adj)

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype = torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i,j] = 1
                        s_onehot[i,j,1] = 1
                    else:
                        s_onehot[i,j,0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def get_speaker_feas_masks(self, speakers, max_dialog_len):
        s_adj = []
        s_feature = []
        s_ids = []
        for speaker in speakers:
            a = torch.zeros(config.max_speaker_num, max_dialog_len)  # M*N
            f = []    # feature for speaker matrix
            s_set = list(set(speaker))
            for i in s_set:
                s_name = self.speaker_vocab['itos'][i]  # speaker name
                if s_name in self.speaker2fea.keys():
                    f.append(self.speaker2fea[s_name])
                    # print(self.speaker2fea[s_name].shape)
                else:
                    f.append(np.random.randn(1024,))
            f.extend((config.max_speaker_num-len(f))*[np.zeros_like(f[0])])
            assert len(f) == config.max_speaker_num
            f = torch.FloatTensor(f)
            s_feature.append(f)
            for i in range(len(speaker)):
                s_id = speaker[i]   # speaker id
                a[s_set.index(s_id), i] = 1   # get utterance-speaker egdes
            # a = a/(torch.sum(a, dim=1).unsqueeze(-1)+1e-9)  # 归一化
            s_adj.append(a)
            s_id = list(s_set) + [0]*(config.max_speaker_num-len(s_set))
            s_ids.append(torch.LongTensor(s_id))
        return torch.stack(s_feature), torch.stack(s_adj), torch.stack(s_ids)

    def collate_fn(self, data):
        '''
        :param data:
            features, labels, speakers, length, utterances
        :return:
            features: (B, N, D) padded
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''
        max_dialog_len = max([d[3] for d in data])  # 最大对话长度
        feaures = pad_sequence([d[0] for d in data], batch_first=True)  # (B, N, D)
        labels = pad_sequence([d[1] for d in data], batch_first=True, padding_value=-1)  # (B, N )
        adj = self.get_adj([d[2] for d in data], max_dialog_len)
        s_mask, s_mask_onehot = self.get_s_mask([d[2] for d in data], max_dialog_len)
        # lengths = torch.LongTensor([d[3] for d in data])
        # speakers = pad_sequence([torch.LongTensor(d[-1]) for d in data], batch_first=True, padding_value=0)
        speakers = [d[-1] for d in data]
        s_feature, s_adj, s_ids = self.get_speaker_feas_masks([d[2] for d in data], max_dialog_len)
        # print(speakers.shape, s_feature.shape, s_adj.shape)
        utterances = [d[4] for d in data]
        return feaures, labels, adj, s_mask, s_feature, s_adj, s_ids, speakers, utterances

if __name__ == '__main__':
    dataset = MyDataset(config.dataset, split='test')
    test_loader = DataLoader(dataset,
                             batch_size=4,
                             collate_fn=dataset.collate_fn)
    for step, data in enumerate(test_loader):
        # print(data)
        break