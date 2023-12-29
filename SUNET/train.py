import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
from tqdm import tqdm
import json
import logging
import os
import dataloader
from model import *

def init_logger(log_name: str = "echo", log_file='log', log_file_level=logging.NOTSET):
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    file_handler = logging.FileHandler(log_file, encoding="utf8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

logger = init_logger()
# seed = random.randint(0, 10000)
def seed_everything(seed=1996):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def eval(model, test_iter, device):
    model.eval()
    targets, preds = [],[]
    for step, data in enumerate(test_iter):
        feaures, labels, adj, s_mask, s_feature, s_adj, speakers, names, utterances = data
        out, p_sim = model(feaures.cuda(), adj.cuda(), s_mask.cuda(), s_feature.cuda(), s_adj.cuda(), speakers.cuda())
        target = list(labels.cpu().detach().numpy())
        pred = list(torch.max(out, dim=-1)[1].cpu().detach().numpy())
        for i,label in enumerate(target):
            for j,l in enumerate(label):
                if l != -1:
                    targets.append(l)
                    preds.append(pred[i][j])
    model.train()
    return targets, preds

def eval_speaker(model, test_iter, device):
    model.eval()
    targets, preds, out_names = [], [], []
    for step, data in enumerate(test_iter):
        feaures, labels, adj, s_mask, s_feature, s_adj, speakers, names, utterances = data
        out, p_sim = model(feaures.cuda(), adj.cuda(), s_mask.cuda(), s_feature.cuda(), s_adj.cuda(), speakers.cuda())
        target = list(labels.cpu().detach().numpy())
        pred = list(torch.max(out, dim=-1)[1].cpu().detach().numpy())
        for i,label in enumerate(target):
            for j,l in enumerate(label):
                if l != -1:
                    targets.append(l)
                    preds.append(pred[i][j])
                    out_names.append(names[i][j])
    model.train()
    assert len(targets) == len(preds) == len(out_names)
    user_label_dict = {}
    for key in config.fix_user:
        user_label_dict[key] = [[], []]
    for i in range(len(targets)):
        if out_names[i] in config.fix_user:
            user_label_dict[out_names[i]][0].append(targets[i])
            user_label_dict[out_names[i]][1].append(preds[i])
    for key in config.fix_user:
        print(key, round(f1_score(user_label_dict[key][0], user_label_dict[key][1], average='weighted') * 100, 2))
        print(key, classification_report(user_label_dict[key][0], user_label_dict[key][1]))

def get_cases(model, test_iter, device):
    model.eval()
    emos = ['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang']
    targets, preds, us = [], [], []
    for step, data in enumerate(test_iter):
        feaures, labels, adj, s_mask, s_feature, s_adj, speakers, names, utterances = data
        out, p_sim = model(feaures.cuda(), adj.cuda(), s_mask.cuda(), s_feature.cuda(), s_adj.cuda(), speakers.cuda())
        target = list(labels.cpu().detach().numpy())
        pred = list(torch.max(out, dim=-1)[1].cpu().detach().numpy())
        for i, label in enumerate(target):
            t, p = [], []
            for j, l in enumerate(label):
                if l != -1:
                    t.append(emos[l])
                    p.append(emos[pred[i][j]])
            targets.append(t)
            preds.append(p)
        us.extend(utterances)
    print(len(targets), len(preds), len(us))
    with open('case.txt', 'w', encoding='utf-8') as wf:
        for i in range(len(targets)):
            wf.write(' '.join(targets[i]))
            wf.write('\n')
            wf.write(' '.join(preds[i]))
            wf.write('\n')
            wf.write('\n'.join(us[i]))
            wf.write('\n')
    model.train()

def train(config):
# start training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(1)
    train_loader, val_loader, test_loader = dataloader.get_My_loaders()
    # model = DAGERC().to(device)
    # model = BLSTMClass().to(device)
    model = PERC().to(device)
    # print(model)

    lr = config.lr
    CE = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.99)
    best_acc, best_micro_f1, best_macro_f1, best_weighted_f1 = 0, 0, 0, 0
    best_val_acc, best_val_micro_f1, best_val_macro_f1, best_val_weighted_f1 = 0, 0, 0, 0
    for epoch in range(config.epoch):
        # if epoch > 50:
        #     lr = lr*0.95
        #     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        for step, data in enumerate(train_loader):
            feaures, labels, adj, s_mask, s_feature, s_adj, speakers, names, utterances = data
            out, p_sim = model(feaures.cuda(), adj.cuda(), s_mask.cuda(), s_feature.cuda(), s_adj.cuda(), speakers.cuda())
            loss = CE(out.view(-1, config.nclass), labels.cuda().view(-1))
            if config.alpha != 0:
                loss = loss + config.alpha*p_sim
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            if step % 100 == 0:
                logger.info("epoch-{}, step-{}, loss:{}".format(epoch, step, loss.data))
        # print('========================VAL============================')
        # targets, preds = eval(model, val_loader, device)
        # avg_accuracy = round(accuracy_score(targets, preds) * 100, 2)
        # avg_micro_f1 = round(f1_score(targets, preds, average='micro', labels=list(range(1, 7))) * 100, 2)
        # avg_macro_f1 = round(f1_score(targets, preds, average='macro') * 100, 2)
        # avg_weighted_f1 = round(f1_score(targets, preds, average='weighted') * 100, 2)
        # if avg_macro_f1 > best_val_macro_f1:
        #     best_val_macro_f1 = avg_macro_f1
        # if avg_micro_f1 > best_val_micro_f1:
        #     best_val_micro_f1 = avg_micro_f1
        # if avg_weighted_f1 > best_val_weighted_f1:
        #     best_val_weighted_f1 = avg_weighted_f1
        # if avg_accuracy > best_val_acc:
        #     best_val_acc = avg_accuracy
        # print('avg_micro_f1-{}, avg_weighted_f1-{}'.format(avg_micro_f1, avg_weighted_f1))
        # print('best_micro_f1-{}, best_weighted_f1-{}'.format(best_val_micro_f1, best_val_weighted_f1))
        # print('best_acc-{}, best_macro_f1-{}'.format(best_val_acc, best_val_macro_f1))
        print('========================TEST============================')
        targets, preds = eval(model, test_loader, device)
        avg_accuracy = round(accuracy_score(targets, preds) * 100, 2)
        avg_micro_f1 = round(f1_score(targets, preds, average='micro', labels=list(range(1, 7))) * 100, 2)
        avg_macro_f1 = round(f1_score(targets, preds, average='macro') * 100, 2)
        avg_weighted_f1 = round(f1_score(targets, preds, average='weighted') * 100, 2)
        confuse = confusion_matrix(targets, preds)
        if avg_macro_f1 > best_macro_f1:
            best_macro_f1 = avg_macro_f1
        if avg_micro_f1 > best_micro_f1:
            best_micro_f1 = avg_micro_f1
        if avg_weighted_f1 > best_weighted_f1:
            best_weighted_f1 = avg_weighted_f1
            # eval_speaker(model, test_loader, device)
            get_cases(model, test_loader, device)
            # torch.save(model.state_dict(), '../results/{}_best_model_wo_speaker'.format(config.dataset))
        if avg_accuracy > best_acc:
            best_acc = avg_accuracy
        print('avg_micro_f1-{}, avg_weighted_f1-{}'.format(avg_micro_f1, avg_weighted_f1))
        print('best_micro_f1-{}, best_weighted_f1-{}'.format(best_micro_f1, best_weighted_f1))
        print('best_acc-{}, best_macro_f1-{}'.format(best_acc, best_macro_f1))

if __name__ == '__main__':
    train(config)