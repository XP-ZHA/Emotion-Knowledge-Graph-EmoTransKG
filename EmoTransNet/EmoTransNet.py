import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from dataloader import IEMOCAPRobertaCometDataset
from model import EmoTransNet
from sklearn.metrics import f1_score, accuracy_score


def seed_everything(seed=100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_IEMOCAP_loaders(batch_size=32, num_workers=0, pin_memory=False):
    trainset = IEMOCAPRobertaCometDataset('train')
    validset = IEMOCAPRobertaCometDataset('valid')

    train_loader = DataLoader(trainset, batch_size=batch_size, collate_fn=trainset.collate_fn, num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset, batch_size=batch_size, collate_fn=validset.collate_fn, num_workers=num_workers,
                              pin_memory=pin_memory)

    return train_loader, valid_loader


def train_or_eval_model(model, dataloader, epoch, optimizer=None, train=False):
    losses, preds, labels, masks, losses_sense = [], [], [], [], []
    assert not train or optimizer != None

    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        features, labels_1, labels_2 = data

        if cuda:
            features = features.cuda()
            labels_1 = np.squeeze(labels_1).cuda()
            labels_2 = np.squeeze(labels_2).cuda()
        log_prob = model(features, labels_1)  # (B, N, C)

        loss_function1 = nn.CrossEntropyLoss()
        loss = loss_function1(log_prob, labels_2)
        labels_2 = labels_2.cpu().numpy().tolist()
        pred_ = torch.argmax(log_prob, 1).cpu().numpy().tolist()  # batch*seq_len

        preds += pred_
        labels += labels_2
        losses.append(loss.item())

        if train:
            total_loss = loss
            total_loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds != []:
        new_preds = []
        new_labels = []
        for i, label in enumerate(labels):
            new_labels.append(label)
            new_preds.append(preds[i])
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)

    avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)
    return avg_loss, avg_accuracy, labels, preds, avg_fscore


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.00002, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0003, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--early_stopping_epoch', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1, metavar='seed', help='seed')
    parser.add_argument('--transformer_layers', type=int, default=6, help='layers of BERT')  
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    emo_gru = True
    n_classes = 6
    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    seed = args.seed
    dropout = args.dropout

    seed_everything(seed)

    model = EmoTransNet(args, n_classes, dropout)

    if cuda:
        model.cuda()

    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    train_loader, valid_loader = get_IEMOCAP_loaders(batch_size=batch_size, num_workers=0)
    valid_losses, valid_fscores = [], []

    best_loss, best_label, best_pred, best_mask = None, None, None, None
    all_fscore = []
    early_stopping_epochs = 0
    best_acc = 0
    for e in range(n_epochs):
        print('Epoch {} start: '.format(e + 1))
        start_time = time.time()
        train_loss, train_acc, _, _, train_fscore = train_or_eval_model(model, loss_function, train_loader, e,
                                                                        optimizer, True)
        valid_loss, valid_acc, _, _, valid_fscore = train_or_eval_model(model, loss_function, valid_loader, e)

        valid_losses.append(valid_loss)
        valid_fscores.append(valid_fscore)
        all_fscore.append([valid_fscore, test_fscore])
        if args.tensorboard:
            writer.add_scalar('train: accuracy/loss', train_acc / train_loss, e)

        x = 'epoch: {}, train_loss: {}, acc: {}, fscore: {}, valid_loss: {}, acc: {}, fscore: {},  time: {} sec'.format(
            e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore,
            round(time.time() - start_time, 2))

        path = "/home/"
        torch.save(model.state_dict(), path + 'EKG_IEMOCAP_pretrained_model.pkl')

        print(x)
        lf.write(x + '\n')

        if valid_acc > best_acc:
            best_acc = valid_acc
            early_stopping_epochs = 0

        else:
            early_stopping_epochs += 1

        if early_stopping_epochs == args.early_stopping_epoch:
            break
    if args.tensorboard:
        writer.close()

    valid_fscores = np.array(valid_fscores).transpose()

    all_fscore = sorted(all_fscore, key=lambda x: (x[0], x[1]), reverse=True)

