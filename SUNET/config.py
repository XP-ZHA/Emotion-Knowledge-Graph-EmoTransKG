# dataset = 'IEMOCAP'  # 50
dataset = 'MELD'   # 10
# dataset = 'DailyDialog'   # 8
# dataset = 'EmoryNLP'   # 14
max_speaker = {
    'IEMOCAP' : 2,
    'MELD': 9,
    'DailyDialog':2,
    'DailyDialog2':2,
    'EmoryNLP':9
}  # 8 for MELD
all_speaker = {
    'IEMOCAP' : 303,
    'MELD': 305,
    'DailyDialog':2,
    'DailyDialog2':2,
    'EmoryNLP':286
}  # 8 for MELD
speaker_vocab = all_speaker[dataset]
max_speaker_num = max_speaker[dataset]
batch_size = 32
if dataset == 'IEMOCAP':
    nclass = 6
    label_names = ['exc', 'neu', 'fru', 'sad', 'hap', 'ang']
else:
    nclass = 7
    label_names = ['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang']
input_dim = 1024
hidden_dim = 300
lr = 1e-4
epoch = 100
cuda = True
w1 = 2   # past window
w2 = 0
layers = 2
dropout = 0.1
att_agg = False
alpha = 1e-10
init_way = 'global'   # local random embed global
fix_user = ['Chandler', 'Ross', 'Phoebe', 'Joey', 'Monica', 'Rachel']