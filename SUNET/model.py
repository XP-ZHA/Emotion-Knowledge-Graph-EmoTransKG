import torch
from torch import nn
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import config
import torch.nn.functional as F

class BLSTMClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.lstm = LSTM(input_size=config.input_dim,
                    hidden_size=config.hidden_dim // 2,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True)
        self.clf = nn.Linear(config.hidden_dim, config.nclass)

    def forward(self, x, adj, a, b, c):
        x = self.lstm(x)[0]
        x = self.dropout(x)
        return self.clf(x)

def mask_logic(alpha, adj):
    '''
    performing mask logic with adj
    :param alpha:
    :param adj:
    :return:
    '''
    return alpha - (1 - adj) * 1e30

class GatDot(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, Q, K, V, adj, s_mask):
        '''
        imformation gatherer with dot product attention
        :param Q: (B, D) # query utterance
        :param K: (B, N, D) # context
        :param V: (B, N, D) # context
        :param adj: (B,  N) # the adj matrix of the i th node
        :return:
        '''
        N = K.size()[1]
        Q = self.linear1(Q).unsqueeze(2) # (B,D,1)
        # K = self.linear2(Q) # (B, N, D)
        K = self.linear2(K) # (B, N, D)
        alpha = torch.bmm(K, Q).permute(0, 2, 1)  # (B, 1, N)
        adj = adj.unsqueeze(1)
        alpha = mask_logic(alpha, adj)  # (B, 1, N)
        attn_weight = F.softmax(alpha, dim=2)  # (B, 1, N)
        attn_sum = torch.bmm(attn_weight, V).squeeze(1)  # (B,  D)
        return attn_weight, attn_sum

class GatDot_rel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.rel_emb = nn.Embedding(2, hidden_size)

    def forward(self, Q, K, V, adj, s_mask):
        '''
        imformation gatherer with dot product attention
        :param Q: (B, D) # query utterance
        :param K: (B, N, D) # context
        :param V: (B, N, D) # context
        :param adj: (B,  N) # the adj matrix of the i th node
        :param s_mask: (B,  N) #  relation mask
        :return:
        '''
        N = K.size()[1]
        rel_emb = self.rel_emb(s_mask)
        Q = self.linear1(Q).unsqueeze(2) # (B,D,1)
        K = self.linear2(K) # (B, N, D)
        y = self.linear3(rel_emb) # (B, N, 1
        alpha = (torch.bmm(K, Q) + y).permute(0, 2, 1)  # (B, 1, N)
        adj = adj.unsqueeze(1)   # adj-> [B,1,i]
        alpha = mask_logic(alpha, adj)  # (B, 1, N)
        attn_weight = F.softmax(alpha, dim=2)  # (B, 1, N)
        attn_sum = torch.bmm(attn_weight, V).squeeze(1)  # (B,  D)
        return attn_weight, attn_sum

class DAGERC(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.input_dim, config.hidden_dim)
        gats1 = []
        for _ in range(config.layers):
            gats1 += [GatDot(config.hidden_dim)]
        self.gather1 = nn.ModuleList(gats1)
        grus = []
        linears = []
        for _ in range(config.layers):
            grus += [nn.GRUCell(config.hidden_dim, config.hidden_dim)]
            linears += [nn.Linear(config.hidden_dim*2, config.hidden_dim)]
        self.grus = nn.ModuleList(grus)
        self.linears = nn.ModuleList(linears)
        # for out put
        in_dim = config.hidden_dim * (config.layers + 1) + config.input_dim
        # output mlp layers
        layers = [nn.Linear(in_dim, config.hidden_dim), nn.ReLU()]
        for _ in range(config.layers - 1):
            layers += [nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(config.hidden_dim, config.nclass)]
        self.out_mlp = nn.Sequential(*layers)

    def forward(self, x, adj, s_mask, s_feature, s_adj):
        '''
        :param x:   feature B,N,D
        :param adj:  B,N,N
        :param s_mask: B,N,N,2
        :param s_feature: B,M,D
        :param s_adj: B,M,N
        :return:
        '''
        num_utter = x.size()[1]
        H0 = F.relu(self.fc1(x)) # (B, N, D)
        s_feature = F.relu(self.fc2(s_feature)) # (B, M, D)   # speaker features
        H = [H0]
        P = [s_feature]
        for l in range(config.layers):
            Mp = torch.bmm(s_adj.permute(0,2,1), P[l])
            H1 = self.grus[l](H[l][:, 0, :]).unsqueeze(1)
            for i in range(1, num_utter):
                _, M = self.gather1[l](H[l][:, i, :], H1, H1, adj[:, i, :i], s_mask)
                Mpi = Mp[:, i, :]  # B,1,D
                M = torch.cat([M, Mpi], dim=-1)
                M = self.linears[l](M)
                H1 = torch.cat((H1, self.grus[l](H[l][:, i, :], M).unsqueeze(1)), dim=1)
            H1 = self.dropout(H1)
            H.append(H1)
            # update P from utterance
            P_l_1 = torch.bmm(s_adj, H1)
            Plm = []
            for m in range(P_l_1.shape[1]):
                Plm.append(self.grus[l](P[l][:, m, :], P_l_1[:, m, :]).unsqueeze(1))
            Pl = torch.cat(Plm, dim=1)  # B,M,D
            P.append(Pl)
        H.append(x)
        H = torch.cat(H, dim=2)  # (B, N, l*D)
        logits = self.out_mlp(H)
        return logits

class RGATLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, input_dim, out_dim, dropout=0.1, alpha=0.1, concat=True):
        super(RGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = input_dim
        self.out_features = out_dim
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(self.in_features, self.out_features))).cuda()
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(self.out_features*3, 1))).cuda()
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.rel_emb = nn.Embedding(2, config.hidden_dim)

    def forward(self, x, adj, s_mask):
        h = torch.matmul(x, self.W)  # B,N,D
        N = h.size()[1]  # nV=30
        batch = h.size()[0]
        h1 = h.repeat(1, 1, N).view(batch, N * N, -1)
        h2 = h.repeat(1, N, 1)
        r = self.rel_emb(s_mask).view(batch, N*N, -1)
        # attention with relation
        a_input = torch.cat([h1, h2, r], dim=2).view(batch, N, N, 3*self.out_features)  # [B, N, N, 3D]
        # a_input = torch.cat([h1, h2]).view(batch, N, N, 2*self.out_features)  # [B, N, N, 3D]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # [128, 30, 30]
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # adj > 0 zero_vec
        attention = F.softmax(attention, dim=2)  # [128, 30, 30]
        self.att = attention
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GRU3d(nn.Module):
    def __init__(self, hidden = config.hidden_dim):
        super().__init__()
        self.hidden = hidden
        self.w_z = nn.Parameter(torch.Tensor(hidden, hidden))
        self.u_z = nn.Parameter(torch.Tensor(hidden, hidden))
        self.w_r = nn.Parameter(torch.Tensor(hidden, hidden))
        self.u_r = nn.Parameter(torch.Tensor(hidden, hidden))
        self.w = nn.Parameter(torch.Tensor(hidden, hidden))
        self.u = nn.Parameter(torch.Tensor(hidden, hidden))
        self.reset_parameters()

    def forward(self, x, c):   #
        '''
        :param x: current hidden state (cat of p_l_1 and aggregation of RGAT) #B,N,D
        :param c: cell state(last hidden state) #B,N,2D
        :return:
        '''
        z = torch.sigmoid(torch.matmul(x, self.w_z) + torch.matmul(c, self.u_z))
        r = torch.sigmoid(torch.matmul(x, self.w_r) + torch.matmul(c, self.u_r))
        h_hat = torch.tanh(torch.matmul(x, self.w) + r*torch.matmul(c, self.u))
        return (1-z)*x + z*h_hat

    def reset_parameters(self):
        std = 0.1
        for weight in self.parameters():
            weight.data.normal_(mean=0.0, std=std)  # 平均数为0方差为0.1的标准正态分布

class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale:
        Return:
            self-attention
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head=4):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x, K, Q):
        batch_size = x.size(0)
        V = self.fc_V(x)
        Q = self.fc_Q(Q)
        K = self.fc_K(K)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V, scale)
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        # out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out

class PERC(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        # self.embedding = nn.Embedding(config.speaker_vocab, config.hidden_dim)
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.input_dim, config.hidden_dim)
        gats = []
        for _ in range(config.layers):
            gats += [RGATLayer(config.hidden_dim, config.hidden_dim)]
        self.gather = nn.ModuleList(gats)
        grus, grus2 = [], []
        Aggs = []
        for _ in range(config.layers):
            grus += [GRU3d(config.hidden_dim)]
            if config.att_agg:
                Aggs += [Multi_Head_Attention(config.hidden_dim)]
            else:
                Aggs += [nn.Linear(config.hidden_dim*3, config.hidden_dim)]
        self.grus = nn.ModuleList(grus)
        self.agg = nn.ModuleList(Aggs)
        self.out_mlp = nn.Linear(config.hidden_dim, config.nclass)
        # self.layer_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x, adj, s_mask, s_feature, s_adj, speaker_id):
        '''
        :param x:   feature B,N,D
        :param adj:  B,N,N
        :param s_mask: B,N,N,2
        :param s_feature: B,M,D
        :param s_adj: B,M,N
        :return:
        '''
        H0 = F.relu(self.fc1(x))  # (B, N, D)
        H0 = self.dropout(H0)
        H = [H0]
        if config.init_way == 'global':
            s_feature = F.relu(self.fc2(s_feature))  # (B, M, D)   # speaker features
        if config.init_way == 'random':
            s_feature = F.relu(self.fc2(s_feature))
            s_feature = torch.rand_like(s_feature)
        if config.init_way == 'local':
            s_feature = torch.bmm(s_adj, x)
            s_feature = F.relu(self.fc2(s_feature))
        # if config.init_way == 'embed':
        #     s_feature = self.embedding(speaker_id)
        #     # print(s_feature.shape)

        P = [s_feature]
        for l in range(config.layers):
            P_l_1 = torch.bmm(s_adj.permute(0, 2, 1), P[l])
            Agg = self.gather[l](H[l], adj, s_mask)
            if config.att_agg:
                H_l = self.agg[l](H[l], P_l_1, Agg)
            else:
                Agg_ = torch.cat([H[l], P_l_1, Agg], dim=-1)  # B,N,D,3
                H_l = F.relu(self.agg[l](Agg_))
            H_l = self.dropout(H_l)
            H.append(H_l)
            # update P_l from all utteracne and P_l_1
            P_l = torch.bmm(s_adj, H[l])
            P_l = self.grus[l](P[l], P_l)
            P.append(P_l)
        p_sim = self.sim_loss(P[-1])
        # print(P[-1].shape, H[-1].shape)
        # H = torch.cat([P_l_1, H[-1]],dim=-1)
        logits = self.out_mlp(H[-1])
        # logits = self.out_mlp(H0)
        return logits, p_sim

    def sim_loss(self, p):
        p_sim = torch.bmm(p, p.permute(0, 2, 1))
        m = p_sim.shape[-1]
        b = p_sim.shape[0]
        mask = torch.ones_like(p_sim)
        eyes = torch.eye(m).repeat(b,1,1).cuda()
        mask = mask - eyes
        p_sim = p_sim*mask
        p_sim = torch.mean(p_sim, dim=-1)
        p_sim = torch.mean(p_sim, dim=-1)
        p_sim = torch.mean(p_sim, dim=-1)
        return p_sim