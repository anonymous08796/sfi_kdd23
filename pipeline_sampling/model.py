from random import random
from typing import Optional
import matplotlib
from torch._C import device
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, SAGEConv
import scipy.stats as stats
from torch.distributions.beta import Beta

from pipeline_sampling.utils import gumbel_softmax_sample, kl_div

EPS = 1e-10


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2, skip=False):
        super(Encoder, self).__init__()
        self.base_model = base_model
        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            # self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            self.conv = [base_model(in_channels, 2 * out_channels)]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


class MyEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, s_channels, y_channels):
        super(MyEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bns1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bns2 = torch.nn.BatchNorm1d(hidden_channels)
        self.prelu = nn.ReLU(hidden_channels)
        self.linear_s = torch.nn.Linear(hidden_channels, s_channels, bias=False)
        self.linear_y = torch.nn.Linear(hidden_channels, y_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.bns1.reset_parameters()
        self.conv2.reset_parameters()
        self.bns2.reset_parameters()
        self.linear_s.reset_parameters()
        self.linear_y.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # x = self.bns1(x)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        # x = self.bns2(x)
        # x = self.prelu(x)
        out_s = self.linear_s(x)
        out_y = self.linear_y(x)
        return out_s, out_y, x


class GRACE(torch.nn.Module):
    def __init__(self, encoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(GRACE, self).__init__()
        self.encoder = encoder
        self.tau: float = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        self.num_hidden = num_hidden

    def forward(self, x: torch.Tensor, edge_index=None):
        if edge_index is None:
            return self.encoder(x)
        else:
            return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, epoch):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = self.sim(z1, z1)
        between_sim = self.sim(z1, z2)
        refl_sim = f(refl_sim)
        between_sim = f(between_sim)
        return -torch.log(between_sim.diag() / (between_sim.sum(1) + refl_sim.sum(1) - refl_sim.diag()))

    def semi_loss_bmm(self, z1, z2, select_feat, args, feat_weight=None, data=None, fit=None):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = self.sim(z1, z1)
        between_sim = self.sim(z1, z2)
        N = between_sim.size(0)
        mask = torch.ones((N, N), dtype=bool).to(z1.device)
        mask[np.eye(N, dtype=bool)] = False
        global B
        kl_loss = None

        if fit == 'div':
            between_sim_norm = between_sim.masked_select(mask).view(N, -1)
            between_sim_norm = (between_sim_norm + 1) / 2
            B = torch.rand((data.x[data.train_mask].shape[0], data.x[data.train_mask].shape[0])).to(z1.device)
            c_feat = data.x[data.train_mask, :]
            c_feat = c_feat[:, data.p_idx]
            c_feat = c_feat[:, select_feat]
            c_dis = np.array(data.dist_l)[select_feat, :]
            c_weight = feat_weight[select_feat]
            # print('Computing s weight, wait...')
            for i in range(B.shape[0]):
                # B[i, :] = torch.logical_xor(p_feature[i, :], p_feature).sum(-1) / num_po
                weighted_sim = torch.logical_xor(c_feat[i, :], c_feat).float() * c_weight
                B[i, :] = weighted_sim.sum(-1)
            B = B.masked_select(mask).view(N, -1) * between_sim_norm
            row_max = torch.max(B.detach(), 1, keepdim=True)[0]
            row_min = torch.min(B.detach(), 1, keepdim=True)[0]
            B = (B - row_min) / (row_max - row_min)
            # sample_res = gumbel_softmax_sample(B)
            sample_res = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(torch.tensor([1]).to(z1.device),
                                                                                B).rsample()
            sample_res = sample_res.sum(1) / sample_res.size(1)
            for idx in range(c_feat.size(1)):
                c = c_feat[:, idx]
                idx_0 = torch.where(c == 0)[0]
                idx_1 = torch.where(c == 1)[0]
                target_dis_0 = torch.Tensor([c_dis[idx][1]]).to(z1.device)
                target_dis_1 = torch.Tensor([c_dis[idx][0]]).to(z1.device)

                # loss_0 = F.kl_div((sample_res[idx_0].mean() + EPS).log(), target_dis_0, reduction='batchmean')
                # loss_1 = F.kl_div((sample_res[idx_1].mean() + EPS).log(), target_dis_1, reduction='batchmean')
                loss_0 = kl_div(sample_res[idx_0].mean() + EPS, target_dis_0)
                loss_1 = kl_div(sample_res[idx_1].mean() + EPS, target_dis_1)
                if idx == 0:
                    kl_loss = (loss_0 + loss_1) * c_weight[idx] * 0.5
                else:
                    kl_loss = kl_loss + (loss_0 + loss_1) * c_weight[idx] * 0.5

        if args.mode == 'weight':
            refl_sim = f(refl_sim)
            between_sim = f(between_sim)
            # ng_bet = (between_sim.masked_select(mask).view(N, -1) * B).sum(1) / B.mean(1)
            # ng_refl = (refl_sim.masked_select(mask).view(N, -1) * B).sum(1) / B.mean(1)
            ng_bet = (between_sim.masked_select(mask).view(N, -1) * B).sum(1)
            ng_refl = (refl_sim.masked_select(mask).view(N, -1) * B).sum(1)
            con_loss = (-torch.log(between_sim.diag() / (between_sim.diag() + ng_bet + ng_refl))).mean()
            if kl_loss is not None:
                # return con_loss
                return con_loss + kl_loss
            else:
                return con_loss

        elif args.mode == 'mix':
            eps = 1e-12
            sorted, indices = torch.sort(B, descending=True)
            N_sel = torch.gather(between_sim[mask].view(N, -1), -1, indices)[:, :args.sel_num]
            random_index = np.random.permutation(np.arange(args.sel_num))
            N_random = N_sel[:, random_index]
            M = sorted[:, :args.sel_num]
            M_random = M[:, random_index]
            M = (N_sel * M + N_random * M_random) / (M + M_random + eps)
            refl_sim = f(refl_sim)
            between_sim = f(between_sim)
            M = f(M)
            return -torch.log(between_sim.diag() / (M.sum(1) + between_sim.sum(1) + refl_sim.sum(1) - refl_sim.diag()))
        else:
            print('Mode Error!')

    def loss(self, z1, z2, epoch, args, select_feat, feat_weight, data, mean: bool = True,
             batch_size: Optional[int] = None):

        feat_weight_t = torch.Tensor(feat_weight).to(z1.device)
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if epoch <= args.epoch_start or args.fit_mode == 'Gen':
            l1 = self.semi_loss(h1, h2, epoch)
            l2 = self.semi_loss(h2, h1, epoch)
            ret = (l1 + l2) * 0.5
            ret = ret.mean() if mean else ret.sum()
        else:
            select_feat_np = select_feat.int().numpy()
            l1 = self.semi_loss_bmm(h1, h2, select_feat_np, args, feat_weight_t, data, fit=args.fit_mode)
            l2 = self.semi_loss_bmm(h2, h1, select_feat_np, args)
            ret = (l1 + l2) * 0.5
            # ret = l1
            ret = ret.mean() if mean else ret.sum()

        return ret


class Encoder2(nn.Module):
    def __init__(self, in_channels, hidden_channels, s_channels):
        super(Encoder2, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bns1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bns2 = torch.nn.BatchNorm1d(hidden_channels)
        self.prelu = nn.ReLU(hidden_channels)
        self.linear_s = torch.nn.Linear(hidden_channels, s_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.bns1.reset_parameters()
        self.conv2.reset_parameters()
        self.bns2.reset_parameters()
        self.linear_s.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # x = self.bns1(x)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        # x = self.bns2(x)
        # x = self.prelu(x)
        out_s = self.linear_s(x)
        return out_s, x
