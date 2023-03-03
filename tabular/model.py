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


# class DCL(nn.Module):
#     def __init__(self, temperature=0.1):
#         super(DCL, self).__init__()
#         self.temp = temperature


def cl_loss(z, temp=0.1, eval=False):
    z = F.normalize(z, p=2, dim=-1)
    z_ori = z[:, 0]  # n,z
    z_trans = z[:, 1:]  # n,k-1, z
    batch_size, num_trans, z_dim = z.shape

    sim_matrix = torch.exp(torch.matmul(z, z.permute(0, 2, 1) / temp))  # n,k,k
    mask = (torch.ones_like(sim_matrix).to(z) - torch.eye(num_trans).unsqueeze(0).to(z)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(batch_size, num_trans, -1)
    trans_matrix = sim_matrix[:, 1:].sum(-1)  # n,k-1

    pos_sim = torch.exp(torch.sum(z_trans * z_ori.unsqueeze(1), -1) / temp)  # n,k-1
    K = num_trans - 1
    scale = 1 / np.abs(K * np.log(1.0 / K))

    loss_tensor = (torch.log(trans_matrix) - torch.log(pos_sim)) * scale

    if eval:
        score = loss_tensor.sum(1)
        return score
    else:
        loss = loss_tensor.sum(1)
        return loss


class TabNeutralAD(nn.Module):
    def __init__(self, model, x_dim, device, config):
        super(TabNeutralAD, self).__init__()

        self.enc, self.trans = model._make_nets(x_dim, config)
        self.num_trans = config['num_trans']
        self.trans_type = config['trans_type']
        self.device = device
        try:
            self.z_dim = config['latent_dim']
        except:
            if 32 <= x_dim <= 300:
                self.z_dim = 32
            elif x_dim < 32:
                self.z_dim = 2 * x_dim
            else:
                self.z_dim = 64

    def forward(self, x):
        x = x.type(torch.FloatTensor).to(self.device)
        x_T = torch.empty(x.shape[0], self.num_trans, x.shape[-1]).to(x)
        for i in range(self.num_trans):
            mask = self.trans[i](x)
            if self.trans_type == 'forward':
                x_T[:, i] = mask
            elif self.trans_type == 'mul':
                mask = torch.sigmoid(mask)
                x_T[:, i] = mask * x
            elif self.trans_type == 'residual':
                x_T[:, i] = mask + x
        x_cat = torch.cat([x.unsqueeze(1), x_T], 1)
        _, _, zs = self.enc(x_cat.reshape(-1, x.shape[-1]))
        zs = zs.reshape(x.shape[0], self.num_trans + 1, self.z_dim)

        return zs


class TabTransformNet(nn.Module):
    def __init__(self, x_dim, h_dim, num_layers):
        super(TabTransformNet, self).__init__()
        net = []
        input_dim = x_dim
        for _ in range(num_layers - 1):
            net.append(nn.Linear(input_dim, h_dim, bias=False))
            # net.append(nn.BatchNorm1d(h_dim,affine=False))
            net.append(nn.ReLU())
            input_dim = h_dim
        net.append(nn.Linear(input_dim, x_dim, bias=False))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = self.net(x)

        return out


class TabEncoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, s_dim, y_dim, num_layers=3, batch_norm=False, bias=False):
        super(TabEncoder, self).__init__()
        enc = []
        input_dim = x_dim
        for _ in range(num_layers - 1):
            enc.append(nn.Linear(input_dim, h_dim, bias=bias))
            if batch_norm:
                enc.append(nn.BatchNorm1d(h_dim, affine=bias))
            enc.append(nn.ReLU())
            input_dim = h_dim

        self.enc = nn.Sequential(*enc)
        self.fc = nn.Linear(input_dim, z_dim, bias=bias)
        self.linear_s = torch.nn.Linear(z_dim, s_dim, bias=False)
        self.linear_y = torch.nn.Linear(z_dim, y_dim, bias=False)

    def forward(self, x):
        z = self.enc(x)
        z = self.fc(z)
        out_s = self.linear_s(z)
        out_y = self.linear_y(z)
        return out_s, out_y, z


class TabNets():

    def _make_nets(self, x_dim, config):
        enc_nlayers = config['enc_nlayers']
        try:
            hdim = config['enc_hdim']
            zdim = config['latent_dim']
            trans_hdim = config['trans_hdim']
        except:
            if 32 <= x_dim <= 300:
                zdim = 32
                hdim = 64
                trans_hdim = x_dim
            elif x_dim < 32:
                zdim = 2 * x_dim
                hdim = 2 * x_dim
                trans_hdim = x_dim
            else:
                zdim = 64
                hdim = 256
                trans_hdim = x_dim
        trans_nlayers = config['trans_nlayers']
        num_trans = config['num_trans']
        batch_norm = config['batch_norm']

        enc = TabEncoder(x_dim, hdim, zdim, enc_nlayers, config['s_dim'], config['y_dim'], config['enc_bias'])
        trans = nn.ModuleList(
            [TabTransformNet(x_dim, trans_hdim, trans_nlayers) for _ in range(num_trans)])

        return enc, trans
