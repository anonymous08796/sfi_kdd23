import sys
import os
import os
from math import exp
import numpy as np
import argparse
import os.path as osp
import random
import torch
from torch_geometric.utils import dropout_adj, degree, to_undirected
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from pipeline_sampling.utils import load_pokec, compute_pr, eigenvector_centrality, accuracy
from simple_param.sp import SimpleParam
from pipeline_sampling.model import GRACE, MyEncoder
from pipeline_sampling.functional import drop_feature, drop_edge_weighted, degree_drop_weights, evc_drop_weights, \
    pr_drop_weights, feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from param.config import *
import torch.nn.functional as F
import os

np.set_printoptions(suppress=True)


def train_y(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    _, out_y, _ = model(data.x, data.edge_index)
    loss = F.cross_entropy(out_y[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def train_s(data, model, optimizer, select_feat, feat_weight, args, tau=10.0, noised_prior_dis=None):
    select_feat_np = select_feat.int().numpy()
    c_feat = np.array(data.p_idx)[select_feat_np]
    c_weight = feat_weight[select_feat_np]
    c_weight = F.softmax(torch.Tensor(c_weight / tau), dim=0).numpy()

    if c_feat.shape[0] == 0:
        return 0

    model.train()
    optimizer.zero_grad()
    out_s, _, _ = model(data.x, data.edge_index)
    loss = None
    for i in range(c_feat.shape[0]):
        c_label = data.x[data.train_mask, c_feat[i]].clone().long()
        if i == 0:
            loss = c_weight[i] * F.cross_entropy(out_s[data.train_mask], c_label)
        else:
            loss = loss + c_weight[i] * F.cross_entropy(out_s[data.train_mask], c_label)

    loss.backward()
    optimizer.step()
    return loss.item()


def train_gcl(x_1, edge_index_1, x_2, edge_index_2, model, optimizer, data, args, select_feat, feat_weight, epoch: int):
    model.train()
    optimizer.zero_grad()
    _, _, z1 = model(x_1, edge_index_1)
    _, _, z2 = model(x_2, edge_index_2)
    loss = model.loss(z1[data.train_mask], z2[data.train_mask], epoch, args, select_feat, feat_weight, data)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data, feat_weight, noised_prior_dis, select_feat, lam, eta, tau=0.7):
    model.eval()
    out_s, out_y, x = model(data.x, data.edge_index)
    select_feat_np = select_feat.int().numpy()

    c_Lxp = []
    for i in range(select_feat_np.shape[0]):
        feat = select_feat_np[i]
        p_label = data.x[data.train_mask, data.p_idx[feat]].clone().long()
        c_Lxp.append(F.cross_entropy(out_s[data.train_mask], p_label).item())

    c_Lxp = np.array(c_Lxp)

    for i in range(select_feat_np.shape[0]):
        feat = select_feat_np[i]
        feat_weight[feat] = exp(
            -1 * ((1 - eta) * jensenshannon(data.dist_l[feat], noised_prior_dis) / lam + eta * c_Lxp[i]))

    out_s = F.softmax(out_s, dim=-1)
    out_y = F.softmax(out_y, dim=-1)
    pred_s = np.argmax(out_s[data.train_mask].cpu().numpy(), axis=1)
    pred_y = np.argmax(out_y[data.train_mask].cpu().numpy(), axis=1)
    s_acc = accuracy(pred_s, data.s[data.train_mask].cpu().numpy())
    y_acc = accuracy(pred_y, data.y[data.train_mask].cpu().numpy())
    return s_acc, y_acc, feat_weight, c_Lxp


def main(args, param, noised_prior_dis):
    print(f'noised prior distribution: {noised_prior_dis}')
    device = torch.device(args.device)
    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    data = load_pokec(dataset=dataset_param['dataset'], sens_attr='region', predict_attr='I_am_working_in_field',
                      path='../datasets/pokec')

    data = data.to(device)

    encoder = MyEncoder(data.x.shape[1], param['num_hidden'], s_channels=2, y_channels=2).to(device)
    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=param['learning_rate'], weight_decay=param['weight_decay'])

    if param['drop_scheme'] == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(device)
    elif param['drop_scheme'] == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
    elif param['drop_scheme'] == 'evc':
        drop_weights = evc_drop_weights(data).to(device)
    else:
        drop_weights = None

    if param['drop_scheme'] == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1])
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    elif param['drop_scheme'] == 'pr':
        node_pr = compute_pr(data.edge_index)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
    elif param['drop_scheme'] == 'evc':
        node_evc = eigenvector_centrality(data)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)

    print("feat_weight")
    feat_weight = np.empty([len(data.p_idx)])
    for i in range(feat_weight.shape[0]):
        feat_weight[i] = exp(-jensenshannon(data.dist_l[i], noised_prior_dis) / args.lam)
        print(data.p_idx[i], feat_weight[i])

    select_mask = torch.bernoulli(torch.Tensor(feat_weight)).bool()
    for i in range(select_mask.shape[0]):
        if select_mask[i] == True and feat_weight[i] < args.cut:
            select_mask[i] = False
    select_feat = torch.where(select_mask)[0].float()

    for epoch in range(1, param['num_epochs'] + 1):
        print(f'(E) | Epoch={epoch:04d}')

        if epoch > args.epoch_start and (epoch - 1) % args.iters == 0:
            print("re-select feat")
            while True:
                select_mask = torch.bernoulli(torch.Tensor(feat_weight)).bool()
                for i in range(select_mask.shape[0]):
                    if select_mask[i] == True and feat_weight[i] < args.cut:
                        select_mask[i] = False
                select_feat = torch.where(select_mask)[0].float()
                if select_feat.shape[0] > 0:
                    break

        # graph augmentation
        def drop_edge(idx: int):
            # global drop_weights
            if param['drop_scheme'] == 'uniform':
                return dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
            elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
                return drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'],
                                          threshold=0.7)
            else:
                raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')

        edge_index_1 = drop_edge(1)
        edge_index_2 = drop_edge(2)

        x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
        x_2 = drop_feature(data.x, param['drop_feature_rate_2'])

        if param['drop_scheme'] in ['pr', 'degree', 'evc']:
            x_1 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_1'])
            x_2 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_2'])

        if epoch > args.epoch_start:
            loss_y = train_y(data, model, optimizer)

        if epoch > args.epoch_start:
            tau_train_s = 10 * (0.999 ** (epoch - args.epoch_start))
            loss_s = train_s(data, model, optimizer, select_feat, feat_weight, args, tau=tau_train_s)

        loss_gcl = train_gcl(x_1, edge_index_1, x_2, edge_index_2, model, optimizer, data, args, select_feat,
                             feat_weight, epoch)

        if epoch > args.epoch_start and epoch % args.iters == 0:
            test_eta = 0.3 * (1 - 0.999 ** (epoch - args.epoch_start))
            s_acc, y_acc, feat_weight, c_Lxp = test(model, data, feat_weight, noised_prior_dis, select_feat, args.lam,
                                                    test_eta)
            print(f'(Test) | s_acc={s_acc:.4f},  y_acc={y_acc:.4f}')

    return s_acc, y_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:5')
    parser.add_argument('--dataset', type=str, default='pokec_z')
    parser.add_argument('--param', type=str, default='local:pokec_z.json')
    parser.add_argument('--epoch_start', type=int, default=200)
    parser.add_argument('--mode', type=str, default='weight')
    parser.add_argument('--fit_mode', type=str, default='div')
    parser.add_argument('--sel_num', type=int, default=1000)
    parser.add_argument('--weight_init', type=float, default=0.05)
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--noise_range', type=float, default=0.01)
    parser.add_argument('--lam', type=float, default=0.01)
    parser.add_argument('--eta', type=float, default=0.99)
    parser.add_argument('--cut', type=float, default=0.4)
    parser.add_argument('--repeat_time', type=int, default=5)
    default_param = {
        'learning_rate': 0.001,
        'num_hidden': 64,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 3000,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()

    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')

    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    use_nni = args.param == 'nni'
    if use_nni and args.device != 'cpu':
        args.device = 'cuda'

    print(args)
    print(param)
    noise_dis_l = []
    random.seed(1)
    for i in range(args.repeat_time):
        dataset_param = dataset_config[args.dataset]
        prior_dis = dataset_param['prior_dis']
        print(f'precise prior distribution: {prior_dis}')
        epsilon = random.random() - 0.5
        print(epsilon)
        noised_prior_dis = np.empty_like(prior_dis)
        noised_prior_dis[0] = prior_dis[0] + epsilon * args.noise_range * 2
        noised_prior_dis[1] = 1 - noised_prior_dis[0]
        noise_dis_l.append(noised_prior_dis)

    s_acc_l = []
    y_acc_l = []
    for i in range(args.repeat_time):
        s_accuracy, y_accuracy = main(args, param, noise_dis_l[i])
        s_acc_l.append(s_accuracy)
        y_acc_l.append(y_accuracy)
        print(i, s_accuracy)

    print("ave res:")
    print("s: ")
    print(np.mean(s_acc_l))
    print(np.std(s_acc_l))
    print("y: ", sum(y_acc_l) / len(y_acc_l))
