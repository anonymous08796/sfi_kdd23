import math
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import iv
from scipy.sparse.linalg import eigsh
import os.path as osp
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import SpectralEmbedding
from tqdm import tqdm
from matplotlib import cm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv, GraphConv, GINConv
from torch_geometric.utils import sort_edge_index, degree, add_remaining_self_loops, remove_self_loops, get_laplacian, \
    to_undirected, to_dense_adj, to_networkx, index_to_mask
from torch_geometric.datasets import KarateClub
from torch_scatter import scatter
import torch_sparse
import torch_geometric.transforms as T
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt


def get_base_model(name: str):
    def gat_wrapper(in_channels, out_channels):
        return GATConv(
            in_channels=in_channels,
            out_channels=out_channels // 4,
            heads=4
        )

    def gin_wrapper(in_channels, out_channels):
        mlp = nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.ELU(),
            nn.Linear(2 * out_channels, out_channels)
        )
        return GINConv(mlp)

    base_models = {
        'GCNConv': GCNConv,
        'SGConv': SGConv,
        'SAGEConv': SAGEConv,
        'GATConv': gat_wrapper,
        'GraphConv': GraphConv,
        'GINConv': gin_wrapper
    }

    return base_models[name]


def get_activation(name: str):
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }

    return activations[name]


def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes,)).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

        x = (1 - damp) * x + damp * agg_msg

    return x


def eigenvector_centrality(data):
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)


def generate_split(num_samples: int, train_ratio: float, val_ratio: float):
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask


def load_pokec(dataset, sens_attr, predict_attr, path):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))

    sens = idx_features_labels[sens_attr].values
    sens_set = set(sens)
    sens_set.discard(-1)
    c_sens = len(sens_set)
    sens = torch.LongTensor(sens)

    labels = idx_features_labels[predict_attr].values
    labels[labels > 1] = 1
    labels = torch.LongTensor(labels)

    idx_train = np.where(labels.numpy() >= 0)[0]
    idx_train = index_to_mask(torch.LongTensor(idx_train), size=labels.shape[0])

    header = list(idx_features_labels.columns)
    header.remove("user_id")
    header.remove(predict_attr)
    features = np.array(idx_features_labels[header])

    binary_feat_l = []
    dist_l = []
    print("selected features' distribution:")
    for i in range(features.shape[1]):
        cand = features[:, i]
        class_set = set(cand)
        class_set.discard(-1)
        class_num = len(class_set)
        if class_num != c_sens:
            continue
        dist_1 = np.where(cand == 1)[0].shape[0] / cand.shape[0]
        dist_0 = np.where(cand == 0)[0].shape[0] / cand.shape[0]
        assert dist_0 + dist_1 == 1
        dist = [dist_0, dist_1]
        print(f'binary variable: {i, dist}')
        dist_l.append(dist)
        binary_feat_l.append(i)

    features = torch.FloatTensor(features)

    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=int)
    edges = torch.tensor(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape).t()

    data = Data(x=features, edge_index=edges, y=labels, s=sens, train_mask=idx_train, p_idx=binary_feat_l,
                dist_l=dist_l)
    data = T.ToUndirected()(data)
    # data = T.ToSparseTensor()(data)

    return data


def accuracy(output, target):
    num_correct = np.sum(output == target)
    res = num_correct / len(target)

    return res


def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape)
    U = U.to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(prob, temperature=1):
    device = prob.device
    prob_hat = 1 / (1 + torch.exp(-1 * (torch.log(prob) + sample_gumbel(prob.size(), device)) / temperature))
    return prob_hat


def kl_div(pred, true):
    return true * (true.log() - pred.log()) + (1 - true) * ((1 - true).log() - (1 - pred).log())
