import math
import os
import random

import torch.distributions.uniform as uniform
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


def load_tabular(dataset, sens_attr, predict_attr, path, device):
    """Load data"""
    print(f'Loading {dataset} dataset from {path}')
    idx_features_labels = pd.read_csv(path)

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

    data = TabularData(x=features.to(device), y=labels.to(device), s=sens.to(device), train_mask=idx_train.to(device),
                       p_idx=binary_feat_l, dist_l=dist_l)
    return data


class TabularData(nn.Module):
    def __init__(self, x, y, s, train_mask, p_idx, dist_l):
        super(TabularData, self).__init__()
        self.x = x
        self.y = y
        self.s = s
        self.train_mask = train_mask
        self.p_idx = p_idx
        self.dist_l = dist_l


def accuracy(output, target):
    num_correct = np.sum(output == target)
    res = num_correct / len(target)

    return res


def DACL_aug(data: torch.Tensor, alpha=0.9):
    N = data.shape[0]
    lower_bound = torch.zeros(N) + alpha
    upper_bound = torch.ones(N, dtype=torch.float)
    lams = uniform.Uniform(lower_bound, upper_bound).sample()
    lams = lams.to(data.device)
    noise_idx = random.choices([i for i in range(N)], k=N)
    aug_data = data * lams.unsqueeze(1) + data[noise_idx, :] * (1 - lams).unsqueeze(1)
    return aug_data
