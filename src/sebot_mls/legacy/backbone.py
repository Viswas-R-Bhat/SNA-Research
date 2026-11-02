# Adapted from 846468230/SEBot, Apache-2.0. See NOTICE.
from .sep_u import SEP_U
from .sep_g import SEP_G
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
from torch_geometric.data import Data
from .rgcn import FACNConv as RGCNConv
# from BackBone.self_attention import SelfAttention
import argparse
import pickle
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

PWD = os.path.dirname(os.path.realpath(__file__))


def edge_mask(edge_index, edge_attr, pe):
    # each edge has a probability of pe to be removed
    edge_index = edge_index.clone()
    edge_num = edge_index.shape[1]
    pre_index = torch.bernoulli(torch.ones(edge_num) * pe) == 0
    pre_index.to(edge_index.device)
    edge_index = edge_index[:, pre_index]
    edge_attr = edge_attr.clone()
    edge_attr = edge_attr[pre_index]
    return edge_index, edge_attr


def edge_dropping(edge_index, edge_attr, edge_weight):
    # each edge has a probability of pe to be removed
    edge_index = edge_index.clone()
    pre_index = torch.bernoulli(edge_weight) == 1
    pre_index.to(edge_index.device)
    edge_index = edge_index[:, pre_index]
    edge_attr = edge_attr.clone()
    edge_attr = edge_attr[pre_index]
    return edge_index, edge_attr


def relatioal_edge_adding(edge_index, edge_attr, pe):
    # for each relation, add pe of the number of edges
    relation_num = edge_attr.max() + 1
    node_num = edge_index.max() + 1
    edge_index = edge_index.clone()
    edge_type = edge_attr.clone()
    for i in range(relation_num):
        relation_edge_num = edge_index[:, edge_type == i].shape[1]
        added_edge_index = torch.randint(
            0, node_num, (2, int(relation_edge_num * pe))).to(edge_attr.device)
        edge_index = torch.cat((edge_index, added_edge_index), dim=1)
        edge_type = torch.cat(
            (edge_type,
             torch.ones(int(relation_edge_num * pe)).to(edge_attr.device) * i),
            dim=0)
    return edge_index, edge_type


def feature_mask(x, drop_prob):
    # each feature channel has a probability of being masked
    drop_mask = torch.empty(
        (x.size(1), ), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(x.device)
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def relational_undirected(edge_index, edge_type):
    device = edge_index.device
    relation_num = edge_type.max() + 1
    edge_index = edge_index.clone()
    edge_type = edge_type.clone()
    r_edge = []
    for i in range(relation_num):
        e1 = edge_index[:, edge_type == i]
        e2 = edge_index[:, edge_type == i].flip(0)
        edges = torch.cat((e1, e2), dim=1).unique(dim=1)
        r_edge.append(edges)
    edge_type = torch.cat(
        [torch.tensor([i] * e.shape[1]) for i, e in enumerate(r_edge)],
        dim=0).to(device)
    edge_index = torch.cat(r_edge, dim=1)

    return edge_index, edge_type


def mask_heterophily(edge_index, edge_type, x, labels, p):
    node_num = len(labels)
    clf = LogisticRegression(random_state=0, max_iter=1000)
    clf.fit(x[:int(0.7 * node_num)], labels[:int(0.7 * node_num)])
    y_pred = clf.predict(x[int(0.7 * node_num):])
    labels = torch.cat(
        (labels[:int(0.7 * node_num)], torch.tensor(y_pred).to(labels.device)))
    mask = labels[edge_index[0]] != labels[edge_index[1]]
    index = torch.nonzero(mask).squeeze()
    n = len(index)
    index = index[torch.randperm(n)]
    index = index[:int(n * p)]  # 去除边的index
    mask = torch.ones(edge_index.size(1),
                      dtype=torch.bool).to(edge_index.device)
    mask[index] = False

    return edge_index[:, mask], edge_type[mask]


class FeatureEncoder(nn.Module):

    def __init__(self, args):
        super(FeatureEncoder, self).__init__()
        self.args = args
        self.des_size = args.des_num
        self.tweet_size = args.tweet_num
        self.num_prop_size = args.prop_num
        self.cat_prop_size = args.cat_num
        self.dropout = args.dropout
        input_dimension = args.input_dim
        embedding_dimension = args.hidden_dim

        self.linear_relu_des = nn.Sequential(
            nn.Linear(self.des_size, int(input_dimension / 4)), nn.LeakyReLU())
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(self.tweet_size, int(input_dimension / 4)),
            nn.LeakyReLU())
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(self.num_prop_size, int(input_dimension / 4)),
            nn.LeakyReLU())
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(self.cat_prop_size, int(input_dimension / 4)),
            nn.LeakyReLU())

        self.linear_relu_input = nn.Sequential(
            nn.Linear(input_dimension, embedding_dimension), nn.LeakyReLU())

    def forward(self, x):
        if self.args.dataset == 'twibot-20':
            num_prop = x[:, :self.num_prop_size]
            tweet = x[:,
                      self.num_prop_size:self.num_prop_size + self.tweet_size]
            cat_prop = x[:, self.num_prop_size +
                         self.tweet_size:self.num_prop_size + self.tweet_size +
                         self.cat_prop_size]
            des = x[:,
                    self.num_prop_size + self.tweet_size + self.cat_prop_size:]
            d = self.linear_relu_des(des)
            t = self.linear_relu_tweet(tweet)
            n = self.linear_relu_num_prop(num_prop)
            c = self.linear_relu_cat_prop(cat_prop)
            x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        return x


class BotRGCN(nn.Module):

    def __init__(self, args):
        super(BotRGCN, self).__init__()
        self.des_size = args.des_num
        self.tweet_size = args.tweet_num
        self.num_prop_size = args.prop_num
        self.cat_prop_size = args.cat_num
        self.dropout = args.dropout
        self.node_num = args.node_num
        self.pe = args.pe
        self.pf = args.pf
        embedding_dimension = args.hidden_dim

        self.feature_encoder = FeatureEncoder(args)
        self.rgcn1 = RGCNConv(embedding_dimension,
                              embedding_dimension,
                              num_relations=args.num_relations)
        self.rgcn2 = RGCNConv(embedding_dimension,
                              embedding_dimension,
                              num_relations=args.num_relations)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension))

        self.relu = nn.LeakyReLU()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type

        if self.training:
            edge_index, edge_type = edge_mask(edge_index, edge_type, self.pe)

        x = self.feature_encoder(x)
        x = self.rgcn1(x, edge_index, edge_type)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn2(x, edge_index, edge_type)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


