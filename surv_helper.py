#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from utils.metrics import ConfusionMatrix
from PIL import Image
import os


# torch.cuda.synchronize()
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

def surv_collate(batch):
    image = [b['image'] for b in batch]  # w, h
    label = [b['label'] for b in batch]
    id = [b['id'] for b in batch]
    c = [b['c'] for b in batch]
    t = [b['t'] for b in batch]
    adj_s = [b['adj_s'] for b in batch]
    return {'image': image, 'label': label, 't': t, 'c': c, 'id': id, 'adj_s': adj_s}


def surv_preparefeatureLabel(batch_graph, batch_label, batch_t, batch_c,  batch_adjs, name, feature_dim):
    batch_size = len(batch_graph)
    labels = torch.LongTensor(batch_size)
    t = torch.LongTensor(batch_t)
    c = torch.LongTensor(batch_c)
    max_node_num = 0

    for i in range(batch_size):
        labels[i] = batch_label[i]
        max_node_num = max(max_node_num, batch_graph[i].shape[0])

    masks = torch.zeros(batch_size, max_node_num)
    adjs = torch.zeros(batch_size, max_node_num, max_node_num)
    batch_node_feat = torch.zeros(batch_size, max_node_num, feature_dim)
    names = []

    for i in range(batch_size):
        cur_node_num = batch_graph[i].shape[0]
        # node attribute feature
        tmp_node_fea = batch_graph[i]
        batch_node_feat[i, 0:cur_node_num] = tmp_node_fea

        # adjs
        adjs[i, 0:cur_node_num, 0:cur_node_num] = batch_adjs[i]

        # masks
        masks[i, 0:cur_node_num] = 1
        names.append(name)

    node_feat = batch_node_feat.cuda()
    labels = labels.cuda()
    adjs = adjs.cuda()
    masks = masks.cuda()

    return node_feat, labels,t , c , adjs, masks, names


class S_Trainer(object):
    def __init__(self, n_class):
        self.metrics = ConfusionMatrix(n_class)

    def get_scores(self):
        acc = self.metrics.get_scores()

        return acc

    def reset_metrics(self):
        self.metrics.reset()

    def plot_cm(self):
        self.metrics.plotcm()

    def train(self, sample, model, dim):

        node_feat, labels,t , c , adjs, masks, names = surv_preparefeatureLabel(sample['image'],
                                                                                sample['label'],
                                                                                sample['t'],
                                                                                sample['c'],
                                                                                sample['adj_s'],
                                                                    sample['id'], feature_dim=dim)
        cls_token, pred, label, loss, patches = model.forward(node_feat, labels, adjs, masks, names, training=True)

        return cls_token, pred, label, t, c, loss, patches


class S_Evaluator(object):
    def __init__(self, n_class):
        self.metrics = ConfusionMatrix(n_class)

    def get_scores(self):
        acc = self.metrics.get_scores()

        return acc

    def reset_metrics(self):
        self.metrics.reset()

    def plot_cm(self):
        self.metrics.plotcm()

    def eval_test(self, sample, model, feature_dim, graphcam_flag=False):
        node_feat, labels, t, c, adjs, masks, names = surv_preparefeatureLabel(sample['image'], sample['label'],
                                                                               sample['t'], sample['c'],
                                                                               sample['adj_s'],
                                                                               sample['id'], feature_dim=feature_dim)


        with torch.no_grad():
                # pred,labels,loss = model.forward(node_feat, labels, adjs, masks)
                cls_token, pred, label, loss, patches = model.forward(node_feat, labels, adjs, masks, names,
                                                                      training=False)

        return cls_token, pred, label, t, c, loss, patches