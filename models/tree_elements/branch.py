import argparse
import copy
import numpy as np
from utils.training_utils import get_cam_1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from .node import Node
import cv2
import matplotlib.pyplot as plt
import os
from math import sqrt
from .attention import Attention_without_Classifier, Attention_Gated
import openslide
from models.ViT import *
torch.backends.cudnn.deterministic = True


class GCB_Simple_Stage(nn.Module):
    def __init__(self, dim):
        super(GCB_Simple_Stage, self).__init__()
        # Single linear transformation and layer normalization
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.ReLU()

    def forward(self, patches):
        x = self.linear(patches)
        x = self.norm(x)
        x = self.activation(x)

        pooled_x_max, _ = torch.max(x, dim=1, keepdim=True)
        pooled_x_max = pooled_x_max.expand(-1, patches.size(1), -1)
        x = pooled_x_max + patches

        return x

class GCB_Simple_Surv(nn.Module):
        def __init__(self, dim):
            super(GCB_Simple_Surv, self).__init__()
            # Single linear transformation and layer normalization
            self.linear = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)
            self.activation = nn.ReLU()

        def forward(self, patches):
            x = self.linear(patches)
            x = self.norm(x)
            x = self.activation(x)



            pooled_x_mean = torch.mean(x, dim=1, keepdim=True)
            pooled_x_mean = pooled_x_mean.expand(-1, patches.size(1), -1)
            x = pooled_x_mean + patches

            return x

class Branch(Node):
    mean = 0.5
    std = 0.1

    def __init__(self, index, l: Node, r: Node, proto_size:list, proto_dim: int, embed_dim: int, task:str):
        super(Branch, self).__init__(index)
        self.current_index = self.index

        self.l = l
        self.r = r

        self.img_size = 448
        self.proto_dim = proto_dim
        self.embed_dim = embed_dim
        self.q = nn.Linear(self.proto_dim, self.proto_dim)
        self.task = task

        self.epsilon = 1e-8
        if self.task == 'survival':
            self.gcb_l = GCB_Simple_Surv(self.proto_dim)
            self.gcb_r = GCB_Simple_Surv(self.proto_dim)
        else:
            self.gcb_l = GCB_Simple_Stage(self.proto_dim)
            self.gcb_r = GCB_Simple_Stage(self.proto_dim)


        self.max_score = float('-inf')
        self.proto_size = proto_size

    def forward(self, logits, patches, training, **kwargs):

            batch_size = patches.size(0)

            node_attr = kwargs.setdefault('attr', dict())
            pa = node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=patches.device))

            task = kwargs['task']

            ps = self.g_map(**kwargs)

            proto = ps.permute(0, 2, 1)

            Q = self.q(patches)

            proto = proto.repeat(batch_size, 1, 1)

            A = torch.bmm(Q, proto.permute(0, 2, 1))/torch.sqrt(torch.tensor(Q.shape[2], dtype=torch.float32, device='cuda'))

            Y_prob = F.sigmoid(A/20)

            if task == 'survival':
                self.maxim = F.adaptive_avg_pool2d(Y_prob, (1, 1)).squeeze(-1)
            else:
                self.maxim = F.adaptive_max_pool2d(Y_prob, (1, 1)).squeeze(-1)

            to_left = self.maxim

            to_right = 1 - to_left


            l_dists, _ = self.l.forward(logits, self.gcb_l(patches), training,  **kwargs)

            r_dists, _ = self.r.forward(logits, self.gcb_r(patches), training,  **kwargs)

            if torch.isnan(self.maxim).any() or torch.isinf(self.maxim).any():
                raise Exception('Error: NaN/INF values!', self.maxim)

            node_attr[self, 'ps'] = self.maxim
            node_attr[self.l, 'pa'] = to_left * pa.unsqueeze(1)
            node_attr[self.r, 'pa'] = to_right * pa.unsqueeze(1)

            return to_left.unsqueeze(-1) * l_dists + to_right.unsqueeze(-1) * r_dists, node_attr


    def g_map(self, **kwargs):
        out_map = kwargs['out_map']  # Mapping from decision nodes (branches) to prototype indices
        prototype_output = kwargs['prototype_map']

        out = prototype_output[str(int(self.index))]
        out = out.unsqueeze(dim=0)
        out = out.unsqueeze(dim=2)

        return out

    def explain_internal(self, logits, patches, training, sizes, id, s_matrix, l_distances, r_distances, y, prefix,
                r_node_id, pool_map, keys={},
                **kwargs):

        out_map = kwargs['out_map']
        node_id = out_map[self.index]
        task = kwargs['task']

        self.epsilon = 1e-8

        batch_size = patches.size(0)

        node_attr = kwargs.setdefault('attr', dict())
        pa = node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=patches.device))

        ps = self.g_map(**kwargs)

        ps = ps.permute(0, 2, 1)
        Q = self.q(patches)

        proto = ps.repeat(batch_size, 1, 1)

        A = torch.bmm(Q, proto.permute(0, 2, 1)) / torch.sqrt(
            torch.tensor(Q.shape[2], dtype=torch.float32, device='cuda'))

        Y_prob = F.sigmoid(A / 20)

        if task == 'survival':
            self.maxim = F.adaptive_avg_pool2d(Y_prob, (1, 1)).squeeze(-1)
        else:
            self.maxim = F.adaptive_max_pool2d(Y_prob, (1, 1)).squeeze(-1)
        #keys[node_id] = Y_prob

        self.l.explain_internal(logits, self.gcb_l(Q), training, sizes, id, s_matrix, l_distances,
                           r_distances, y, prefix,
                           r_node_id * 2 + 1, pool_map,
                           keys,
                           **kwargs)

        self.r.explain_internal(logits, self.gcb_r(Q), training, sizes, id, s_matrix, l_distances,
                           r_distances, y, prefix,
                           r_node_id * 2 + 2, pool_map,
                           keys,
                           **kwargs)
        return keys


    @property
    def size(self) -> int:
        return 1 + self.l.size + self.r.size

    @property
    def leaves(self) -> set:
        return self.l.leaves.union(self.r.leaves)

    @property
    def branches(self) -> set:
        return {self}.union(self.l.branches).union(self.r.branches)

    @property
    def nodes_by_index(self) -> dict:
        return {self.index: self, **self.l.nodes_by_index, **self.r.nodes_by_index}

    @property
    def num_leaves(self) -> int:
        return self.l.num_leaves + self.r.num_leaves

    @property
    def depth(self) -> int:
        return self.l.depth + 1