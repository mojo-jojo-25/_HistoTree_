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

def find_high_activation_crop(mask, threshold):
    threshold = 1. - threshold
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > threshold:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > threshold:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > threshold:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > threshold:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1

class GCB_AttnPool(nn.Module):
    def __init__(self, dim, patch_dim=49):
        super(GCB_AttnPool, self).__init__()
        exp_dim = int(dim * 1.0)

        self.cm = nn.Linear(dim, 1)
        self.wv1 = nn.Linear(dim, exp_dim)
        self.norm = nn.LayerNorm(exp_dim)
        self.gelu = nn.GELU()
        self.wv2 = nn.Linear(exp_dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, patches):
        h = patches
        attention_scores = self.cm(h)
        attention_weights = F.softmax(attention_scores, dim=1)
        context_vector = torch.bmm(attention_weights.permute(0, 2, 1), h).squeeze(1)

        x = self.wv1(context_vector)
        x = self.norm(x)
        x = self.gelu(x)
        x = self.wv2(x)

        # Expand context vector to match original dimensions
        x = x.unsqueeze(1).expand(-1, h.size(1), -1)
        x = self.ffn_norm(x + h)

        return x

class GCB_AvgPool(nn.Module):
    def __init__(self, dim, patch_dim=49):
        super(GCB_AvgPool, self).__init__()
        exp_dim = int(dim * 1.0)

        self.wv1 = nn.Linear(dim, exp_dim)
        self.norm = nn.LayerNorm(exp_dim)
        self.gelu = nn.GELU()
        self.wv2 = nn.Linear(exp_dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, patches):
        h = patches
        x = self.wv1(h)
        x = self.norm(x)
        x = self.gelu(x)
        x = self.wv2(x)

        # Average pooling across patches
        pooled_x = torch.mean(x, dim=1, keepdim=True)
        pooled_x = pooled_x.expand(-1, h.size(1), -1)  # Expand to match original dimensions
        x = self.ffn_norm(pooled_x + h)

        return x


class GCB_GAP(nn.Module):
    def __init__(self, dim, patch_dim=49):
        super(GCB_GAP, self).__init__()
        exp_dim = int(dim * 1.0)

        self.wv1 = nn.Linear(dim, exp_dim)
        self.norm = nn.LayerNorm(exp_dim)
        self.gelu = nn.GELU()
        self.wv2 = nn.Linear(exp_dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, patches):
        h = patches
        x = self.wv1(h)
        x = self.norm(x)
        x = self.gelu(x)
        x = self.wv2(x)

        # Global Average Pooling across all dimensions except batch and channel
        pooled_x = torch.mean(x, dim=[1, 2], keepdim=True)
        pooled_x = pooled_x.expand_as(h)  # Expand to match original dimensions

        # Add residual connection and normalize
        x = self.ffn_norm(pooled_x + h)

        return x
class GCB_MaxPool(nn.Module):
    def __init__(self, dim, patch_dim=49):
        super(GCB_MaxPool, self).__init__()
        exp_dim = int(dim * 1.0)

        self.wv1 = nn.Linear(dim, exp_dim)
        self.norm = nn.LayerNorm(exp_dim)
        self.gelu = nn.GELU()
        self.wv2 = nn.Linear(exp_dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, patches):
        h = patches
        x = self.wv1(h)
        x = self.norm(x)
        x = self.gelu(x)
        x = self.wv2(x)

        # Max pooling across patches
        pooled_x, _ = torch.max(x, dim=1, keepdim=True)
        pooled_x = pooled_x.expand(-1, h.size(1), -1)  # Expand to match original dimensions
        x = self.ffn_norm(pooled_x + h)

        return x
class GCB_Residual(nn.Module):
    def __init__(self, dim, patch_dim=49):
        super(GCB_Residual, self).__init__()
        exp_dim = int(dim * 1.0)
        self.wv1 = nn.Linear(dim, exp_dim)
        self.norm = nn.LayerNorm(exp_dim)
        self.gelu = nn.GELU()
        self.wv2 = nn.Linear(exp_dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, patches):
        h = patches
        x = self.wv1(h) + h  # Residual connection
        x = self.norm(x)
        x = self.gelu(x)
        x = self.wv2(x) + h  # Second residual connection

        # Average pooling across patches
        pooled_x = torch.mean(x, dim=1, keepdim=True)
        pooled_x = pooled_x.expand(-1, h.size(1), -1)
        x = self.ffn_norm(pooled_x + h)

        return x

class GCB(nn.Module):
    mean = 0.5
    std = 0.1

    def __init__(self, dim, patch_dim=49):
        super(GCB, self).__init__()

        exp_dim = int(dim * 1.)

        self.cm = nn.Linear(dim, 1)
        self.wv1 = nn.Linear(dim, exp_dim)
        self.norm = nn.LayerNorm(exp_dim)
        self.gelu = nn.GELU()
        self.wv2 = nn.Linear(exp_dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, patches):

        h = patches
        x = self.cm(patches)
        x = torch.bmm(h.permute(0, 2, 1), F.softmax(x, 1)).squeeze(-1)
        x = self.wv1(x)
        x = h + x.unsqueeze(1)

        return x

class GCB_Simple(nn.Module):
        def __init__(self, dim):
            super(GCB_Simple, self).__init__()
            # Single linear transformation and layer normalization
            self.linear = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)
            self.activation = nn.ReLU()  # Simple ReLU activation

        def forward(self, patches):
            x = self.linear(patches)
            x = self.norm(x)
            x = self.activation(x)

            # Average pooling across patches
            pooled_x = torch.mean(x, dim=1, keepdim=True)
            pooled_x = pooled_x.expand(-1, patches.size(1), -1)
            x = pooled_x + patches  # Residual connection with the input

            return x

class Branch(Node):
    mean = 0.5
    std = 0.1

    def __init__(self, index, l: Node, r: Node, proto_size:list, proto_dim: int, embed_dim: int):
        super(Branch, self).__init__(index)

        self.current_index = self.index

        self.l = l
        self.r = r

        self.img_size = 448
        self.proto_dim = proto_dim
        self.embed_dim = embed_dim
        self.q = nn.Linear(self.proto_dim, self.proto_dim )

        self.fcc = nn.Conv1d(1, 1, kernel_size=32)

        self.epsilon = 1e-8

        self.gcb_l = GCB_Simple(self.proto_dim)
        self.gcb_r = GCB_Simple(self.proto_dim)

        self.max_score = float('-inf')
        self.proto_size = proto_size

    def forward(self, logits, patches, training, **kwargs):

            batch_size = patches.size(0)

            node_attr = kwargs.setdefault('attr', dict())
            pa = node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=patches.device))

            ps = self.g(**kwargs)

            ps = ps.permute(0, 2, 1)

            Q = self.q(patches)

            proto = ps.expand(batch_size, 20, self.embed_dim)

            #proto = self.w(proto)

            A = torch.bmm(Q, proto.permute(0, 2, 1))/torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device='cuda'))

            A = torch.mean(A, dim=2, keepdim=True)

            Y_prob = torch.sigmoid(A)

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

    def g(self, **kwargs):
        out_map = kwargs['out_map']  # Obtain the mapping from decision nodes to conv net outputs
        conv_net_output = kwargs['conv_net_output']
        out = conv_net_output[out_map[self.index]] # Obtain the output corresponding to this decision node
        return out.squeeze(dim=1)


    def explain_internal(self, logits, patches, training, sizes, id, s_matrix, l_distances, r_distances, y, prefix,
                r_node_id, pool_map, keys={},
                **kwargs):

        out_map = kwargs['out_map']
        node_id = out_map[self.index]



        self.epsilon = 1e-8

        batch_size = patches.size(0)

        node_attr = kwargs.setdefault('attr', dict())
        pa = node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=patches.device))

        ps = self.g(**kwargs)


        ps = ps.permute(0, 2, 1)
        Q = self.q(patches)

        proto = ps.expand(batch_size, 1, Q.shape[2])

        A = torch.bmm(Q, proto.permute(0, 2, 1)) / torch.sqrt(
            torch.tensor(Q.shape[1], dtype=torch.float32, device='cuda'))

        A = torch.mean(A, dim=2, keepdim=True)

        Y_prob = torch.sigmoid(A)

        self.maxim = F.adaptive_max_pool2d(Y_prob, (1, 1)).squeeze(-1)

        keys[node_id] = Y_prob


        self.l.explain_internal(logits, self.gcb_l(patches), training, sizes, id, s_matrix, l_distances,
                           r_distances, y, prefix,
                           r_node_id * 2 + 1, pool_map,
                           keys,
                           **kwargs)

        self.r.explain_internal(logits, self.gcb_r(patches), training, sizes, id, s_matrix, l_distances,
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