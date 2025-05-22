import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .node import Node
from .attention import Attention_with_Classifier, Attention2

class Leaf(Node):
    mean = 0.5
    std = 0.1

    def __init__(self, index, num_classes, embed_dim):
        super(Leaf, self).__init__(index)
        self.proto_dim = embed_dim
        self.pred = nn.Linear(self.proto_dim, num_classes)


        self.norm_o = nn.LayerNorm(num_classes)

        self.attention_head = nn.Sequential(
            nn.Linear(self.proto_dim, self.proto_dim),
            nn.Tanh(),
            nn.Linear(self.proto_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, logits, patches, training, **kwargs):
        batch_size = patches.size(0)

        node_attr = kwargs.setdefault('attr', dict())

        node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=patches.device))

        tree_logits = self.pred(patches)
        self.dists = logits + tree_logits

        node_attr[self, 'ds'] = self.dists

        return self.dists, node_attr

    def hard_forward(self, logits, patches, **kwargs):
        return self(logits, patches, **kwargs)

    def explain_internal(self, logits, patches, training, sizes, id, s_matrix, l_distances, r_distances, y, prefix,
                r_node_id, pool_map,keys={},
                **kwargs):
        batch_size = patches.size(0)

        out_map = kwargs['leaf_out_map']

        node_id = out_map[self.index]

        node_attr = kwargs.setdefault('attr', dict())

        node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=patches.device))

        tree_logits = self.pred(patches)

        self.dists = logits + tree_logits

        value = torch.mean(self.dists , dim=2)

        keys[node_id] = value.squeeze(0)

        #self.dists = logits + tree_logits

        return tree_logits, patches, node_attr

    @property
    def requires_grad(self) -> bool:
        return self._dist_params.requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool):
        self._dist_params.requires_grad = val

    @property
    def size(self) -> int:
        return 1

    @property
    def leaves(self) -> set:
        return {self}

    @property
    def branches(self) -> set:
        return set()

    @property
    def nodes_by_index(self) -> dict:
        return {self.index: self}

    @property
    def num_branches(self) -> int:
        return 0

    @property
    def num_leaves(self) -> int:
        return 1

    @property
    def depth(self) -> int:
        return 0
