import sys
import os
import torch
import random
import numpy as np
from .tree_elements import *
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, MiniBatchKMeans
import torch.optim as optim
import matplotlib.pyplot as plt
from nystrom_attention import NystromAttention
from utils.training_utils import get_cam_1d
from .ViT import *
from .gcn import GCNBlock
from sklearn.decomposition import PCA
from torch_geometric.nn import GCNConv, DenseGraphConv, dense_mincut_pool
from sklearn.feature_selection import VarianceThreshold
from torch.nn import Linear
from models.tree_elements.attention import Attention_with_Classifier, Attention2
from torch_geometric.nn import TGNMemory, TransformerConv


class Classifier(nn.Module):
    def __init__(self, config, args):
        super(Classifier, self).__init__()

        self.feature_dim = config.model.feature_dim
        self.embed_dim = config.model.embed_dim
        self.num_layers = config.model.num_layers
        self.node_cluster_num = config.model.node_cluster_num
        self.n_classes = args.n_class
        self.vis_folder= args.vis_folder

        self._fc1 = nn.Sequential(nn.Linear(self.feature_dim, 512), nn.ReLU())

        self.transformer = VisionTransformer(num_classes=self.n_classes, embed_dim=self.embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1
        self.conv1 = GCNBlock(512, self.embed_dim, self.bn, self.add_self, self.normalize_embedding, 0., 0)  # 64->128
        self.pool1 = Linear(self.embed_dim, self.node_cluster_num)  # 100-> 20

        self.pred = nn.Linear(self.embed_dim, self.n_classes)

        self.w = nn.Linear(self.embed_dim, self.embed_dim)
        self.q = nn.Linear(self.embed_dim, self.embed_dim)

        self.num_prototypes = (2**config.tree.depth)-1

        prototype_shape = [self.num_prototypes, config.model.embed_dim]

        self.proto = nn.Parameter(torch.randn(prototype_shape), requires_grad=True)

        self.explain = args.explain

        self.attention_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim,),
            nn.Tanh(),
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, node_feat, labels, adj, mask ,file_names):

        X = node_feat
        X = self._fc1(X)
        X = mask.unsqueeze(2) * X

        X = self.conv1(X, adj, mask)
        s = self.pool1(X)

        X, adj, mc1, o1 = dense_mincut_pool(X, adj, s, mask)

        b, _, _ = X.shape

        cls_token = self.cls_token.repeat(b, 1, 1)

        X = torch.cat([cls_token, X], dim=1)

        out, X = self.transformer(X)

        proto = self.w(self.proto)

        attention_scores = torch.matmul(X, proto.T) / torch.sqrt(
            torch.tensor(proto.shape[-1], dtype=torch.float32, device=proto.device)
        )

        attention_weights = F.softmax(attention_scores, dim=-1)

        context_vector = torch.matmul(attention_weights, self.proto)

        instance_logits = self.pred(context_vector)

        attn = self.attention_head(context_vector)

        weighted_instance_logits = attn * instance_logits

        loss = mc1 + o1

        if self.explain:

            os.makedirs(self.vis_folder, exist_ok=True)

            torch.save(s[0], os.path.join(self.vis_folder, '{}_s_matrix_ori.pt'.format(
                file_names[0][0])))

            torch.save(instance_logits[:, 1:, :].squeeze(0), os.path.join(self.vis_folder, '{}_inst_logits.pt'.format(
                file_names[0][0])))

            torch.save(attn[:, 1:, :].squeeze(-1), os.path.join(self.vis_folder, '{}_attn.pt'.format(
                file_names[0][0])))

            torch.save(attention_weights, os.path.join(self.vis_folder, '{}_bag_scores.pt'.format(
                file_names[0][0])))

            s_patches = torch.bmm(s, X[:, 1:])

            torch.save(s_patches, os.path.join(self.vis_folder, '{}_feats.pt'.format(
                file_names[0][0])))

            m = nn.Softmax(dim=1)
            assign_matrix = m(s[0])

            att_matrix = torch.mm(assign_matrix, attn[:, 1:, :].squeeze(-1).transpose(1, 0))

            inst_logits = torch.mm(assign_matrix, instance_logits[:, 1:, :].squeeze(0))

            weighted_logits = att_matrix * inst_logits

            m = nn.Softmax(dim=1)
            weighted_instance_logits = m(weighted_logits)

        return weighted_instance_logits, labels, X, loss, s, self.proto


class DTree(nn.Module):

    def __init__(self, num_classes, tree_depth, proto_size, proto_dim, embed_dim):
        super(DTree, self).__init__()

        self.proto_size = proto_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self._parents = dict()
        self.proto_dim = proto_dim
        self._root = self._init_tree(num_classes, tree_depth, proto_size, self.proto_dim, self.embed_dim)

        self._set_parents()

        sorted_branches = sorted(self.branches, key=lambda x: x.index)

        self._out_map = {n.index: i for i, n in zip(range(2 ** (tree_depth) - 1), sorted_branches)}

        self._maxims = {n.index: float('-inf') for n in self.branches}

        self.num_prototypes = self.num_branches
        self.num_leaves = len(self.leaves)
        self._leaf_map = {n.index: i for i, n in zip(range(self.num_leaves), self.leaves)}

        prototype_shape = [self.num_prototypes, self.embed_dim] + self.proto_size

        self.attrib_pvec = nn.Parameter(torch.randn(prototype_shape), requires_grad=True)

    def forward(self, logits, patches,pool_map, attn_weights, training, **kwargs):

        kwargs['conv_net_output'] = self.attrib_pvec.chunk(self.num_prototypes, dim=0)
        kwargs['out_map'] = dict(self._out_map)
        kwargs['attn_weights'] = attn_weights
        kwargs['prototype'] = self.attrib_pvec

        out, attr = self._root.forward(logits, patches, training, **kwargs)

        return out

    def explain(self, logits, patches, training, sizes, id, s_matrix, y, pool_map, prefix: str, **kwargs):

        kwargs['conv_net_output'] = self.attrib_pvec.chunk(self.num_prototypes, dim=0)
        kwargs['out_map'] = dict(self._out_map)
        kwargs['img_size'] = self.img_size

        r_node_id = 0
        l_sim = None
        r_sim = None

        self._root.explain(logits, patches, training, sizes, id, s_matrix, l_sim, r_sim, y, prefix, r_node_id, pool_map,keys={},
                           **kwargs)

    def explain_internal(self, logits, patches, training, sizes, id, s_matrix, y, pool_map, prefix: str, **kwargs):
            kwargs['conv_net_output'] = self.attrib_pvec.chunk(self.num_prototypes, dim=0)
            kwargs['out_map'] = dict(self._out_map)


            r_node_id = 0
            l_sim = None
            r_sim = None

            keys = self._root.explain_internal(logits, patches, training, sizes, id, s_matrix, l_sim, r_sim, y, prefix, r_node_id,
                               pool_map, keys={},
                               **kwargs)
            return keys

    def get_min_by_ind(self, left_distance, right_distance):
        B, Br, W, H = left_distance.shape

        relative_distance = left_distance / (left_distance + right_distance)
        relative_distance = relative_distance.view(B, Br, -1)
        _, min_dist_idx = relative_distance.min(-1)
        min_left_distance = left_distance.view(B, Br, -1).gather(-1, min_dist_idx.unsqueeze(-1))
        return min_left_distance

    def _init_tree(self, num_classes, tree_depth, proto_size, proto_dim, embed_dim):
        def _init_tree_recursive(i, d):
            if d == tree_depth:
                return Leaf(i, num_classes, proto_dim)
            else:
                left = _init_tree_recursive(i + 1, d + 1)
                right = _init_tree_recursive(i + left.size + 1, d + 1)
                return Branch(i, left, right, proto_size, proto_dim, embed_dim)

        root = _init_tree_recursive(0, 0)

        return root
    def _initialize_branch_stats(self, node):
        """
        Recursively initialize statistics for each branch in the tree.
        """
        branch_stats = {}
        if isinstance(node, Branch):
            branch_stats[node.index] = {'mean': 0, 'var': 1, 'gamma':1, 'beta':0 }
            # Recursively initialize stats for child nodes
            branch_stats.update(self._initialize_branch_stats(node.l))
            branch_stats.update(self._initialize_branch_stats(node.r))
        return branch_stats

    def _set_parents(self):
        def _set_parents_recursively(node):
            if isinstance(node, Branch):
                self._parents[node.r] = node
                self._parents[node.l] = node
                _set_parents_recursively(node.r)
                _set_parents_recursively(node.l)
                return
            if isinstance(node, Leaf):
                return
            raise Exception('Unrecognized node type!')

        self._parents.clear()
        self._parents[self._root] = None
        _set_parents_recursively(self._root)

    @property
    def stats(self):
        return self.branch_stats

    @property
    def root(self):
        return self._root

    @property
    def leaves_require_grad(self) -> bool:
        return any([leaf.requires_grad for leaf in self.leaves])

    @leaves_require_grad.setter
    def leaves_require_grad(self, val: bool):
        for leaf in self.leaves:
            leaf.requires_grad = val

    @property
    def prototypes_require_grad(self) -> bool:
        return self.prototype_layer.prototype_vectors.requires_grad

    @prototypes_require_grad.setter
    def prototypes_require_grad(self, val: bool):
        self.prototype_layer.prototype_vectors.requires_grad = val

    @property
    def features_require_grad(self) -> bool:
        return any([param.requires_grad for param in self.backbone.parameters()])

    @features_require_grad.setter
    def features_require_grad(self, val: bool):
        for param in self.backbone.parameters():
            param.requires_grad = val

    @property
    def add_on_layers_require_grad(self) -> bool:
        return any([param.requires_grad for param in self._add_on.parameters()])

    @add_on_layers_require_grad.setter
    def add_on_layers_require_grad(self, val: bool):
        for param in self._add_on.parameters():
            param.requires_grad = val

    @property
    def depth(self) -> int:
        d = lambda node: 1 if isinstance(node, Leaf) else 1 + max(d(node.l), d(node.r))
        return d(self._root)

    @property
    def size(self) -> int:
        return self._root.size

    @property
    def nodes(self) -> set:
        return self._root.nodes

    @property
    def nodes_by_index(self) -> dict:
        return self._root.nodes_by_index

    @property
    def node_depths(self) -> dict:

        def _assign_depths(node, d):
            if isinstance(node, Leaf):
                return {node: d}
            if isinstance(node, Branch):
                return {node: d, **_assign_depths(node.r, d + 1), **_assign_depths(node.l, d + 1)}

        return _assign_depths(self._root, 0)

    @property
    def branches(self) -> set:
        return self._root.branches

    @property
    def leaves(self) -> set:
        return self._root.leaves

    @property
    def num_branches(self) -> int:
        return self._root.num_branches

class GraphTree(nn.Module):
    def __init__(self, config, args):
        super(GraphTree, self).__init__()
        self.config = config
        self.patch_embedding = Classifier(config, args)

        self.add_on = nn.Sequential(
            nn.Linear(config.tree.embed_dim, config.tree.proto_dim, bias=False)
        )
        self.tree = DTree(num_classes=args.n_class, tree_depth=config.tree.depth,
                          proto_size=config.tree.proto_size, proto_dim=config.tree.proto_dim,
                          embed_dim=self.config.model.embed_dim)

        self.leaves = self.tree.leaves
        self.branches = self.tree.branches
        self.nodes = self.tree.nodes
        self._root = self.tree._root
        self._parents = self.tree._parents
        self.nodes_by_index= self._root.nodes_by_index
        self.is_initialized = False
        self.pca = PCA(n_components=config.model.embed_dim, svd_solver='full')

        self.num_prototypes = self.tree.num_branches

        if config.model.mode =='kmeans':

            self.kmeans = KMeans(n_clusters=self.num_prototypes, max_iter=50, random_state=1)
        else:
            try:
                import faiss
            except ImportError:
                print("FAISS not installed. Please use KMeans option!")
                raise

                self.kmeans  = faiss.Clustering(1024, self.num_prototypes)
                self.index = faiss.IndexFlatL2(config.tree.proto_dim)


    def forward(self, node_feat, labels, adjs, masks, file_names, training):

        logits, label, patches, mincut_loss, s_matrix, proto = self.patch_embedding(node_feat, labels, adjs, masks,  file_names)

        patches = self.add_on(patches[:, 1:])

        self.tree.attrib_pvec = torch.nn.Parameter(proto.unsqueeze(2).clone())

        logits = self.tree(logits[:, 1:, :], patches, None, None, training)

        logits = torch.mean(logits, dim=1)

        pred = torch.argmax(logits, dim=1)

        return logits, pred, label, mincut_loss, patches

    def explain(self, x, y, adjs, masks, file_names, sizes, id, prefix, training=False):
        logits, label, patches, mincut_loss, s_matrix, proto = self.patch_embedding(x, y, adjs, masks, file_names)

        patches = self.add_on(patches[:, 1:])

        #self.tree.attrib_pvec = torch.nn.Parameter(proto.unsqueeze(2).clone())

        self.tree.explain(logits, patches, training, sizes, id, s_matrix, None, None, prefix)

    def explain_internal(self, x, y, adjs, masks, file_names, sizes, id, prefix,training=False ):
        logits, label, patches, mincut_loss, s_matrix, proto  = self.patch_embedding(x, y, adjs, masks, file_names)

        patches = self.add_on(patches[:, 1:])

        self.tree.attrib_pvec = torch.nn.Parameter(proto.unsqueeze(2).clone())

        s_patches = torch.bmm(s_matrix, patches)

        keys = self.tree.explain_internal(logits, s_patches, training, sizes, id, s_matrix, None, None, prefix)

        return keys

    def _init_param(self):
        def init_weights_xavier(m):
            if type(m) == torch.nn.Linear:
                # torch.nn.init.xavier_normal_(m.weight, #gain=torch.nn.init.calculate_gain('sigmoid'))
                nn.init.xavier_normal_(m.weight)

        with torch.no_grad():
            # torch.nn.init.normal_(self.attrib_pvec, mean=self.mean, std=self.std)
            self.add_on.apply(init_weights_xavier)

    def update_dtree_prototypes(self, patches):

        patches = self.pca.fit_transform(patches)

        if self.config.model.mode =='kmeans':
            kmeans = self.kmeans.fit(patches)
            self.initial_prototypes = kmeans.cluster_centers_
        else:
            try:
                import faiss
            except ImportError:
                print("FAISS not installed. Please use KMeans option!")
                raise
                for batch in patches_array:
                    self.kmeans.train(batch, self.index)
                self.initial_prototypes = faiss.vector_to_array(self.kmeans.centroids).reshape(self.num_prototypes, -1)

        with torch.no_grad():
            self.initial_prototypes = torch.tensor(self.initial_prototypes, dtype=torch.float32, device='cuda')
            self.patch_embedding.proto.copy_(self.initial_prototypes)