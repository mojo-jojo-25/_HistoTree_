import sys
import os
import torch
import random
import numpy as np
from .tree_elements import *
from .ViT import *
from .gcn import GCNBlock
from torch_geometric.nn import GCNConv, DenseGraphConv, dense_mincut_pool
from torch.nn import Linear
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from utils.dataset import GraphDataset
from utils.survival_dataset import Surv_GraphDataset
from helper import collate
from surv_helper import surv_collate
from collections import Counter
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from sklearn.cluster import AgglomerativeClustering

class Classifier(nn.Module):
    def __init__(self, config, args):
        super(Classifier, self).__init__()

        self.feature_dim = config.model.feature_dim
        self.embed_dim = config.model.embed_dim
        self.num_layers = config.model.num_layers
        self.node_cluster_num = config.model.node_cluster_num
        self.n_classes = args.n_class
        self.vis_folder = args.vis_folder

        self._fc1 = nn.Sequential(nn.Linear(self.feature_dim, 512), nn.ReLU())

        self.transformer = VisionTransformer(num_classes=self.n_classes, embed_dim=self.embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1
        self.conv1 = GCNBlock(512, self.embed_dim, self.bn, self.add_self, self.normalize_embedding, 0., 0)
        self.pool1 = Linear(self.embed_dim, self.node_cluster_num)

        self.pred = nn.Linear(self.embed_dim, self.n_classes)

        self.num_prototypes = (2**config.tree.depth)-1

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

        attn = self.attention_head(X)

        instance_logits = self.pred(X)

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

        return weighted_instance_logits, labels, X, loss, s

class FeatureGatingTransformer(nn.Module):
    def __init__(self, dim=1024):
        super(FeatureGatingTransformer, self).__init__()
        self.gate = nn.Linear(dim * 2, dim)  # Learnable gating function
        self.norm = nn.LayerNorm(dim)

    def forward(self, left_proto, right_proto):
        # Concatenate prototypes along feature dimension (1, 2048)
        merged_input = torch.cat([left_proto, right_proto], dim=-1)

        #merged_input = F.layer_norm(merged_input, normalized_shape=[merged_input.shape[-1]])

        # Compute feature-wise gate values (sigmoid activation ensures values between 0 and 1)
        gate_values = torch.sigmoid(self.gate(merged_input))

        merged_proto = gate_values * left_proto + (1 - gate_values) * right_proto  # Feature-wise selection

        return self.norm(merged_proto)

class AttentionFusion(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, left, right):
        # Stack as sequence: [left, right]
        inputs = torch.stack([left, right], dim=0)  # Shape: [2, D]
        q = self.query(inputs.mean(dim=0, keepdim=True))  # Global query from mean
        k = self.key(inputs)  # Keys
        v = self.value(inputs)  # Values

        attn_weights = torch.softmax(q @ k.T / (k.size(-1) ** 0.5), dim=-1)  # [1, 2]
        fused = (attn_weights @ v).squeeze(0)  # [D]
        return self.norm(fused)

class CrossGatingFusion(nn.Module):
        def __init__(self, dim=1024):
            super(CrossGatingFusion, self).__init__()
            self.gate_left = nn.Linear(dim, dim)
            self.gate_right = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)

        def forward(self, left_proto, right_proto):
            gate_left = torch.sigmoid(self.gate_left(left_proto))  # Decide importance of left
            gate_right = torch.sigmoid(self.gate_right(right_proto))  # Decide importance of right

            merged_proto = gate_left * left_proto + gate_right * right_proto  # Cross-gating

            return self.norm(merged_proto)

class NeuralMerge(nn.Module):
    def __init__(self, dim=1024):
        super(NeuralMerge, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, left_proto, right_proto):
        merged_input = torch.cat([left_proto, right_proto], dim=-1)  # (1, 2048)
        merged_proto = self.mlp(merged_input)  # Learn a nonlinear merge
        return self.norm(merged_proto)


class DTree(nn.Module):

    def __init__(self, num_classes, task, proto_size, proto_dim, embed_dim, linkage_matrix, cluster_centers):
        super(DTree, self).__init__()

        self.proto_size = proto_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self._parents = dict()
        self.proto_dim = proto_dim
        self.task = task

        self._root = self._init_tree_recursive(linkage_matrix, proto_size, proto_dim, embed_dim)

        self._set_parents()

        sorted_branches = sorted(self.branches, key=lambda x: x.index)

        self._out_map = {n.index: i for i, n in zip(range(self.num_branches), sorted_branches)}

        self._maxims = {n.index: float('-inf') for n in self.branches}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_prototypes = self.num_branches
        self.num_leaves = len(self.leaves)
        self.merging_layer = CrossGatingFusion(dim=1024).to(self.device)

        sorted_leaves = sorted(self.leaves, key=lambda x: x.index)
        self._leaf_map = {n.index: i for i, n in zip(range(self.num_leaves), sorted_leaves)}

        self.attrib_pvec = nn.Parameter(cluster_centers, requires_grad=False)
        self.linkage_matrix = linkage_matrix
        self.prototype_map = nn.ParameterDict()
        self.prototype_map = self.assign_prototypes_to_branches(self.linkage_matrix, self.attrib_pvec, init=True)

    def forward(self, logits, patches, pool_map, attn_weights, training, **kwargs):

        self.prototype_map = self.assign_prototypes_to_branches(self.linkage_matrix, self.attrib_pvec, init=False)
        kwargs['conv_net_output'] = tuple(chunk for chunk in self.attrib_pvec.chunk(self.num_prototypes, dim=0))
        kwargs['out_map'] = dict(self._out_map)
        kwargs['attn_weights'] = attn_weights
        kwargs['prototype'] = self.attrib_pvec
        kwargs['prototype_map'] = self.prototype_map
        kwargs['task'] = self.task

        out, attr = self._root.forward(logits, patches, training, **kwargs)

        return out

    def explain_internal(self, logits, patches, training, sizes, id, s_matrix, y, pool_map, prefix: str, **kwargs):
            self.prototype_map = self.assign_prototypes_to_branches(self.linkage_matrix, self.attrib_pvec, init=False)
            kwargs['conv_net_output'] = self.attrib_pvec.chunk(self.num_prototypes, dim=0)
            kwargs['out_map'] = dict(self._out_map)
            kwargs['leaf_out_map'] = dict(self._leaf_map )
            kwargs['prototype'] = self.attrib_pvec
            kwargs['prototype_map'] = self.prototype_map
            kwargs['task'] = self.task

            r_node_id = 0
            l_sim = None
            r_sim = None

            keys = self._root.explain_internal(logits, patches, training, sizes, id, s_matrix, l_sim, r_sim, y, prefix, r_node_id,
                               pool_map, keys={},
                               **kwargs)
            return keys

    def assign_prototypes_to_branches(self, linkage_matrix, initial_prototypes, init):
            """
            Assign prototypes to branches based on the linkage matrix and initial prototypes.

            Args:
                linkage_matrix (ndarray): Linkage matrix from hierarchical clustering.
                initial_prototypes (torch.Tensor): Prototypes for leaves, shape (N, 1024).
                branch_indices (dict): Mapping of branches to their indices in the tree.

            Returns:
                prototype_map (dict): Mapping from branch indices to assigned prototypes.
            """

            num_samples = linkage_matrix.shape[0] + 1


            if init:
                for i in sorted(range(num_samples)):
                    param_name = str(i)
                    param = nn.Parameter(initial_prototypes[i].to(self.device), requires_grad=True)
                    self.prototype_map[param_name] = param

            for branch_idx, (left_idx, right_idx, _, _) in enumerate(linkage_matrix, start=num_samples):
                left_key = str(int(left_idx))  # ✅ Ensure correct string conversion
                right_key = str(int(right_idx))
                branch_key = str(branch_idx)

                left_prototype = self.prototype_map[left_key].to(self.device)
                right_prototype = self.prototype_map[right_key].to(self.device)

                #merged_prototype = self.merging_layer(left_prototype, right_prototype).to(self.device)
                merged_prototype = (right_prototype + left_prototype) / 2

                param = nn.Parameter(merged_prototype, requires_grad=True)
                self.prototype_map[branch_key] = param

            return self.prototype_map

    def _init_tree_recursive(self, linkage_matrix, proto_size, proto_dim, embed_dim):
            """
            Recursively builds the hierarchical binary tree, stopping when the depth limit is reached.
            """

            num_samples = linkage_matrix.shape[0] + 1  # Original number of samples
            max_depth = linkage_matrix.shape[0]  # Number of merges

            def _recursive_helper(i, d):
                """
                Inner recursive function to construct the tree.
                """
                # Base case: If reaching max depth or original samples, return a Leaf
                if i < num_samples or d == max_depth:
                    print(f"Creating Leaf {i}")
                    return Leaf(i, self.num_classes, proto_dim)

                # Ensure valid index in linkage matrix
                if i - num_samples >= len(linkage_matrix):
                    return Leaf(i, self.num_classes, proto_dim)

                # Get child indices from linkage matrix
                left_index = int(linkage_matrix[i - num_samples][0])
                right_index = int(linkage_matrix[i - num_samples][1])

                print(f"Creating Branch {i}: Left={left_index}, Right={right_index}")

                # Recursively build left and right children
                left = _recursive_helper(left_index, d + 1)
                right = _recursive_helper(right_index, d + 1)

                return Branch(i, left, right, proto_size, proto_dim, embed_dim, self.task)

            # Start from the last merged node (root)
            root = _recursive_helper(num_samples + max_depth - 1, 0)

            return root

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
    def __init__(self, config, args, linkage_matrix=None, initial_prototypes=None):
        super(GraphTree, self).__init__()
        self.args = args
        self.config = config

        self.feature_dim = config.model.feature_dim
        self.task_name = args.task_name
        self.train_flag = args.train
        self.test_flag = args.test
        self.seed = args.seed
        self.data = args.dataset
        self.task = args.task
        self.n_proto =  config.tree.n_proto
        self.n_proto_patches = 20000

        if args.batch_size is None:
            self.batch_size = self.config.training.batch_size
        else:
            self.batch_size = args.batch_size

        if self.train_flag:
            ids_train = open(args.train_set).readlines()
            if self.task == 'survival':
                dataset_train = Surv_GraphDataset(os.path.join(self.config.data.feature_path, ""), ids_train,
                                                  self.feature_dim, self.data)
                self.cluster_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=1,
                                                                  num_workers=8,
                                                                  collate_fn=surv_collate, shuffle=True, pin_memory=True,
                                                                  drop_last=True)


            else:

                dataset_train = GraphDataset(os.path.join(self.config.data.feature_path, ""), ids_train,
                                             self.feature_dim, self.data)
                self.cluster_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=1,
                                                         num_workers=8,
                                                         collate_fn=collate, shuffle=True, pin_memory=True,
                                                         drop_last=True)

            self.kmeans = KMeans(n_clusters=self.n_proto, max_iter=50, random_state=1)
            linkage_matrix, initial_prototypes = self.compute_centroids()

            self.linkage_matrix = linkage_matrix

            self.initial_prototypes = initial_prototypes

            self.patch_embedding = Classifier(config, args)
            self.add_on = nn.Sequential(
                nn.Linear(config.tree.embed_dim, config.tree.proto_dim, bias=False)
            )
        else:

            self.patch_embedding = Classifier(config, args)
            self.add_on = nn.Sequential(
                nn.Linear(config.tree.embed_dim, config.tree.proto_dim, bias=False)
            )
            self.linkage_matrix = linkage_matrix
            self.initial_prototypes = initial_prototypes


        self.tree = DTree(num_classes = args.n_class, task = self.task,
                              proto_size = config.tree.proto_size, proto_dim = config.tree.proto_dim,
                              embed_dim = self.config.model.embed_dim, linkage_matrix = self.linkage_matrix,
                              cluster_centers = self.initial_prototypes)


    def forward(self, node_feat, labels, adjs, masks, file_names, training):

        logits, label, patches, mincut_loss, s_matrix = self.patch_embedding(node_feat, labels, adjs, masks,  file_names)

        patches = self.add_on(patches[:, 1:])

        logits = self.tree(logits[:, 1:, :], patches, None, None, training)

        logits = torch.mean(logits, dim=1)

        pred = torch.argmax(logits, dim=1)

        return logits, pred, label, mincut_loss, patches

    def explain_internal(self, x, y, adjs, masks, file_names, sizes, id, prefix,training=False ):
        logits, label, patches, mincut_loss, s_matrix  = self.patch_embedding(x, y, adjs, masks, file_names)

        patches = self.add_on(patches[:, 1:])

        s_patches = torch.bmm(s_matrix, patches)

        keys = self.tree.explain_internal(logits, s_patches, training, sizes, id, s_matrix, None, None, prefix)

        return keys

    def _init_param(self):
        def init_weights_xavier(m):
            if type(m) == torch.nn.Linear:
                nn.init.xavier_normal_(m.weight)

        with torch.no_grad():
            self.add_on.apply(init_weights_xavier)


    def compute_centroids(self):
            n_patches = 0
            n_total = self.n_proto * self.n_proto_patches

            seed = 42
            torch.manual_seed(seed)
            np.random.seed(seed)

            try:
                n_patches_per_batch = (n_total + len(self.cluster_loader) - 1) // len(self.cluster_loader)
            except:
                n_patches_per_batch = 1000

            patches = torch.Tensor(n_total, self.feature_dim)

            for i_batch, sample_batched in enumerate(self.cluster_loader):
                if n_patches >= n_total:
                    continue

                data, label = sample_batched['image'][0], sample_batched['label'][0]

                n_samples = int(n_patches_per_batch)

                indices = torch.randperm(len(sample_batched))[:n_samples]

                with torch.no_grad():
                    out = data[indices].reshape(-1, data.shape[-1])

                size = out.size(0)
                if n_patches + size > n_total:
                    size = n_total - n_patches
                    out = out[:size]
                patches[n_patches: n_patches + size] = out
                n_patches += size

            kmeans = self.kmeans.fit(patches)
            initial_prototypes = kmeans.cluster_centers_
            import matplotlib.pyplot as plt
            from scipy.cluster.hierarchy import linkage, dendrogram

            linkage_matrix = linkage(initial_prototypes, method='ward')

            # plt.figure(figsize=(10, 5))
            # dendrogram(linkage_matrix)
            # plt.title('Tree structure')
            # # plt.xlabel('Prototype Index')
            # # plt.ylabel('Distance')
            # plt.yticks([])
            # plt.ylabel('')
            # plt.savefig('prototype_dendrogram.png', dpi=300, bbox_inches='tight')

            centroids = torch.tensor(initial_prototypes, dtype=torch.float32, device='cuda')

            return linkage_matrix, centroids

