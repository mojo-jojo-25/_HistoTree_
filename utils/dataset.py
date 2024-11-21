import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFile
import random
from torchvision.transforms import ToTensor
from torchvision import transforms
import cv2
import torch.nn.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True

def preparefeatureLabel(batch_graph, batch_label, batch_adjs, name, feature_dim):
    batch_size = len(batch_graph)
    labels = torch.LongTensor(batch_size)
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

    return node_feat, labels, adjs, masks, names


def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]


def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    trnsfrms_val = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )
    return trnsfrms_val


class GraphDataset(data.Dataset):
    """input and label image dataset"""

    def __init__(self, root, ids, feature_dim, dataset, target_patch_size=-1):
        super(GraphDataset, self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.root = root
        self.ids = ids

        self.dataset = dataset
        if self.dataset=='lung':
            self.classdict = {'LUAD': 0, 'LUSC': 1}
        elif self.dataset=='breast':
            self.classdict = {'IDC': 1, 'ILC': 0}
        elif self.dataset == 'kidney':
            self.classdict  = {'ccRCC': 0, 'PRCC': 1, 'ChRCC': 2}
        self.feature_dim = feature_dim

    def __getitem__(self, index):
        sample = {}
        info = self.ids[index].replace('\n', '')
        file_name, label = info.split('\t')[0], info.split('\t')[1]

        file_path = self.root + 'graphs/simclr_files/'  # _resnet_with

        sample['label'] = self.classdict[label]
        sample['id'] = file_name

        # feature_path = os.path.join(self.root, file_name, 'features.pt')
        feature_path = os.path.join(file_path, file_name, 'features.pt')

        if os.path.exists(feature_path):
            features = torch.load(feature_path, map_location=lambda storage, loc: storage)
        else:
            features = torch.zeros(1, self.feature_dim)

        # adj_s_path = os.path.join(self.root, file_name, 'adj_s.pt')
        adj_s_path = os.path.join(file_path, file_name, 'adj_s.pt')
        if os.path.exists(adj_s_path):
            adj_s = torch.load(adj_s_path, map_location=lambda storage, loc: storage)
        else:
            print(adj_s_path + ' not exists')
            adj_s = torch.ones(features.shape[0], features.shape[0])

        # features = features.unsqueeze(0)
        sample['image'] = features
        sample['adj_s'] = adj_s  # adj_s.to(torch.double)
        # return {'image': image.astype(np.float32), 'label': label.astype(np.int64)}

        return sample

    def __len__(self):
        return len(self.ids)