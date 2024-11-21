import logging
import time
import gc
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from helper import Trainer, Evaluator, collate
from utils.training_utils import EarlyStopper, get_optimizer, get_grad_norm, SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from models.GraphTransformer import GraphTree
from helper import collate
import pandas as pd
from utils.dataset import GraphDataset, preparefeatureLabel
from timm.utils import AverageMeter
import torchmetrics
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from copy import deepcopy
import itertools
import torch.nn.functional as F

plt.style.use('ggplot')


def seed_torch(seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class HistoTree(object):
    def __init__(self, args, config, device=None):

        self.args = args
        self.task_name = args.task_name
        self.train = args.train
        self.test = args.test
        self.seed = args.seed
        self.data = args.dataset

        self.vis_folder = args.vis_folder
        self.is_initialized = False

        self.config = config
        if args.batch_size is None:
            self.batch_size=self.config.training.batch_size
        else:
            self.batch_size = args.batch_size

        self.feature_dim = config.model.feature_dim
        self.n_class = args.n_class

        self.model_path = os.path.join(args.exp, 'saved_models')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.device = device

        seed_torch(self.seed, self.device)

        if args.train:
            ids_train = open(args.train_set).readlines()
            dataset_train = GraphDataset(os.path.join(self.config.data.feature_path, ""), ids_train,  self.feature_dim, self.data)
            self.train_dataloader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=self.batch_size,
                                                           num_workers=8,
                                                           collate_fn=collate, shuffle=True, pin_memory=True,
                                                           drop_last=True)
            self.cluster_loader=torch.utils.data.DataLoader(dataset=dataset_train, batch_size=1,
                                                           num_workers=8,
                                                           collate_fn=collate, shuffle=True, pin_memory=True,
                                                           drop_last=True)
            self.total_train_num = len(self.train_dataloader) * self.batch_size

        ids_val = open(args.val_set).readlines()
        dataset_val = GraphDataset(os.path.join(self.config.data.feature_path, ""), ids_val,  self.feature_dim, self.data)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=self.batch_size, num_workers=1,
                                                     collate_fn=collate, shuffle=False, pin_memory=True)
        self.total_val_num = len(self.val_dataloader) * self.batch_size

        self.model = GraphTree(self.config, self.args )

        self.prototypes = self.model.num_prototypes
        self.prototype_shape = [self.prototypes, config.model.feature_dim]

        self.n_proto_patches = 10000

        self.epochs = args.n_epochs

        self.model = nn.DataParallel(self.model)

        if args.test or args.explain:
            checkpoint = torch.load(os.path.join(self.model_path, "{}.pth".format(self.task_name)),
                                    map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            # for index, stats in checkpoint['stats'].items():
            #     self.model.module.stats[index] = stats

        if torch.cuda.is_available():
            self.model = self.model.cuda()


    def training(self):

        args = self.args
        config = self.config
        #tb_logger = self.config.tb_logger

        model = self.model
        learning_rate = args.lr

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"number of params: {n_parameters}")

        self.criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 100], gamma=0.1)

        min_valid_loss = np.inf
        early_stopper = EarlyStopper(patience=20)

        evaluator = Evaluator(self.n_class)
        trainer = Trainer(self.n_class)

        start_time = time.time()
        for epoch in range(self.epochs):
            model.train()
            train_loss = 0
            val_loss = 0
            total = 0
            val_total = 0
            current_lr = optimizer.param_groups[0]['lr']
            print('\n=>Epoches %i, learning rate = %.7f, previous best = %.4f' % (epoch + 1, current_lr, min_valid_loss))

            if self.train:
                if not self.is_initialized:

                    n_patches = 0
                    n_total = self.prototypes * self.n_proto_patches
                    # Sample equal number of patch features from each WSI
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

                    self.model.module.update_dtree_prototypes(patches)

                    self.is_initialized = True

                for i_batch, sample_batched in enumerate(self.train_dataloader):

                    cls_token, pred, label, loss, patches = trainer.train(sample_batched, model, self.feature_dim)

                    train_slide_loss = self.criterion(cls_token, label)

                    loss = loss + train_slide_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    lr_scheduler.step()

                    train_loss += loss
                    total += 1

                    trainer.metrics.update(label, pred)

                if epoch % config.training.logging_freq == 0:
                            logging.info("EPOCH: {},  training loss: {}".format(epoch, train_loss / total))
                            logging.info("EPOCH: {},  training accuracy: {}".format(epoch, trainer.get_scores()))

            if epoch % config.training.validation_freq == 0:
                model.eval()
                with torch.no_grad():

                    slide_labels = []
                    slide_preds = []
                    slide_probs = []
                    for i_batch, sample_batched in enumerate(self.val_dataloader):
                        cls_token, pred, label, loss, _= evaluator.eval_test(sample_batched, model, self.feature_dim )

                        slide_labels.append(label.cpu().detach().numpy().tolist())
                        slide_preds.append(pred.cpu().detach().numpy().tolist())
                        slide_probs.append(nn.Softmax(dim=1)(cls_token).cpu().detach().numpy().tolist())

                        eval_slide_loss = self.criterion(cls_token, label)

                        loss = loss+eval_slide_loss

                        evaluator.metrics.update(label, pred)

                        val_loss += loss
                        val_total += 1

                slide_probs = np.vstack((slide_probs))
                slide_labels = list(itertools.chain(*slide_labels))
                slide_probs = list(itertools.chain(*slide_probs))
                slide_preds = list(itertools.chain(*slide_preds))
                slide_probs = np.reshape(slide_probs, (len(slide_labels), self.n_class))

                if self.data=='kidney':
                    auc = roc_auc_score(slide_labels, slide_probs, average="macro", multi_class='ovr')
                    fscore = f1_score(slide_labels, slide_preds , average="weighted")
                else:
                    auc = roc_auc_score(slide_labels, slide_probs[:, 1].reshape(-1, 1))
                    fscore = f1_score(slide_labels, np.clip(slide_preds, 0, 1), average="macro")


                print('[%d/%d] val acc: %.3f' % (self.total_val_num, self.total_val_num, evaluator.get_scores()))
                print('[%d/%d] val AUC: %.3f' % (self.total_val_num, self.total_val_num, auc))
                print('[%d/%d] val fscore: %.3f' % (self.total_val_num, self.total_val_num, fscore))


                evaluator.plot_cm()
                val_loss = val_loss/val_total
                print(f'Validation Loss ---> {val_loss:.6f}')


                if self.train:
                    if min_valid_loss > val_loss:
                        logging.info(
                            f'Validation Loss Decreased ({min_valid_loss:.6f}--->{val_loss:.6f}) \t Saving The Model'
                                    )
                        min_valid_loss = val_loss
                        # Saving State Dict
                        states = {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                        }
                        torch.save(states, os.path.join(self.model_path,"{}.pth".format(self.task_name)))

            trainer.reset_metrics()
            evaluator.reset_metrics()

            if early_stopper.early_stop(val_loss):
                break

        end_time = time.time()
        logging.info("\nTraining of  classifier took {:.4f} minutes.\n".format(
                (end_time - start_time) / 60))


    def explain(self):

        config = self.config
        model = self.model

        os.makedirs(self.vis_folder, exist_ok=True)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"number of params: {n_parameters}")


        model.eval()
        with torch.no_grad():
                for i_batch, sample_batched in enumerate(self.val_dataloader):
                                node_feat, labels, adjs, masks, names = preparefeatureLabel(sample_batched['image'],
                                                                                            sample_batched['label'],
                                                                                            sample_batched['adj_s'],
                                                                                            sample_batched['id'],
                                                                                            self.feature_dim)

                                sizes = [len(bag) for bag in sample_batched['image']]

                                keys = model.module.explain_internal(node_feat,
                                                                     labels,
                                                                     adjs, masks,
                                                                     names, sizes,
                                                                     sample_batched['id'],
                                                                     f'{i_batch}-{labels[0]}')
                                tensor_values = list(keys.values())
                                concatenated_tensor = torch.stack(tensor_values, dim=1)
                                numpy_array = concatenated_tensor.cpu().numpy()
                                numpy_array = np.transpose(np.squeeze(numpy_array))
                                #normalized_values = normalize(numpy_array, norm='l1')
                                # filename = os.path.join(self.vis_folder, '{}_distances.npy'.format(
                                #                names[0][0]))
                                # with open(filename, 'wb') as f:
                                #     np.save(f, numpy_array)