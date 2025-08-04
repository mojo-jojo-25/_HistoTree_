import logging
import time
import gc
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import os
from sklearn.metrics import roc_auc_score, f1_score
from helper import Trainer, Evaluator, collate, preparefeatureLabel
from surv_helper import  S_Trainer, S_Evaluator, surv_collate
from utils.training_utils import EarlyStopper
from models.GraphTransformer import GraphTree
from helper import collate
from utils.dataset import GraphDataset
from utils.survival_dataset import Surv_GraphDataset
import random
import itertools
from utils.metrics import nll_loss
from sksurv.metrics import concordance_index_censored
from surv_helper import surv_preparefeatureLabel
from models.GraphTransformer import DTree

def seed_torch(seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_prototypes(epoch, model, task_name,  save_dir="proto_logs"):
        import os
        proto_dir = os.path.join(save_dir, task_name)
        os.makedirs(proto_dir, exist_ok=True)

        dtree = model.module.tree if hasattr(model.module, "tree") else model.module
        prototype_map = dtree.prototype_map

        sorted_keys = sorted(prototype_map.keys(), key=lambda x: int(x))
        protos = [prototype_map[k].detach().cpu().unsqueeze(0) for k in sorted_keys]
        proto_tensor = torch.cat(protos, dim=0)  # [num_prototypes, feature_dim]


        torch.save(proto_tensor, os.path.join(save_dir, f"epoch_{epoch:03d}.pt"))

class HistoTree(object):
    def __init__(self, args, config, device=None):

        self.args = args
        self.task_name = args.task_name
        self.train = args.train
        self.test = args.test
        self.seed = args.seed
        self.data = args.dataset
        self.task = args.task

        self.vis_folder = args.vis_folder
        self.is_initialized = False

        self.config = config
        if args.batch_size is None:
            self.batch_size = self.config.training.batch_size
        else:
            self.batch_size = args.batch_size

        self.feature_dim = config.model.feature_dim
        self.n_class = args.n_class

        self.model_path = os.path.join(args.exp, 'saved_models')
        self.proto_path = os.path.join(args.exp, 'proto_logs')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            os.makedirs(self.proto_path)

        self.device = device

        seed_torch(self.seed, self.device)

        if self.train:
            ids_train = open(args.train_set).readlines()
            if self.task == 'survival':
                dataset_train = Surv_GraphDataset(os.path.join(self.config.data.feature_path, ""), ids_train,
                                             self.feature_dim, self.data)
                self.train_dataloader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=self.batch_size,
                                                                    num_workers=8,
                                                                    collate_fn=surv_collate, shuffle=True, pin_memory=True,
                                                                    drop_last=True)

            else:

                dataset_train = GraphDataset(os.path.join(self.config.data.feature_path, ""), ids_train,  self.feature_dim, self.data)

                self.train_dataloader = torch.utils.data.DataLoader(dataset=dataset_train , batch_size=self.batch_size,
                                                                   num_workers=8,
                                                                   collate_fn=collate, shuffle=True, pin_memory=True,
                                                                   drop_last=True)

            self.total_train_num = len(self.train_dataloader) * self.batch_size

        ids_val = open(args.val_set).readlines()
        if self.task == 'survival':
            dataset_val = Surv_GraphDataset(os.path.join(self.config.data.feature_path, ""), ids_val, self.feature_dim,
                                       self.data)
            self.val_dataloader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=self.batch_size,
                                                              num_workers=1,
                                                              collate_fn=surv_collate, shuffle=False, pin_memory=True)
        else:
            dataset_val = GraphDataset(os.path.join(self.config.data.feature_path, ""), ids_val,  self.feature_dim, self.data)
            self.val_dataloader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=self.batch_size,
                                                              num_workers=1,
                                                              collate_fn=collate, shuffle=False, pin_memory=True)
        self.total_val_num = len(self.val_dataloader) * self.batch_size


        if args.test or args.explain:

            checkpoint = torch.load(os.path.join(self.model_path, "{}.pth".format(self.task_name)),
                                    map_location=self.device)

            linkage_matrix = checkpoint.get('linkage_matrix', None)

            proto = checkpoint.get('prototypes', None)

            self.model = GraphTree(self.config, self.args, linkage_matrix=linkage_matrix, initial_prototypes=proto)

            self.model = nn.DataParallel(self.model)

            self.model.load_state_dict(checkpoint['model_state_dict'])

            self.epochs = 1

        else:

            self.model = GraphTree(self.config, self.args)

            self.epochs = args.n_epochs

            self.model = nn.DataParallel(self.model)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        print (os.path.join(self.model_path, "{}.pth".format(self.task_name)))



    def training(self):

        args = self.args
        config = self.config
        #tb_logger = self.config.tb_logger

        model = self.model
        learning_rate = args.lr

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"number of params: {n_parameters}")

        if self.task != 'survival':
            self.criterion = nn.CrossEntropyLoss()


        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)


        min_valid_loss = np.inf
        early_stopper = EarlyStopper(patience=20)

        if self.task == 'survival':
            evaluator = S_Evaluator(self.n_class)
            trainer = S_Trainer(self.n_class)
        else:
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

            if self.task =='survival':
                    if self.train:
                        all_censorships = []
                        all_event_times = []
                        all_risk_scores = []
                        for i_batch, sample_batched in enumerate(self.train_dataloader):

                            cls_token, pred, label, t, c, loss, patches = trainer.train(sample_batched, model, self.feature_dim)

                            hazards = torch.sigmoid(cls_token)

                            S = torch.cumprod(1 - hazards, dim=1)

                            risk = -torch.sum(S, dim=1)

                            train_slide_loss = nll_loss(hazards=hazards.cuda(), S=S.cuda(), Y=label.cuda(), c=c.cuda())
                            loss = loss + train_slide_loss

                            optimizer.zero_grad()

                            loss.backward()

                            optimizer.step()

                            lr_scheduler.step()

                            all_risk_scores.append(risk.cpu().detach().numpy())
                            all_event_times.append(t.cpu().detach().numpy())
                            all_censorships.append(c.cpu().detach().numpy())

                            train_loss += loss
                            total += 1

                        all_risk_scores = np.concatenate(all_risk_scores).squeeze()
                        all_event_times = np.concatenate(all_event_times).squeeze()
                        all_censorships = np.concatenate(all_censorships).squeeze()

                        c_index = concordance_index_censored(
                                event_indicator=(1 - all_censorships).astype(bool),
                                event_time=all_event_times,
                                estimate=all_risk_scores)[0]

                        c_index = torch.tensor((c_index))

                        if epoch % config.training.logging_freq == 0:
                                                 log_prototypes(epoch, model=self.model, save_dir= self.proto_path, task_name = self.task_name)
                                                 logging.info("EPOCH: {},  "
                                                              "training loss: {}, "
                                                              "c_index :{}".format(epoch+1,
                                                               train_loss / total, c_index))

                    if epoch % config.training.validation_freq == 0:
                            model.eval()
                            with torch.no_grad():

                                all_censorships = []
                                all_event_times = []
                                all_risk_scores = []
                                for i_batch, sample_batched in enumerate(self.val_dataloader):
                                    cls_token, pred, label, t, c, loss, patches = evaluator.eval_test(sample_batched, model, self.feature_dim )

                                    Y_hat = torch.argmax(cls_token, dim=1)

                                    hazards = torch.sigmoid(cls_token)

                                    S = torch.cumprod(1 - hazards, dim=1)

                                    risk = -torch.sum(S, dim=1)

                                    eval_slide_loss = nll_loss(hazards=hazards.cuda(), S=S.cuda(), Y=label.cuda(),
                                                                c=c.cuda())

                                    loss = loss + eval_slide_loss

                                    all_risk_scores.append(risk.cpu().detach().numpy())
                                    all_event_times.append(t.cpu().detach().numpy())
                                    all_censorships.append(c.cpu().detach().numpy())


                                    val_loss += loss
                                    val_total += 1

                            all_risk_scores = np.concatenate(all_risk_scores).squeeze()
                            all_event_times = np.concatenate(all_event_times).squeeze()
                            all_censorships = np.concatenate(all_censorships).squeeze()

                            c_index = concordance_index_censored(
                                event_indicator=(1 - all_censorships).astype(bool),
                                event_time=all_event_times,
                                estimate=all_risk_scores
                            )[0]

                            c_index = torch.tensor((c_index))

                            valid_loss = val_loss / total

                            if epoch % config.training.logging_freq == 0:
                                logging.info(
                                            "EPOCH: {}, "
                                            " validation loss: {},"
                                            " c_index :{}".format(epoch+1,
                                                                  valid_loss,
                                                                  c_index))
                                print(' EPOCH: {}, c_index {}'.format(epoch+1, c_index))

                            if self.train:
                                states = {
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'linkage_matrix': self.model.module.linkage_matrix,
                                    'prototypes': self.model.module.initial_prototypes
                                    }

                                torch.save(states, os.path.join(self.model_path, "{}.pth".format(self.task_name)))

            else:
                    if self.train:
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
                                    logging.info("EPOCH: {},  training loss: {}".format(epoch+1, train_loss / total))
                                    logging.info("EPOCH: {},  training accuracy: {}".format(epoch+1, trainer.get_scores()))

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

                        auc = roc_auc_score(slide_labels, slide_probs, average="macro", multi_class='ovr')
                        fscore = f1_score(slide_labels, slide_preds , average="macro")

                        print('[%d/%d] val acc: %.3f' % (self.total_val_num, self.total_val_num, evaluator.get_scores()))
                        print('[%d/%d] val AUC: %.3f' % (self.total_val_num, self.total_val_num, auc))
                        print('[%d/%d] val fscore: %.3f' % (self.total_val_num, self.total_val_num, fscore))

                        evaluator.plot_cm()
                        val_loss = val_loss/val_total
                        print(f'Validation Loss ---> {val_loss:.6f}')

                        trainer.reset_metrics()
                        evaluator.reset_metrics()

                        if early_stopper.early_stop(val_loss):
                            break

                        end_time = time.time()
                        logging.info("\nTraining of  classifier took {:.4f} minutes.\n".format(
                                (end_time - start_time) / 60))

                        if self.train:
                            if min_valid_loss > val_loss:
                                logging.info(
                                    f'Validation Loss Decreased ({min_valid_loss:.6f}--->{val_loss:.6f}) \t Saving The Model'
                                )
                                min_valid_loss = val_loss
                                # Saving State Dict
                                states = {
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'linkage_matrix': self.model.module.linkage_matrix,
                                    'prototypes': self.model.module.initial_prototypes
                                }
                                torch.save(states, os.path.join(self.model_path, "{}.pth".format(self.task_name)))


    def explain(self):

        config = self.config
        model = self.model

        os.makedirs(self.vis_folder, exist_ok=True)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"number of params: {n_parameters}")

        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # subtract max per row
            return e_x / np.sum(e_x, axis=1, keepdims=True)

        if self.task == 'survival':

            model.eval()
            with torch.no_grad():
                for i_batch, sample in enumerate(self.val_dataloader):
                    node_feat, labels, t, c, adjs, masks, names = surv_preparefeatureLabel(sample['image'],
                                                                                           sample['label'],
                                                                                           sample['t'], sample['c'],
                                                                                           sample['adj_s'],
                                                                                           sample['id'],
                                                                                           feature_dim=self.feature_dim)

                    sizes = [len(bag) for bag in sample['image']]

                    keys = model.module.explain_internal(node_feat,
                                                         labels,
                                                         adjs, masks,
                                                         names, sizes,
                                                         sample['id'],
                                                         f'{i_batch}-{labels[0]}')


                    tensor_values = list(keys.values())

                    concatenated_tensor = torch.stack(tensor_values, dim=1)
                    numpy_array = concatenated_tensor.cpu().numpy()
                    numpy_array = np.transpose(np.squeeze(numpy_array))

                    numpy_array = softmax(numpy_array)

                    #labels = np.argmax(numpy_array, axis=0)

                    filename = os.path.join(self.vis_folder, '{}_distances.npy'.format(
                                   names[0][0]))
                    with open(filename, 'wb') as f:
                        np.save(f, numpy_array)
        else:

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
                                    filename = os.path.join(self.vis_folder, '{}_distances.npy'.format(
                                                   names[0][0]))
                                    with open(filename, 'wb') as f:
                                        np.save(f, numpy_array)