# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
import torch

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)

    S_padded = torch.cat([torch.ones_like(c, device = 'cuda'), S], 1)

    #S_padded = torch.cat([torch.ones(S.size(0), 1, device=S.device), S], dim=1)
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)

    Y = Y.long()  # Ensure Y is an integer tensor
    Y = Y.clamp(0, S_padded.shape[1] - 1)  # Clamp indices for S_padded
    Y_hazards = Y.clamp(0, hazards.shape[1] - 1)  # Clamp indices for hazards

    uncensored_loss = -(1 - c) * (
            torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) +
            torch.log(torch.gather(hazards, 1, Y_hazards).clamp(min=eps))
    )

    #uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss



class ConfusionMatrix(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        # axis = 0: prediction
        # axis = 1: target
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        hist = np.zeros((n_class, n_class))
        hist[label_pred, label_true] += 1

        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            tmp = self._fast_hist(lt.item(), lp.item(), self.n_classes)    #lt.item(), lp.item()
            self.confusion_matrix += tmp

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        # accuracy is recall/sensitivity for each class, predicted TP / all real positives
        # axis in sum: perform summation along

        if sum(hist.sum(axis=1)) != 0:
            acc = sum(np.diag(hist)) / sum(hist.sum(axis=1))
        else:
            acc = 0.0
        
        return acc
    
    def plotcm(self):
        print(self.confusion_matrix)

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))