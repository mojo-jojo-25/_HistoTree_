import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABC
from typing import Dict, Optional

import torch
from torch import nn

class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x

class Attention2(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention2, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x, isNorm=True):
        ## x: N x L
        A = self.attention(x)  ## N x K
        A = torch.transpose(A, 1, 0)  # KxN
        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N
        return A  ### K x N


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N
        return A  ### K x N

class Attention_without_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, isNorm=True):
        super(Attention_without_Classifier, self).__init__()
        self.attention = Attention_Gated(L, D, K)
        self.wv1 = nn.Linear(L, D)
        self.wv2 = nn.Linear(L,D)
        self.isNorm = isNorm
        self.gelu = nn.GELU()
    def forward(self, x): ## x: N x L
        AA = self.attention(x, self.isNorm)

        afeat = AA * x
        h = self.wv1(x)

        afeat = afeat + h
        afeat = torch.relu(self.wv2(afeat))
        return afeat

class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0, isNorm=True):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)
        self.isNorm=isNorm

    def forward(self, x): ## x: N x L
        AA = self.attention(x, self.isNorm )  ## K x N

        afeat = AA * x ## K x L
        #afeat= torch.sum(afeat, dim=1)

        pred = self.classifier(afeat) ## K x num_cls

        return pred


