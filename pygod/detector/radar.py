# -*- coding: utf-8 -*-
""" Residual Analysis for Anomaly Detection in Attributed Networks
"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import time
import torch
import warnings
from torch_geometric.utils import to_dense_adj

from . import Detector
from ..utils import logger, validate_device


class Radar(Detector):
    """
    Residual Analysis for Anomaly Detection in Attributed Networks

    Radar is an anomaly detector with residual analysis.

    .. note::
        This detector is transductive only. Using ``predict`` with
        unseen data will train the detector from scratch.

    See :cite:`li2017radar` for details.

    Parameters
    ----------
    gamma : float, optional
        Weight of node feature w.r.t. structure.
        Default: ``1``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``100``.
    gpu : int
        GPU Index, -1 for using CPU. Default: ``-1``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    """

    def __init__(self,
                 gamma=1.,
                 weight_decay=0.,
                 lr=0.004,
                 epoch=100,
                 gpu=-1,
                 contamination=0.1,
                 verbose=0):
        super(Radar, self).__init__(contamination=contamination,
                                    verbose=verbose)

        # model param
        self.gamma = gamma
        self.weight_decay = weight_decay

        # training param
        self.lr = lr
        self.epoch = epoch
        self.device = validate_device(gpu)

        self.model = None

    def fit(self, data, label=None, train_mask=None):

        data.s = to_dense_adj(data.edge_index)[0]
        num_nodes = data.x.shape[0]
        train_mask = self._validate_train_mask(train_mask, num_nodes)
        self.train_mask_ = None if torch.all(train_mask) else train_mask

        x, s, l, w_init, r_init = self.process_graph(data)

        self.model = RadarBase(w_init, r_init)
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        for epoch in range(self.epoch):
            start_time = time.time()
            x_, r = self.model(x)
            loss = self._masked_loss(x, x_, r, l, train_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            decision_score = torch.sum(torch.pow(r, 2),
                                       dim=1)[train_mask].detach().cpu()

            logger(epoch=epoch,
                   loss=loss.item(),
                   score=decision_score,
                   target=None if label is None else label[train_mask],
                   time=time.time() - start_time,
                   verbose=self.verbose,
                   train=True)

        self.decision_score_ = decision_score
        self._process_decision_score()
        return self

    def decision_function(self, data, label=None, train_mask=None):
        if data is not None:
            warnings.warn("This detector is transductive only. "
                          "Training from scratch with the input data.")
            self.fit(data, label, train_mask)
        return self.decision_score_

    def process_graph(self, data):

        x = data.x.to(self.device)
        s = data.s.to(self.device)

        s = torch.max(s, s.T)
        laplacian = torch.diag(torch.sum(s, dim=1)) - s

        w_init = torch.eye(x.shape[0]).to(self.device)
        r_init = torch.inverse((1 + self.weight_decay) *
            torch.eye(x.shape[0]).to(self.device) + self.gamma * laplacian) @ x

        return x, s, laplacian, w_init, r_init

    def _masked_loss(self, x, x_, r, l, train_mask):
        # node-wise residuals are aggregated over the training seeds
        # only; the graph-Laplacian smoothness term is retained as a
        # global structural regularizer over the whole graph
        node_term = torch.norm((x - x_ - r)[train_mask], 2)
        return node_term + self.gamma * torch.trace(r.T @ l @ r)

    def _loss(self, x, x_, r, l):
        return torch.norm(x - x_ - r, 2) + \
               self.gamma * torch.trace(r.T @ l @ r)


class RadarBase(torch.nn.Module):
    def __init__(self, w, r):
        super(RadarBase, self).__init__()
        self.w = torch.nn.Parameter(w)
        self.r = torch.nn.Parameter(r)

    def forward(self, x):
        return self.w @ x, self.r
