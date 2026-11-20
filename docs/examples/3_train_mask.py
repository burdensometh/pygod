"""
Training Seed Mask
==================

This tutorial shows how to fit a detector on a subset of training
seed nodes while non-seed nodes only provide contextual information
through message passing.

(Time estimate: 3 minutes)
"""

import torch

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

from pygod.generator import gen_contextual_outlier, gen_structural_outlier
from pygod.detector import DOMINANT
from pygod.metric import eval_roc_auc

#######################################################################
# Build a graph with injected outliers, as in the introductory
# tutorial.

data = Planetoid('./data/Cora', 'Cora',
                  transform=T.NormalizeFeatures())[0]
data, ya = gen_contextual_outlier(data, n=100, k=50)
data, ys = gen_structural_outlier(data, m=10, n=10)
data.y = torch.logical_or(ys, ya).long()

#######################################################################
# Define the training seed population. Only these nodes are treated as
# eligible training data. The remaining nodes can still be pulled in as
# neighbors and provide context, but they never enter the optimization
# loss, the decision scores or the contamination threshold.

num_nodes = data.num_nodes
generator = torch.Generator().manual_seed(42)
train_mask = torch.rand(num_nodes, generator=generator) > 0.3

detector = DOMINANT(hid_dim=64, num_layers=4, epoch=100)
detector.fit(data, train_mask=train_mask)

#######################################################################
# ``decision_score_``, ``threshold_`` and ``label_`` are defined on the
# eligible training population only.

print('number of training seeds:', int(train_mask.sum()))
print('decision_score_ length:', detector.decision_score_.shape[0])
print('label_ length:', detector.label_.shape[0])

#######################################################################
# Inference is not restricted by the mask: every node of a graph,
# including the non-seed nodes, receives an outlier score. Here the
# fitted detector scores the whole graph.

score = detector.predict(data, return_pred=False, return_score=True)
print('inference score length:', score.shape[0])

#######################################################################
# Evaluation on the training seed population uses the same compact
# population as the loss and the threshold fitting.

auc = eval_roc_auc(data.y[train_mask], detector.decision_score_)
print('AUC on training seeds:', auc)

#######################################################################
# Detectors whose fitting contract is intrinsically defined over the
# whole node population (e.g. GAAN, GADNR, SCAN) reject a partial mask
# explicitly with ``NotImplementedError`` instead of silently letting
# non-seed nodes contaminate the loss and the threshold. An empty mask
# is always rejected with ``ValueError``.
