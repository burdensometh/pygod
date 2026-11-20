# -*- coding: utf-8 -*-
"""Tests for missing-value handling: train-fitted imputation/scaling,
observation-masked reconstruction loss, and the DOMINANT integration.
"""
import os
import unittest
import numpy as np
from numpy.testing import assert_raises

import torch
from torch_geometric.seed import seed_everything

from pygod.detector import DOMINANT
from pygod.metric import eval_roc_auc
from pygod.nn.functional import double_recon_loss
from pygod.utils import FeatureImputer

seed_everything(717)

NAN = float('nan')


class TestFeatureImputer(unittest.TestCase):
    def setUp(self):
        # col 0 observed: [1, 3, 3]; col 1 observed: [2, 4, 4];
        # col 2 observed: [6, 6]
        self.x = torch.tensor([[1., 2., NAN],
                               [3., NAN, 6.],
                               [NAN, 4., 6.],
                               [3., 4., NAN]])

    def test_fit_uses_observed_entries_only(self):
        imp = FeatureImputer().fit(self.x)
        assert torch.allclose(imp.mean_,
                              torch.tensor([7. / 3., 10. / 3., 6.]))
        assert torch.equal(imp.n_obs_, torch.tensor([3, 3, 2]))
        # population std over observed entries only
        col0 = torch.tensor([1., 3., 3.])
        assert torch.allclose(imp.std_[0],
                              col0.std(unbiased=False))

    def test_transform_reuses_fitted_state(self):
        imp = FeatureImputer().fit(self.x)
        x_new = torch.tensor([[100., NAN, NAN],
                              [NAN, -50., 12.]])
        mean_before = imp.mean_.clone()
        std_before = imp.std_.clone()
        n_obs_before = imp.n_obs_.clone()

        out, mask = imp.transform(x_new)

        # validation data must not affect the training statistics
        assert torch.equal(imp.mean_, mean_before)
        assert torch.equal(imp.std_, std_before)
        assert torch.equal(imp.n_obs_, n_obs_before)

        # no NaN left, and not by zero-filling: observed entries are
        # standardized with the training statistics
        assert not torch.isnan(out).any()
        assert torch.allclose(out[0, 0],
                              (100. - mean_before[0]) / std_before[0])
        assert torch.allclose(out[1, 2],
                              (12. - mean_before[2]) / std_before[2])
        # missing entries become the training mean (0 after scaling)
        assert out[0, 1] == 0 and out[0, 2] == 0 and out[1, 0] == 0
        # the observation mask marks exactly the observed entries
        assert mask.tolist() == [[1., 0., 0.], [0., 1., 1.]]

    def test_fully_missing_column_and_row(self):
        x = self.x.clone()
        x[:, 2] = NAN  # feature never observed in training
        x[0, :] = NAN  # node with all features missing
        imp = FeatureImputer().fit(x)

        # explicit policy: never-observed feature gets mean 0, std 1
        assert imp.mean_[2] == 0 and imp.std_[2] == 1

        out, mask = imp.transform(x)
        # explicit policy: fully-missing node is imputed to the
        # training means (0 after scaling), with an all-zero mask
        assert torch.equal(out[0], torch.zeros(3))
        assert torch.equal(mask[0], torch.zeros(3))

    def test_not_fitted_and_dim_mismatch(self):
        imp = FeatureImputer()
        with assert_raises(AssertionError):
            imp.transform(self.x)
        imp.fit(self.x)
        with assert_raises(ValueError):
            imp.transform(torch.ones(2, self.x.shape[1] + 1))


class TestMaskedReconLoss(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([[1., 2.],
                               [3., 4.],
                               [5., 6.]])
        self.x_ = torch.tensor([[1.5, 1.],
                                [2., 4.],
                                [0., 0.]])
        self.s = torch.tensor([[0., 1., 0.],
                               [1., 0., 1.],
                               [0., 1., 0.]])
        self.s_ = torch.tensor([[0.1, 0.9, 0.2],
                                [0.8, 0.2, 0.7],
                                [0.3, 0.6, 0.1]])
        # node 0 fully observed, node 1 half observed, node 2 unobserved
        self.mask = torch.tensor([[1., 1.],
                                  [1., 0.],
                                  [0., 0.]])

    def test_no_mask_keeps_legacy_behavior(self):
        score = double_recon_loss(self.x, self.x_, self.s, self.s_, 0.5)
        attr = torch.sqrt(((self.x - self.x_) ** 2).sum(1))
        stru = torch.sqrt(((self.s - self.s_) ** 2).sum(1))
        assert torch.allclose(score, 0.5 * attr + 0.5 * stru)

    def test_masked_dimensions_do_not_change_score(self):
        x2 = self.x.clone()
        x2[1, 1] = 1e6  # corrupt an unobserved dimension
        x2[2] = -1e6  # corrupt a fully unobserved node
        s1 = double_recon_loss(self.x, self.x_, self.s, self.s_, 0.5,
                               x_mask=self.mask)
        s2 = double_recon_loss(x2, self.x_, self.s, self.s_, 0.5,
                               x_mask=self.mask)
        assert torch.allclose(s1, s2)

    def test_normalized_by_valid_dimensions(self):
        score = double_recon_loss(self.x, self.x_, self.s, self.s_, 1.0,
                                  x_mask=self.mask)
        # node 0: RMS over 2 observed dims
        e0 = torch.sqrt(((self.x[0] - self.x_[0]) ** 2).sum() / 2)
        # node 1: only dim 0 observed
        e1 = torch.sqrt(((self.x[1, 0] - self.x_[1, 0]) ** 2).sum() / 1)
        assert torch.allclose(score[0], e0)
        assert torch.allclose(score[1], e1)

    def test_fully_missing_node_scored_by_structure_only(self):
        score = double_recon_loss(self.x, self.x_, self.s, self.s_, 0.9,
                                  x_mask=self.mask)
        stru = torch.sqrt(((self.s[2] - self.s_[2]) ** 2).sum())
        assert torch.allclose(score[2], stru)

    def test_structure_score_is_not_masked(self):
        # with weight 0 the score is the pure structure error,
        # with or without an observation mask
        stru = torch.sqrt(((self.s - self.s_) ** 2).sum(1))
        s_nomask = double_recon_loss(self.x, self.x_, self.s, self.s_, 0.)
        s_masked = double_recon_loss(self.x, self.x_, self.s, self.s_, 0.,
                                     x_mask=self.mask)
        assert torch.allclose(s_nomask, stru)
        assert torch.allclose(s_masked, stru)


class TestDOMINANTMissing(unittest.TestCase):
    def setUp(self):
        self.roc_floor = 0.60

        self.train_data = torch.load(os.path.join('pygod/test/train_graph.pt'))
        self.test_data = torch.load(os.path.join('pygod/test/test_graph.pt'))

        # inject 20% missing values plus one fully-missing node per graph
        g = torch.Generator().manual_seed(717)
        for d in (self.train_data, self.test_data):
            miss = torch.rand(d.x.shape, generator=g) < 0.2
            d.x = d.x.masked_fill(miss, NAN)
        self.train_data.x[0] = NAN
        self.test_data.x[1] = NAN

    def test_fit_predict_with_missing(self):
        detector = DOMINANT(epoch=5, num_layers=3)
        detector.fit(self.train_data)

        # imputation/scaling state fitted on the training graph only
        obs = ~torch.isnan(self.train_data.x)
        obs_mean = torch.nansum(self.train_data.x, 0) / obs.sum(0)
        assert torch.allclose(detector.imputer_.mean_, obs_mean)

        # the user's graph is not modified by fitting
        assert torch.isnan(self.train_data.x).any()
        assert not hasattr(self.train_data, 'x_mask')
        assert not hasattr(self.train_data, 's')

        # scores and labels are complete and finite: no node is masked
        # out, including the fully-missing node 0
        assert detector.decision_score_.shape[0] == self.train_data.num_nodes
        assert torch.isfinite(detector.decision_score_).all()
        assert detector.label_.shape[0] == self.train_data.num_nodes
        assert detector.label_.sum() > 0
        assert np.isfinite(detector.threshold_)

        score = detector.predict(return_pred=False, return_score=True)
        assert (eval_roc_auc(self.train_data.y, score) >= self.roc_floor)

        mean_before = detector.imputer_.mean_.clone()
        std_before = detector.imputer_.std_.clone()
        pred, score = detector.predict(self.test_data,
                                       return_pred=True,
                                       return_score=True)

        # the testing data does not affect the training statistics
        assert torch.equal(detector.imputer_.mean_, mean_before)
        assert torch.equal(detector.imputer_.std_, std_before)
        assert torch.isnan(self.test_data.x).any()

        assert pred.shape[0] == self.test_data.y.shape[0]
        assert torch.isfinite(score).all()
        assert (eval_roc_auc(self.test_data.y, score) >= self.roc_floor)

    def test_minibatch_with_missing(self):
        detector = DOMINANT(hid_dim=32,
                            num_layers=2,
                            epoch=2,
                            batch_size=16,
                            num_neigh=[2, 1])
        detector.fit(self.train_data)
        pred, score = detector.predict(self.test_data,
                                       return_pred=True,
                                       return_score=True)
        assert pred.shape[0] == self.test_data.y.shape[0]
        assert torch.isfinite(score).all()

    def test_imputed_x_and_mask_enter_detector(self):
        detector = DOMINANT(epoch=3, num_layers=2)
        detector.fit(self.train_data)

        # replay what decision_function does, keeping the tensors
        data = self.test_data.clone()
        data.x, data.x_mask = detector.imputer_.transform(data.x)
        detector.process_graph(data)
        data.batch_size = data.num_nodes
        data.n_id = torch.arange(data.num_nodes)

        detector.model.eval()
        with torch.no_grad():
            _, score = detector.forward_model(data)
            x_, s_ = detector.model(data.x, data.edge_index)

        # the network consumed the imputed features (no NaN anywhere)
        assert not torch.isnan(data.x).any()
        assert not torch.isnan(x_).any()

        # the score is exactly the masked reconstruction loss
        manual = double_recon_loss(data.x, x_, data.s, s_,
                                   detector.weight, x_mask=data.x_mask)
        assert torch.allclose(score, manual)

        # the fully-missing node 1 is scored by structure error only
        stru = torch.sqrt(((data.s - s_) ** 2).sum(1))
        assert torch.allclose(score[1], stru[1])

        # and the mask genuinely matters for the score
        unmasked = double_recon_loss(data.x, x_, data.s, s_,
                                     detector.weight)
        assert not torch.allclose(score[1], unmasked[1])

    def test_decision_function_before_fit(self):
        detector = DOMINANT(epoch=1)
        with assert_raises(AssertionError):
            detector.decision_function(self.test_data)


if __name__ == '__main__':
    unittest.main()
