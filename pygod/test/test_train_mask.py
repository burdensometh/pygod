# -*- coding: utf-8 -*-
"""Tests for the explicit training seed mask contract.

A training seed mask restricts the eligible training population: only
seed nodes are centers of neighbor sampling, enter the optimization
loss, and define the decision scores / contamination threshold.
Non-seed nodes may still be sampled as contextual neighbors but must
never participate in the loss or the threshold fitting. Inference
still scores every node.
"""
import os
import unittest

import numpy as np
import torch
from torch_geometric.seed import seed_everything

from pygod.detector import (DOMINANT, GAE, CoLA, CONAD, OCGNN, DMGD,
                            DONE, AdONE, Radar, ANOMALOUS, ONE, SCAN,
                            GAAN, GADNR, CARD, GUIDE, AnomalyDAE)
from pygod.metric import eval_roc_auc

seed_everything(717)


class TestTrainMask(unittest.TestCase):
    def setUp(self):
        self.train_data = torch.load(
            os.path.join('pygod/test/train_graph.pt'))
        self.test_data = torch.load(
            os.path.join('pygod/test/test_graph.pt'))
        n = self.train_data.num_nodes
        # 70% of the nodes are training seeds, the rest are context
        g = torch.Generator().manual_seed(42)
        self.train_mask = torch.rand(n, generator=g) > 0.3
        self.num_seeds = int(self.train_mask.sum())
        self.assertLess(self.num_seeds, n)
        self.assertGreater(self.num_seeds, 0)

    def _check_seed_population(self, detector, num_seeds):
        # scores, threshold and labels share the same eligible
        # population
        self.assertEqual(detector.decision_score_.shape[0], num_seeds)
        self.assertEqual(detector.label_.shape[0], num_seeds)
        expected_thr = np.percentile(
            detector.decision_score_.numpy(),
            100 * (1 - detector.contamination))
        self.assertAlmostEqual(detector.threshold_, expected_thr)
        expected_label = (detector.decision_score_ >
                          detector.threshold_).long()
        torch.testing.assert_close(detector.label_, expected_label)
        # the eligible population is remembered explicitly
        self.assertIsNotNone(detector.train_mask_)
        self.assertEqual(int(detector.train_mask_.sum()), num_seeds)

    def test_dominant_full_batch_mask(self):
        detector = DOMINANT(epoch=3, num_layers=2)
        detector.fit(self.train_data, train_mask=self.train_mask)

        self._check_seed_population(detector, self.num_seeds)
        self.assertEqual(detector.decision_score_.shape[0],
                         self.num_seeds)

        # training-data prediction only covers the eligible population
        train_pred = detector.predict(return_pred=True,
                                      return_score=False)
        self.assertEqual(train_pred.shape[0], self.num_seeds)

        # inference still scores every node of a new graph
        pred, score = detector.predict(self.test_data,
                                       return_pred=True,
                                       return_score=True)
        self.assertEqual(score.shape[0], self.test_data.num_nodes)
        self.assertEqual(pred.shape[0], self.test_data.num_nodes)

    def test_dominant_mini_batch_mask(self):
        detector = DOMINANT(epoch=2,
                            num_layers=2,
                            batch_size=16,
                            num_neigh=[2, 1],
                            save_emb=True)
        detector.fit(self.train_data, train_mask=self.train_mask)

        self._check_seed_population(detector, self.num_seeds)
        # embeddings of non-seed nodes stay at their zero initialization
        # because only seed nodes were sampled as targets
        non_seed = ~self.train_mask
        self.assertEqual(
            float(detector.emb[non_seed].abs().sum()), 0.0)
        self.assertGreater(
            float(detector.emb[self.train_mask].abs().sum()), 0.0)

        # full-graph inference fills embeddings for every node
        detector.predict(self.test_data, return_pred=False,
                         return_score=True, return_emb=True)
        self.assertEqual(detector.emb.shape[0],
                         self.test_data.num_nodes)

    def test_all_true_mask_equals_none(self):
        seed_everything(1)
        d1 = DOMINANT(epoch=2, num_layers=2)
        d1.fit(self.train_data)
        seed_everything(1)
        d2 = DOMINANT(epoch=2, num_layers=2)
        d2.fit(self.train_data,
               train_mask=torch.ones(self.train_data.num_nodes,
                                     dtype=torch.bool))
        torch.testing.assert_close(d1.decision_score_,
                                   d2.decision_score_)
        self.assertIsNone(d2.train_mask_)

    def test_bad_mask_shape_rejected(self):
        detector = DOMINANT(epoch=1, num_layers=2)
        bad = torch.ones(self.train_data.num_nodes + 1, dtype=torch.bool)
        with self.assertRaises(ValueError):
            detector.fit(self.train_data, train_mask=bad)

    def test_bad_mask_type_rejected(self):
        detector = DOMINANT(epoch=1, num_layers=2)
        with self.assertRaises(TypeError):
            detector.fit(self.train_data,
                         train_mask=[True] * self.train_data.num_nodes)

    def test_unsupported_detectors_reject_mask(self):
        for cls in (GADNR, GAAN, SCAN):
            detector = cls(epoch=1) if cls is not SCAN else cls()
            # a partial seed mask is an explicit contract violation
            with self.assertRaises(NotImplementedError):
                detector.fit(self.train_data,
                             train_mask=self.train_mask)

    def test_unsupported_detectors_accept_full_mask(self):
        # an all-True mask is semantically identical to None
        full = torch.ones(self.train_data.num_nodes, dtype=torch.bool)
        for cls in (GADNR, GAAN, SCAN):
            detector = cls(epoch=1) if cls is not SCAN else cls()
            detector.fit(self.train_data, train_mask=full)
            self.assertEqual(detector.decision_score_.shape[0],
                             self.train_data.num_nodes)

    def test_empty_mask_rejected_everywhere(self):
        empty = torch.zeros(self.train_data.num_nodes, dtype=torch.bool)
        for cls in (DOMINANT, Radar, ONE, GAAN, GADNR, SCAN):
            detector = cls(epoch=1) if cls is not SCAN else cls()
            with self.assertRaises(ValueError):
                detector.fit(self.train_data, train_mask=empty)

    def test_masked_logging_runs(self):
        # verbose=3 exercises every logging metric on the masked
        # population; labels must be filtered to the seeds
        detector = DOMINANT(epoch=2, num_layers=2, verbose=3)
        detector.fit(self.train_data,
                     label=self.train_data.y,
                     train_mask=self.train_mask)
        self._check_seed_population(detector, self.num_seeds)

    def test_non_deep_detectors_mask(self):
        for cls in (Radar, ANOMALOUS, ONE):
            detector = cls(epoch=3)
            detector.fit(self.train_data, train_mask=self.train_mask)
            self._check_seed_population(detector, self.num_seeds)
            # inference on the training graph returns seed scores only
            score = detector.predict(return_pred=False,
                                     return_score=True)
            self.assertEqual(score.shape[0], self.num_seeds)

    def test_deep_detectors_mask_shapes(self):
        # every mask-supporting deep detector must run end to end and
        # keep the seed-only population contract
        fast = dict(epoch=1, num_layers=2)
        detectors = [
            GAE(**fast),
            AnomalyDAE(**fast),
            GUIDE(**fast),
            CoLA(**fast),
            CONAD(**fast),
            OCGNN(**fast),
            DMGD(**fast),
            DONE(**fast),
            AdONE(**fast),
        ]
        for detector in detectors:
            detector.fit(self.train_data,
                         train_mask=self.train_mask)
            self.assertEqual(
                detector.decision_score_.shape[0],
                self.num_seeds,
                msg=detector.__class__.__name__)
            self.assertEqual(
                detector.label_.shape[0],
                self.num_seeds,
                msg=detector.__class__.__name__)
            score = detector.predict(self.test_data,
                                     return_pred=False,
                                     return_score=True)
            self.assertEqual(score.shape[0],
                             self.test_data.num_nodes,
                             msg=detector.__class__.__name__)

    def test_masked_fit_keeps_reasonable_auc(self):
        detector = DOMINANT(epoch=5, num_layers=2)
        detector.fit(self.train_data, train_mask=self.train_mask)
        # evaluate on the seed population only
        score = detector.decision_score_
        auc = eval_roc_auc(self.train_data.y[self.train_mask], score)
        self.assertGreater(auc, 0.55)

    def test_card_mask(self):
        detector = CARD(epoch=1, num_layers=2,
                        subgraph_num_neigh=2)
        detector.fit(self.train_data, train_mask=self.train_mask)
        self.assertEqual(detector.decision_score_.shape[0],
                         self.num_seeds)
        score = detector.predict(self.test_data,
                                 return_pred=False,
                                 return_score=True)
        self.assertEqual(score.shape[0], self.test_data.num_nodes)


if __name__ == '__main__':
    unittest.main()
