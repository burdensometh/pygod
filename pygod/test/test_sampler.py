# -*- coding: utf-8 -*-

import numpy as np
import torch
import unittest

from numpy.testing import assert_raises
from numpy.testing import assert_warns

from torch_geometric.data import Data, Batch

from pygod.detector import DOMINANT
from pygod.utils import BudgetSampler, ReviewSample


def make_graph(n=40, num_feat=8):
    """Ring graph over n nodes (links the two embedding clusters)."""
    src = list(range(n)) + list(range(1, n)) + [0]
    dst = list(range(1, n)) + [0] + list(range(n))
    edge_index = torch.tensor([src + dst, dst + src])
    x = torch.arange(n * num_feat, dtype=torch.float).reshape(n, num_feat)
    return Data(x=x, edge_index=edge_index)


def make_score_emb(n=40):
    """Two separated embedding clusters; the top scores all live in
    the first cluster, so naive top-k review picks one cluster only."""
    rs = np.random.RandomState(0)
    emb = np.vstack([rs.randn(n // 2, 4) * 0.01,
                     rs.randn(n // 2, 4) * 0.01 + 10.0])
    score = np.zeros(n)
    score[:n // 2] = np.linspace(0.5, 1.0, n // 2)
    score[n // 2:] = np.linspace(0.0, 0.4, n // 2)
    return score, emb


class TestBudgetSampler(unittest.TestCase):

    def setUp(self):
        self.n = 40
        self.data = make_graph(self.n)
        self.score, self.emb = make_score_emb(self.n)

    def test_cluster_diversity(self):
        # naive top-6 would be nodes 14..19, all from cluster 0
        sampler = BudgetSampler(budget=6, quota_by='none', seed=42)
        result = sampler.sample(self.data, self.score, emb=self.emb)

        assert isinstance(result, ReviewSample)
        assert result.method == 'cluster'
        assert len(result) == 6
        assert len(set(result.node_id.tolist())) == 6
        # the sample covers both embedding clusters
        picked = result.node_id.numpy()
        assert (picked < self.n // 2).any()
        assert (picked >= self.n // 2).any()
        # representatives are the highest-score nodes of their clusters
        assert (result.cluster_id >= 0).all()

    def test_quota_by_graph(self):
        g1 = make_graph(20)
        g2 = make_graph(24)
        batch = Batch.from_data_list([g1, g2])
        score = np.random.RandomState(1).rand(batch.num_nodes)
        emb = np.random.RandomState(2).randn(batch.num_nodes, 4)

        sampler = BudgetSampler(budget=6, quota_by='graph', seed=0)
        result = sampler.sample(batch, score, emb=emb)

        # proportional allocation over the two graphs: 20/44 and 24/44
        assert result.quota == {0: 3, 1: 3}
        assert len(result) == 6
        graph_of = batch.batch[result.node_id]
        assert (graph_of == 0).sum() == 3
        assert (graph_of == 1).sum() == 3

    def test_quota_by_entity(self):
        node_type = np.array([0] * 10 + [1] * 10 + [2] * 20)
        self.data.node_type = torch.from_numpy(node_type)

        sampler = BudgetSampler(budget=8, quota_by='entity', seed=0)
        result = sampler.sample(self.data, self.score, emb=self.emb)

        # sizes 10, 10, 20 -> quotas 2, 2, 4
        assert result.quota == {0: 2, 1: 2, 2: 4}
        assert len(result) == 8
        for t, q in zip([0, 1, 2], [2, 2, 4]):
            assert (node_type[result.node_id.numpy()] == t).sum() == q

        # explicit group argument overrides quota_by
        sampler2 = BudgetSampler(budget=4, quota_by='none', seed=0)
        result2 = sampler2.sample(self.data, self.score, emb=self.emb,
                                  group=node_type)
        assert len(result2.quota) == 3

    def test_fallback_without_emb(self):
        sampler = BudgetSampler(budget=6, quota_by='none', seed=42)
        with assert_warns(UserWarning):
            result = sampler.sample(self.data, self.score, emb=None)

        assert result.method == 'fallback'
        assert (result.cluster_id == -1).all()
        # fallback selects the unique highest-score nodes of the group
        assert set(result.node_id.tolist()) == {14, 15, 16, 17, 18, 19}

    def test_subgraph_targets_and_context(self):
        sampler = BudgetSampler(budget=5, quota_by='none', num_hops=1,
                                seed=42)
        result = sampler.sample(self.data, self.score, emb=self.emb)
        sub = result.subgraph

        # targets are marked separately from context-only neighbors
        assert int(sub.is_target.sum()) == 5
        assert torch.equal(sub.n_id[sub.is_target].sort().values,
                           result.node_id)
        assert sub.num_nodes > 5  # context neighbors are included
        # every context node is a 1-hop neighbor of some target
        edge_index = self.data.edge_index
        for nid in sub.n_id[~sub.is_target].tolist():
            is_neigh = ((edge_index[0] == nid).any(dim=0) &
                        torch.isin(edge_index[1], result.node_id)).any()
            assert is_neigh
        # original scores are carried over for every subgraph node
        assert np.allclose(sub.score.numpy(), self.score[sub.n_id.numpy()])
        # subgraph is a valid relabeled graph
        assert int(sub.edge_index.max()) < sub.num_nodes
        assert sub.x.shape[0] == sub.num_nodes
        # node identity is stable: n_id maps back to the input graph
        assert torch.equal(sub.x, self.data.x[sub.n_id])

    def test_num_hops_zero(self):
        sampler = BudgetSampler(budget=5, quota_by='none', num_hops=0,
                                seed=42)
        result = sampler.sample(self.data, self.score, emb=self.emb)
        assert result.subgraph.num_nodes == 5
        assert bool(result.subgraph.is_target.all())

    def test_determinism(self):
        sampler = BudgetSampler(budget=6, quota_by='none', seed=7)
        r1 = sampler.sample(self.data, self.score, emb=self.emb)
        r2 = sampler.sample(self.data, self.score, emb=self.emb)
        assert torch.equal(r1.node_id, r2.node_id)
        assert torch.equal(r1.subgraph.n_id, r2.subgraph.n_id)

        sampler2 = BudgetSampler(budget=6, quota_by='none', seed=7)
        r3 = sampler2.sample(self.data, self.score, emb=self.emb)
        assert torch.equal(r1.node_id, r3.node_id)

    def test_no_duplicate_padding(self):
        # duplicated candidates are deduplicated, never used as padding
        sampler = BudgetSampler(budget=10, quota_by='none', seed=0)
        result = sampler.sample(self.data, self.score, emb=self.emb,
                                candidates=[0, 0, 1, 1, 2])
        assert len(result) == 3
        assert set(result.node_id.tolist()) == {0, 1, 2}

        # budget larger than the node set returns every node exactly once
        sampler2 = BudgetSampler(budget=100, quota_by='none', seed=0)
        result2 = sampler2.sample(self.data, self.score, emb=self.emb)
        assert len(result2) == self.n
        assert len(set(result2.node_id.tolist())) == self.n

    def test_identity_and_scores(self):
        sampler = BudgetSampler(budget=7, quota_by='none', seed=1)
        result = sampler.sample(self.data, self.score, emb=self.emb)
        node_id = result.node_id.numpy()
        # stable identity: sorted original node ids
        assert (np.diff(node_id) > 0).all()
        # original scores are returned unchanged
        assert np.allclose(result.score.numpy(), self.score[node_id])
        # quota keys are plain python scalars
        for k in result.quota:
            assert isinstance(k, int)

    def test_validation(self):
        with assert_raises(ValueError):
            BudgetSampler(budget=0)
        with assert_raises(ValueError):
            BudgetSampler(budget=2.5)
        with assert_raises(ValueError):
            BudgetSampler(budget=5, quota_by='bad')
        with assert_raises(ValueError):
            BudgetSampler(budget=5, num_hops=-1)

        sampler = BudgetSampler(budget=5, quota_by='none')
        # score length mismatch
        with assert_raises(ValueError):
            sampler.sample(self.data, self.score[:10], emb=self.emb)
        # non-finite scores
        bad_score = self.score.copy()
        bad_score[0] = np.inf
        with assert_raises(ValueError):
            sampler.sample(self.data, bad_score, emb=self.emb)
        # embedding shape mismatch
        with assert_raises(ValueError):
            sampler.sample(self.data, self.score, emb=self.emb[:10])
        # entity quota without group information
        sampler_e = BudgetSampler(budget=5, quota_by='entity')
        with assert_raises(ValueError):
            sampler_e.sample(self.data, self.score, emb=self.emb)
        # candidates out of range
        with assert_raises(ValueError):
            sampler.sample(self.data, self.score, emb=self.emb,
                           candidates=[0, self.n])
        # empty candidates
        with assert_raises(ValueError):
            sampler.sample(self.data, self.score, emb=self.emb,
                           candidates=[])

    def test_tuple_emb(self):
        sampler = BudgetSampler(budget=6, quota_by='none', seed=42)
        result = sampler.sample(self.data, self.score,
                                emb=(self.emb[:, :2], self.emb[:, 2:]))
        assert len(result) == 6

    def test_with_detector(self):
        data = torch.load('pygod/test/test_graph.pt', weights_only=False)
        detector = DOMINANT(epoch=2, save_emb=True)
        detector.fit(data)

        sampler = BudgetSampler(budget=3, quota_by='graph', seed=0)
        result = sampler.sample(data, detector.decision_score_,
                                emb=detector.emb)
        assert len(result) == 3
        assert result.node_id.max() < data.num_nodes
        assert np.allclose(result.score.numpy(),
                           detector.decision_score_[result.node_id].numpy())
        assert int(result.subgraph.is_target.sum()) == 3


if __name__ == '__main__':
    unittest.main()
