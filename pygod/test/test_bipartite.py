# -*- coding: utf-8 -*-
"""Tests for the bipartite one-mode projection adapter."""
import os
import unittest

import numpy as np
import scipy.sparse as sp
import torch
from numpy.testing import assert_raises, assert_warns, assert_allclose
from torch_geometric.seed import seed_everything

from pygod.utils import (
    BipartiteAdapter,
    BipartiteProjection,
    BipartiteVocabulary,
    BudgetExceededError,
    PROJECTION_VERSION,
)
from pygod.detector import DOMINANT, SCAN

seed_everything(717)


def toy_edges():
    # users U0..U4, items I0..I3. Item I0 is deliberately very popular.
    # U0 and U1 share I0, I1 -> must be linked.
    # String ids prove the two sides never collide (users also have an
    # item sharing the same raw value below via numeric ids test).
    return [
        ("U0", "I0"), ("U0", "I1"),
        ("U1", "I0"), ("U1", "I1"), ("U1", "I2"),
        ("U2", "I0"), ("U2", "I2"),
        ("U3", "I3"),
        # U4 is isolated in this relation
    ], ["U0", "U1", "U2", "U3", "U4"], ["I0", "I1", "I2", "I3"]


class TestVocabulary(unittest.TestCase):
    def test_separate_vocabularies_same_raw_id(self):
        # Raw id 7 exists on both sides but must mean different entities.
        vocab = BipartiteVocabulary(left_ids=[7, 8], right_ids=[7, 9])
        assert vocab.to_index("left", [7])[0] == 0
        assert vocab.to_index("right", [7])[0] == 0
        assert vocab.to_index("left", [8])[0] == 1
        assert vocab.to_index("right", [9])[0] == 1
        assert vocab.left_size == 2 and vocab.right_size == 2

    def test_duplicate_and_unknown(self):
        with assert_raises(ValueError):
            BipartiteVocabulary([1, 1], [2])
        vocab = BipartiteVocabulary([1, 2], [3, 4])
        with assert_raises(KeyError):
            vocab.to_index("left", [9])
        with assert_raises(ValueError):
            vocab.to_index("middle", [1])


class TestProjectionRules(unittest.TestCase):
    def setUp(self):
        self.edges, self.users, self.items = toy_edges()

    def test_basic_count_projection(self):
        adapter = BipartiteAdapter(target_side="left")
        proj = adapter.fit_transform(
            self.edges, left_ids=self.users, right_ids=self.items)

        self.assertIsInstance(proj, BipartiteProjection)
        # One node per target entity, isolated entities retained.
        assert proj.num_nodes == 5
        assert proj.x.shape[0] == 5  # default constant features
        assert proj.edge_index.shape[0] == 2
        # Undirected: every edge stored both ways, no self loops.
        ei = proj.edge_index.numpy()
        assert (ei[0] != ei[1]).all()

        # U0-U1 share two items -> weight 2.
        u0, u1 = proj.to_index("U0"), proj.to_index("U1")
        u2 = proj.to_index("U2")
        adj = proj.adjacency()
        assert adj[u0, u1] == 2
        assert adj[u1, u2] == 2  # U1/U2 share I0 and I2
        # U3 shares nothing with the others.
        u3 = proj.to_index("U3")
        assert adj[u3].nnz == 0
        # Isolated U4 retained as a node.
        assert proj.to_index("U4") == 4

    def test_min_shared_neighbors_rule(self):
        adapter = BipartiteAdapter(target_side="left",
                                   min_shared_neighbors=2)
        proj = adapter.fit_transform(
            self.edges, left_ids=self.users, right_ids=self.items)
        # Only pairs with >=2 shared neighbors survive.
        assert proj.num_projected_edges == 2

    def test_degree_correction_dampens_popular_item(self):
        plain = BipartiteAdapter(weighting="count").fit_transform(
            self.edges, left_ids=self.users, right_ids=self.items)
        corrected = BipartiteAdapter(
            weighting="degree").fit_transform(
            self.edges, left_ids=self.users, right_ids=self.items)

        # U1-U2 share I0 (degree 3) and I2 (degree 2):
        # corrected weight = 1/3 + 1/2
        u1, u2 = plain.to_index("U1"), plain.to_index("U2")
        assert_allclose(corrected.adjacency()[u1, u2], 1 / 3 + 1 / 2)
        # Degree correction must reduce the influence of shared popular
        # items relative to raw counts.
        assert corrected.adjacency()[u1, u2] < plain.adjacency()[u1, u2]

    def test_cosine_weighting(self):
        proj = BipartiteAdapter(weighting="cosine").fit_transform(
            self.edges, left_ids=self.users, right_ids=self.items)
        u0, u1 = proj.to_index("U0"), proj.to_index("U1")
        # degrees: U0=2, U1=3, shared=2
        assert_allclose(proj.adjacency()[u0, u1],
                        2 / np.sqrt(2 * 3))

    def test_right_side_projection(self):
        adapter = BipartiteAdapter(target_side="right")
        proj = adapter.fit_transform(
            self.edges, left_ids=self.users, right_ids=self.items)
        assert proj.num_nodes == 4
        assert proj.target_side == "right"
        assert set(proj.entity_ids) == set(self.items)
        # Item co-occurrences through shared users: I0-I1, I0-I2, I1-I2
        assert proj.num_projected_edges == 3
        adj = proj.adjacency()
        i0, i2 = proj.to_index("I0"), proj.to_index("I2")
        assert adj[i0, i2] == 2  # both U1 and U2 bought I0 and I2

    def test_sparse_matrix_input_and_orientation(self):
        vocab = BipartiteVocabulary(self.users, self.items)
        rows = vocab.to_index("left", [e[0] for e in self.edges])
        cols = vocab.to_index("right", [e[1] for e in self.edges])
        incidence = sp.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(5, 4))
        # Sparse input is already index space; its shape defines both
        # universes, so the isolated target node is retained.
        proj = BipartiteAdapter().fit_transform(incidence)
        assert proj.num_nodes == 5
        assert proj.edge_index.shape[1] > 0

    def test_torch_tensor_input(self):
        edges = np.array(self.edges, dtype=object)
        # Universe inferred from edges excludes the isolated U4 unless
        # explicitly declared.
        proj = BipartiteAdapter().fit_transform(edges.T)
        assert isinstance(proj, BipartiteProjection)
        assert proj.num_nodes == 4

        proj_full = BipartiteAdapter().fit_transform(
            edges.T, left_ids=self.users, right_ids=self.items)
        assert proj_full.num_nodes == 5


class TestBudget(unittest.TestCase):
    def setUp(self):
        self.edges, self.users, self.items = toy_edges()

    def test_max_edges_raise(self):
        with assert_raises(BudgetExceededError):
            BipartiteAdapter(max_edges=1, on_budget="raise").fit_transform(
                self.edges, left_ids=self.users, right_ids=self.items)

    def test_max_edges_truncate(self):
        proj = BipartiteAdapter(
            max_edges=1, on_budget="truncate").fit_transform(
            self.edges, left_ids=self.users, right_ids=self.items)
        assert proj.num_projected_edges == 1
        # The retained edge is the highest weighted one (weight 2).
        assert proj.edge_weight.max() == 2

    def test_max_neighbors_cap(self):
        proj = BipartiteAdapter(max_neighbors=1).fit_transform(
            self.edges, left_ids=self.users, right_ids=self.items)
        adj = proj.adjacency()
        degrees = np.diff(adj.indptr)
        # Undirected consistency: no node can keep more neighbors than
        # the cap plus edges it keeps because the opposite endpoint
        # ranked it first. With cap=1 the maximum degree is 2 (a node
        # mutually retained by two distinct neighbors).
        assert degrees.max() <= 2
        # Every surviving edge must be the top choice of at least one
        # of its endpoints; the low-weight U0-U2 edge (weight 1) is
        # pruned while both weight-2 edges survive.
        u0, u1, u2 = (proj.to_index("U0"), proj.to_index("U1"),
                      proj.to_index("U2"))
        assert adj[u0, u2] == 0
        assert adj[u0, u1] == 2 and adj[u1, u2] == 2
        assert proj.num_projected_edges == 2

    def test_popular_item_never_densifies(self):
        # One item connected to 2000 users: naively this creates a
        # 2000x2000 dense co-occurrence block; the adapter must stay
        # sparse and bounded.
        rng = np.random.RandomState(0)
        edges = [("U%d" % i, "POP") for i in range(2000)]
        edges += [("U%d" % i, "I%d" % j)
                  for i, j in zip(rng.randint(0, 2000, 400),
                                  rng.randint(0, 50, 400))]
        with assert_raises(BudgetExceededError):
            BipartiteAdapter(
                max_edges=100, on_budget="raise").fit_transform(edges)

        proj = BipartiteAdapter(
            max_edges=100, on_budget="truncate").fit_transform(edges)
        assert proj.num_projected_edges <= 100
        # No dense member ever materialized on the object.
        assert sp.issparse(proj.adjacency())

    def test_popular_item_bounded_memory_all_weightings(self):
        # A viral item alone produces C(n,2) candidate pairs. For every
        # weighting scheme the truncation path must keep both the edge
        # count and the working set bounded by the declared budget.
        rng = np.random.RandomState(2)
        n = 3000
        edges = [("U%d" % i, "POP") for i in range(n)]
        edges += [("U%d" % i, "I%d" % j)
                  for i, j in zip(rng.randint(0, n, 100),
                                  rng.randint(0, 20, 100))]
        for weighting in ("count", "degree", "cosine"):
            proj = BipartiteAdapter(
                weighting=weighting,
                max_edges=64,
                on_budget="truncate").fit_transform(edges)
            assert proj.num_projected_edges <= 64
            # Features never blow up into an N x N dense matrix.
            assert proj.x.shape == (n, 1)

    def test_invalid_arguments(self):
        with assert_raises(ValueError):
            BipartiteAdapter(weighting="bogus")
        with assert_raises(ValueError):
            BipartiteAdapter(min_shared_neighbors=0)
        with assert_raises(ValueError):
            BipartiteAdapter(on_budget="ignore")
        with assert_raises(ValueError):
            BipartiteAdapter(target_side="top")

    def test_weighted_bipartite_edges(self):
        edges = np.array(
            [("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")],
            dtype=object).T
        weights = np.array([2.0, 1.0, 3.0, 1.0])
        proj = BipartiteAdapter(weighting="count").fit_transform(
            edges, edge_weight=weights)
        # shared x: 2*3, shared y: 1*1 -> 7
        assert_allclose(proj.adjacency()[0, 1], 7.0)

    def test_self_loops_semantics(self):
        edges = np.array(
            [("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")],
            dtype=object).T
        with_loops = BipartiteAdapter(add_self_loops=True).fit_transform(edges)
        diag = with_loops.adjacency().diagonal()
        # each entity has degree 2 -> count self-loop weight = 2
        assert_allclose(diag, [2.0, 2.0])
        # self loops counted once, not duplicated by symmetrization
        assert with_loops.num_projected_edges == 3

        degree_loops = BipartiteAdapter(
            weighting="degree", add_self_loops=True).fit_transform(edges)
        # 1/d(x) + 1/d(y) = 1/2 + 1/2 = 1
        assert_allclose(degree_loops.adjacency().diagonal(), [1.0, 1.0])

        cosine_loops = BipartiteAdapter(
            weighting="cosine", add_self_loops=True).fit_transform(edges)
        # cosine self-similarity is exactly 1
        assert_allclose(cosine_loops.adjacency().diagonal(), [1.0, 1.0])

        without = BipartiteAdapter(add_self_loops=False).fit_transform(edges)
        assert_allclose(without.adjacency().diagonal(), [0.0, 0.0])


class TestAttributes(unittest.TestCase):
    def setUp(self):
        self.edges, self.users, self.items = toy_edges()

    def test_dict_join_by_target_id_only(self):
        attrs = {"U0": [1.0, 0.0], "U1": [0.0, 1.0],
                 "U2": [1.0, 1.0], "U3": [0.5, 0.5]}
        proj = BipartiteAdapter().fit_transform(
            self.edges, left_ids=self.users,
            right_ids=self.items, target_attributes=attrs)
        # U4 missing -> zero row with warning.
        with assert_warns(UserWarning):
            BipartiteAdapter(strict_attributes=False).fit_transform(
                self.edges, left_ids=self.users,
                right_ids=self.items, target_attributes=attrs)
        u0 = proj.to_index("U0")
        assert_allclose(proj.x[u0].numpy(), [1.0, 0.0])
        assert (proj.x[proj.to_index("U4")] == 0).all()

    def test_strict_missing_rejected(self):
        attrs = {"U0": [1.0]}
        with assert_raises(KeyError):
            BipartiteAdapter(strict_attributes=True).fit_transform(
                self.edges, left_ids=self.users,
                right_ids=self.items, target_attributes=attrs)

    def test_aligned_array_and_pair_input(self):
        x = np.arange(10, dtype=float).reshape(5, 2)
        proj = BipartiteAdapter().fit_transform(
            self.edges, left_ids=self.users,
            right_ids=self.items, target_attributes=x)
        assert_allclose(proj.x.numpy(), x)

        pair = (self.users, x)
        proj2 = BipartiteAdapter().fit_transform(
            self.edges, left_ids=self.users,
            right_ids=self.items, target_attributes=pair)
        assert_allclose(proj2.x.numpy(), x)

    def test_unknown_attribute_id_rejected(self):
        attrs = {"U0": [1.0], "GHOST": [2.0]}
        with assert_raises(KeyError):
            BipartiteAdapter().fit_transform(
                self.edges, left_ids=self.users,
                right_ids=self.items, target_attributes=attrs)

    def test_source_attributes_are_not_used(self):
        # A feature table keyed by item ids (source side) must not leak
        # into the user (target) projection; only target ids are valid.
        item_attrs = {"I0": [9.0], "I1": [8.0]}
        with assert_raises(KeyError):
            BipartiteAdapter(target_side="left").fit_transform(
                self.edges, left_ids=self.users,
                right_ids=self.items, target_attributes=item_attrs)


class TestEdgeCases(unittest.TestCase):
    def test_empty_relation_keeps_universe(self):
        proj = BipartiteAdapter().fit_transform(
            np.empty((2, 0), dtype=object),
            left_ids=["a", "b"], right_ids=["x"])
        assert proj.num_nodes == 2
        assert proj.num_projected_edges == 0
        assert proj.edge_index.shape == (2, 0)
        assert proj.x.shape == (2, 1)

    def test_single_entity(self):
        proj = BipartiteAdapter().fit_transform(
            [("a", "x")], left_ids=["a"], right_ids=["x"])
        assert proj.num_nodes == 1
        assert proj.edge_index.shape == (2, 0)

    def test_duplicate_bipartite_edges_deduplicated(self):
        proj = BipartiteAdapter().fit_transform(
            [("a", "x"), ("a", "x"), ("b", "x")])
        # Binary incidence: a and b share exactly one distinct neighbor.
        assert proj.adjacency()[0, 1] == 1.0

    def test_rule_filters_everything(self):
        proj = BipartiteAdapter(
            min_shared_neighbors=5,
            max_edges=1, on_budget="truncate").fit_transform(
            [("a", "x"), ("b", "x")])
        assert proj.num_projected_edges == 0

    def test_unknown_projection_id(self):
        proj = BipartiteAdapter().fit_transform([("a", "x"), ("b", "x")])
        with assert_raises(KeyError):
            proj.to_index("ghost")


class TestIdentityRetention(unittest.TestCase):
    def test_version_and_scores_by_id(self):
        edges, users, items = toy_edges()
        proj = BipartiteAdapter().fit_transform(
            edges, left_ids=users, right_ids=items)
        assert proj.projection_version == PROJECTION_VERSION
        assert "weighting" in proj.projection_config

        raw = proj.entity_ids
        assert raw[proj.to_index("U1")] == "U1"
        assert proj.to_raw_id(0) == "U0"

        score = torch.arange(proj.num_nodes, dtype=torch.float)
        scored = proj.scores_by_id(score)
        assert set(scored.keys()) == set(users)
        assert scored["U1"] == float(proj.to_index("U1"))

        reordered = proj.align_scores(["U2", "U0"], score)
        assert reordered.tolist() == [float(proj.to_index("U2")),
                                      float(proj.to_index("U0"))]


class TestDetectorIntegration(unittest.TestCase):
    def test_projection_enters_gcn_detector(self):
        # Every user shares at least one item so the projection has no
        # isolated nodes: deep detectors that derive the dense adjacency
        # from edge_index alone (DOMINANT) require a connected node set.
        edges = [
            ("U0", "I0"), ("U0", "I1"),
            ("U1", "I0"), ("U1", "I1"), ("U1", "I2"),
            ("U2", "I0"), ("U2", "I2"),
            ("U3", "I2"), ("U3", "I3"),
            ("U4", "I3"),
        ]
        users = ["U0", "U1", "U2", "U3", "U4"]
        items = ["I0", "I1", "I2", "I3"]
        attrs = {u: np.random.RandomState(i).randn(4)
                 for i, u in enumerate(users)}
        proj = BipartiteAdapter(
            max_edges=100, on_budget="truncate").fit_transform(
            edges, left_ids=users, right_ids=items,
            target_attributes=attrs)
        assert proj.adjacency().diagonal().sum() == 0

        detector = DOMINANT(epoch=3, num_layers=2, hid_dim=8)
        detector.fit(proj)
        pred, score = detector.predict(return_pred=True,
                                       return_score=True)
        # One score per original target entity.
        assert score.shape[0] == proj.num_nodes == len(users)
        assert pred.shape[0] == len(users)
        # Scores can be mapped back to original identities.
        scored = proj.scores_by_id(score)
        assert set(scored.keys()) == set(users)

    def test_projection_enters_structure_detector_scan(self):
        edges, users, items = toy_edges()
        proj = BipartiteAdapter().fit_transform(
            edges, left_ids=users, right_ids=items)
        detector = SCAN(eps=0.2, mu=2)
        detector.fit(proj)
        score = detector.predict(return_pred=False, return_score=True)
        assert score.shape[0] == proj.num_nodes

    def test_loaded_graph_pipeline_end_to_end(self):
        # Real PyG graph: treat its edges as an arbitrary bipartite
        # relation between even/odd indexed node ids, which overlap on
        # purpose (both sides use ids 0..5) to prove no id collision.
        data = torch.load(os.path.join(
            "pygod/test/train_graph.pt"), weights_only=False)
        ei = data.edge_index.numpy()
        left = ei[0]
        right = ei[1]
        ids = np.arange(data.num_nodes)

        proj = BipartiteAdapter(
            target_side="left",
            max_edges=50,
            on_budget="truncate").fit_transform(
            np.vstack([left, right]), left_ids=ids, right_ids=ids)

        assert proj.num_nodes == data.num_nodes
        # SCAN scores pure structure and tolerates the isolated nodes
        # that edge-budget truncation can leave behind; DOMINANT is
        # covered separately on a fully connected projection.
        detector = SCAN(eps=0.2, mu=2)
        detector.fit(proj)
        score = detector.predict(return_pred=False, return_score=True)
        assert score.shape[0] == data.num_nodes
        assert proj.projection_version == PROJECTION_VERSION


if __name__ == "__main__":
    unittest.main()
