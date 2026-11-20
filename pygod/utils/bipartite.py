# -*- coding: utf-8 -*-
"""Bipartite (two-mode) graph adapter for one-mode entity projection.

A user-item bipartite graph must not be collapsed into a single node
universe: a user id and an item id that happen to share the same raw
value are different entities. This module therefore keeps **two
independent vocabularies** (one per side) and builds a sparse one-mode
projection on a declared target side by the *shared-neighbor rule*:
two target entities are connected when they share at least
``min_shared_neighbors`` neighbors on the opposite side.

The projection is always constructed as a real sparse graph
(``torch_geometric.data.Data`` with ``edge_index``/``edge_weight`` and
one node per target entity), never as a dense similarity table. All
intermediate algebra uses ``scipy.sparse`` matrices, so a single
popular item with a very high degree cannot materialize an unbounded
dense matrix, and explicit edge budgets are either enforced by
truncation or rejected with :class:`BudgetExceededError`.

Node attributes are joined strictly from the target-side entity table
by raw entity id; the source-side table is never used as a source of
target features.
"""
# Author: PyGOD contributors
# License: BSD 2 clause

import warnings

import numpy as np
import scipy.sparse as sp
import torch

from torch_geometric.data import Data

# Bumped whenever the on-disk/in-memory projection format changes.
PROJECTION_VERSION = "bipartite-one-mode-projection-v1"

_VALID_WEIGHTING = ("count", "degree", "cosine")
_VALID_ON_BUDGET = ("raise", "truncate")
_VALID_DEFAULT_FEATURES = ("ones", "identity")


class BudgetExceededError(ValueError):
    """Raised when a projected graph exceeds the declared edge budget
    and ``on_budget='raise'`` is configured."""

    def __init__(self, num_edges, max_edges):
        self.num_edges = num_edges
        self.max_edges = max_edges
        super().__init__(
            "Bipartite projection produces %d undirected edges, which "
            "exceeds the declared budget max_edges=%d. Raise the budget "
            "or pass on_budget='truncate' to keep the highest-weighted "
            "candidates." % (num_edges, max_edges))


class BipartiteVocabulary:
    """Two independent id-to-index vocabularies for a bipartite graph.

    The left and right side are mapped into separate index spaces, so
    identical raw ids on the two sides never collide into one node.

    Parameters
    ----------
    left_ids : array-like
        All raw entity ids on the left side, in canonical order. Local
        index positions follow this order.
    right_ids : array-like
        All raw entity ids on the right side, in canonical order.
    """

    def __init__(self, left_ids, right_ids):
        self.left_ids = np.asarray(list(left_ids), dtype=object)
        self.right_ids = np.asarray(list(right_ids), dtype=object)

        if len(np.unique(self.left_ids)) != len(self.left_ids):
            raise ValueError("left_ids contain duplicate raw ids.")
        if len(np.unique(self.right_ids)) != len(self.right_ids):
            raise ValueError("right_ids contain duplicate raw ids.")

        self._left_to_index = {
            raw: i for i, raw in enumerate(self.left_ids)}
        self._right_to_index = {
            raw: i for i, raw in enumerate(self.right_ids)}

    @property
    def left_size(self):
        return self.left_ids.shape[0]

    @property
    def right_size(self):
        return self.right_ids.shape[0]

    def to_index(self, side, raw_ids):
        """Map raw ids of one side to local integer indices.

        Parameters
        ----------
        side : {'left', 'right'}
            Which vocabulary to use.
        raw_ids : array-like
            Raw entity ids. The same raw value on the opposite side
            resolves to a different index.

        Returns
        -------
        indices : numpy.ndarray of int
        """
        if side == "left":
            mapping = self._left_to_index
            side_name = "left"
        elif side == "right":
            mapping = self._right_to_index
            side_name = "right"
        else:
            raise ValueError("side must be 'left' or 'right', got %r" % side)

        raw_ids = np.asarray(list(raw_ids), dtype=object)
        try:
            return np.array([mapping[v] for v in raw_ids], dtype=np.int64)
        except KeyError as e:
            raise KeyError(
                "Raw id %s is not present in the %s-side vocabulary."
                % (e.args[0], side_name))

    def to_raw_id(self, side, indices):
        """Inverse of :meth:`to_index`."""
        indices = np.asarray(indices, dtype=np.int64)
        if side == "left":
            return self.left_ids[indices]
        elif side == "right":
            return self.right_ids[indices]
        raise ValueError("side must be 'left' or 'right', got %r" % side)

    def ids(self, side):
        if side == "left":
            return self.left_ids
        if side == "right":
            return self.right_ids
        raise ValueError("side must be 'left' or 'right', got %r" % side)

    def __repr__(self):
        return "BipartiteVocabulary(left=%d, right=%d)" % (
            self.left_size, self.right_size)


class BipartiteProjection(Data):
    """A sparse one-mode projection of a bipartite graph.

    This is a regular :class:`torch_geometric.data.Data` graph (one node
    per target-side entity, ``edge_index`` with local target indices and
    optional ``edge_weight``) and can be passed directly to any PyGOD
    detector that consumes sparse graph structure. Original entity
    identities and the projection version are retained alongside the
    local indices.

    Properties stored under underscore-prefixed names are kept out of
    PyG's tensor store so that data loaders do not try to batch them.
    """

    def __init__(self,
                 edge_index,
                 edge_weight,
                 x,
                 entity_ids,
                 target_side,
                 config,
                 num_nodes):
        # num_nodes is always set explicitly so downstream code that
        # derives adjacency from edge_index alone (e.g. to_dense_adj)
        # cannot shrink the graph when isolated entities are present.
        if x.shape[0] != num_nodes or len(entity_ids) != num_nodes:
            raise ValueError(
                "Inconsistent projection size: num_nodes=%d but got "
                "%d feature rows and %d entity ids."
                % (num_nodes, x.shape[0], len(entity_ids)))
        super().__init__(x=x,
                         edge_index=edge_index,
                         edge_weight=edge_weight,
                         num_nodes=num_nodes)
        # Private attributes: ignored by PyG's data store / loaders.
        self._entity_ids = list(entity_ids)
        self._id_to_index = {
            raw: i for i, raw in enumerate(self._entity_ids)}
        self._target_side = target_side
        self._projection_version = PROJECTION_VERSION
        self._projection_config = dict(config)

    @property
    def entity_ids(self):
        """Raw target-side entity id at each local node index."""
        return list(self._entity_ids)

    @property
    def target_side(self):
        return self._target_side

    @property
    def projection_version(self):
        return self._projection_version

    @property
    def projection_config(self):
        return dict(self._projection_config)

    @property
    def num_projected_edges(self):
        """Number of unique undirected projected edges (self loops,
        when present, count once)."""
        ei = self.edge_index
        if ei.shape[1] == 0:
            return 0
        is_loop = ei[0] == ei[1]
        return int((ei.shape[1] - int(is_loop.sum())) // 2
                   + int(is_loop.sum()))

    def to_index(self, raw_id):
        """Return the local projected-graph index of a raw entity id."""
        try:
            return self._id_to_index[raw_id]
        except KeyError:
            raise KeyError(
                "Raw entity id %r is not a node of the projected graph."
                % (raw_id,))

    def to_raw_id(self, index):
        """Return the raw entity id behind a local projected-graph index."""
        index = int(index)
        if index < 0 or index >= self.num_nodes:
            raise IndexError(index)
        return self._entity_ids[index]

    def scores_by_id(self, score):
        """Attach detector scores back to the original entity identities.

        Parameters
        ----------
        score : array-like, shape [num_nodes]
            Per-node detector output in local projected-graph order.

        Returns
        -------
        scored : dict
            Mapping ``raw_entity_id -> outlier_score``.
        """
        score = np.asarray(torch.as_tensor(score).detach().cpu()).ravel()
        if score.shape[0] != self.num_nodes:
            raise ValueError(
                "Expected %d scores (one per projected entity), got %d."
                % (self.num_nodes, score.shape[0]))
        return {raw: float(s) for raw, s in zip(self._entity_ids, score)}

    def align_scores(self, raw_ids, score):
        """Reorder a detector score vector onto a requested id order."""
        score = torch.as_tensor(score).detach().cpu()
        idx = np.array([self._id_to_index[v] for v in raw_ids],
                       dtype=np.int64)
        return score[torch.as_tensor(idx)]

    def adjacency(self):
        """Return the projected graph as a weighted ``scipy.sparse.csr``
        matrix (no densification is performed anywhere in the adapter)."""
        ei = self.edge_index.detach().cpu().numpy()
        ew = (self.edge_weight.detach().cpu().numpy()
              if self.edge_weight is not None
              else np.ones(ei.shape[1]))
        return sp.csr_matrix(
            (ew, (ei[0], ei[1])),
            shape=(self.num_nodes, self.num_nodes)).tocsr()

    def __repr__(self):
        return ("BipartiteProjection(version=%r, target_side=%r, "
                "num_nodes=%d, num_edges=%d)"
                % (self._projection_version, self._target_side,
                   self.num_nodes, self.num_projected_edges))


class BipartiteAdapter:
    """Build sparse one-mode entity graphs from a bipartite relation.

    The two sides of the relation keep separate vocabularies, so raw
    ids are never merged across sides. A target side is projected into
    a sparse entity graph under a declared shared-neighbor rule, with
    optional source-degree (e.g. item-degree) weight correction,
    per-node candidate caps and explicit global edge budgets.

    Parameters
    ----------
    target_side : {'left', 'right'}, optional
        The side whose entities become projected-graph nodes. Shared
        neighbors are taken from the opposite side. Default: ``'left'``.
    weighting : {'count', 'degree', 'cosine'}, optional
        Edge weighting scheme of the projection. Default: ``'count'``.

        - ``'count'``: weight is the number of shared neighbors.
        - ``'degree'``: item/source-degree-corrected weight,
          :math:`\\sum_{j \\in N(u) \\cap N(v)} 1/d(j)`, where
          :math:`d(j)` is the bipartite degree of the shared
          source-side entity. Popular items contribute less per
          co-occurrence.
        - ``'cosine'``: shared count normalized by target degrees,
          :math:`|N(u) \\cap N(v)| / \\sqrt{d(u)d(v)}`.
    min_shared_neighbors : int, optional
        Declared shared-neighbor rule: a projected edge is emitted only
        when the two target entities share at least this many opposite
        side neighbors. Default: ``1``.
    max_neighbors : int or None, optional
        Per-node candidate cap. When set, every target node keeps at
        most this many projected neighbors, retaining the highest
        weights. Applied on sparse arrays only. Default: ``None``.
    max_edges : int or None, optional
        Global budget on the number of unique undirected projected
        edges. When the projection exceeds it, ``on_budget`` decides
        whether to truncate or raise. Default: ``None`` (no cap).
    on_budget : {'raise', 'truncate'}, optional
        Behavior when ``max_edges`` is exceeded: ``'raise'`` throws
        :class:`BudgetExceededError`, ``'truncate'`` keeps the
        ``max_edges`` highest-weighted edges. Default: ``'raise'``.
    add_self_loops : bool, optional
        Whether to keep self loops in the projection. Default: ``False``.
    default_features : {'ones', 'identity'}, optional
        Feature matrix synthesized when no target attribute table is
        supplied. ``'ones'`` gives a single constant feature (structure
        only, stays tiny for large graphs); ``'identity'`` gives a
        one-hot matrix and is only suitable for small graphs.
        Default: ``'ones'``.
    strict_attributes : bool, optional
        If ``True``, a target entity missing from the attribute table
        raises ``KeyError``. Otherwise a zero feature row is filled and
        a warning is emitted. Default: ``False``.

    Attributes
    ----------
    vocabulary_ : BipartiteVocabulary
        Fitted separate vocabularies for the two relation sides.
    projection_ : BipartiteProjection
        The projected graph produced by the last ``fit_transform`` call.
    """

    def __init__(self,
                 target_side="left",
                 weighting="count",
                 min_shared_neighbors=1,
                 max_neighbors=None,
                 max_edges=None,
                 on_budget="raise",
                 add_self_loops=False,
                 default_features="ones",
                 strict_attributes=False):

        if target_side not in ("left", "right"):
            raise ValueError(
                "target_side must be 'left' or 'right', got %r" % target_side)
        if weighting not in _VALID_WEIGHTING:
            raise ValueError("weighting must be one of %s, got %r"
                             % (_VALID_WEIGHTING, weighting))
        if not isinstance(min_shared_neighbors, (int, np.integer)) or \
                min_shared_neighbors < 1:
            raise ValueError("min_shared_neighbors must be a positive int, "
                             "got %r" % min_shared_neighbors)
        if max_neighbors is not None and (
                not isinstance(max_neighbors, (int, np.integer))
                or max_neighbors < 1):
            raise ValueError("max_neighbors must be a positive int or None, "
                             "got %r" % max_neighbors)
        if max_edges is not None and (
                not isinstance(max_edges, (int, np.integer))
                or max_edges < 1):
            raise ValueError("max_edges must be a positive int or None, "
                             "got %r" % max_edges)
        if on_budget not in _VALID_ON_BUDGET:
            raise ValueError("on_budget must be one of %s, got %r"
                             % (_VALID_ON_BUDGET, on_budget))
        if default_features not in _VALID_DEFAULT_FEATURES:
            raise ValueError(
                "default_features must be one of %s, got %r"
                % (_VALID_DEFAULT_FEATURES, default_features))

        self.target_side = target_side
        self.weighting = weighting
        self.min_shared_neighbors = int(min_shared_neighbors)
        self.max_neighbors = (None if max_neighbors is None
                              else int(max_neighbors))
        self.max_edges = None if max_edges is None else int(max_edges)
        self.on_budget = on_budget
        self.add_self_loops = add_self_loops
        self.default_features = default_features
        self.strict_attributes = strict_attributes

        self.vocabulary_ = None
        self.projection_ = None

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_edges(edge_index, edge_weight):
        """Return ``(left_raw, right_raw, weight, shape)``.

        ``shape`` is non-None only for sparse-matrix input, whose shape
        already defines both index-space universes (including rows or
        columns that have no non-zero entries).
        """
        if sp.issparse(edge_index):
            matrix = edge_index.tocoo()
            left_raw = matrix.row
            right_raw = matrix.col
            if edge_weight is not None:
                raise ValueError(
                    "edge_weight must be embedded in the sparse matrix, "
                    "not passed separately.")
            edge_weight = matrix.data
            return left_raw, right_raw, edge_weight, matrix.shape

        if isinstance(edge_index, torch.Tensor):
            arr = edge_index.detach().cpu().numpy()
        else:
            arr = np.asarray(edge_index, dtype=object)

        if arr.ndim != 2:
            raise ValueError(
                "edge_index must be a 2D array of shape (2, E) or (E, 2).")
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            left_raw, right_raw = arr[0], arr[1]
        elif arr.shape[1] == 2:
            left_raw, right_raw = arr[:, 0], arr[:, 1]
        else:
            raise ValueError(
                "edge_index must be a 2D array of shape (2, E) or (E, 2).")

        if edge_weight is not None:
            edge_weight = np.asarray(edge_weight, dtype=np.float64).ravel()
            if edge_weight.shape[0] != len(left_raw):
                raise ValueError(
                    "edge_weight length %d does not match %d edges."
                    % (edge_weight.shape[0], len(left_raw)))
        return (np.asarray(left_raw, dtype=object),
                np.asarray(right_raw, dtype=object),
                edge_weight, None)

    def _build_vocabulary(self, left_raw, right_raw, left_ids, right_ids,
                          sparse_shape=None):
        if sparse_shape is not None:
            # Sparse incidence matrices live in integer index space and
            # their shape defines both universes, so isolated rows and
            # columns are retained.
            if left_ids is not None or right_ids is not None:
                raise ValueError(
                    "left_ids/right_ids cannot be supplied together with a "
                    "sparse incidence matrix: its shape already defines both "
                    "index-space universes.")
            left_ids = np.arange(sparse_shape[0], dtype=np.int64)
            right_ids = np.arange(sparse_shape[1], dtype=np.int64)
            return BipartiteVocabulary(left_ids, right_ids)

        if left_ids is None:
            left_ids = np.unique(left_raw)
        else:
            left_ids = np.asarray(list(left_ids), dtype=object)
            unknown = np.setdiff1d(np.unique(left_raw),
                                   np.asarray(left_ids, dtype=object))
            if unknown.size:
                raise KeyError(
                    "Edges contain left-side ids absent from left_ids: %s"
                    % list(unknown)[:5])

        if right_ids is None:
            right_ids = np.unique(right_raw)
        else:
            right_ids = np.asarray(list(right_ids), dtype=object)
            unknown = np.setdiff1d(np.unique(right_raw),
                                   np.asarray(right_ids, dtype=object))
            if unknown.size:
                raise KeyError(
                    "Edges contain right-side ids absent from right_ids: %s"
                    % list(unknown)[:5])

        return BipartiteVocabulary(left_ids, right_ids)

    # ------------------------------------------------------------------
    # Attribute joining (target table only)
    # ------------------------------------------------------------------
    def _join_attributes(self, target_attributes, target_ids, n_target):
        if target_attributes is None:
            if self.default_features == "identity":
                return torch.eye(n_target, dtype=torch.float)
            return torch.ones((n_target, 1), dtype=torch.float)

        # (ids, X) tuple/list form.
        if isinstance(target_attributes, (tuple, list)) and \
                len(target_attributes) == 2 and not isinstance(
                target_attributes[0], (int, np.integer, str, bytes)):
            attr_ids, attr_x = target_attributes
            attr_ids = np.asarray(list(attr_ids), dtype=object)
        elif isinstance(target_attributes, dict):
            if not target_attributes:
                raise ValueError("target_attributes dict is empty.")
            attr_ids = np.asarray(list(target_attributes.keys()), dtype=object)
            attr_x = np.asarray(list(target_attributes.values()),
                                dtype=np.float64)
        elif isinstance(target_attributes, (torch.Tensor, np.ndarray)):
            attr_x = target_attributes.detach().cpu().numpy() \
                if isinstance(target_attributes, torch.Tensor) \
                else np.asarray(target_attributes, dtype=np.float64)
            if attr_x.ndim != 2 or attr_x.shape[0] != n_target:
                raise ValueError(
                    "Array target_attributes must be 2D with one row per "
                    "target entity (%d rows), got shape %s."
                    % (n_target, attr_x.shape))
            # Aligned with the declared target id universe.
            return torch.as_tensor(attr_x, dtype=torch.float)
        else:
            raise TypeError(
                "Unsupported target_attributes type: %r. Use a dict, an "
                "(ids, X) pair, or an aligned 2D array."
                % type(target_attributes))

        attr_x = np.asarray(attr_x, dtype=np.float64)
        if attr_x.ndim == 1:
            attr_x = attr_x.reshape(-1, 1)
        if attr_x.shape[0] != attr_ids.shape[0]:
            raise ValueError(
                "Attribute table has %d rows but %d ids."
                % (attr_x.shape[0], attr_ids.shape[0]))

        # Attributes referencing unknown target entities signal a broken
        # join and are always rejected.
        unknown = np.setdiff1d(attr_ids, np.asarray(target_ids, dtype=object))
        if unknown.size:
            raise KeyError(
                "Attribute table contains ids absent from the target "
                "entity table: %s" % list(unknown)[:5])

        feature_dim = attr_x.shape[1]
        x = np.zeros((n_target, feature_dim), dtype=np.float64)
        index_by_id = {raw: i for i, raw in enumerate(attr_ids)}
        target_index = {raw: i for i, raw in enumerate(target_ids)}

        filled = np.zeros(n_target, dtype=bool)
        for raw, row_idx in index_by_id.items():
            x[target_index[raw]] = attr_x[row_idx]
            filled[target_index[raw]] = True

        if not filled.all():
            missing = [raw for raw in target_ids
                       if not filled[target_index[raw]]]
            if self.strict_attributes:
                raise KeyError(
                    "Target entities missing from the attribute table: %s"
                    % missing[:5])
            warnings.warn(
                "%d target entities have no attributes; filling zero "
                "feature rows (first missing ids: %s). Pass "
                "strict_attributes=True to reject missing attributes."
                % (len(missing), missing[:5]),
                stacklevel=2)

        return torch.as_tensor(x, dtype=torch.float)

    # ------------------------------------------------------------------
    # Sparse projection
    # ------------------------------------------------------------------
    @staticmethod
    def _cap_neighbors(upper_coo, max_neighbors, n_target):
        """Keep at most ``max_neighbors`` highest-weighted neighbors per
        node while preserving undirected edge consistency.

        Operates on COO arrays only; no dense matrix is formed. The
        input holds unique canonical (upper-triangle) edges.
        """
        rows, cols, data = upper_coo.row, upper_coo.col, upper_coo.data
        # Symmetrize so every node sees all of its incident edges.
        sym_rows = np.concatenate([rows, cols])
        sym_cols = np.concatenate([cols, rows])
        sym_data = np.concatenate([data, data])

        # Sort by node, then by descending weight within each node.
        order = np.lexsort((-sym_data, sym_rows))
        sym_rows, sym_cols, sym_data = (sym_rows[order], sym_cols[order],
                                        sym_data[order])
        segment_starts = np.flatnonzero(
            np.r_[True, sym_rows[1:] != sym_rows[:-1]])
        local_rank = np.arange(sym_rows.shape[0]) - np.repeat(
            segment_starts,
            np.diff(np.r_[segment_starts, sym_rows.shape[0]]))
        keep_directed = local_rank < max_neighbors

        # An undirected edge survives only if both endpoints keep it,
        # otherwise the resulting graph would be asymmetric.
        keep_key = set()
        for r, c, k in zip(sym_rows[keep_directed],
                           sym_cols[keep_directed],
                           sym_data[keep_directed]):
            keep_key.add((int(min(r, c)), int(max(r, c))))

        kept_r, kept_c, kept_d = [], [], []
        for r, c, v in zip(rows, cols, data):
            if (int(r), int(c)) in keep_key:
                kept_r.append(r)
                kept_c.append(c)
                kept_d.append(v)
        return sp.coo_matrix(
            (np.asarray(kept_d),
             (np.asarray(kept_r), np.asarray(kept_c))),
            shape=(n_target, n_target))

    class _BoundedPairAccumulator:
        """Streaming canonical-pair weight accumulator with a hard size cap.

        A single high-degree source can emit more candidate pairs than
        the whole declared edge budget. Without a cap, accumulating
        those pairs would grow without bound even when the final graph
        is going to be truncated. This accumulator therefore prunes to
        the top ``capacity`` weights whenever its size exceeds
        ``2 * capacity``, giving amortized O(capacity) memory regardless
        of the largest source degree. Pruned edges can never return
        (weights are non-negative), so keeping the current leaders is a
        valid bound for a final top-``capacity`` selection.
        """

        def __init__(self, capacity):
            self.capacity = None if capacity is None else int(capacity)
            self.pairs = {}
            self._prune_threshold = (
                None if capacity is None else 2 * int(capacity))

        def add(self, keys, values):
            pairs = self.pairs
            for key, val in zip(keys, values):
                pairs[key] = pairs.get(key, 0.0) + float(val)
            if self.capacity is not None and \
                    len(pairs) > self._prune_threshold:
                # np.partition on the values avoids a full sort.
                vals = np.fromiter(pairs.values(), dtype=np.float64,
                                   count=len(pairs))
                cutoff = np.partition(
                    vals, -self.capacity)[-self.capacity]
                self.pairs = {k: v for k, v in pairs.items()
                              if v >= cutoff}
                # Ties may leave the dict slightly above capacity; the
                # final argpartition enforces the exact budget.
                if len(self.pairs) > self._prune_threshold:
                    top = sorted(self.pairs.items(),
                                 key=lambda kv: kv[1],
                                 reverse=True)[:self.capacity]
                    self.pairs = dict(top)

        def items(self):
            return self.pairs.items()

        def keys(self):
            return self.pairs.keys()

        def values(self):
            return self.pairs.values()

        def __len__(self):
            return len(self.pairs)

        def __contains__(self, key):
            return key in self.pairs

        def __getitem__(self, key):
            return self.pairs[key]

    def fit_transform(self,
                      edge_index,
                      edge_weight=None,
                      left_ids=None,
                      right_ids=None,
                      target_attributes=None):
        """Build the projected sparse entity graph.

        Parameters
        ----------
        edge_index : array-like or scipy.sparse matrix
            Bipartite edges. Either a ``(2, E)`` / ``(E, 2)`` array of
            raw entity ids (torch tensor, numpy array or list of tuples)
            or a sparse ``(n_left, n_right)`` incidence matrix.
        edge_weight : array-like or None, optional
            Optional weights of the bipartite edges. Ignored when
            ``edge_index`` is itself a sparse matrix.
        left_ids, right_ids : array-like or None, optional
            Explicit id universes of the two sides, including isolated
            entities. When omitted, the universe is inferred from the
            edges. Two separate vocabularies are always built, so equal
            raw values across sides remain distinct entities.
        target_attributes : dict, tuple, numpy.ndarray, torch.Tensor or None
            Target-side entity feature table, joined by raw entity id
            (``{id: feature}`` or ``(ids, X)`` or an array aligned with
            the target id universe). Source-side attributes are never
            read. ``None`` synthesizes constant default features.

        Returns
        -------
        projection : BipartiteProjection
            A scorable sparse graph with one node per target entity and
            the original entity ids retained.
        """
        left_raw, right_raw, bip_weight, sparse_shape = \
            self._normalize_edges(edge_index, edge_weight)

        vocabulary = self._build_vocabulary(
            left_raw, right_raw, left_ids, right_ids,
            sparse_shape=sparse_shape)
        self.vocabulary_ = vocabulary

        left_idx = vocabulary.to_index("left", left_raw)
        right_idx = vocabulary.to_index("right", right_raw)
        n_left, n_right = vocabulary.left_size, vocabulary.right_size

        # Binary incidence (for rule/degree statistics) and optionally
        # weighted incidence. Everything below stays in scipy.sparse.
        ones = np.ones(left_idx.shape[0], dtype=np.float64)
        b_binary = sp.csr_matrix(
            (ones, (left_idx, right_idx)), shape=(n_left, n_right))
        b_binary.sum_duplicates()
        b_binary.data[:] = 1.0

        if bip_weight is not None and self.weighting == "count":
            b_used = sp.csr_matrix(
                (bip_weight.astype(np.float64), (left_idx, right_idx)),
                shape=(n_left, n_right))
            b_used.sum_duplicates()
        else:
            b_used = b_binary

        # Orient the incidence as (n_target, n_source).
        if self.target_side == "left":
            target_ids = vocabulary.left_ids
            n_target, n_source = n_left, n_right
            b_target, b_bin_t = b_used, b_binary
        else:
            target_ids = vocabulary.right_ids
            n_target, n_source = n_right, n_left
            b_target, b_bin_t = b_used.T.tocsr(), b_binary.T.tocsr()

        target_degree = np.asarray(b_bin_t.sum(axis=1)).ravel()
        source_degree = np.asarray(b_bin_t.sum(axis=0)).ravel().astype(
            np.int64)

        # ---- Budget pre-check on source degrees (O(nnz), no blow-up) ----
        # Each source entity j with bipartite degree d(j) makes every pair
        # in its target neighborhood share one neighbor. The number of
        # unique undirected candidate pairs is therefore bounded by
        # sum_j C(d(j), 2); int64 avoids overflow on high-degree items.
        candidate_bound = int(
            np.sum(source_degree * (source_degree - 1) // 2))
        if self.max_edges is not None and \
                self.on_budget == "raise" and \
                candidate_bound > self.max_edges and \
                self.min_shared_neighbors == 1 and \
                self.max_neighbors is None:
            # The bound is exact for the default shared-neighbor rule.
            raise BudgetExceededError(candidate_bound, self.max_edges)

        # ---- Streaming per-source neighbor accumulation ----
        # We never form B @ B.T: a single popular source would make scipy
        # materialize all C(d(j),2) pairs of that source at once. Instead
        # the source-side CSC columns are walked one neighbor list at a
        # time, and each source's contributions are merged into a
        # dictionary keyed by the canonical (min, max) target pair.
        b_csc = b_bin_t.tocsc()
        b_used_csc = b_target.tocsc()
        csc_p, csc_i = b_csc.indptr, b_csc.indices
        used_p, used_i, used_x = (b_used_csc.indptr, b_used_csc.indices,
                                  b_used_csc.data)

        # pair_key (encoded int64) -> accumulated weight. Encoding keeps
        # the dictionary compact even for very large target universes.
        # In truncate mode the accumulator is hard-capped at the edge
        # budget, so a single popular source cannot exhaust memory.
        accum_capacity = (self.max_edges
                          if (self.max_edges is not None
                              and self.on_budget == "truncate")
                          else None)
        pair_weights = self._BoundedPairAccumulator(accum_capacity)
        n_target_int = int(n_target)

        if self.weighting == "degree":
            source_correction = 1.0 / np.maximum(source_degree, 1.0)

        # Largest neighborhood whose complete pair set fits the budget.
        def _max_degree_for_budget(capacity):
            return int(np.floor((1 + np.sqrt(1 + 8 * capacity)) / 2))

        for j in range(csc_p.shape[0] - 1):
            start, end = csc_p[j], csc_p[j + 1]
            neighbors = csc_i[start:end]
            d = neighbors.shape[0]
            if d < 2:
                continue

            local_values = None
            # A popular source alone can emit C(d,2) pairs. When a hard
            # truncation budget exists and this source exceeds it, cap
            # its neighborhood first so even one source stays bounded.
            if accum_capacity is not None and \
                    (d * (d - 1) // 2) > accum_capacity:
                keep_d = _max_degree_for_budget(accum_capacity)
                if self.weighting == "count":
                    u_start, u_end = used_p[j], used_p[j + 1]
                    if (u_end - u_start) == d and np.array_equal(
                            used_i[u_start:u_end], neighbors):
                        strength = used_x[u_start:u_end]
                    else:
                        strength = np.ones(d)
                    sel = np.argpartition(strength, keep_d - 1)[:keep_d]
                    sel.sort()
                    neighbors = neighbors[sel]
                    local_values = strength[sel]
                else:
                    # Degree/cosine weight every endpoint equally within
                    # one source, so take a deterministic prefix.
                    neighbors = neighbors[:keep_d]
                d = neighbors.shape[0]

            if self.weighting == "count":
                if local_values is not None:
                    col_values = local_values
                else:
                    u_start, u_end = used_p[j], used_p[j + 1]
                    if (u_end - u_start) == d and \
                            np.array_equal(used_i[u_start:u_end], neighbors):
                        col_values = used_x[u_start:u_end]
                    else:
                        col_values = np.ones(d)
            elif self.weighting == "degree":
                col_values = np.full(d, source_correction[j])
            else:  # cosine: raw shared count, normalized later
                col_values = np.ones(d)

            # All unordered pairs within this source's target neighborhood.
            if self.weighting == "degree":
                # Degree correction contributes the constant 1/d(j) to
                # every pair sharing source j; it must not be multiplied
                # per endpoint.
                col_contrib = np.full(d - 1, source_correction[j])
                for p_idx in range(d - 1):
                    u = int(neighbors[p_idx])
                    v_arr = neighbors[p_idx + 1:].astype(np.int64)
                    lo = np.minimum(u, v_arr)
                    hi = np.maximum(u, v_arr)
                    keys = (lo * n_target_int + hi).tolist()
                    pair_weights.add(
                        keys, col_contrib[:v_arr.shape[0]].tolist())
                continue

            for p_idx in range(d - 1):
                u = int(neighbors[p_idx])
                wu = col_values[p_idx]
                v_arr = neighbors[p_idx + 1:].astype(np.int64)
                # Neighborhood indices are not globally sorted, so
                # canonicalize every (u, v) pair to (min, max).
                lo = np.minimum(u, v_arr)
                hi = np.maximum(u, v_arr)
                keys = (lo * n_target_int + hi).tolist()
                contrib = (wu * col_values[p_idx + 1:]).tolist()
                pair_weights.add(keys, contrib)

        # Self loops: a target entity "shares" all of its own source
        # neighbors with itself. Emit them only when requested, with the
        # same weighting semantics as off-diagonal edges.
        if self.add_self_loops:
            if self.weighting == "degree":
                # sum_j 1/d(j) over the target's own neighbors
                b_csr = b_bin_t.tocsr()
                loop_keys = (np.arange(n_target_int, dtype=np.int64)
                             * n_target_int
                             + np.arange(n_target_int, dtype=np.int64))
                loop_vals = np.zeros(n_target_int, dtype=np.float64)
                for t in range(n_target_int):
                    s0, s1 = b_csr.indptr[t], b_csr.indptr[t + 1]
                    srcs = b_csr.indices[s0:s1]
                    loop_vals[t] = float(
                        np.sum(source_correction[srcs])) if srcs.size else 0.0
                pair_weights.add(loop_keys.tolist(), loop_vals.tolist())
            else:
                # count (or weighted count): the target's own incidence
                # total; cosine self-similarity normalizes to 1 later.
                loop_keys = (np.arange(n_target_int, dtype=np.int64)
                             * n_target_int
                             + np.arange(n_target_int, dtype=np.int64))
                if self.weighting == "count":
                    loop_vals = np.asarray(
                        b_target.sum(axis=1)).ravel().astype(np.float64)
                else:
                    loop_vals = target_degree.astype(np.float64)
                pair_weights.add(loop_keys.tolist(), loop_vals.tolist())

        if not pair_weights:
            upper_rows = np.empty(0, dtype=np.int64)
            upper_cols = np.empty(0, dtype=np.int64)
            weights = np.empty(0, dtype=np.float64)
            shared_counts = np.empty(0, dtype=np.int64)
        else:
            encoded = np.fromiter(pair_weights.keys(), dtype=np.int64,
                                  count=len(pair_weights))
            weights = np.fromiter(pair_weights.values(), dtype=np.float64,
                                  count=len(pair_weights))
            upper_rows = (encoded // n_target_int).astype(np.int64)
            upper_cols = (encoded % n_target_int).astype(np.int64)

            # Shared-neighbor counts are required both for the declared
            # rule and for cosine normalization; recompute them with a
            # second bounded streaming pass only when needed. The same
            # source-level neighborhood cap is applied, and pairs absent
            # from the bounded accumulator are skipped, keeping this pass
            # linear in the retained candidate set.
            if self.min_shared_neighbors > 1 or \
                    self.weighting == "cosine":
                counts = np.zeros(encoded.shape[0], dtype=np.int64)
                key_position = {k: i for i, k in enumerate(encoded.tolist())}
                # Self loops share all of the entity's own neighbors.
                if self.add_self_loops:
                    for t in range(n_target_int):
                        pos = key_position.get(
                            t * n_target_int + t)
                        if pos is not None:
                            counts[pos] = int(target_degree[t])
                for j in range(csc_p.shape[0] - 1):
                    start, end = csc_p[j], csc_p[j + 1]
                    neighbors = csc_i[start:end]
                    d_j = neighbors.shape[0]
                    if d_j < 2:
                        continue
                    if accum_capacity is not None and \
                            (d_j * (d_j - 1) // 2) > accum_capacity:
                        neighbors = neighbors[
                            :_max_degree_for_budget(accum_capacity)]
                        d_j = neighbors.shape[0]
                    for p_idx in range(d_j - 1):
                        u = int(neighbors[p_idx])
                        for v in neighbors[p_idx + 1:]:
                            v = int(v)
                            key = (min(u, v) * n_target_int
                                   + max(u, v))
                            pos = key_position.get(key)
                            if pos is not None:
                                counts[pos] += 1
                shared_counts = counts
            else:
                shared_counts = np.ones(encoded.shape[0], dtype=np.int64)

            if self.weighting == "cosine":
                denom = np.sqrt(
                    target_degree[upper_rows] * target_degree[upper_cols])
                denom[denom == 0] = 1.0
                weights = shared_counts / denom

            keep = shared_counts >= self.min_shared_neighbors
            if not self.add_self_loops:
                keep &= (upper_rows != upper_cols)
            upper_rows = upper_rows[keep]
            upper_cols = upper_cols[keep]
            weights = weights[keep]
            shared_counts = shared_counts[keep]

        projected = sp.csr_matrix(
            (weights, (upper_rows, upper_cols)),
            shape=(n_target, n_target))
        projected.eliminate_zeros()

        # Per-node candidate cap on sparse arrays. Operate on canonical
        # undirected edges and re-symmetrize inside the helper.
        if self.max_neighbors is not None:
            canonical = sp.triu(
                projected,
                k=0 if self.add_self_loops else 1,
                format="coo")
            upper = self._cap_neighbors(
                canonical, self.max_neighbors, n_target)
        else:
            upper = sp.triu(projected,
                            k=0 if self.add_self_loops else 1,
                            format="coo")
        num_edges = upper.nnz

        # Explicit global budget: reject or truncate, never silently
        # explode into a dense structure.
        if self.max_edges is not None and num_edges > self.max_edges:
            if self.on_budget == "raise":
                raise BudgetExceededError(num_edges, self.max_edges)
            top = np.argpartition(-upper.data, self.max_edges - 1)
            top = top[:self.max_edges]
            upper = sp.coo_matrix(
                (upper.data[top], (upper.row[top], upper.col[top])),
                shape=upper.shape)
            num_edges = upper.nnz

        # Symmetrize into an undirected torch edge_index. Self loops are
        # stored once; every other canonical edge gets both directions.
        is_loop = upper.row == upper.col
        off_row = upper.row[~is_loop]
        off_col = upper.col[~is_loop]
        off_data = upper.data[~is_loop]
        loop_idx = upper.row[is_loop]
        loop_data = upper.data[is_loop]

        row = np.concatenate([off_row, off_col, loop_idx])
        col = np.concatenate([off_col, off_row, loop_idx])
        data = np.concatenate([off_data, off_data, loop_data])
        order = np.lexsort((col, row))
        edge_index = torch.as_tensor(
            np.vstack([row[order], col[order]]), dtype=torch.long)
        edge_weight_t = torch.as_tensor(data[order], dtype=torch.float)

        x = self._join_attributes(
            target_attributes, target_ids, n_target)

        config = {
            "weighting": self.weighting,
            "min_shared_neighbors": self.min_shared_neighbors,
            "max_neighbors": self.max_neighbors,
            "max_edges": self.max_edges,
            "on_budget": self.on_budget,
            "add_self_loops": self.add_self_loops,
            "num_source_entities": int(n_source),
            "num_bipartite_edges": int(b_bin_t.nnz),
        }
        projection = BipartiteProjection(
            edge_index=edge_index,
            edge_weight=edge_weight_t,
            x=x,
            entity_ids=target_ids,
            target_side=self.target_side,
            config=config,
            num_nodes=n_target)
        self.projection_ = projection
        return projection

    # sklearn-style alias
    def transform(self, *args, **kwargs):
        return self.fit_transform(*args, **kwargs)


__all__ = [
    "PROJECTION_VERSION",
    "BudgetExceededError",
    "BipartiteVocabulary",
    "BipartiteProjection",
    "BipartiteAdapter",
]
