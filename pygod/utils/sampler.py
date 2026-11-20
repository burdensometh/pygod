# -*- coding: utf-8 -*-
"""Budget-aware sampler for manual review of outlier detection results.

Manual review can only inspect a fixed number of nodes, and simply
taking the highest outlier scores often returns near-duplicates from
a single similar cluster. The sampler in this module allocates the
review budget as quotas over graphs or entity layers, and then
selects high-score representatives of clusters in the node embedding
space, so the reviewed nodes cover diverse anomaly types. It returns
the real review subgraph in which review targets and context-only
neighbors are marked separately, together with the selected node ids
and their original scores.
"""
# Author: PyGOD Team
# License: BSD 2 clause

import warnings
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph

from .utility import check_parameter


@dataclass
class ReviewSample:
    """Container for the output of :meth:`BudgetSampler.sample`.

    Attributes
    ----------
    node_id : torch.LongTensor
        Sorted, unique indices of the selected review targets in the
        input graph. Node identity is stable: ids always refer to the
        node order of the input graph, never to subgraph positions.
    score : torch.Tensor
        The original outlier scores of the review targets, aligned
        with ``node_id``.
    subgraph : torch_geometric.data.Data
        The real review subgraph induced by the targets and their
        context neighbors. Its attributes are:

        - ``n_id``: original node id of each subgraph node;
        - ``is_target``: bool mask that is ``True`` for review targets
          and ``False`` for context-only neighbors;
        - ``score``: original outlier score of each subgraph node;
        - ``x``, ``edge_index``, ``edge_attr``, ``y``: carried over
          from the input graph when available.
    quota : dict
        The quota allocated to each graph/entity group.
    group : numpy.ndarray
        The group key of each review target, aligned with ``node_id``.
    cluster_id : numpy.ndarray
        The embedding cluster of each review target, aligned with
        ``node_id``. ``-1`` means the node was selected without
        clustering (fallback selection or take-all quota).
    method : str
        ``'cluster'`` if embeddings were used, ``'fallback'`` if the
        deterministic score-only fallback was used.
    """

    node_id: torch.Tensor
    score: torch.Tensor
    subgraph: Data
    quota: dict
    group: np.ndarray
    cluster_id: np.ndarray
    method: str

    def __len__(self):
        return int(self.node_id.shape[0])

    def __repr__(self):
        return ('ReviewSample(method={}, targets={}, groups={}, '
                'subgraph_nodes={})'.format(self.method, len(self),
                                            len(self.quota),
                                            self.subgraph.num_nodes))


class BudgetSampler:
    """Select a bounded, diverse set of high-score nodes for manual review.

    The sampler first allocates the review budget as quotas over
    groups (graphs of a batch, or entity layers), then picks
    high-score representatives of embedding clusters inside each
    group, so that the reviewed nodes are not all drawn from one
    similar cluster. It completes the full selection and returns the
    selected node ids, their original scores, and the real review
    subgraph with review targets and context-only neighbors marked
    separately.

    Parameters
    ----------
    budget : int
        Maximum number of review targets to select. Must be at least
        1. Fewer than ``budget`` targets are returned when the unique
        candidates are exhausted; duplicates are never used to pad
        the budget.
    quota_by : str, optional
        How to allocate the budget before selection:

        - ``'graph'``: one quota group per graph, read from
          ``data.batch`` (merge multiple graphs with
          ``torch_geometric.data.Batch``; a single graph forms one
          group);
        - ``'entity'``: one quota group per entity layer, read from
          the ``group`` argument of :meth:`sample` or from
          ``data.node_type``;
        - ``'none'``: no allocation, all nodes form a single group.

        Default: ``'graph'``.
    num_hops : int, optional
        Number of hops (traversed in either edge direction) used to
        expand the review targets with context neighbors when building
        the review subgraph. ``0`` means the subgraph contains the
        targets only. Default: ``1``.
    seed : int, optional
        Random seed of the clustering. The same input and seed always
        produce the same sample. Default: ``42``.
    n_init : int, optional
        Number of k-means restarts. Default: ``10``.

    Examples
    --------
    >>> import torch
    >>> from pygod.utils import BudgetSampler
    >>> sampler = BudgetSampler(budget=4, quota_by='none', seed=42)
    >>> data = ...  # a torch_geometric.data.Data graph
    >>> score = torch.rand(data.num_nodes)  # outlier score per node
    >>> emb = torch.randn(data.num_nodes, 16)  # detector embedding
    >>> result = sampler.sample(data, score, emb=emb)  # doctest: +SKIP
    >>> result.node_id  # ids of the review targets  # doctest: +SKIP
    """

    def __init__(self,
                 budget,
                 quota_by='graph',
                 num_hops=1,
                 seed=42,
                 n_init=10):

        check_parameter(budget, 1, param_name='budget', include_left=True)
        check_parameter(num_hops, 0, param_name='num_hops',
                        include_left=True)
        check_parameter(seed, 0, param_name='seed', include_left=True)
        check_parameter(n_init, 1, param_name='n_init', include_left=True)
        if not (float(budget).is_integer() and float(num_hops).is_integer()
                and float(seed).is_integer() and float(n_init).is_integer()):
            raise ValueError('budget, num_hops, seed and n_init must be '
                             'integers')
        if quota_by not in ('graph', 'entity', 'none'):
            raise ValueError("quota_by must be one of 'graph', 'entity' or "
                             "'none', got: {}".format(quota_by))

        self.budget = int(budget)
        self.quota_by = quota_by
        self.num_hops = int(num_hops)
        self.seed = int(seed)
        self.n_init = int(n_init)

    def sample(self, data, score, emb=None, group=None, candidates=None):
        """Select the review targets and build the review subgraph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input graph.
        score : torch.Tensor or numpy.ndarray
            The outlier scores of shape :math:`N`. Higher scores are
            more suspicious.
        emb : torch.Tensor or numpy.ndarray or tuple, optional
            The node embeddings of shape :math:`N \\times D` used for
            clustering, e.g. the ``emb`` attribute of a fitted deep
            detector. A tuple of embeddings is concatenated along the
            feature dimension. If ``None``, a deterministic fallback
            selects the highest-score unique candidates inside each
            quota group. Default: ``None``.
        group : torch.Tensor or numpy.ndarray, optional
            Explicit group (entity layer) of each node, of shape
            :math:`N`. Overrides ``quota_by`` when given.
            Default: ``None``.
        candidates : torch.Tensor or numpy.ndarray, optional
            The node ids allowed to be selected. Duplicated ids are
            removed and never count towards the budget. If ``None``,
            all nodes are candidates. Default: ``None``.

        Returns
        -------
        result : ReviewSample
            The selected review targets, their original scores, the
            review subgraph, and the allocation metadata.
        """

        score_np = self._to_numpy(score, 'score').astype(np.float64).ravel()
        if not np.isfinite(score_np).all():
            raise ValueError('score must contain only finite values')

        num_nodes = data.num_nodes
        if num_nodes is None:
            num_nodes = score_np.shape[0]
        if score_np.shape[0] != num_nodes:
            raise ValueError('score has {} entries but the graph has {} '
                             'nodes'.format(score_np.shape[0], num_nodes))

        # unique candidates only; duplicates never pad the budget
        if candidates is None:
            cand = np.arange(num_nodes, dtype=np.int64)
        else:
            cand = np.unique(self._to_numpy(candidates, 'candidates')
                             .astype(np.int64).ravel())
            if cand.size > 0 and (cand[0] < 0 or cand[-1] >= num_nodes):
                raise ValueError('candidates contain node ids outside '
                                 '[0, {})'.format(num_nodes))
        if cand.size == 0:
            raise ValueError('no candidates to sample from')

        group_keys, group_idx = self._resolve_groups(data, group, num_nodes)

        emb_np = self._resolve_emb(emb, num_nodes)
        method = 'cluster' if emb_np is not None else 'fallback'
        if method == 'fallback':
            warnings.warn('emb is not provided: falling back to '
                          'deterministic top-score selection within each '
                          'quota group')

        # allocate the budget over groups (largest remainder, capped by
        # the number of unique candidates of each group)
        sizes = np.bincount(group_idx[cand], minlength=len(group_keys))
        quota = self._allocate_quota(sizes, min(self.budget, cand.size))

        sel_id, sel_group, sel_cluster = [], [], []
        for gi in range(len(group_keys)):
            q = int(quota[gi])
            if q == 0:
                continue
            idx = cand[group_idx[cand] == gi]
            chosen, cl = self._select_group(idx, q, score_np, emb_np, method)
            sel_id.append(chosen)
            sel_group.append(np.full(chosen.shape[0], gi, dtype=np.int64))
            sel_cluster.append(cl)

        node_id = np.concatenate(sel_id)
        group_of = np.concatenate(sel_group)
        cluster_of = np.concatenate(sel_cluster)

        # stable identity: sort by the original node id
        order = np.argsort(node_id, kind='stable')
        node_id = node_id[order]
        group_of = group_of[order]
        cluster_of = cluster_of[order]

        node_id_t = torch.from_numpy(node_id).long()
        sub = self._build_subgraph(data, node_id_t, score_np, num_nodes)

        return ReviewSample(
            node_id=node_id_t,
            score=torch.from_numpy(score_np[node_id]),
            subgraph=sub,
            quota={self._py_key(group_keys[i]): int(quota[i])
                   for i in range(len(group_keys))},
            group=group_keys[group_of],
            cluster_id=cluster_of,
            method=method)

    def _resolve_groups(self, data, group, num_nodes):
        """Resolve the quota group of every node."""
        if group is not None:
            grp = self._to_numpy(group, 'group').ravel()
        elif self.quota_by == 'graph':
            batch = getattr(data, 'batch', None)
            grp = np.zeros(num_nodes, dtype=np.int64) if batch is None \
                else self._to_numpy(batch, 'data.batch').ravel()
        elif self.quota_by == 'entity':
            node_type = getattr(data, 'node_type', None)
            if node_type is None:
                raise ValueError("quota_by='entity' requires the `group` "
                                 "argument or a `data.node_type` attribute")
            grp = self._to_numpy(node_type, 'data.node_type').ravel()
        else:  # 'none'
            grp = np.zeros(num_nodes, dtype=np.int64)

        if grp.shape[0] != num_nodes:
            raise ValueError('group has {} entries but the graph has {} '
                             'nodes'.format(grp.shape[0], num_nodes))
        keys, idx = np.unique(grp, return_inverse=True)
        return keys, idx

    @staticmethod
    def _resolve_emb(emb, num_nodes):
        """Validate the embedding, concatenating tuples if given."""
        if emb is None:
            return None
        if isinstance(emb, (list, tuple)):
            emb = np.concatenate(
                [BudgetSampler._to_numpy(e, 'emb') for e in emb], axis=1)
        else:
            emb = BudgetSampler._to_numpy(emb, 'emb')
        if emb.ndim == 1:
            emb = emb.reshape(-1, 1)
        if emb.ndim != 2 or emb.shape[0] != num_nodes:
            raise ValueError('emb must have shape ({}, D), got {}'.format(
                num_nodes, emb.shape))
        return emb.astype(np.float64)

    def _select_group(self, idx, q, score_np, emb_np, method):
        """Select q unique nodes from ascending candidate ids ``idx``."""
        if q >= idx.shape[0]:
            return idx.copy(), np.full(idx.shape[0], -1, dtype=np.int64)

        if method == 'cluster':
            labels = KMeans(n_clusters=q, random_state=self.seed,
                            n_init=self.n_init).fit_predict(emb_np[idx])
            chosen = []
            for c in range(q):
                pos = np.nonzero(labels == c)[0]
                if pos.size == 0:
                    # empty clusters are possible with duplicate
                    # embedding rows; the top-up below backfills them
                    continue
                # highest score first; ties broken by the lowest node id
                chosen.append(pos[np.argmax(score_np[idx[pos]])])
            chosen = np.asarray(chosen, dtype=np.int64)
            if chosen.shape[0] < q:
                # top up from unchosen candidates, highest score first
                # (unique nodes only, never duplicates)
                mask = np.ones(idx.shape[0], dtype=bool)
                mask[chosen] = False
                rest = np.nonzero(mask)[0]
                ranked = rest[np.lexsort((idx[rest], -score_np[idx[rest]]))]
                chosen = np.concatenate([chosen, ranked[:q - chosen.shape[0]]])
            chosen = np.unique(chosen)
            return idx[chosen], labels[chosen]

        # fallback: highest scores first, ties broken by the lowest node id
        ranked = np.lexsort((idx, -score_np[idx]))
        pos = np.sort(ranked[:q])
        return idx[pos], np.full(q, -1, dtype=np.int64)

    @staticmethod
    def _allocate_quota(sizes, budget):
        """Largest-remainder allocation of ``budget`` over group sizes,
        capped by the size of each group, with deterministic ties."""
        sizes = np.asarray(sizes, dtype=np.int64)
        quota = np.zeros(sizes.shape[0], dtype=np.int64)
        budget = int(min(budget, sizes.sum()))
        if budget <= 0:
            return quota

        raw = budget * sizes / sizes.sum()
        quota = np.minimum(np.floor(raw).astype(np.int64), sizes)
        left = budget - int(quota.sum())

        # largest fractional remainder first; ties broken by group order
        frac = raw - np.floor(raw)
        order = np.argsort(-frac, kind='stable')
        while left > 0:
            progressed = False
            for g in order:
                if left == 0:
                    break
                if quota[g] < sizes[g]:
                    quota[g] += 1
                    left -= 1
                    progressed = True
            if not progressed:
                break
        return quota

    def _build_subgraph(self, data, node_id_t, score_np, num_nodes):
        """Induce the review subgraph on targets + context neighbors."""
        edge_index = getattr(data, 'edge_index', None)
        device = edge_index.device if edge_index is not None \
            else torch.device('cpu')
        seeds = node_id_t.to(device)

        if edge_index is not None and self.num_hops > 0:
            subset, _, _, _ = k_hop_subgraph(seeds, self.num_hops, edge_index,
                                             relabel_nodes=False,
                                             num_nodes=num_nodes)
            subset = torch.unique(torch.cat([subset, seeds]))
        else:
            subset = torch.unique(seeds)

        sub = Data()
        sub.num_nodes = int(subset.numel())
        sub.n_id = subset
        # review targets are marked separately from context-only neighbors
        sub.is_target = torch.isin(subset, seeds)
        sub.score = torch.from_numpy(score_np[subset.cpu().numpy()]).to(device)

        if getattr(data, 'x', None) is not None:
            sub.x = data.x[subset]
        if edge_index is not None:
            edge_attr = getattr(data, 'edge_attr', None)
            new_ei, new_ea = subgraph(subset, edge_index, edge_attr,
                                      relabel_nodes=True,
                                      num_nodes=num_nodes)
            sub.edge_index = new_ei
            if new_ea is not None:
                sub.edge_attr = new_ea
        y = getattr(data, 'y', None)
        if y is not None and y.shape[0] == num_nodes:
            sub.y = y[subset]
        return sub

    @staticmethod
    def _to_numpy(x, name):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        try:
            return np.asarray(x)
        except Exception:
            raise TypeError('{} must be a torch.Tensor or '
                            'array-like'.format(name))

    @staticmethod
    def _py_key(key):
        return key.item() if hasattr(key, 'item') else key

    def __repr__(self):
        return ('BudgetSampler(budget={}, quota_by={!r}, num_hops={}, '
                'seed={})'.format(self.budget, self.quota_by, self.num_hops,
                                  self.seed))
