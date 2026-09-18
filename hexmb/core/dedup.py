"""Topological node deduplication via batched connected components.

Assembling a multi-block mesh means merging the shared nodes of every block
interface. Each interface contributes a set of node-pair equivalences; the
merged node set is the connected components of the graph of all those pairs.

This used to be a per-pair Python union-find, which did not scale (a Python
loop over every shared node, plus a `find` over every slot). It is now
edge-accumulating: unions are buffered as edges and resolved in ONE vectorised
pass -- `scipy.sparse.csgraph.connected_components` when SciPy is available,
else a numpy label-propagation fallback -- so meshes with millions of nodes
assemble quickly. The public API (`union`, `union_arrays`, `compact_labels`)
is unchanged.
"""
from __future__ import annotations

import numpy as np


class DSU:
    """Union-find over ``n`` slots, resolved as connected components.

    Unions are accumulated as edges and resolved lazily in ``compact_labels``
    (or ``find``). Isolated slots become their own component, so every slot
    maps to a dense label ``0..K-1``.
    """

    def __init__(self, n: int):
        self.n = int(n)
        self._src: list = []
        self._dst: list = []
        self._labels: np.ndarray | None = None

    def union(self, a: int, b: int) -> None:
        self._src.append(np.int64(a))
        self._dst.append(np.int64(b))
        self._labels = None

    def union_arrays(self, a: np.ndarray, b: np.ndarray) -> None:
        """Buffer an element-wise union of two equal-length id arrays (O(1))."""
        self._src.append(np.asarray(a, dtype=np.int64).ravel())
        self._dst.append(np.asarray(b, dtype=np.int64).ravel())
        self._labels = None

    def _edges(self):
        if not self._src:
            e = np.empty(0, np.int64)
            return e, e
        s = np.concatenate([np.atleast_1d(x) for x in self._src]).astype(np.int64)
        d = np.concatenate([np.atleast_1d(x) for x in self._dst]).astype(np.int64)
        return s, d

    def compact_labels(self) -> np.ndarray:
        """Map every slot to a dense root label ``0..K-1`` (cached)."""
        if self._labels is not None:
            return self._labels
        s, d = self._edges()
        try:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            g = coo_matrix((np.ones(len(s), np.int8), (s, d)), shape=(self.n, self.n))
            _, labels = connected_components(g, directed=False)
            labels = labels.astype(np.int64)
        except Exception:
            labels = self._labels_numpy(s, d)
        self._labels = labels
        return labels

    def _labels_numpy(self, s: np.ndarray, d: np.ndarray) -> np.ndarray:
        """SciPy-free connected components by min-label propagation."""
        labels = np.arange(self.n, dtype=np.int64)
        if len(s):
            u = np.concatenate([s, d])
            v = np.concatenate([d, s])
            while True:
                new = labels.copy()
                np.minimum.at(new, u, labels[v])
                if np.array_equal(new, labels):
                    break
                labels = new
            _, labels = np.unique(labels, return_inverse=True)
            labels = labels.astype(np.int64)
        return labels

    def find(self, x: int) -> int:
        """Dense component label of slot ``x`` (resolves + caches on first call)."""
        return int(self.compact_labels()[x])
