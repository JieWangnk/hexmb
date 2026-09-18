"""Face keys, the fixed local hex vertex order, and 2D face-grid orientation.

The engine's cells are always built in one fixed local vertex order (the order
`cfd/morph_demo.py::scaled_jac` and the OpenFOAM face template both assume), so
the quality gate and the exporter apply unchanged.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

# ---------------------------------------------------------------------------
# Fixed local hex vertex order (offsets in (di, dj, dk) from cell corner (i,j,k))
# matches cfd/morph_demo.py:
#   v0..v3 = k-layer 0 loop, v4..v7 = k-layer 1 loop
# ---------------------------------------------------------------------------
HEX_OFFSETS = np.array(
    [
        (0, 0, 0),  # v0
        (0, 1, 0),  # v1
        (1, 1, 0),  # v2
        (1, 0, 0),  # v3
        (0, 0, 1),  # v4
        (0, 1, 1),  # v5
        (1, 1, 1),  # v6
        (1, 0, 1),  # v7
    ],
    dtype=int,
)

# The 6 faces of the hex as 4-cycles of local vertex indices (each a genuine
# edge loop of the hex). Loop direction is not relied upon; the exporter
# re-orients every face outward from its cell centroid.
HEX_FACES = (
    (0, 1, 2, 3),  # dk = 0
    (4, 5, 6, 7),  # dk = 1
    (0, 4, 5, 1),  # di = 0
    (3, 2, 6, 7),  # di = 1
    (0, 3, 7, 4),  # dj = 0
    (1, 5, 6, 2),  # dj = 1
)

# The 12 edges of the hex (local vertex index pairs), for adjacency-based smoothing.
# The first 8 lie within a k-layer (in-plane); the last 4 connect the two layers.
HEX_INPLANE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
)
HEX_VERTICAL_EDGES = ((0, 4), (1, 5), (2, 6), (3, 7))
HEX_EDGES = HEX_INPLANE_EDGES + HEX_VERTICAL_EDGES


class FaceKey(str, Enum):
    """The 6 logical faces of a structured block."""

    I0 = "i0"
    I1 = "i1"
    J0 = "j0"
    J1 = "j1"
    K0 = "k0"
    K1 = "k1"


# For each face: (axis fixed, index picks 0 or -1). The remaining two axes, in
# increasing order, form the face's 2D (a, b) grid.
_FACE_SPEC = {
    FaceKey.I0: (0, 0),
    FaceKey.I1: (0, -1),
    FaceKey.J0: (1, 0),
    FaceKey.J1: (1, -1),
    FaceKey.K0: (2, 0),
    FaceKey.K1: (2, -1),
}


def face_index_grid(gid: np.ndarray, key: FaceKey) -> np.ndarray:
    """Return the 2D (na, nb) sub-array of a block's (ni,nj,nk) id/scalar grid
    lying on face `key`, in natural increasing order of the two free axes."""
    axis, end = _FACE_SPEC[key]
    sl = [slice(None), slice(None), slice(None)]
    sl[axis] = 0 if end == 0 else gid.shape[axis] - 1
    return gid[tuple(sl)]


def face_coord_grid(nodes: np.ndarray, key: FaceKey) -> np.ndarray:
    """Return the (na, nb, 3) physical coordinates on face `key`."""
    axis, end = _FACE_SPEC[key]
    sl = [slice(None), slice(None), slice(None)]
    sl[axis] = 0 if end == 0 else nodes.shape[axis] - 1
    return nodes[tuple(sl)]


def orient_transforms(a: np.ndarray):
    """Yield (name, transformed) for all 8 dihedral orientations of a 2D grid
    (last axis, if any, is treated as a trailing component axis and preserved)."""
    yield "identity", a
    yield "flipa", a[::-1]
    yield "flipb", a[:, ::-1]
    yield "flipab", a[::-1, ::-1]
    t = np.swapaxes(a, 0, 1)
    yield "T", t
    yield "T.flipa", t[::-1]
    yield "T.flipb", t[:, ::-1]
    yield "T.flipab", t[::-1, ::-1]


def match_orientation(coords_a: np.ndarray, coords_b: np.ndarray, tol: float):
    """Find the dihedral transform of `coords_b` (na,nb,3) that makes it coincide
    with `coords_a` (na,nb,3) within `tol`. Returns (name, transformed_b) or
    (None, None) if no orientation matches (the two faces are not geometrically
    the same surface -> a crack)."""
    for name, tb in orient_transforms(coords_b):
        if tb.shape != coords_a.shape:
            continue
        if np.max(np.linalg.norm(tb - coords_a, axis=-1)) <= tol:
            return name, tb
    return None, None
