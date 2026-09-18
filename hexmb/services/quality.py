"""Cell quality: the per-hex scaled Jacobian (min over the 8 corner tetrahedra;
<= 0 means an inverted/degenerate cell). Vertex order matches the OpenFOAM hex."""
from __future__ import annotations

import numpy as np

# corner-tet list keyed to the fixed HEX_OFFSETS vertex order
_CORNERS = [
    (0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7),
    (4, 7, 5, 0), (5, 4, 6, 1), (6, 5, 7, 2), (7, 6, 4, 3),
]


def scaled_jac(points: np.ndarray, hexes: np.ndarray) -> np.ndarray:
    """Return the per-cell minimum-corner scaled Jacobian in [-1,1].
    <= 0 marks an inverted (degenerate/tangled) cell."""
    h = points[hexes]  # (ncell, 8, 3)
    mn = np.full(len(hexes), np.inf)
    for a, b, c, e in _CORNERS:
        u = h[:, b] - h[:, a]
        v = h[:, c] - h[:, a]
        w = h[:, e] - h[:, a]
        det = np.einsum("ij,ij->i", u, np.cross(v, w))
        nrm = (
            np.linalg.norm(u, axis=1)
            * np.linalg.norm(v, axis=1)
            * np.linalg.norm(w, axis=1)
            + 1e-12
        )
        mn = np.minimum(mn, det / nrm)
    return mn
