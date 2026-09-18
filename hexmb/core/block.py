"""Block: one logical structured hexahedral region."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .topology import FaceKey, HEX_OFFSETS, face_coord_grid, face_index_grid
from .tfi import tfi3d


@dataclass
class BoundaryFace:
    """A face of a block. `patch` names the OpenFOAM patch it exports to; a face
    consumed by a Join is an interface (patch=None) and is never exported."""

    patch: str | None = None
    kind: str = "grid"  # 'grid' | 'analytic' | 'interface' | 'stl'


class Block:
    """A structured block with node counts (ni,nj,nk). `nodes` holds physical
    coordinates in millimetres, filled by a template (directly or via TFI)."""

    def __init__(self, name: str, shape: tuple[int, int, int]):
        self.name = name
        self.shape = tuple(int(s) for s in shape)
        if any(s < 2 for s in self.shape):
            raise ValueError(f"block {name}: each dimension needs >=2 nodes, got {shape}")
        self.nodes = np.zeros(self.shape + (3,), float)
        self.bfaces: dict[FaceKey, BoundaryFace] = {}
        self.gid: np.ndarray | None = None  # (ni,nj,nk) filled at assembly

    # -- geometry --------------------------------------------------------
    @property
    def npoints(self) -> int:
        ni, nj, nk = self.shape
        return ni * nj * nk

    def set_nodes(self, nodes: np.ndarray) -> "Block":
        nodes = np.asarray(nodes, float)
        if nodes.shape != self.shape + (3,):
            raise ValueError(f"block {self.name}: nodes {nodes.shape} != {self.shape + (3,)}")
        self.nodes = nodes
        return self

    def set_face(self, key: FaceKey, patch: str | None = None, kind: str = "grid") -> "Block":
        self.bfaces[FaceKey(key)] = BoundaryFace(patch=patch, kind=kind)
        return self

    def fill_tfi(self) -> "Block":
        """Fill the interior from the 6 current boundary faces by Coons TFI."""
        f = {
            "i0": self.nodes[0, :, :, :],
            "i1": self.nodes[-1, :, :, :],
            "j0": self.nodes[:, 0, :, :],
            "j1": self.nodes[:, -1, :, :],
            "k0": self.nodes[:, :, 0, :],
            "k1": self.nodes[:, :, -1, :],
        }
        self.nodes = tfi3d(f)
        return self

    def face_coords(self, key: FaceKey) -> np.ndarray:
        return face_coord_grid(self.nodes, FaceKey(key))

    # -- topology --------------------------------------------------------
    def local_ids(self) -> np.ndarray:
        """(ni,nj,nk) array of local flat node ids: (i*nj+j)*nk+k."""
        ni, nj, nk = self.shape
        return np.arange(ni * nj * nk).reshape(ni, nj, nk)

    def face_local_ids(self, key: FaceKey) -> np.ndarray:
        return face_index_grid(self.local_ids(), FaceKey(key))

    def cells_local(self) -> np.ndarray:
        """(ncell, 8) local vertex ids in the fixed HEX_OFFSETS order."""
        ni, nj, nk = self.shape
        lid = self.local_ids()
        ii, jj, kk = np.meshgrid(
            np.arange(ni - 1), np.arange(nj - 1), np.arange(nk - 1), indexing="ij"
        )
        ii = ii.ravel()
        jj = jj.ravel()
        kk = kk.ravel()
        cols = []
        for di, dj, dk in HEX_OFFSETS:
            cols.append(lid[ii + di, jj + dj, kk + dk])
        return np.stack(cols, axis=1)
