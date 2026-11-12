"""Static cavities without cardiac boundary assumptions. Coordinates are in mm.

Box cavities preserve sharp corners. Ring cavities require ordered, planar,
star-shaped sections; their two end sections are capped with flat O-grid discs.
No geometry repair or trimming is implicit.
"""
from __future__ import annotations

import numpy as np

from ..assembly.ogrid import build_lv_ogrid
from ..core.block import Block
from ..core.multiblock import MultiBlockMesh
from ..core.topology import FaceKey


class BoxCavityTemplate:
    """Rectangular cavity, with cell counts (not node counts) and wall clustering.

    ``stretch`` in [0, 5] applies symmetric tanh spacing on x/y/z; zero is
    uniform. y-max is a separate ``lid`` wall. ``two_d`` requires one z cell.
    """

    def __init__(self, size=(100, 100, 100), cells=(24, 24, 24),
                 stretch=(0, 0, 0), two_d=False):
        self.size = np.asarray(size, float)
        self.cells = np.asarray(cells)
        self.stretch = np.asarray(stretch, float)
        self.two_d = two_d
        if self.size.shape != (3,) or not np.all(np.isfinite(self.size) & (self.size > 0)):
            raise ValueError("size must contain three finite positive lengths")
        if (self.cells.shape != (3,) or not np.all(np.isfinite(self.cells))
                or not np.all((self.cells >= 1) & (self.cells == self.cells.astype(int)))):
            raise ValueError("cells must contain three positive integers")
        self.cells = self.cells.astype(int)
        if (self.stretch.shape != (3,) or not np.all(np.isfinite(self.stretch))
                or np.any((self.stretch < 0) | (self.stretch > 5))):
            raise ValueError("stretch must contain three values between 0 and 5")
        if two_d and self.cells[2] != 1:
            raise ValueError("two_d requires exactly one cell in z")
        self.patch_types = {p: "wall" for p in ("left", "right", "bottom", "lid", "front", "back")}
        if two_d:
            self.patch_types.update(front="empty", back="empty")

    def build(self):
        coords = []
        for length, n, beta in zip(self.size, self.cells, self.stretch):
            t = np.linspace(-1, 1, n + 1)
            x = (t + 1) / 2 if beta == 0 else (1 + np.tanh(beta * t) / np.tanh(beta)) / 2
            coords.append(length * x)
        nodes = np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1)
        b = Block("cavity", tuple(self.cells + 1)).set_nodes(nodes)
        for face, patch in zip((FaceKey.I0, FaceKey.I1, FaceKey.J0, FaceKey.J1, FaceKey.K0, FaceKey.K1),
                               ("left", "right", "bottom", "lid", "front", "back")):
            b.set_face(face, patch=patch)
        m = MultiBlockMesh()
        m.add_block(b)
        return m.assemble()


class RingCavityTemplate:
    """One cavity from rings, with explicit closed or open end caps.

    ``ends``: closed (both wall), open (last cap opening), through (inlet/outlet).
    ``start``/``end`` are retained axial fractions; a pointed pole needs an
    explicit start > 0. This changes the geometry and is recorded by the CLI.
    """

    def __init__(self, grid, axis, e1, e2, NR=6, core_frac=0.35,
                 wall_ratio=1, ends="closed", start=0, end=1):
        G = np.asarray(grid, float)
        if (G.ndim != 3 or G.shape[2] != 3 or G.shape[0] < 2
                or G.shape[1] < 4 or G.shape[1] % 4 or not np.isfinite(G).all()):
            raise ValueError("grid must be finite (stations>=2, angles>=4 divisible by 4, 3)")
        frame = np.asarray([e1, e2, axis], float)
        if (frame.shape != (3, 3) or not np.isfinite(frame).all()
                or not np.allclose(frame @ frame.T, np.eye(3), atol=1e-6)
                or np.linalg.det(frame) < 0):
            raise ValueError("e1, e2, axis must form a right-handed orthonormal frame")
        if not np.isfinite(core_frac) or not 0 < core_frac < 1:
            raise ValueError("core_frac must be between 0 and 1")
        if not isinstance(NR, (int, np.integer)) or NR < 2:
            raise ValueError("NR must be an integer >= 2")
        if not np.isfinite(wall_ratio) or wall_ratio <= 0:
            raise ValueError("wall_ratio must be finite and positive")
        if not (0 <= start < end <= 1):
            raise ValueError("need 0 <= start < end <= 1")
        if ends not in ("closed", "open", "through"):
            raise ValueError("ends must be closed, open, or through")
        s = (G.mean(1) - G[0].mean(0)) @ frame[2]
        if np.any(np.diff(s) <= 0):
            raise ValueError("ring centres must advance along the axis; folded rows are not removed automatically")
        extent = s[-1]
        off_plane = (G - G.mean(1)[:, None]) @ frame[2]
        if np.max(np.abs(off_plane)) > 1e-6 * extent:
            raise ValueError("rings must lie in planes normal to the supplied axis")
        if start != 0 or end != 1:
            from scipy.interpolate import interp1d
            G = interp1d(s / extent, G, axis=0)(np.linspace(start, end, len(G)))
        # bad_rings deliberately skips row zero in its LV API. Prepend a copy
        # so that every supplied cavity ring is screened here.
        from .conformal_lv import bad_rings
        bad = bad_rings(np.concatenate([G[:1], G]), frame[2], frame[0], frame[1])
        if bad:
            raise ValueError(f"non-star-shaped or collapsed rings: {[k - 1 for k in bad]}; "
                             "a pointed end needs explicit --start/--end trimming")
        self.grid, self.frame = G, frame
        self.NR, self.core_frac, self.wall_ratio = NR, core_frac, wall_ratio
        self.ends = ends
        self.patch_types = {"wall": "wall", "opening": "patch", "inlet": "patch", "outlet": "patch"}

    def build(self):
        e1, e2, axis = self.frame
        m = build_lv_ogrid(self.grid, axis, e1, e2, self.NR, self.core_frac, self.wall_ratio)
        first, last = {"closed": ("wall", "wall"), "open": ("wall", "opening"),
                       "through": ("inlet", "outlet")}[self.ends]
        for k in range(4):
            m.blocks[f"p{k}"].set_face(FaceKey.I1, patch="wall")
        for b in m.blocks.values():
            b.set_face(FaceKey.K0, patch=first)
            b.set_face(FaceKey.K1, patch=last)
        return m.assemble()
