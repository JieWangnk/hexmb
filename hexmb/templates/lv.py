"""LV chamber template: the first hexmb geometry template.

Builds the 5-block LV O-grid from an endocardial surface grid, labels the outer
petal faces + apex disc as `wall`, and partitions the base cap into `inlet`
(mitral), `outlet1` (aortic) and `wall` (annulus) by angular sector.
"""
from __future__ import annotations

import numpy as np

from ..assembly.ogrid import build_lv_ogrid
from ..core.multiblock import MultiBlockMesh
from ..core.topology import FaceKey

_BLOCKS = ["core", "p0", "p1", "p2", "p3"]


class LVChamberTemplate:
    def __init__(
        self,
        grid,
        axis,
        e1,
        e2,
        mu=None,
        NR: int = 6,
        core_frac: float = 0.35,
        mitral_deg: tuple = (110.0, 320.0),
        aortic_deg: tuple = (0.0, 90.0),
        smooth_iters: int = 0,
        wall_ratio: float = 1.0,
    ):
        self.grid = np.asarray(grid, float)
        self.axis = np.asarray(axis, float)
        self.e1 = np.asarray(e1, float)
        self.e2 = np.asarray(e2, float)
        self.mu = np.asarray(mu, float) if mu is not None else self.grid[-1].mean(axis=0)
        self.NR = NR
        self.core_frac = core_frac
        self.mitral_deg = mitral_deg
        self.aortic_deg = aortic_deg
        self.smooth_iters = smooth_iters
        self.wall_ratio = wall_ratio  # <1 clusters petal cells at the endocardial wall

    def _base_patch(self, xyz):
        d = xyz - self.mu
        ang = np.degrees(np.arctan2(d @ self.e2, d @ self.e1)) % 360.0
        a0, a1 = self.aortic_deg
        if a0 <= ang <= a1:
            return "outlet1"
        m0, m1 = self.mitral_deg
        if m0 <= ang <= m1:
            return "inlet"
        return "wall"

    def build(self) -> MultiBlockMesh:
        m = build_lv_ogrid(self.grid, self.axis, self.e1, self.e2, self.NR, self.core_frac,
                           wall_ratio=self.wall_ratio)
        # outer petal walls
        for k in range(4):
            m.blocks[f"p{k}"].set_face(FaceKey.I1, patch="wall")
        # apex disc (K0 of every block)
        for name in _BLOCKS:
            m.blocks[name].set_face(FaceKey.K0, patch="wall")
        # base cap (K1 of every block) -> angular predicate
        for name in _BLOCKS:
            m.set_predicate_face(name, FaceKey.K1, self._base_patch)
        m.assemble()
        if self.smooth_iters:
            m.smooth(self.smooth_iters)
        return m
