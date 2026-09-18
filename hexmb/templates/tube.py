"""Tube templates: straight pipe and U-bend/elbow. Both are one `sweep_ogrid`
with a different centreline. Patches: inlet (start cap), outlet (end cap),
wall (swept lateral surface)."""
from __future__ import annotations

import numpy as np

from ..assembly.ogrid import graded_radial
from ..assembly.sweep import sweep_ogrid
from ..core.multiblock import MultiBlockMesh
from ..core.topology import FaceKey

_BLOCKS = ["core", "p0", "p1", "p2", "p3"]


def _finish(m: MultiBlockMesh) -> MultiBlockMesh:
    for k in range(4):
        m.blocks[f"p{k}"].set_face(FaceKey.I1, patch="wall")
    for name in _BLOCKS:
        m.blocks[name].set_face(FaceKey.K0, patch="inlet")
        m.blocks[name].set_face(FaceKey.K1, patch="outlet")
    m.assemble()
    return m


class PipeTemplate:
    """Straight circular pipe. length, radius in mm."""

    def __init__(self, length=100.0, radius=10.0, n_axial=40, NT=32, NR=6, core_frac=0.42,
                 wall_ratio=1.0):
        self.length = length
        self.radius = radius
        self.n_axial = n_axial
        self.NT = NT
        self.NR = NR
        self.core_frac = core_frac
        self.wall_ratio = wall_ratio

    def build(self) -> MultiBlockMesh:
        z = np.linspace(0, self.length, self.n_axial)
        path = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
        rd = graded_radial(self.NR, self.wall_ratio)
        m = sweep_ogrid(path, self.radius, self.NT, self.NR, self.core_frac, radial_dist=rd)
        return _finish(m)


class ElbowTemplate:
    """U-bend / elbow: two straight legs joined by a circular arc.
    bend_radius = centreline radius of the arc; angle in degrees (90 or 180)."""

    def __init__(
        self, radius=10.0, bend_radius=40.0, angle_deg=180.0, leg=40.0,
        n_leg=16, n_arc=40, NT=32, NR=6, core_frac=0.42, wall_ratio=1.0,
    ):
        self.radius = radius
        self.bend_radius = bend_radius
        self.angle = np.radians(angle_deg)
        self.leg = leg
        self.n_leg = n_leg
        self.n_arc = n_arc
        self.NT = NT
        self.NR = NR
        self.core_frac = core_frac
        self.wall_ratio = wall_ratio

    def build(self) -> MultiBlockMesh:
        R = self.bend_radius
        # inlet leg along -y into the arc, arc in the x-y plane about centre (R,0)
        t_in = np.linspace(-self.leg, 0, self.n_leg, endpoint=False)
        leg_in = np.stack([np.zeros_like(t_in), t_in, np.zeros_like(t_in)], axis=1)
        phi = np.linspace(0, self.angle, self.n_arc)
        arc = np.stack([R - R * np.cos(phi), R * np.sin(phi), np.zeros_like(phi)], axis=1)
        # outlet leg tangent to arc end
        tend = np.array([np.sin(self.angle), np.cos(self.angle), 0.0])
        s_out = np.linspace(0, self.leg, self.n_leg + 1)[1:]
        leg_out = arc[-1] + np.outer(s_out, tend)
        path = np.vstack([leg_in, arc, leg_out])
        rd = graded_radial(self.NR, self.wall_ratio)
        m = sweep_ogrid(path, self.radius, self.NT, self.NR, self.core_frac, radial_dist=rd)
        return _finish(m)
