"""STL tube template: single-lumen tubular STL -> O-grid.

Reports the applicability diagnostics (branch stations, missed rays). A geometry
with branches (a section that has more than one loop) is out of scope for the
single-lumen O-grid sweep; only the main trunk is meshed and the branch is
reported.
"""
from __future__ import annotations

from ..core.multiblock import MultiBlockMesh
from ..core.topology import FaceKey
from ..project.stl_tube import ogrid_from_stl

_BLOCKS = ["core", "p0", "p1", "p2", "p3"]


class NotSingleLumen(ValueError):
    pass


class TubeSTLTemplate:
    def __init__(self, path, ns=40, NT=32, NR=6, core_frac=0.42, wall_ratio=1.0,
                 inlet_stl=None, step=2.0, centreline=None, allow_branches=False):
        self.path = path
        self.ns = ns
        self.NT = NT
        self.NR = NR
        self.core_frac = core_frac
        self.wall_ratio = wall_ratio
        self.inlet_stl = inlet_stl
        self.step = step
        self.centreline = centreline
        self.allow_branches = allow_branches
        self.diag = None

    @classmethod
    def from_skeleton(cls, wall_path, inlet_stl, outlet_paths, pitch=1.2, **kw):
        """Build using a voxel-skeleton centreline (handles arches + branches).
        Uses the main (longest) inlet->outlet path."""
        from ..project.skeleton import skeleton_centrelines
        paths = skeleton_centrelines(wall_path, inlet_stl, outlet_paths, pitch=pitch)
        if not paths:
            raise ValueError("skeleton produced no centreline")
        return cls(wall_path, inlet_stl=inlet_stl, centreline=paths[0], **kw)

    def build(self) -> MultiBlockMesh:
        m, diag = ogrid_from_stl(
            self.path, ns=self.ns, NT=self.NT, NR=self.NR,
            core_frac=self.core_frac, wall_ratio=self.wall_ratio,
            inlet_stl=self.inlet_stl, step=self.step, centreline=self.centreline,
            return_diag=True,
        )
        self.diag = diag
        if diag["branch_stations"] > 0 and not self.allow_branches:
            raise NotSingleLumen(
                f"{diag['branch_stations']}/{diag['ns']} stations have a branch "
                f"(section with >1 loop). This geometry is a junction/tree, which "
                f"the single-lumen O-grid sweep does not support. Only the main trunk "
                f"can be meshed (pass allow_branches=True to mesh the trunk anyway)."
            )
        for k in range(4):
            m.blocks[f"p{k}"].set_face(FaceKey.I1, patch="wall")
        for name in _BLOCKS:
            m.blocks[name].set_face(FaceKey.K0, patch="inlet")
            m.blocks[name].set_face(FaceKey.K1, patch="outlet")
        m.assemble()
        return m
