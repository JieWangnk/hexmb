"""O-grid (butterfly) cross-section and its 3D swept assembly.

A butterfly cross-section is one square core block + 4 petal blocks. The core
block edge subdivisions are forced equal to the petal circumferential
subdivisions (ncore = Q) so the assembly is conforming. Sweeping the
cross-section along a stack of boundary rings gives a 5-block 3D O-grid with no
singular polar axis. Pipe, U-bend, aorta and LV are all this one operation with
different rings + per-station frames.
"""
from __future__ import annotations

import numpy as np

from ..core.block import Block
from ..core.multiblock import MultiBlockMesh
from ..core.topology import FaceKey

# 8 fixed joins for the butterfly (petal inner edge -> core edge; petal seam -> petal seam).
OGRID_JOINS = [
    ("p0", FaceKey.I0, "core", FaceKey.J0),
    ("p1", FaceKey.I0, "core", FaceKey.I1),
    ("p2", FaceKey.I0, "core", FaceKey.J1),
    ("p3", FaceKey.I0, "core", FaceKey.I0),
    ("p0", FaceKey.J1, "p1", FaceKey.J0),
    ("p1", FaceKey.J1, "p2", FaceKey.J0),
    ("p2", FaceKey.J1, "p3", FaceKey.J0),
    ("p3", FaceKey.J1, "p0", FaceKey.J0),
]


def _plane_project(pts, c, e1, e2):
    d = pts - c
    return np.stack([d @ e1, d @ e2], axis=-1)


def graded_radial(NR: int, wall_ratio: float = 1.0) -> np.ndarray:
    """Radial node positions in [0,1] (r=0 inner/core edge, r=1 outer/wall).
    `wall_ratio` = size of the cell at the wall relative to the cell at the core
    edge. wall_ratio < 1 clusters cells at the wall (a boundary layer);
    wall_ratio = 1 is uniform."""
    if NR < 2:
        raise ValueError("NR must be >= 2")
    if abs(wall_ratio - 1.0) < 1e-9:
        return np.linspace(0.0, 1.0, NR)
    ncell = NR - 1
    g = wall_ratio ** (1.0 / (ncell - 1)) if ncell > 1 else 1.0  # per-cell size ratio
    sizes = g ** np.arange(ncell)
    pos = np.concatenate([[0.0], np.cumsum(sizes)])
    return pos / pos[-1]


def add_ogrid_tube(mesh: MultiBlockMesh, rings, e1s, e2s, NR: int = 6, core_frac: float = 0.42,
                   radial_dist=None, prefix: str = "") -> MultiBlockMesh:
    """Add one O-grid tube (5 prefixed blocks + 8 joins) to an existing mesh, so
    several tubes can live in one MultiBlockMesh (e.g. a bifurcation tree)."""
    rings = np.asarray(rings, float)
    ns, NT, _ = rings.shape
    if NT % 4 != 0:
        raise ValueError(f"NT={NT} must be divisible by 4")
    Q = NT // 4
    e1s = np.asarray(e1s, float)
    e2s = np.asarray(e2s, float)

    def nm(x):
        return f"{prefix}{x}"

    core = Block(nm("core"), (Q + 1, Q + 1, ns))
    petals = [Block(nm(f"p{k}"), (NR, Q + 1, ns)) for k in range(4)]

    uu = (np.arange(Q + 1) / Q)[:, None]
    vv = (np.arange(Q + 1) / Q)[None, :]
    if radial_dist is None:
        radial_dist = np.linspace(0.0, 1.0, NR)
    radial_dist = np.asarray(radial_dist, float)
    if radial_dist.shape != (NR,):
        raise ValueError(f"radial_dist must have shape ({NR},)")
    rr = radial_dist[:, None]

    for s in range(ns):
        ring = rings[s]
        c = ring.mean(axis=0)
        e1, e2 = e1s[s], e2s[s]
        S = np.empty((4, 3))
        for k in range(4):
            corner = ring[(k * Q) % NT]
            p2 = _plane_project(corner, c, e1, e2)
            S[k] = c + core_frac * (p2[0] * e1 + p2[1] * e2)
        core3d = (
            S[0] * (1 - uu)[..., None] * (1 - vv)[..., None]
            + S[1] * uu[..., None] * (1 - vv)[..., None]
            + S[2] * uu[..., None] * vv[..., None]
            + S[3] * (1 - uu)[..., None] * vv[..., None]
        )
        core.nodes[:, :, s, :] = core3d
        for k in range(4):
            a0 = k * Q
            idx = [(a0 + t) % NT for t in range(Q + 1)]
            outer = ring[idx]
            inner = (1 - vv.T) * S[k] + vv.T * S[(k + 1) % 4]
            pet = (1 - rr)[..., None] * inner[None, :, :] + rr[..., None] * outer[None, :, :]
            petals[k].nodes[:, :, s, :] = pet

    mesh.add_block(core)
    for p in petals:
        mesh.add_block(p)
    for (ba, fa, bb, fb) in OGRID_JOINS:
        mesh.join(nm(ba), fa, nm(bb), fb)
    return mesh


def build_ogrid_from_rings(rings, e1s, e2s, NR: int = 6, core_frac: float = 0.42,
                           radial_dist=None) -> MultiBlockMesh:
    """Sweep a butterfly cross-section along a stack of boundary rings.

    rings: (ns, NT, 3) boundary ring at each station (NT divisible by 4).
    e1s, e2s: (ns, 3) in-plane frame at each station (used to place the core
              square; for a bending path use rotation-minimizing frames).
    Returns a MultiBlockMesh with blocks {core,p0,p1,p2,p3} + OGRID_JOINS (not
    assembled, no patches). Block axes: core (u,v,station); petals
    (r=inner..outer, c=circumferential, station).
    """
    m = MultiBlockMesh()
    return add_ogrid_tube(m, rings, e1s, e2s, NR=NR, core_frac=core_frac, radial_dist=radial_dist)


def build_lv_ogrid(grid, axis, e1, e2, NR: int = 6, core_frac: float = 0.42,
                   wall_ratio: float = 1.0) -> MultiBlockMesh:
    """LV special case: rings = surface levels, a single fixed in-plane frame
    (e1,e2) for every level (the LV long axis is nearly straight, so a fixed
    frame avoids per-level twist).

    `wall_ratio` < 1 clusters the petal cells at the endocardial wall (a
    boundary layer); 1 is uniform. NOTE: a wall layer thins the near-wall cells,
    which raises their aspect ratio and their risk of inverting under a large
    moving-mesh morph -- gate quality across the cycle before relying on it.
    """
    grid = np.asarray(grid, float)
    ns = grid.shape[0]
    e1s = np.tile(np.asarray(e1, float), (ns, 1))
    e2s = np.tile(np.asarray(e2, float), (ns, 1))
    rd = graded_radial(NR, wall_ratio)
    return build_ogrid_from_rings(grid, e1s, e2s, NR=NR, core_frac=core_frac, radial_dist=rd)
