"""Conformal split-hex 2:1 transition + telescoping O-grid (one conservative region).

This ports snappyHexMesh's ``hexRef8`` octree-refinement idea to a *structured*
O-grid.  Where a COARSE O-grid segment meets a 2x-FINE one, each coarse cell's
interface face is split into 2x2 sub-faces so it matches the four fine cells
that abut it.  The coarse cell becomes a **split-hex polyhedron**; the fine
cells stay hexes; the interface is **conformal** (the cells share faces), so
flux is conserved across it.

Why it matters
--------------
The alternative coupling for a coarse<->fine O-grid join is an AMI /
``nonConformalCyclic`` interface, which *interpolates* fluxes and is therefore
**non-conservative** for the pressure/continuity equation.  On a moving left
ventricle that stalls the diastolic (filling) pressure solve.  A conformal
split-hex join keeps one conservative region, so the moving filling runs.

What you build with it
-----------------------
A **telescoping O-grid**: a swept O-grid whose ends are coarse and whose middle
is fine (coarse cap - fine body - coarse cap), in ONE region, with no
interpolation coupling.  This breaks the single-resolution cell-count ceiling of
a plain morphing O-grid: refine only where you need it.

The exactness trick
-------------------
The fine ring is the coarse ring **subdivided** (each edge gets a midpoint), and
the O-grid cross-section is built with a *bilinear* core and *linear* petals.
Both are affine, so the coarse nodes are an exact subset of the fine nodes and
the geometric split points (edge midpoints, face centre) coincide with the fine
nodes.  That is what makes the join watertight.

Requirements
------------
* fine ring size ``NT_f = 2 * NT_c`` and fine radial ``NR_f = 2 * NR_c - 1``;
* ``NT`` divisible by 4 (O-grid has four petals).

Known limits
------------
Split cells inside strongly-curved petals carry residual skewness (~5-6) and
non-orthogonality (~75-80 deg); skewness-corrected FV schemes tolerate them but
they are real.  Refining the coarse ring refines the apex ring too, which
pinches at end-systole on a contracting ventricle; refine the body axially or
use :func:`build_multilevel_telescoping_ogrid` to keep the apex coarse.

This is the mesh route behind the validated moving-ventricle CFD runs (the
AMI-coupled :mod:`hexmb.templates.telescoping_lv` is the other route; its
interpolated coupling is not flux-conservative and stalls the filling solve).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

__all__ = [
    "PolyMesh",
    "ogrid_section",
    "subdivide_ring",
    "emit_ogrid_cell",
    "emit_ogrid_layer",
    "build_telescoping_ogrid",
    "build_multilevel_telescoping_ogrid",
]


# --------------------------------------------------------------------------- #
#  Polyhedral mesh container + OpenFOAM writer                                 #
# --------------------------------------------------------------------------- #
class PolyMesh:
    """A polyhedral mesh built face-list by face-list, with coordinate dedup.

    Cells are added as a list of faces, each face a list of xyz points.  Points
    are de-duplicated by rounded coordinate so shared faces between neighbouring
    cells collapse to one internal face.  ``write`` emits an OpenFOAM
    ``constant/polyMesh`` with owner<neighbour ordering and boundary faces
    grouped per patch.

    Unlike the all-hex :class:`hexmb.core.multiblock.MultiBlockMesh`, this holds
    general polyhedra, which is what the split-hex transition cells are.
    """

    def __init__(self, round_dp: int = 4):
        self._round = round_dp
        self._index: dict[tuple, int] = {}
        self.P: list[tuple] = []
        self.cells: list[list[list[int]]] = []
        self.cell_patch: list[dict[int, str]] = []
        self._classify = None          # default boundary classifier (write may override)
        self.patches: list[str] = []   # patch names in the order the last write() used
        self.frame = None              # optional (axis, e1, e2, mu) stashed by a template
        self.top_faces: set = set()    # face keys on the last axial section (valve plane), tagged at emit

    @property
    def points(self) -> np.ndarray:
        """Point coordinates as an ``(n_points, 3)`` array (mm)."""
        return np.asarray(self.P, float)

    @property
    def n_hex(self) -> int:
        """Number of plain hexahedra (6 faces)."""
        return sum(1 for c in self.cells if len(c) == 6)

    @property
    def n_split(self) -> int:
        """Number of split-hex transition polyhedra (9 faces)."""
        return sum(1 for c in self.cells if len(c) != 6)

    def same_topology(self, other: "PolyMesh") -> bool:
        """True when both meshes have identical cell/face/point connectivity (the
        condition for using them as phases of one moving mesh)."""
        return len(self.P) == len(other.P) and self.cells == other.cells

    def pid(self, xyz) -> int:
        k = tuple(round(float(v), self._round) for v in xyz)
        i = self._index.get(k)
        if i is None:
            i = len(self.P)
            self._index[k] = i
            self.P.append(k)
        return i

    def add(self, faces, patch_faces=None) -> None:
        """Add one cell.  ``faces`` is a list of faces (each a list of xyz).
        ``patch_faces`` optionally tags boundary faces of THIS cell by their
        local index: ``{local_face_index: patch_name}``."""
        self.cells.append([[self.pid(p) for p in f] for f in faces])
        self.cell_patch.append(patch_faces or {})
        self._ft_key = None

    # -- geometry helpers --------------------------------------------------- #
    def _face_table(self):
        """Flat arrays over every (cell, face): ``cid`` (n_faces,), and per face
        length a dict ``{k: (rows, vid (m, k))}``.  Cached until cells change."""
        key = (len(self.cells), len(self.P))
        if getattr(self, "_ft_key", None) == key:
            return self._ft
        cid, by_len = [], {}
        for ci, c in enumerate(self.cells):
            for f in c:
                by_len.setdefault(len(f), ([], []))
                by_len[len(f)][0].append(ci)
                by_len[len(f)][1].append(f)
        ft = {k: (np.asarray(r, int), np.asarray(v, int)) for k, (r, v) in by_len.items()}
        self._ft, self._ft_key = ft, key
        return ft

    def _cell_centres(self) -> np.ndarray:
        P = np.asarray(self.P, float)
        pairs = np.concatenate([np.stack([np.repeat(cid, v.shape[1]), v.ravel()], 1)
                                for cid, v in self._face_table().values()])
        pairs = np.unique(pairs, axis=0)                      # (cell, unique vertex)
        cc = np.zeros((len(self.cells), 3)); cnt = np.zeros(len(self.cells))
        np.add.at(cc, pairs[:, 0], P[pairs[:, 1]]); np.add.at(cnt, pairs[:, 0], 1.0)
        return cc / cnt[:, None]

    def _face_normal(self, f) -> np.ndarray:
        v = [np.asarray(self.P[i], float) for i in f]
        n = np.zeros(3)
        for i in range(len(v)):
            n += np.cross(v[i], v[(i + 1) % len(v)])
        return n / (np.linalg.norm(n) + 1e-30)

    def _face_vectors(self):
        """Per (cell, face) entry in face-table order: cell id, face centroid, area
        vector (right-hand rule on the stored vertex order)."""
        P = np.asarray(self.P, float)
        cids, fcs, sfs = [], [], []
        for k, (cid, vid) in self._face_table().items():
            pts = P[vid]                                      # (m, k, 3)
            fc = pts.mean(1)
            s = np.zeros((len(cid), 3))
            for i in range(k):
                s += np.cross(pts[:, i] - fc, pts[:, (i + 1) % k] - fc)
            cids.append(cid); fcs.append(fc); sfs.append(0.5 * s)
        return np.concatenate(cids), np.concatenate(fcs), np.concatenate(sfs)

    def pyramid_volumes(self):
        """Signed volume of the pyramid (cell centre, face) for every (cell, face)
        entry, using the stored face orientation (outward for a right-handed cell
        as emitted by :func:`emit_ogrid_cell`).  A negative value means the face
        is folded past the cell centre or the cell is inverted; OpenFOAM's
        checkMesh fails such cells ("open cells" / negative pyramids) even when
        the unsigned volume looks fine.  Returns ``(cell_ids, pyramid_volumes)``."""
        cid, fc, sf = self._face_vectors()
        cc = self._cell_centres()
        return cid, np.einsum("ij,ij->i", fc - cc[cid], sf) / 3.0

    def cell_volumes(self) -> np.ndarray:
        """Signed cell volumes: the sum of the signed pyramid volumes, so an
        inverted cell is negative and a folded cell is less than its convex hull."""
        cid, pv = self.pyramid_volumes()
        vols = np.zeros(len(self.cells))
        np.add.at(vols, cid, pv)
        return vols

    def min_pyramid_per_cell(self) -> np.ndarray:
        """Smallest signed pyramid volume of each cell (the hard validity gate:
        every entry must be > 0)."""
        cid, pv = self.pyramid_volumes()
        out = np.full(len(self.cells), np.inf)
        np.minimum.at(out, cid, pv)
        return out

    def face_areas(self) -> np.ndarray:
        """Area of every (cell, face) entry, flattened in face-table order.  A zero
        area means a degenerate face (coincident points), which OpenFOAM rejects
        even when every cell volume is positive.  An internal face appears twice
        here (once per cell); :meth:`n_zero_area_faces` counts each face once."""
        return np.linalg.norm(self._face_vectors()[2], axis=1)

    def n_zero_area_faces(self, tol: float = 1e-9) -> int:
        """Number of distinct faces whose area is at most ``tol`` (mm^2).  A face
        shared by two cells is counted once."""
        areas = np.linalg.norm(self._face_vectors()[2], axis=1)
        keys = set()
        pos = 0
        for k, (cid, vid) in self._face_table().items():
            small = np.flatnonzero(areas[pos:pos + len(cid)] <= tol)
            for i in small:
                keys.add(frozenset(vid[i].tolist()))
            pos += len(cid)
        return len(keys)

    def smooth(self, iters: int = 3, step: float = 0.35, radial_constrain: bool = False,
               axis=None) -> None:
        """Gentle interior-point Laplacian relaxation.

        Boundary points are held fixed; only points interior to the volume move,
        and shared interface nodes move consistently for every cell that owns
        them, so conformity is preserved.  Use it to relax the skew that the
        split-hex transition cells introduce in curved regions.

        ``radial_constrain`` freezes each moving node's **radial distance** from
        the centreline (its move is projected to the tangential+axial plane), so a
        boundary layer (wall-clustered radial spacing) survives the smoothing.
        ``axis`` is the centreline direction; if omitted it is estimated by PCA."""
        P = np.asarray(self.P, float)
        adj: dict[int, set] = defaultdict(set)
        fcount: dict[frozenset, int] = defaultdict(int)
        for c in self.cells:
            for f in c:
                fcount[frozenset(f)] += 1
                for i in range(len(f)):
                    adj[f[i]].add(f[(i + 1) % len(f)])
                    adj[f[i]].add(f[i - 1])
        boundary = {i for c in self.cells for f in c if fcount[frozenset(f)] == 1 for i in f}
        interior = [i for i in range(len(P)) if i not in boundary]
        if radial_constrain:
            mu0 = P.mean(0)
            if axis is None:
                axis = np.linalg.svd(P - mu0, full_matrices=False)[2][0]
            axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
        for _ in range(iters):
            new = P.copy()
            for i in interior:
                nb = list(adj[i])
                if not nb:
                    continue
                target = (1 - step) * P[i] + step * P[np.array(nb)].mean(0)
                if radial_constrain:
                    c = mu0 + np.dot(P[i] - mu0, axis) * axis      # centreline point at this axial pos
                    rvec = P[i] - c
                    rn = np.linalg.norm(rvec)
                    if rn > 1e-9:
                        rhat = rvec / rn
                        move = target - P[i]
                        target = P[i] + (move - np.dot(move, rhat) * rhat)   # drop the radial move
                new[i] = target
            P = new
        self.P = [tuple(p) for p in P]

    # -- OpenFOAM export ---------------------------------------------------- #
    def topology(self, classify=None):
        """Assemble the OpenFOAM face topology without writing it.

        Returns ``(faces, owner, neighbour, patches)``: internal faces first
        (owner < neighbour, sorted by owner then neighbour, each face oriented
        from owner to neighbour), then the boundary faces grouped per patch and
        oriented outward; ``patches`` is an ordered ``{name: (start, n_faces)}``.
        ``classify`` is as in :meth:`write`.  :meth:`write`, the VTK writers and
        :func:`hexmb.services.quality.polymesh_metrics` all work from this.
        """
        classify = classify or self._classify or (lambda fc, nrm: "wall")
        prepare = getattr(classify, "prepare", None)
        if prepare is not None:
            prepare()                       # recompute from the points as they are NOW (a phase may have been refilled)
        P = np.asarray(self.P, float)
        cc = self._cell_centres()

        fmap: dict[frozenset, list] = {}
        face_patch: dict[frozenset, str] = {}
        for ci, (c, pp) in enumerate(zip(self.cells, self.cell_patch)):
            for fi, f in enumerate(c):
                key = frozenset(f)
                fmap.setdefault(key, []).append((ci, f))
                if fi in pp:
                    face_patch[key] = pp[fi]

        internal, boundary = [], []
        for key, occ in fmap.items():
            if len(occ) == 2:
                (a, fa), (b, fb) = occ
                own, nei = (a, b) if a < b else (b, a)
                f = fa if own == a else fb
                if np.dot(self._face_normal(f), cc[nei] - cc[own]) < 0:
                    f = f[::-1]
                internal.append((own, nei, f))
            else:
                if len(occ) > 2:
                    raise ValueError(f"face {sorted(key)} is shared by {len(occ)} cells "
                                     f"(coincident points after rounding?); the mesh is not a valid polyMesh")
                ci, f = occ[0]
                fc = P[f].mean(0)
                nrm = self._face_normal(f)
                if np.dot(nrm, fc - cc[ci]) < 0:
                    f = f[::-1]
                    nrm = -nrm
                if getattr(classify, "uses_top_tag", False):
                    name = face_patch.get(key) or classify(fc, nrm, key in self.top_faces)
                else:
                    name = face_patch.get(key) or classify(fc, nrm)
                boundary.append((ci, f, name))
        internal.sort(key=lambda t: (t[0], t[1]))

        groups: dict[str, list] = {}
        for ci, f, name in boundary:
            groups.setdefault(name, []).append((ci, f))
        faces = [f for _, _, f in internal] + [f for pl in groups.values() for _, f in pl]
        owner = np.array([o for o, _, _ in internal] + [ci for pl in groups.values() for ci, _ in pl], int)
        neigh = np.array([n for _, n, _ in internal], int)
        patches: dict[str, tuple] = {}
        start = len(internal)
        for name, fl in groups.items():
            patches[name] = (start, len(fl))
            start += len(fl)
        return faces, owner, neigh, patches

    def write(self, case_dir, classify=None, scale: float = 1e-3):
        """Write ``constant/polyMesh``.

        Parameters
        ----------
        case_dir : path-like
            Case directory; ``constant/polyMesh`` is created under it.
        classify : callable, optional
            ``classify(face_centroid_mm, outward_normal) -> patch_name`` for
            boundary faces not already tagged via ``add(..., patch_faces=...)``.
            Defaults to everything ``"wall"``.
        scale : float
            Applied to point coordinates on write (mm -> m by default).

        Returns ``(n_cells, n_internal_faces, n_points)``.
        """
        case_dir = Path(case_dir)
        P = np.asarray(self.P, float)
        faces, owner, neigh, patches = self.topology(classify)
        n_internal = len(neigh)

        pm = case_dir / "constant" / "polyMesh"
        pm.mkdir(parents=True, exist_ok=True)
        hdr = (lambda cls, obj: f'FoamFile{{version 2.0;format ascii;class {cls};'
                                f'location "constant/polyMesh";object {obj};}}\n')
        (pm / "points").write_text(
            hdr("vectorField", "points") + f"{len(P)}\n(\n"
            + "\n".join(f"({p[0]:.7g} {p[1]:.7g} {p[2]:.7g})" for p in P * scale) + "\n)\n")
        (pm / "faces").write_text(
            hdr("faceList", "faces") + f"{len(faces)}\n(\n"
            + "\n".join(f"{len(f)}({' '.join(map(str, f))})" for f in faces) + "\n)\n")
        (pm / "owner").write_text(
            hdr("labelList", "owner") + f"{len(owner)}\n(\n" + "\n".join(map(str, owner)) + "\n)\n")
        (pm / "neighbour").write_text(
            hdr("labelList", "neighbour") + f"{len(neigh)}\n(\n" + "\n".join(map(str, neigh)) + "\n)\n")
        blk = ""
        for name, (start, n) in patches.items():
            typ = "wall" if name == "wall" else "patch"
            blk += (f"    {name}\n    {{\n        type {typ};\n"
                    f"        nFaces {n};\n        startFace {start};\n    }}\n")
        (pm / "boundary").write_text(
            hdr("polyBoundaryMesh", "boundary") + f"{len(patches)}\n(\n{blk})\n")
        self.patches = list(patches.keys())
        return len(self.cells), n_internal, len(P)


# --------------------------------------------------------------------------- #
#  O-grid cross-section (exactly subdivisible)                                 #
# --------------------------------------------------------------------------- #
def ogrid_section(ring, core_frac: float, radial):
    """O-grid (butterfly) cross-section for a boundary ``ring`` of shape (NT,3).

    Returns ``(core, petals)`` where ``core`` has shape ``(Q+1, Q+1, 3)`` and
    ``petals`` is a list of four arrays of shape ``(NR, Q+1, 3)`` with ``Q =
    NT//4`` and ``NR = len(radial)``.  The core is **bilinear** and the petals
    are **linear** in (radius, tangent), so subdividing the ring subdivides the
    section exactly -- the property the split-hex join relies on.

    ``radial`` is the 1-D list of normalised radial positions (0 at core edge, 1
    at wall); pass e.g. ``np.linspace(0, 1, NR)``.
    """
    ring = np.asarray(ring, float)
    NT = len(ring)
    if NT % 4:
        raise ValueError(f"ring size NT={NT} must be divisible by 4 (four petals)")
    Q = NT // 4
    NR = len(radial)
    c = ring.mean(0)
    S = [c + core_frac * (ring[(k * Q) % NT] - c) for k in range(4)]      # 4 core corners
    uu = (np.arange(Q + 1) / Q)[:, None, None]
    vv = (np.arange(Q + 1) / Q)[None, :, None]
    core = (S[0] * (1 - uu) * (1 - vv) + S[1] * uu * (1 - vv)
            + S[2] * uu * vv + S[3] * (1 - uu) * vv)
    petals = []
    for k in range(4):
        pet = np.zeros((NR, Q + 1, 3))
        for t in range(Q + 1):
            outer = ring[(k * Q + t) % NT]
            inner = (1 - t / Q) * S[k] + (t / Q) * S[(k + 1) % 4]
            for r in range(NR):
                pet[r, t] = (1 - radial[r]) * inner + radial[r] * outer
        petals.append(pet)
    return core, petals


def subdivide_ring(coarse_ring):
    """Return the 2x-fine ring: the coarse nodes plus one midpoint per edge.

    Shape goes ``(NT_c, 3)`` -> ``(2*NT_c, 3)`` with even indices the coarse
    nodes and odd indices the edge midpoints, so coarse nodes are an exact subset
    of the fine ones.
    """
    cr = np.asarray(coarse_ring, float)
    n = len(cr)
    fr = np.zeros((2 * n, 3))
    fr[0::2] = cr
    fr[1::2] = 0.5 * (cr + np.roll(cr, -1, axis=0))
    return fr


# --------------------------------------------------------------------------- #
#  Split-hex cell primitive                                                    #
# --------------------------------------------------------------------------- #
def emit_ogrid_cell(mesh: PolyMesh, hexc, top_split: bool = False, bot_split: bool = False,
                    tag_top: str | None = None):
    """Emit one cell into ``mesh``: a plain hex, or a split-hex if one of its
    z-faces meets a 2x-finer neighbour.

    ``hexc = [b0, b1, b2, b3, t0, t1, t2, t3]`` -- the bottom quad then the top
    quad, in the usual hex vertex order.  When ``top_split`` (or ``bot_split``)
    is set, the fine-facing quad is split into a 2x2 grid using its edge
    midpoints and centre; the four side faces of the coarse cell become
    pentagons, and the coarse cell is a split-hex polyhedron.

    Every face is emitted with an OUTWARD orientation for a right-handed cell
    (bottom quad counter-clockwise seen from the top), so the signed pyramid
    volumes in :meth:`PolyMesh.pyramid_volumes` are an absolute inversion /
    fold test.  ``tag_top`` records the top face(s) in ``mesh.top_faces``.
    """
    b, t = hexc[:4], hexc[4:]
    if not (top_split or bot_split):
        faces = [b[::-1], t,
                 [b[0], b[1], t[1], t[0]], [b[1], b[2], t[2], t[1]],
                 [b[2], b[3], t[3], t[2]], [b[3], b[0], t[0], t[3]]]
        mesh.add(faces)
        if tag_top:
            mesh.top_faces.add(frozenset(mesh.cells[-1][1]))
        return

    def split(q):
        A, B, C, D = (np.asarray(x, float) for x in q)
        AB, BC, CD, DA = (A + B) / 2, (B + C) / 2, (C + D) / 2, (D + A) / 2
        O = (A + B + C + D) / 4
        subs = [[A, AB, O, DA], [AB, B, BC, O], [O, BC, C, CD], [DA, O, CD, D]]
        return (AB, BC, CD, DA, O), subs

    if top_split and bot_split:
        # a one-layer coarse zone between two finer zones: both end faces are split and
        # every side face carries the mid-edge node of both ends (six vertices)
        (tAB, tBC, tCD, tDA, _), tsubs = split(t)
        (bAB, bBC, bCD, bDA, _), bsubs = split(b)
        faces = tsubs + [q[::-1] for q in bsubs] + [
            [b[0], bAB, b[1], t[1], tAB, t[0]], [b[1], bBC, b[2], t[2], tBC, t[1]],
            [b[2], bCD, b[3], t[3], tCD, t[2]], [b[3], bDA, b[0], t[0], tDA, t[3]]]
        top_local = range(0, 4)
    elif top_split:
        (AB, BC, CD, DA, O), subs = split(t)
        faces = [b[::-1]] + subs + [
            [b[0], b[1], t[1], AB, t[0]], [b[1], b[2], t[2], BC, t[1]],
            [b[2], b[3], t[3], CD, t[2]], [b[3], b[0], t[0], DA, t[3]]]
        top_local = range(1, 5)
    else:
        (AB, BC, CD, DA, O), subs = split(b)
        faces = [t] + [q[::-1] for q in subs] + [
            [b[0], AB, b[1], t[1], t[0]], [b[1], BC, b[2], t[2], t[1]],
            [b[2], CD, b[3], t[3], t[2]], [b[3], DA, b[0], t[0], t[3]]]
        top_local = [0]
    mesh.add(faces)
    if tag_top:
        for lf in top_local:
            mesh.top_faces.add(frozenset(mesh.cells[-1][lf]))


def emit_ogrid_layer(mesh: PolyMesh, sec0, sec1, split_top: bool = False, split_bot: bool = False,
                     tag_top: str | None = None):
    """Emit one axial layer of O-grid cells between two cross-sections
    ``sec0`` and ``sec1`` (each an ``(core, petals)`` pair from
    :func:`ogrid_section`).  Cells that face a finer neighbour across the top
    (``split_top``) or bottom (``split_bot``) plane are emitted as split-hex.
    """
    (c0, p0), (c1, p1) = sec0, sec1
    Q = c0.shape[0] - 1
    NR = p0[0].shape[0]
    for i in range(Q):                                        # core cells
        for j in range(Q):
            hexc = [c0[i, j], c0[i + 1, j], c0[i + 1, j + 1], c0[i, j + 1],
                    c1[i, j], c1[i + 1, j], c1[i + 1, j + 1], c1[i, j + 1]]
            emit_ogrid_cell(mesh, hexc, split_top, split_bot, tag_top)
    for k in range(4):                                        # petal cells
        for r in range(NR - 1):
            for t in range(Q):
                hexc = [p0[k][r, t], p0[k][r + 1, t], p0[k][r + 1, t + 1], p0[k][r, t + 1],
                        p1[k][r, t], p1[k][r + 1, t], p1[k][r + 1, t + 1], p1[k][r, t + 1]]
                emit_ogrid_cell(mesh, hexc, split_top, split_bot, tag_top)


# --------------------------------------------------------------------------- #
#  Telescoping O-grid builder                                                  #
# --------------------------------------------------------------------------- #
def graded_radial(NR, wall_ratio=1.0):
    """Radial node positions in [0,1], from the O-grid core edge (0) to the wall (1).

    ``wall_ratio < 1`` clusters cells toward the **wall** (a boundary layer): the
    wall interval is ``wall_ratio`` times the core-edge interval, spaced
    geometrically. ``wall_ratio == 1`` is uniform. This is what makes the mesh
    fine where the flow gradients are (near the wall / WSS), not in the core.
    """
    if NR < 3 or wall_ratio == 1.0:
        return np.linspace(0.0, 1.0, NR)
    n = NR - 1                                   # number of intervals
    g = wall_ratio ** (1.0 / (n - 1))            # geometric ratio per interval
    d = g ** np.arange(n)                        # interval sizes, shrinking toward the wall
    r = np.concatenate([[0.0], np.cumsum(d)])
    return r / r[-1]


def _subdivide_1d(x):
    """Insert the midpoint of every interval -> the 2:1 fine distribution.
    Keeps grading, and keeps the coarse nodes an exact subset (split-hex exactness)."""
    y = np.zeros(2 * len(x) - 1)
    y[0::2] = x
    y[1::2] = 0.5 * (x[:-1] + x[1:])
    return y


def build_telescoping_ogrid(ring_at, *, NT_c=24, NR_c=4, core_frac=0.42, wall_ratio=1.0,
                            apex=(0.0, 0.24, 10), body=(0.24, 0.80, 30),
                            base=(0.80, 1.0, 12), classify=None, smooth=True, smooth_iters=3):
    """Build a telescoping O-grid: coarse cap - fine body - coarse cap, joined
    by conformal split-hex transitions, as ONE region.

    Parameters
    ----------
    ring_at : callable
        ``ring_at(frac) -> (M, 3)`` boundary contour at normalised axial
        position ``frac`` in [0, 1].  It is resampled internally to ``NT_c``
        (coarse) or ``2*NT_c`` (fine).
    NT_c, NR_c : int
        Coarse circumferential / radial resolution.  The fine body is ``2*NT_c``
        circumferential and ``2*NR_c-1`` radial.
    core_frac : float
        O-grid core size as a fraction of the local radius.
    wall_ratio : float
        Boundary-layer clustering (``< 1`` clusters cells toward the wall; ``1``
        uniform). Use it so the mesh is fine at the wall (WSS) rather than in the
        low-gradient core. Grading is preserved through the split-hex transition.
    apex, body, base : (start_frac, end_frac, n_levels)
        The three axial zones.  ``apex``/``base`` are coarse; ``body`` is fine.
    classify : callable, optional
        Boundary-face patch classifier passed to :meth:`PolyMesh.write`.
    smooth : bool
        Relax the transition skew with a light interior Laplacian.

    Returns the :class:`PolyMesh`.
    """
    from scipy.signal import resample

    rad_c = graded_radial(NR_c, wall_ratio)          # coarse radial (wall-clustered if wall_ratio<1)
    rad_f = _subdivide_1d(rad_c)                      # fine = coarse subdivided -> split-hex exact

    # normalise the contour direction once: counter-clockwise about the apex->base
    # direction, so every cell is emitted right-handed whatever the surface reader's
    # angular convention was (a clockwise grid would otherwise mirror every cell)
    r_lo, r_hi = np.asarray(ring_at(apex[0]), float), np.asarray(ring_at(base[1]), float)
    ax = r_hi.mean(0) - r_lo.mean(0); ax /= (np.linalg.norm(ax) + 1e-30)
    rm = np.asarray(ring_at(0.5 * (body[0] + body[1])), float); d = rm - rm.mean(0)
    flip_ring = float(np.cross(d, np.roll(d, -1, axis=0)).sum(0) @ ax) < 0

    def cring(f):
        r = resample(ring_at(f), NT_c, axis=0)
        return r[::-1] if flip_ring else r

    def fring(f):
        return subdivide_ring(cring(f))

    def csec(f):
        return ogrid_section(cring(f), core_frac, rad_c)

    def fsec(f):
        return ogrid_section(fring(f), core_frac, rad_f)

    mesh = PolyMesh()

    if not (apex[0] < apex[1] <= body[0] < body[1] <= base[0] < base[1]):
        raise ValueError(f"axial zones must advance: apex {apex[:2]}, body {body[:2]}, base {base[:2]}")

    def add_segment(fracs, sec_fn, split_top=False, split_bot=False, last=False):
        secs = [sec_fn(f) for f in fracs]
        for L in range(len(fracs) - 1):
            emit_ogrid_layer(mesh, secs[L], secs[L + 1],
                             split_top=(split_top and L == len(fracs) - 2),
                             split_bot=(split_bot and L == 0),
                             tag_top="top" if (last and L == len(fracs) - 2) else None)

    add_segment(np.linspace(apex[0], apex[1], apex[2]), csec, split_top=True)
    add_segment(np.linspace(body[0], body[1], body[2]), fsec)
    add_segment(np.linspace(base[0], base[1], base[2]), csec, split_bot=True, last=True)
    if smooth:
        # always constrain the smoother radially: it relaxes the transition skew
        # without flattening the graded near-wall spacing (a boundary layer), and it
        # is stable at high NR (the unconstrained Laplacian blows up there)
        mesh.smooth(iters=smooth_iters, radial_constrain=True)
    mesh._classify = classify           # stash for convenience; write() takes it too
    return mesh


def build_multilevel_telescoping_ogrid(ring_at, *, NT_c=24, NR_c=4, core_frac=0.42,
                                       wall_ratio=1.0, zones, classify=None,
                                       smooth=True, smooth_iters=3):
    """Telescoping O-grid with an arbitrary number of refinement levels.

    Where :func:`build_telescoping_ogrid` has one fine body between two coarse
    caps (a single 2:1 step on each side), this takes a list of axial ``zones``
    at chosen refinement levels and joins each pair of neighbouring zones with a
    conformal split-hex transition.  It exists for the moving left ventricle: the
    apex ring must stay **coarse** (a fine apex ring pinches to near-zero cell
    size when the cavity contracts at end-systole), while the mid-body -- which
    keeps a large radius through the whole cycle -- can be refined circumferentially
    for the diastolic vortex.  So you nest the resolution: coarse apex cap, then a
    step up to the body, then a further step up in the mid-body, then back down.

    Parameters
    ----------
    ring_at : callable
        ``ring_at(frac) -> (M, 3)`` boundary contour at axial position ``frac``.
    NT_c, NR_c : int
        Level-0 (coarsest) circumferential / radial resolution.  A zone at level
        ``L`` has ``NT_c * 2**L`` circumferential cells and its radial line is the
        level-0 line subdivided ``L`` times, so the coarse nodes stay an exact
        subset of the fine ones (the split-hex exactness the join relies on).
    zones : list of (start_frac, end_frac, n_levels, L)
        Ordered apex->base.  ``n_levels`` axial sections span ``[start, end]``;
        ``L`` is the refinement level of the zone.  **Neighbouring zones must
        differ by at most one level** (a 2:1 face split); a jump of two or more is
        rejected.  Put the apex and base zones at ``L=0``.
    core_frac, wall_ratio, classify, smooth, smooth_iters
        As in :func:`build_telescoping_ogrid`.

    Returns the :class:`PolyMesh`.
    """
    from scipy.signal import resample

    levels = [z[3] for z in zones]
    for a, b in zip(levels, levels[1:]):
        if abs(a - b) > 1:
            raise ValueError(f"neighbouring zones must differ by <=1 refine level; got {a} and {b}")
    maxL = max(levels)
    rad = [graded_radial(NR_c, wall_ratio)]              # radial line per level (subdivided)
    for _ in range(maxL):
        rad.append(_subdivide_1d(rad[-1]))

    def sec_at(f, L):
        ring = resample(ring_at(f), NT_c, axis=0)        # level-0 ring
        for _ in range(L):
            ring = subdivide_ring(ring)                  # subdivide L times -> exact fine ring
        return ogrid_section(ring, core_frac, rad[L])

    mesh = PolyMesh()
    for zi, (s, e, nlev, L) in enumerate(zones):
        fracs = np.linspace(s, e, nlev)
        secs = [sec_at(f, L) for f in fracs]
        # split toward whichever neighbour is finer (the split lives on the coarse side)
        split_bot = zi > 0 and L < levels[zi - 1]
        split_top = zi < len(zones) - 1 and L < levels[zi + 1]
        for Ln in range(len(fracs) - 1):
            emit_ogrid_layer(mesh, secs[Ln], secs[Ln + 1],
                             split_top=(split_top and Ln == len(fracs) - 2),
                             split_bot=(split_bot and Ln == 0))
    if smooth:
        mesh.smooth(iters=smooth_iters, radial_constrain=True)
    mesh._classify = classify
    return mesh
