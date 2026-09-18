"""MultiBlockMesh: assemble blocks into one conforming all-hex mesh."""
from __future__ import annotations

import numpy as np

from .block import Block
from .dedup import DSU
from .topology import FaceKey, match_orientation


def _apply_named(arr: np.ndarray, name: str) -> np.ndarray:
    """Apply one of the 8 dihedral transforms (see topology.orient_transforms)."""
    if name.startswith("T"):
        arr = np.swapaxes(arr, 0, 1)
        name = name[1:].lstrip(".")
    if name == "identity" or name == "":
        return arr
    if name == "flipa":
        return arr[::-1]
    if name == "flipb":
        return arr[:, ::-1]
    if name == "flipab":
        return arr[::-1, ::-1]
    raise ValueError(f"unknown transform {name!r}")


class SubdivisionMismatch(ValueError):
    pass


class MultiBlockMesh:
    def __init__(self, tol: float = 1e-6):
        self.blocks: dict[str, Block] = {}
        self.joins: list[tuple] = []  # (bA, fA, bB, fB)
        self.tol = tol
        self.points: np.ndarray | None = None
        self.hexes: np.ndarray | None = None
        self._base: dict[str, int] = {}
        # predicate patches: list of (block, facekey, fn(centroid)->name)
        self.predicate_faces: list[tuple] = []

    # -- construction ----------------------------------------------------
    def add_block(self, b: Block) -> Block:
        if b.name in self.blocks:
            raise ValueError(f"duplicate block name {b.name}")
        self.blocks[b.name] = b
        return b

    def join(self, block_a, face_a, block_b, face_b) -> None:
        self.joins.append((block_a, FaceKey(face_a), block_b, FaceKey(face_b)))

    def set_predicate_face(self, block, face, fn) -> None:
        """Assign a boundary block-face whose quads are labelled per centroid by
        fn(xyz)->patch name."""
        self.predicate_faces.append((block, FaceKey(face), fn))

    # -- assembly --------------------------------------------------------
    def assemble(self) -> "MultiBlockMesh":
        # 1. slot offsets
        base = {}
        off = 0
        for name, b in self.blocks.items():
            base[name] = off
            off += b.npoints
        total = off
        self._base = base
        dsu = DSU(total)

        # 2. unions across joins
        for (na, fa, nb, fb) in self.joins:
            A, B = self.blocks[na], self.blocks[nb]
            cA = A.face_coords(fa)
            cB = B.face_coords(fb)
            name, tB = match_orientation(cA, cB, self.tol)
            if name is None:
                raise SubdivisionMismatch(
                    f"join {na}.{fa.value} <-> {nb}.{fb.value}: faces do not "
                    f"coincide (shapes {cA.shape[:2]} vs {cB.shape[:2]}) -> crack "
                    f"or wrong subdivision"
                )
            idA = A.face_local_ids(fa) + base[na]
            idB = _apply_named(B.face_local_ids(fb) + base[nb], name)
            dsu.union_arrays(idA, idB)

        # 3. compact labels -> global point ids
        labels = dsu.compact_labels()  # (total,)
        n_point = int(labels.max()) + 1

        # 4. build points, verify coincidence within tol
        psum = np.zeros((n_point, 3))
        pcnt = np.zeros(n_point, np.int64)
        for name, b in self.blocks.items():
            sl = slice(base[name], base[name] + b.npoints)
            lab = labels[sl]
            flat = b.nodes.reshape(-1, 3)
            np.add.at(psum, lab, flat)
            np.add.at(pcnt, lab, 1)
            b.gid = lab.reshape(b.shape)
        points = psum / pcnt[:, None]
        # coincidence check
        for name, b in self.blocks.items():
            sl = slice(base[name], base[name] + b.npoints)
            lab = labels[sl]
            flat = b.nodes.reshape(-1, 3)
            dev = np.linalg.norm(flat - points[lab], axis=1).max()
            if dev > 10 * self.tol + 1e-9:
                raise ValueError(
                    f"block {name}: merged nodes disagree by {dev:.3g} mm "
                    f"(> tol) -> inconsistent shared geometry"
                )
        self.points = points

        # 5. build hexes, normalise handedness per block
        from ..services.quality import scaled_jac

        allhex = []
        for name, b in self.blocks.items():
            cl = b.cells_local() + base[name]
            gh = labels[cl]  # (ncell,8) global
            sj = scaled_jac(points, gh)
            if np.median(sj) <= 0:
                gh = gh[:, [4, 5, 6, 7, 0, 1, 2, 3]]  # reverse k-layer -> flip orientation
            allhex.append(gh)
        self.hexes = np.vstack(allhex)
        return self

    # -- patches ---------------------------------------------------------
    def _interface_faces(self) -> set:
        s = set()
        for (na, fa, nb, fb) in self.joins:
            s.add((na, fa))
            s.add((nb, fb))
        return s

    def _face_quads(self, block: Block, key: FaceKey):
        """Yield (quad_global_ids (4,), centroid (3,)) for each cell-face on
        block face `key`."""
        gidface = _face_from_grid(block.gid, key)  # (na,nb)
        na, nb = gidface.shape
        for a in range(na - 1):
            for c in range(nb - 1):
                quad = np.array(
                    [gidface[a, c], gidface[a + 1, c], gidface[a + 1, c + 1], gidface[a, c + 1]]
                )
                cen = self.points[quad].mean(axis=0)
                yield quad, cen

    def patch_key_map(self) -> tuple[dict, dict]:
        """Return (patch_of, patch_types): tuple(sorted(ids))->name, name->type."""
        interfaces = self._interface_faces()
        patch_of = {}
        patch_types = {}
        pred_lookup = {(b, f): fn for (b, f, fn) in self.predicate_faces}
        for name, b in self.blocks.items():
            for key in (FaceKey.I0, FaceKey.I1, FaceKey.J0, FaceKey.J1, FaceKey.K0, FaceKey.K1):
                if (name, key) in interfaces:
                    continue
                bf = b.bfaces.get(key)
                fn = pred_lookup.get((name, key))
                if bf is None and fn is None:
                    continue  # unassigned & not an interface -> not exported here
                for quad, cen in self._face_quads(b, key):
                    if fn is not None:
                        pname = fn(cen)
                    else:
                        pname = bf.patch
                    if pname is None:
                        continue
                    k = tuple(sorted(int(x) for x in quad))
                    patch_of[k] = pname
        return patch_of, patch_types

    # -- services --------------------------------------------------------
    def scaled_jacobian(self) -> np.ndarray:
        from ..services.quality import scaled_jac

        return scaled_jac(self.points, self.hexes)

    def smooth(self, n: int = 50, in_plane: bool = True) -> "MultiBlockMesh":
        from ..services.smooth import laplacian_smooth

        boundary = self._boundary_node_mask()
        self.points = laplacian_smooth(
            self.points, self.hexes, boundary, iters=n, in_plane=in_plane
        )
        return self

    def _boundary_node_mask(self) -> np.ndarray:
        """Nodes lying on any exported (non-interface) block face are fixed."""
        interfaces = self._interface_faces()
        mask = np.zeros(len(self.points), bool)
        pred_faces = {(b, f) for (b, f, _) in self.predicate_faces}
        for name, b in self.blocks.items():
            for key in (FaceKey.I0, FaceKey.I1, FaceKey.J0, FaceKey.J1, FaceKey.K0, FaceKey.K1):
                if (name, key) in interfaces:
                    continue
                if b.bfaces.get(key) is None and (name, key) not in pred_faces:
                    continue
                gidface = _face_from_grid(b.gid, key)
                mask[gidface.ravel()] = True
        return mask

    def write_polymesh(self, case_dir, scale: float = 1e-3):
        from ..services.export_foam import write_polymesh

        patch_of, patch_types = self.patch_key_map()
        return write_polymesh(case_dir, self.points, self.hexes, patch_of, patch_types, scale)

    def morph(self, ctrl_pts, ctrl_disp, kernel: str = "biharmonic") -> np.ndarray:
        from ..services.morph import RBFMorpher

        m = RBFMorpher(kernel).fit(np.asarray(ctrl_pts, float), np.asarray(ctrl_disp, float))
        return self.points + m.displacement(self.points)


def _face_from_grid(grid, key: FaceKey):
    from .topology import face_index_grid

    return face_index_grid(grid, key)
