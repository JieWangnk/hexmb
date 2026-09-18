"""Core engine tests: TFI, dedup, handedness, export round-trip."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hexmb import Block, FaceKey, MultiBlockMesh  # noqa: E402
from hexmb.core.tfi import tfi3d  # noqa: E402
from hexmb.services.quality import scaled_jac  # noqa: E402


def _unit_block(name, x0, nx=3, ny=3, nz=3):
    """A rectilinear block spanning [x0,x0+1] x [0,1] x [0,1]."""
    b = Block(name, (nx, ny, nz))
    xs = np.linspace(x0, x0 + 1, nx)
    ys = np.linspace(0, 1, ny)
    zs = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    b.set_nodes(np.stack([X, Y, Z], axis=-1))
    return b


def test_tfi_affine_exact():
    # affine image of the unit cube must be reproduced exactly by TFI interior
    n = 4
    lin = np.linspace(0, 1, n)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    P = np.stack([X, Y, Z], axis=-1)
    M = np.array([[1.0, 0.3, -0.2], [0.0, 2.0, 0.1], [0.4, 0.0, 1.5]])
    t = np.array([5.0, -1.0, 2.0])
    aff = P @ M.T + t
    faces = {
        "i0": aff[0], "i1": aff[-1],
        "j0": aff[:, 0], "j1": aff[:, -1],
        "k0": aff[:, :, 0], "k1": aff[:, :, -1],
    }
    out = tfi3d(faces)
    assert np.max(np.linalg.norm(out - aff, axis=-1)) < 1e-10


def test_two_block_dedup_and_export(tmp_path):
    m = MultiBlockMesh()
    a = _unit_block("A", 0.0)
    b = _unit_block("B", 1.0)
    m.add_block(a)
    m.add_block(b)
    m.join("A", FaceKey.I1, "B", FaceKey.I0)  # shared face at x=1
    # patches: everything not the shared interface is 'wall'
    for blk in (a, b):
        for key in [FaceKey.I0, FaceKey.I1, FaceKey.J0, FaceKey.J1, FaceKey.K0, FaceKey.K1]:
            blk.set_face(key, patch="wall")
    m.assemble()

    # 3x3x3 nodes each = 27; shared face 3x3=9 merged -> 27+27-9 = 45
    assert len(m.points) == 45
    # 2 blocks x (2x2x2 cells) = 16 hexes
    assert len(m.hexes) == 16
    # all cells valid
    sj = m.scaled_jacobian()
    assert sj.min() > 0, f"min SJ {sj.min()}"

    info = m.write_polymesh(tmp_path / "constant" / "polyMesh")
    # internal faces: the 4 interior faces per block along x + the shared 1
    # shared interface (4 quads) is internal
    assert info["nInternalFaces"] > 0
    assert info["nCells"] == 16

    # readback round-trip
    sys.path.insert(0, str(Path.home() / "GitHub" / "betaFlow" / "tools"))
    import foam_mesh as fm

    pm = tmp_path / "constant" / "polyMesh"
    pts = fm.read_points(pm / "points")
    faces = fm.read_faces(pm / "faces")
    owner = fm.read_labels(pm / "owner")
    neigh = fm.read_labels(pm / "neighbour")
    bnd = fm.read_boundary(pm / "boundary")
    assert len(pts) == 45
    assert len(owner) == len(faces)
    assert len(neigh) == info["nInternalFaces"]
    # owner < neighbour on all internal faces
    assert np.all(owner[: len(neigh)] < neigh)
    # closed cells: sum of signed area vectors per cell ~ 0
    sf, cf = fm.face_area_vectors(pts, faces)
    cell_sf = np.zeros((info["nCells"], 3))
    np.add.at(cell_sf, owner, sf)
    np.add.at(cell_sf[: 0 + info["nCells"]], neigh, -sf[: len(neigh)])
    assert np.abs(cell_sf).max() < 1e-9, f"open cells, max |sum Sf| = {np.abs(cell_sf).max()}"
    assert set(bnd) == {"wall"}


if __name__ == "__main__":
    test_tfi_affine_exact()
    print("TFI affine-exact: OK")
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_two_block_dedup_and_export(Path(d))
    print("two-block dedup + export round-trip: OK")
