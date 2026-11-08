"""O-grid assembly + single-region LV template tests on a synthetic prolate LV.

Self-contained (no data files): the single O-grid over the whole cavity is the
baseline the telescoping template improves on; the exported polyMesh is read
back with hexmb's own reader and every cell is checked to close.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hexmb.assembly.ogrid import build_lv_ogrid  # noqa: E402
from hexmb.templates.lv import LVChamberTemplate  # noqa: E402
from _synthetic import synthetic_lv  # noqa: E402


def _load():
    G, axis, e1, e2, _mu = synthetic_lv()
    return G, axis, e1, e2


def test_ogrid_assembles_valid():
    G, axis, e1, e2 = _load()
    NL, NT, _ = G.shape
    Q = NT // 4
    m = build_lv_ogrid(G, axis, e1, e2, NR=6, core_frac=0.35)
    m.assemble()
    # all-hex, no polar core; a clean prolate cavity meshes fully valid
    sj = m.scaled_jacobian()
    assert (sj > 0).mean() > 0.999, f"unexpected inverted cells: min SJ {sj.min()}"
    # expected hex count: core Q*Q + 4 petals (NR-1)*Q, per level (NL-1)
    assert len(m.hexes) == (Q * Q + 4 * (6 - 1) * Q) * (NL - 1)


def test_lv_template_export_patches(tmp_path):
    G, axis, e1, e2 = _load()
    m = LVChamberTemplate(G, axis, e1, e2, NR=6, core_frac=0.35).build()
    assert (m.scaled_jacobian() > 0).mean() > 0.999
    info = m.write_polymesh(tmp_path / "constant" / "polyMesh")
    names = {p[0] for p in info["patches"]}
    assert names == {"wall", "inlet", "outlet1"}
    nb = sum(p[1] for p in info["patches"])
    assert nb == info["nFaces"] - info["nInternalFaces"]

    # read the exported polyMesh back and confirm every cell closes
    from hexmb.io.foam import read_polymesh
    from hexmb.services.quality import _face_geometry

    pm = tmp_path / "constant" / "polyMesh"
    pts, faces, owner, neigh = read_polymesh(pm)
    assert len(pts) == info["nPoints"]
    assert np.all(owner[: len(neigh)] < neigh)
    _, sf = _face_geometry(pts, faces)
    cell_sf = np.zeros((info["nCells"], 3))
    np.add.at(cell_sf, owner, sf)
    np.add.at(cell_sf, neigh, -sf[: len(neigh)])
    assert np.abs(cell_sf).max() < 1e-9  # cells closed


def test_lv_wall_ratio_builds_valid():
    G, axis, e1, e2 = _load()
    m = LVChamberTemplate(G, axis, e1, e2, NR=8, core_frac=0.42, wall_ratio=0.5).build()
    sj = m.scaled_jacobian()
    assert (sj > 0).mean() > 0.99, f"LV wall-layer: too many inverted, min SJ {sj.min()}"
