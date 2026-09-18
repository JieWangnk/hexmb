"""Pipe + elbow templates: the engine's generality beyond the LV."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hexmb.templates.tube import ElbowTemplate, PipeTemplate  # noqa: E402


def _check_valid(m):
    sj = m.scaled_jacobian()
    assert sj.min() > 0, f"min SJ {sj.min()}"
    # all cells hexes: hexes array is (n,8)
    assert m.hexes.shape[1] == 8
    return sj


def test_pipe_valid_and_export(tmp_path):
    m = PipeTemplate(length=100, radius=10, n_axial=30, NT=32, NR=6).build()
    sj = _check_valid(m)
    assert sj.min() > 0.5  # a straight pipe is high quality
    info = m.write_polymesh(tmp_path / "constant" / "polyMesh")
    assert {p[0] for p in info["patches"]} == {"wall", "inlet", "outlet"}


def test_elbow_valid_no_twist(tmp_path):
    m = ElbowTemplate(radius=10, bend_radius=35, angle_deg=180, leg=40, NT=32, NR=6).build()
    sj = _check_valid(m)
    # rotation-minimizing frames keep the bend well-shaped
    assert sj.min() > 0.4
    info = m.write_polymesh(tmp_path / "constant" / "polyMesh")
    assert {p[0] for p in info["patches"]} == {"wall", "inlet", "outlet"}


def test_ogrid_node_count_matches_formula():
    # pipe with ns stations: (Q*Q + 4*(NR-1)*Q) * (ns-1) hexes
    NT, NR, ns = 32, 6, 30
    Q = NT // 4
    m = PipeTemplate(length=50, radius=5, n_axial=ns, NT=NT, NR=NR).build()
    assert len(m.hexes) == (Q * Q + 4 * (NR - 1) * Q) * (ns - 1)


def test_graded_radial_clusters_at_wall():
    from hexmb.assembly.ogrid import graded_radial
    import numpy as np
    u = graded_radial(8, 1.0)
    assert np.allclose(u, np.linspace(0, 1, 8))
    g = graded_radial(8, 0.25)
    assert (1 - g[-2]) < (1 - u[-2])          # wall cell thinner
    assert np.all(np.diff(g) > 0)              # monotonic




def test_graded_radial_boundary_layer():
    """wall_ratio < 1 must cluster cells at the wall; = 1 must be uniform."""
    from hexmb.assembly.ogrid import graded_radial
    NR = 9
    uni = graded_radial(NR, 1.0)
    assert np.allclose(uni, np.linspace(0, 1, NR))
    g = graded_radial(NR, 0.3)                      # 0.3 -> strong wall clustering
    assert g[0] == 0.0 and abs(g[-1] - 1.0) < 1e-9  # spans [0,1]
    assert np.all(np.diff(g) > 0)                   # monotone
    d = np.diff(g)
    assert d[-1] < d[0]                             # last (wall) cell thinner than first (core)


def test_pipe_wall_ratio_builds_valid():
    m = PipeTemplate(length=60.0, radius=10.0, n_axial=30, NT=48, NR=8,
                     core_frac=0.42, wall_ratio=0.4).build()
    sj = m.scaled_jacobian()
    assert sj.min() > 0 and int((sj <= 0).sum()) == 0


def test_large_pipe_assembles_millions():
    """Vectorised assembly must handle a >1e6-cell mesh (scaling regression)."""
    m = PipeTemplate(length=100.0, radius=10.0, n_axial=300, NT=160, NR=16,
                     core_frac=0.42, wall_ratio=0.6).build()
    assert len(m.hexes) > 1_000_000
    sj = m.scaled_jacobian()
    assert int((sj <= 0).sum()) == 0
