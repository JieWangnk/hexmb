"""Cavity acceptance: physical volume, boundaries, units and failure exits."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from hexmb.cli import main
from hexmb.io.foam import read_boundary, read_polymesh
from hexmb.services.quality import scaled_jac, polymesh_metrics
from hexmb.templates.cavity import BoxCavityTemplate, RingCavityTemplate


def test_box_volume_patches_and_units(tmp_path):
    outputs = []
    for unit, size in [("m", [1, 2, 3]), ("mm", [1000, 2000, 3000])]:
        out = tmp_path / unit
        assert main(["cavity", "--size", *map(str, size), "--units", unit,
                     "--cells", "3", "4", "5", "--stretch", "1", "2", "0", "-o", str(out)]) == 0
        pts, faces, own, nei = read_polymesh(out / "constant/polyMesh")
        q = polymesh_metrics(pts, faces, own, nei)["summary"]
        assert q["total_volume"] == pytest.approx(6)
        assert q["n_cells"] == 60
        assert q["max_non_orthogonality_deg"] < 1e-5
        patches = read_boundary(out / "constant/polyMesh/boundary")
        assert set(patches) == {"left", "right", "bottom", "lid", "front", "back"}
        assert all(p["type"] == "wall" for p in patches.values())
        assert sum(p["nFaces"] for p in patches.values()) == 2 * (12 + 15 + 20)
        report = json.loads((out / "mesh_report.json").read_text())
        assert report["accepted"] and report["output_units"] == "m"
        outputs.append(pts)
    np.testing.assert_allclose(*outputs)


def test_two_d_and_no_overwrite(tmp_path):
    out = tmp_path / "case"
    args = ["cavity", "--size", "1", "1", ".01", "--units", "m", "--cells", "8", "8", "1",
            "--two-d", "-o", str(out)]
    assert main(args) == 0
    boundary = read_boundary(out / "constant/polyMesh/boundary")
    assert boundary["front"]["type"] == boundary["back"]["type"] == "empty"
    before = (out / "constant/polyMesh/points").read_bytes()
    assert main(args) == 1
    assert before == (out / "constant/polyMesh/points").read_bytes()


def rings():
    t = np.arange(32) * 2 * np.pi / 32
    z = np.linspace(0, 40, 12)
    return np.array([np.column_stack([10*np.cos(t), 7*np.sin(t), np.full(32, zi)]) for zi in z])


@pytest.mark.parametrize("ends,patches", [("closed", {"wall"}), ("open", {"wall", "opening"}),
                                         ("through", {"wall", "inlet", "outlet"})])
def test_ring_cavity_exact_prism_volume(tmp_path, ends, patches):
    from hexmb.io.surface import write_surface_npz
    G = rings()
    file = tmp_path / "rings.npz"
    write_surface_npz(file, G, [0, 0, 1], [1, 0, 0], [0, 1, 0])
    out = tmp_path / ends
    assert main(["cavity", "--surface", str(file), "--units", "mm", "--ends", ends,
                 "-o", str(out)]) == 0
    bc = read_boundary(out / "constant/polyMesh/boundary")
    assert set(bc) == patches
    pts, faces, own, nei = read_polymesh(out / "constant/polyMesh")
    vol = polymesh_metrics(pts, faces, own, nei)["summary"]["total_volume"]
    exact_polygon_volume = 32 / 2 * 10 * 7 * np.sin(2*np.pi/32) * 40 * 1e-9
    assert vol == pytest.approx(exact_polygon_volume, rel=1e-8)


@pytest.mark.parametrize("kw", [dict(size=(0, 1, 1)), dict(size=(np.nan, 1, 1)),
                               dict(cells=(1.5, 2, 3)), dict(stretch=(6, 0, 0)),
                               dict(two_d=True)])
def test_invalid_box_options(kw):
    with pytest.raises(ValueError):
        BoxCavityTemplate(**kw)


def test_pointed_and_folded_rings_rejected():
    G = rings()
    G[0, :, :2] = 0
    with pytest.raises(ValueError, match="collapsed"):
        RingCavityTemplate(G, [0, 0, 1], [1, 0, 0], [0, 1, 0])
    m = RingCavityTemplate(G, [0, 0, 1], [1, 0, 0], [0, 1, 0], start=.1).build()
    assert np.all(m.scaled_jacobian() > 0)
    G = rings()
    G[-1, :, 2] = G[-2, :, 2] - 1
    with pytest.raises(ValueError, match="advance"):
        RingCavityTemplate(G, [0, 0, 1], [1, 0, 0], [0, 1, 0])


def test_quality_rejection_writes_report_only(tmp_path):
    from hexmb.io.surface import write_surface_npz
    G = rings()
    G[:, :, 0] += G[:, :, 2] * .5  # sheared prism, finite positive cells
    path = tmp_path / "shear.npz"
    write_surface_npz(path, G, [0, 0, 1], [1, 0, 0], [0, 1, 0])
    out = tmp_path / "case"
    assert main(["cavity", "--surface", str(path), "--units", "mm", "--max-nonorth", "1",
                 "-o", str(out)]) == 1
    assert not (out / "constant/polyMesh").exists()
    r = json.loads((out / "mesh_report.json").read_text())
    assert r["geometrically_valid"] and not r["accepted"]


def test_jacobian_scale_invariance_and_degeneracy():
    mesh = BoxCavityTemplate(cells=(1, 1, 1)).build()
    for scale in (1e-12, 1e-6, 1, 1e6):
        np.testing.assert_allclose(scaled_jac(mesh.points * scale, mesh.hexes), 1)
    mesh.points[:, 2] = 0
    assert scaled_jac(mesh.points, mesh.hexes)[0] == 0
    mesh.points[0, 0] = np.nan
    assert np.isnan(scaled_jac(mesh.points, mesh.hexes)[0])


def test_verify_readback_failure(tmp_path):
    pm = tmp_path / "constant/polyMesh"
    pm.mkdir(parents=True)
    (pm / "points").write_text("broken")
    assert main(["verify", "-o", str(tmp_path)]) == 1


@pytest.mark.parametrize("code,text", [(1, "failure"), (0, "Failed 1 mesh checks."), (0, "no result")])
def test_verify_propagates_external_failure(tmp_path, monkeypatch, code, text):
    assert main(["cavity", "--size", "1", "1", "1", "--units", "m", "--cells", "2", "2", "2",
                 "-o", str(tmp_path)]) == 0
    monkeypatch.setattr("hexmb.cli.shutil.which", lambda _: "/fake/checkMesh")
    monkeypatch.setattr("hexmb.cli.subprocess.run", lambda *a, **k:
                        SimpleNamespace(returncode=code, stdout=text, stderr=""))
    assert main(["verify", "-o", str(tmp_path)]) == 1


def test_surface_sac_in_metres(tmp_path):
    pytest.importorskip("trimesh")
    pytest.importorskip("rtree")
    from _synthetic import synthetic_lv
    from hexmb.io.stl_surface import write_surface_stl
    G, *_ = synthetic_lv(NL=60, NT=64)
    path = tmp_path / "sac.stl"
    write_surface_stl(G, path, scale=.001)
    out = tmp_path / "case"
    assert main(["cavity", "--surface", str(path), "--units", "m", "--start", ".05",
                 "--ends", "open", "--apex", "0,0,0", "--base", "0,0,.08", "--vtk", "-o", str(out)]) == 0
    report = json.loads((out / "mesh_report.json").read_text())
    assert report["source"]["retained_axial_fractions"] == [.05, 1]
    assert report["source"]["sampling"]["axis_length"] == pytest.approx(80, rel=1e-5)
    assert (out / "vtk/motion.pvd").exists()


@pytest.mark.parametrize("ratio", [0, -1, np.nan, np.inf])
def test_invalid_wall_grading(ratio):
    from hexmb.assembly.ogrid import graded_radial
    with pytest.raises(ValueError, match="positive"):
        graded_radial(6, ratio)


def test_verify_external_success(tmp_path, monkeypatch):
    assert main(["cavity", "--size", "1", "1", "1", "--units", "m", "--cells", "2", "2", "2",
                 "-o", str(tmp_path)]) == 0
    monkeypatch.setattr("hexmb.cli.shutil.which", lambda _: "/fake/checkMesh")
    def run(command, **kwargs):
        assert command == ["/fake/checkMesh", "-case", str(tmp_path), "-constant"]
        return SimpleNamespace(returncode=0, stdout="Mesh OK.\n", stderr="")
    monkeypatch.setattr("hexmb.cli.subprocess.run", run)
    assert main(["verify", "-o", str(tmp_path)]) == 0
