"""VTK / Abaqus export, checkMesh-style metrics and the wall-driven flow case."""
import base64
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hexmb.services.quality import polymesh_metrics, polymesh_metrics_of
from hexmb.io.vtk import (write_vtu, write_pvd, write_abaqus_inp, case_to_vtk, abaqus_from_case,
                          hex_corners, cells_from_polymesh)
from hexmb.io.foam import read_polymesh
from _synthetic import synthetic_lv

SMALL = dict(NT_c=8, NR_c=2, apex_levels=4, body_levels=8, base_levels=4)


def _two_cubes(shear=0.0):
    """Two unit cubes along x sharing the face x = 1; ``shear`` shifts the x = 2 face in y."""
    P = np.array([[x, y, z] for x in (0, 1, 2) for y in (0, 1) for z in (0, 1)], float)
    P[P[:, 0] == 2, 1] += shear
    pid = lambda x, y, z: x * 4 + y * 2 + z
    q = lambda a, b, c, d: [pid(*a), pid(*b), pid(*c), pid(*d)]
    faces = [q((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))]            # internal, owner 0 -> neighbour 1
    owner, neigh = [0], [1]
    for cx in (0, 1):
        x0, x1 = cx, cx + 1
        faces += [q((x0, 0, 0), (x0, 0, 1), (x0, 1, 1), (x0, 1, 0)) if cx == 0
                  else q((x1, 0, 0), (x1, 1, 0), (x1, 1, 1), (x1, 0, 1)),
                  q((x0, 0, 0), (x1, 0, 0), (x1, 0, 1), (x0, 0, 1)), q((x0, 1, 0), (x0, 1, 1), (x1, 1, 1), (x1, 1, 0)),
                  q((x0, 0, 0), (x0, 1, 0), (x1, 1, 0), (x1, 0, 0)), q((x0, 0, 1), (x1, 0, 1), (x1, 1, 1), (x0, 1, 1))]
        owner += [cx] * 5
    return P, faces, np.array(owner), np.array(neigh)


def test_metrics_are_exact_on_cubes_and_a_sheared_cell():
    P, faces, own, nei = _two_cubes()
    s = polymesh_metrics(P, faces, own, nei)["summary"]
    assert s["n_cells"] == 2 and s["n_internal_faces"] == 1
    assert s["max_non_orthogonality_deg"] < 1e-9 and s["max_skewness"] < 1e-9
    assert abs(s["max_aspect_ratio"] - 1) < 1e-9 and abs(s["total_volume"] - 2) < 1e-12
    # shift the far face by one unit: the owner-neighbour line tilts by atan(0.5) = 26.57 deg
    # and crosses the shared face 0.25 off its centre, 0.5 of the half-face -> skewness 0.5
    P, faces, own, nei = _two_cubes(shear=1.0)
    m = polymesh_metrics(P, faces, own, nei)
    assert abs(m["summary"]["max_non_orthogonality_deg"] - np.degrees(np.arctan(0.5))) < 1e-6
    assert abs(m["summary"]["max_skewness_internal"] - 0.5) < 1e-9
    assert np.allclose(m["volume"], 1.0)


def test_metrics_of_a_conformal_mesh_match_the_gate_and_are_finite():
    from hexmb.templates.conformal_lv import ConformalLVConfig, build_conformal_lv, polymesh_quality
    G, axis, e1, e2 = synthetic_lv()[:4]
    mesh = build_conformal_lv(G, axis, e1, e2, ConformalLVConfig(**SMALL))
    m = polymesh_metrics_of(mesh)
    s = m["summary"]
    assert s["n_cells"] == len(mesh.cells) and s["n_negative_volume"] == 0
    assert abs(s["total_volume"] * 1e-3 - polymesh_quality(mesh)["volume_mL"]) < 1e-6
    assert 0 < s["max_non_orthogonality_deg"] < 90 and s["max_skewness"] < 10


def _decode(vtu_text, name):
    m = re.search(rf'<DataArray type="(\w+)" Name="{re.escape(name)}" NumberOfComponents="(\d+)" format="binary">([^<]+)<', vtu_text)
    raw = base64.b64decode(m.group(3))
    dt = {"Float64": "<f8", "Int64": "<i8", "UInt8": "u1"}[m.group(1)]
    arr = np.frombuffer(raw[8:], dt)
    return arr.reshape(-1, int(m.group(2))) if int(m.group(2)) > 1 else arr


def test_hex_corners_are_right_handed_and_vtu_round_trips(tmp_path):
    P, faces, own, nei = _two_cubes(shear=0.3)
    cells = [[faces[fi] for fi in c] for c in cells_from_polymesh(faces, own, nei)]
    for c in cells:
        hc = hex_corners(c, P)
        assert hc is not None and len(set(hc)) == 8
        b, t = P[hc[:4]], P[hc[4:]]
        nrm = np.cross(b[1] - b[0], b[2] - b[0])
        assert nrm @ (t.mean(0) - b.mean(0)) > 0          # bottom normal points into the cell (VTK order)
        assert np.linalg.det(np.stack([P[hc[1]] - P[hc[0]], P[hc[3]] - P[hc[0]], P[hc[4]] - P[hc[0]]])) > 0
        for k in range(4):                                 # (bottom k, top k) is an edge of the cell
            assert any(abs(f.index(hc[k]) - f.index(hc[4 + k])) in (1, 3) for f in c if hc[k] in f and hc[4 + k] in f)
    out = write_vtu(tmp_path / "two.vtu", P, cells, point_data={"d": np.ones((12, 3))}, cell_data={"v": np.ones(2)})
    txt = out.read_text()
    assert _decode(txt, "types").tolist() == [12, 12]
    assert _decode(txt, "connectivity").shape == (16,) and _decode(txt, "offsets").tolist() == [8, 16]
    assert _decode(txt, "Points").shape == (12, 3) and _decode(txt, "d").shape == (12, 3)
    ascii_out = write_vtu(tmp_path / "two_ascii.vtu", P, cells, binary=False)
    assert 'format="ascii"' in ascii_out.read_text()
    # a polyhedron (a cell with 9 faces) gets type 42 with a faces array
    cells9 = [cells[0], cells[1][:5] + [cells[1][5][:2] + [cells[1][5][2]], cells[1][5]] + [cells[1][5]] * 2]
    txt = write_vtu(tmp_path / "poly.vtu", P, cells9).read_text()
    assert _decode(txt, "types").tolist() == [12, 42] and "faceoffsets" in txt


def test_pvd_lists_times(tmp_path):
    p = write_pvd(tmp_path / "m.pvd", [(0.0, "a.vtu"), (0.05, "b.vtu")])
    txt = p.read_text()
    assert 'timestep="0"' in txt and 'timestep="0.05"' in txt and 'file="b.vtu"' in txt


def test_case_to_vtk_and_abaqus_on_written_cases(tmp_path):
    from hexmb.cli import main
    # all-hex pipe: VTK + Abaqus
    pipe = tmp_path / "pipe"
    assert main(["pipe", "-o", str(pipe), "--NT", "8", "--NR", "2", "--n-axial", "4"]) == 0
    out = case_to_vtk(pipe, verbose=False)
    assert (out / "mesh.vtu").exists() and (out / "motion.pvd").exists()
    pats = sorted(p.name for p in out.glob("boundary_*.vtp"))
    assert pats and any("wall" in p for p in pats)
    pts, faces, own, nei = read_polymesh(pipe / "constant" / "polyMesh")
    inp = abaqus_from_case(pipe, tmp_path / "pipe.inp", scale=1000.0)
    txt = inp.read_text()
    n_el = len(re.findall(r"^\d+, \d+, \d+, \d+, \d+, \d+, \d+, \d+, \d+$", txt, re.M))
    assert n_el == int(own.max()) + 1
    assert "*ELEMENT, TYPE=C3D8" in txt and "*NSET, NSET=wall" in txt
    # the conformal LV has split-hex cells: VTK yes, Abaqus refused
    from hexmb.templates.conformal_lv import ConformalLVConfig, build_conformal_lv
    G, axis, e1, e2 = synthetic_lv()[:4]
    mesh = build_conformal_lv(G, axis, e1, e2, ConformalLVConfig(**SMALL))
    lv = tmp_path / "lv"
    mesh.write(lv)
    out = case_to_vtk(lv, verbose=False)
    txt = (out / "mesh.vtu").read_text()
    types = _decode(txt, "types")
    assert (types == 12).sum() == mesh.n_hex and (types == 42).sum() == mesh.n_split
    with pytest.raises(ValueError, match="not a hexahedron"):
        abaqus_from_case(lv, tmp_path / "lv.inp")


def _moving_case(tmp_path, reference="es"):
    from hexmb.services.motion import build_lv_motion
    from hexmb.templates.conformal_lv import ConformalLVConfig
    G0, axis, e1, e2 = synthetic_lv()[:4]
    w = np.linspace(1.0, 0.0, G0.shape[0])[:, None, None]          # contract the apex, hold the base
    grids = []
    for k in range(6):
        s = 1.0 - 0.3 * (0.5 - 0.5 * np.cos(2 * np.pi * k / 6))      # 70 % radius at phase 3
        grids.append(np.concatenate([G0[:, :, :2] * (1 - w * (1 - s)), G0[:, :, 2:]], axis=2))
    times = 0.15 * np.arange(6)
    case = tmp_path / f"mov_{reference}"
    res = build_lv_motion(grids, times, cfg=ConformalLVConfig(**SMALL), reference=reference, period=0.9,
                          out=case, frame=(axis, e1, e2, grids[0].reshape(-1, 3).mean(0)), verbose=False)
    assert res["valid"]
    return case, res


def test_flow_case_windows_and_dictionaries(tmp_path):
    from hexmb.services.flow import write_flow_case, case_phase_volumes, find_window
    case, res = _moving_case(tmp_path, "es")
    times, vols = case_phase_volumes(case)
    assert len(times) == len(res["volumes_mL"]) + 4          # the wrap repeats the phases up to the reference
    assert np.allclose(np.sort(np.unique(np.round(vols * 1e6, 3))), np.sort(np.unique(np.round(res["volumes_mL"], 3))))
    t0, t1 = find_window(times, vols, "diastole", times[np.argmin(vols)])
    assert t1 > t0
    with pytest.raises(ValueError, match="reference es"):
        find_window(times, vols, "diastole", times[np.argmax(vols)])
    with pytest.raises(ValueError, match="reference ed"):
        write_flow_case(case, "systole", verbose=False)
    r = write_flow_case(case, "diastole", n_procs=2, verbose=False)
    assert r["open"] == "inlet" and r["closed"] == ["outlet1"]
    tdir = [d for d in case.iterdir() if d.is_dir() and (d / "U").exists()]
    assert len(tdir) == 1
    U, p = (tdir[0] / "U").read_text(), (tdir[0] / "p").read_text()
    assert "pressureInletOutletVelocity" in U.split("inlet")[1] and "movingWallVelocity" in U.split("outlet1")[1]
    assert "totalPressure" in p.split("inlet")[1] and "fixedFluxPressure" in p.split("outlet1")[1]
    cd = (case / "system" / "controlDict").read_text()
    assert "incompressibleFluid" in cd and f"endTime         {t1:g}" in cd and "maxCo" in cd and "Qout" in cd
    assert "correctMeshPhi  yes" in (case / "system" / "fvSolution").read_text()
    assert "linearUpwind grad(U)" in (case / "system" / "fvSchemes").read_text()
    assert "numberOfSubdomains 2" in (case / "system" / "decomposeParDict").read_text()
    assert (case / "Allrun").stat().st_mode & 0o111
    assert "interpolationScheme spline" in (case / "constant" / "dynamicMeshDict").read_text()
    run = (case / "Allrun").read_text()
    assert "decomposePar -force -time '0:'" in run and "CHECK: Courant max is 0" in run
    # systole from an end-diastole reference
    case_ed, _ = _moving_case(tmp_path, "ed")
    r = write_flow_case(case_ed, "systole", verbose=False)
    assert r["open"] == "outlet1" and r["closed"] == ["inlet"]



def test_openness_and_severe_non_orthogonality_follow_checkmesh():
    """checkMesh's openness is the worst COMPONENT of |sum Sf| / sum |Sf|: a unit
    cube missing its +x face is fully open (1.0) in x.  Shearing the far face by
    6 tilts the owner-neighbour line by atan(3) = 71.6 deg: one face above 70."""
    P = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    pid = lambda x, y, z: x * 4 + y * 2 + z
    q = lambda a, b, c, d: [pid(*a), pid(*b), pid(*c), pid(*d)]
    faces = [q((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
             q((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)), q((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),
             q((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)), q((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))]
    s = polymesh_metrics(P, faces, np.zeros(5, int), np.array([], int), n_cells=1)["summary"]
    assert abs(s["max_openness"] - 1.0) < 1e-12
    s = polymesh_metrics(P, faces + [q((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))],
                         np.zeros(6, int), np.array([], int), n_cells=1)["summary"]
    assert s["max_openness"] < 1e-12
    P, faces, own, nei = _two_cubes(shear=6.0)
    s = polymesh_metrics(P, faces, own, nei)["summary"]
    assert abs(s["max_non_orthogonality_deg"] - np.degrees(np.arctan(3.0))) < 1e-6
    assert s["n_non_orthogonality_gt70"] == 1 and s["n_non_orthogonality_gt90"] == 0
