"""Anatomical anchor of the mesh frame: Atlas models carry it, other surfaces take --septum."""
import json
import re

import numpy as np
import pytest

from _synthetic import synthetic_lv
from hexmb import cli
from hexmb.io.surface import anatomical_frame

pytest.importorskip("trimesh")


def test_anatomical_frame_projects_and_refuses_axial_landmarks():
    e1, e2 = anatomical_frame([0, 0, 1], [3.0, 4.0, 7.0])
    assert np.allclose(e1, [0.6, 0.8, 0.0]) and np.allclose(e2, [-0.8, 0.6, 0.0])
    with pytest.raises(ValueError, match="long axis"):
        anatomical_frame([0, 0, 1], [0.0, 0.0, 5.0])


def test_cap_frame_is_model_y_and_matches_the_old_frame():
    from importlib import resources
    from hexmb.io.cap import cap_endo_surface, cap_frame, surface_frame
    f = str(resources.files("hexmb") / "data/cap/SCD0000101/SCD0000101_1.model.exnode")
    G = cap_endo_surface(f, NL=40, NT=56)
    axis, e1, e2, _ = cap_frame(G)
    y = np.array([0.0, 1.0, 0.0])
    assert np.allclose(e1, (y - axis * (y @ axis)) / np.linalg.norm(y - axis * (y @ axis)))
    assert np.allclose(e1, surface_frame(G)[1])          # published Atlas meshes do not move


def _points(case):
    txt = (case / "constant/polyMesh/points").read_text()
    return np.array(re.findall(r"\(([-\d.eE+]+) ([-\d.eE+]+) ([-\d.eE+]+)\)", txt), float)


def test_septum_landmark_restores_correspondence_after_a_rotation(tmp_path, capsys):
    from hexmb.io.stl_surface import write_surface_stl
    G = synthetic_lv(NL=24, NT=32)[0]
    G[:, :, 0] *= 1.3                                    # elliptic, so the angle matters
    write_surface_stl(G, tmp_path / "ref.stl")
    c, s = np.cos(0.9), np.sin(0.9)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    t = np.array([5.0, -20.0, 30.0])
    write_surface_stl(G @ R.T + t, tmp_path / "rot.stl")
    assert cli.main(["lv", "--surface", str(tmp_path / "ref.stl"), "--septum", "40,0,40",
                     "-o", str(tmp_path / "a")]) == 0
    sept = R @ np.array([40.0, 0.0, 40.0]) + t
    assert cli.main(["lv", "--surface", str(tmp_path / "rot.stl"), "--septum", ",".join(map(str, sept)),
                     "-o", str(tmp_path / "b")]) == 0
    assert cli.main(["lv", "--surface", str(tmp_path / "rot.stl"), "-o", str(tmp_path / "c")]) == 0
    out = capsys.readouterr().out
    assert "angle zero from septum landmark" in out and "not anatomical" in out
    A, B, C = (_points(tmp_path / k) * 1000 for k in "abc")
    back = lambda P: (P - t) @ R
    assert np.abs(back(B) - A).max() < 0.05                          # node for node, mm
    assert np.median(np.linalg.norm(back(C) - A, axis=1)) > 1.0      # without the landmark it does not
    rec = json.loads((tmp_path / "b/hexmb_frame.json").read_text())
    assert rec["anatomical"] and rec["anchor"] == "septum landmark"
    assert not json.loads((tmp_path / "c/hexmb_frame.json").read_text())["anatomical"]


def test_septum_is_refused_for_atlas_models(tmp_path, capsys):
    from importlib import resources
    f = str(resources.files("hexmb") / "data/cap/SCD0000101/SCD0000101_1.model.exnode")
    assert cli.main(["lv", "--surface", f, "--septum", "1,2,3", "-o", str(tmp_path / "x")]) == 1
    assert "own anatomical frame" in capsys.readouterr().err
