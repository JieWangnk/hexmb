"""Input handling of the command line as a new user meets it (units, missing files, demo surfaces, OpenFOAM lookup)."""
import pytest

from _synthetic import synthetic_lv
from hexmb import cli
from hexmb.io.stl_surface import write_surface_stl

pytest.importorskip("trimesh")


@pytest.fixture
def lv_stl(tmp_path):
    """An 80 mm synthetic ventricle written in mm, cm and m."""
    G = synthetic_lv(NL=24, NT=32)[0]
    paths = {}
    for unit, f in (("mm", 1.0), ("cm", 0.1), ("m", 0.001)):
        paths[unit] = tmp_path / f"lv_{unit}.stl"
        write_surface_stl(G * f, paths[unit], scale=1.0)
    return paths


@pytest.mark.parametrize("unit, hint", [("cm", "--units cm"), ("m", "--units m")])
def test_wrong_unit_is_stopped_with_the_right_hint(lv_stl, tmp_path, capsys, unit, hint):
    rc = cli.main(["lv", "--surface", str(lv_stl[unit]), "-o", str(tmp_path / "case")])
    err = capsys.readouterr().err
    assert rc == 1 and hint in err and "human left ventricle" in err
    assert not (tmp_path / "case").exists()


@pytest.mark.parametrize("unit", ["mm", "cm", "m"])
def test_units_option_gives_the_same_mesh(lv_stl, tmp_path, capsys, unit):
    rc = cli.main(["lv", "--surface", str(lv_stl[unit]), "--units", unit, "-o", str(tmp_path / unit)])
    out = capsys.readouterr().out
    assert rc == 0 and "(VALID)" in out
    ref = cli.main(["lv", "--surface", str(lv_stl["mm"]), "-o", str(tmp_path / "ref")])
    assert ref == 0
    assert out.split("cavity")[1].split()[0] == capsys.readouterr().out.split("cavity")[1].split()[0]


def test_any_size_lets_a_small_heart_through(lv_stl, tmp_path, capsys):
    rc = cli.main(["lv", "--surface", str(lv_stl["cm"]), "--any-size", "-o", str(tmp_path / "small")])
    assert rc == 0 and "(VALID)" in capsys.readouterr().out


def test_missing_file_and_empty_pattern_are_named(tmp_path, capsys):
    assert cli.main(["lv", "--surface", str(tmp_path / "nothere.stl"), "-o", str(tmp_path / "a")]) == 1
    assert "nothere.stl: no such file" in capsys.readouterr().err
    assert cli.main(["motion", "--surfaces", str(tmp_path / "nope_*.stl"), "-o", str(tmp_path / "b")]) == 1
    assert "no file matches the pattern" in capsys.readouterr().err


def test_one_phase_is_refused_before_meshing(lv_stl, tmp_path, capsys):
    assert cli.main(["motion", "--surfaces", str(lv_stl["mm"]), "-o", str(tmp_path / "c")]) == 1
    cap = capsys.readouterr()
    assert "at least two phases" in cap.err and "template" not in cap.out


def test_demo_writes_the_phase_surfaces(tmp_path, capsys):
    rc = cli.main(["demo", "--phases", "3", "-o", str(tmp_path / "demo"), "--phases-stl", str(tmp_path / "ph")])
    assert rc == 0
    files = sorted((tmp_path / "ph").glob("lv_phase_*.stl"))
    assert [f.name for f in files] == ["lv_phase_00.stl", "lv_phase_01.stl", "lv_phase_02.stl"]
    assert "hexmb motion --surfaces" in capsys.readouterr().out


def test_foam_bashrc_lookup_order(tmp_path, monkeypatch):
    rc = tmp_path / "of" / "etc" / "bashrc"; rc.parent.mkdir(parents=True); rc.write_text("")
    monkeypatch.setenv("HEXMB_FOAM_BASHRC", str(rc))
    assert cli._foam_bashrc() == rc
    monkeypatch.setenv("HEXMB_FOAM_BASHRC", str(tmp_path / "missing"))
    monkeypatch.setenv("WM_PROJECT_DIR", str(tmp_path / "of"))
    assert cli._foam_bashrc() == rc
