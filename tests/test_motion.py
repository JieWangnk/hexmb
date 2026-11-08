"""Moving-mesh writer: per-phase builds, gate, reference phase, time folders."""
import os

import numpy as np
import pytest

from _synthetic import synthetic_lv
from hexmb.services.motion import build_lv_motion, hex_volumes
from hexmb.templates.conformal_lv import ConformalLVConfig


def _phases(scales=(1.0, 0.88, 0.78, 0.9)):
    G, *_ = synthetic_lv()
    w = np.linspace(1.0, 0.0, G.shape[0])[:, None, None]       # contract the apex, hold the base
    return [np.concatenate([G[:, :, :2] * (1 - w * (1 - s)), G[:, :, 2:]], axis=2) for s in scales]


def _read_field(path):
    txt = open(path).read()
    body = txt.split("(\n", 1)[1].split("\n);")[0]
    return np.array([[float(v) for v in ln.strip("()").split()] for ln in body.splitlines()])


def test_hex_volumes_unit_cube():
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float) * 2.0
    assert np.isclose(hex_volumes(pts, [[0, 1, 2, 3, 4, 5, 6, 7]])[0], 8.0)


def test_single_builder_case(tmp_path):
    grids = _phases()
    times = np.array([0.0, 0.2, 0.4, 0.6])
    res = build_lv_motion(grids, times, builder="single", cfg=dict(NR=5, core_frac=0.35),
                          reference="es", period=0.8, out=tmp_path / "mov", verbose=False)
    assert res["valid"] and res["ref"] == 2                     # smallest cavity is phase 2
    assert res["points"].shape[0] == 4
    v = res["volumes_mL"]
    assert v[0] > v[1] > v[2] < v[3]
    assert set(res["patches"]) == {"wall", "inlet", "outlet1"}
    dirs = sorted(d for d in os.listdir(tmp_path / "mov") if d[0].isdigit())
    assert dirs == ["0", "0.2", "0.4", "0.6", "0.8", "1", "1.2"]   # phases + wrap to ref + period
    d_ref = _read_field(tmp_path / "mov" / "0.4" / "wantedDisplacement")
    assert np.abs(d_ref).max() == 0.0                           # reference displacement exactly zero
    d0 = _read_field(tmp_path / "mov" / "0" / "wantedDisplacement")
    assert np.allclose(d0, (res["points"][0] - res["points"][2]) * 1e-3)
    assert (tmp_path / "mov" / "constant" / "dynamicMeshDict").exists()
    assert "startTime       0.4;" in (tmp_path / "mov" / "system" / "controlDict").read_text()
    assert (tmp_path / "mov" / "motion_quality.csv").read_text().count("\n") == 5
    d_wrap = _read_field(tmp_path / "mov" / "1.2" / "wantedDisplacement")
    assert np.abs(d_wrap).max() == 0.0                          # reference again at ref + period
    assert "endTime         1.2;" in (tmp_path / "mov" / "system" / "controlDict").read_text()


def test_conformal_builder_phases(tmp_path):
    grids = _phases((1.0, 0.8))
    cfg = ConformalLVConfig(NT_c=16, NR_c=3, apex_levels=4, body_levels=6, base_levels=4)
    res = build_lv_motion(grids, [0.0, 0.4], builder="conformal", cfg=cfg, reference="ed",
                          out=tmp_path / "conf", verbose=False)
    assert res["valid"] and res["ref"] == 0
    assert all(r["n_nonpositive"] == 0 for r in res["quality"])
    assert (tmp_path / "conf" / "constant" / "polyMesh" / "faces").exists()


def test_rejects_bad_input():
    grids = _phases((1.0, 0.9))
    with pytest.raises(ValueError):
        build_lv_motion(grids, [0.0], builder="single", verbose=False)
    with pytest.raises(ValueError):
        build_lv_motion(grids, [0.0, 0.0], builder="single", verbose=False)
    with pytest.raises(ValueError):
        build_lv_motion(grids, [0.0, 0.5], builder="nope", verbose=False)


def test_single_builder_gate_uses_scaled_jacobian_and_trims_point_apex():
    """A grid whose first ring is a point inverts the first ring of hexes in the
    single template; the gate must see it (SJ) and the default apex trim fix it."""
    grids = _phases((1.0, 0.9))
    for G in grids:
        G[0] = G[0].mean(0)                                        # collapse the apex ring to a point
    res = build_lv_motion(grids, [0.0, 0.4], builder="single", cfg=dict(NR=4, apex_trim=0.0), verbose=False,
                          auto_trim=False, fill_failed=False)
    assert not res["valid"] and all(r["n_inverted"] > 0 for r in res["quality"])
    res = build_lv_motion(grids, [0.0, 0.4], builder="single", cfg=dict(NR=4), verbose=False, auto_trim=False)
    assert res["valid"] and not res["adjustments"]
    # auto_trim finds the same fix itself: the failing cells sit in the apex zone, so apex_trim goes up
    res = build_lv_motion(grids, [0.0, 0.4], builder="single", cfg=dict(NR=4, apex_trim=0.0), verbose=False)
    assert res["valid"] and res["cfg"]["apex_trim"] > 0 and "apex_trim" in res["adjustments"][0]


def test_refuses_to_delete_a_non_case_directory(tmp_path):
    grids = _phases((1.0, 0.9))
    d = tmp_path / "work"; d.mkdir(); (d / "notes.txt").write_text("keep me")
    with pytest.raises(ValueError):
        build_lv_motion(grids, [0.0, 0.4], builder="single", cfg=dict(NR=4), out=d, verbose=False)
    assert (d / "notes.txt").exists()
    with pytest.raises(ValueError):
        build_lv_motion(grids[:1], [0.0], builder="single", verbose=False)   # one phase is not motion


# --------------------------------------------------------------------------- #
#  ring screen, automatic rim trimming, filling a degenerate phase            #
# --------------------------------------------------------------------------- #
from hexmb.services import motion as motion_mod
from hexmb.services.motion import build_phases
from hexmb.templates.conformal_lv import bad_rings, axial_fractions

SMALL = dict(NT_c=8, NR_c=2, apex_levels=4, body_levels=8, base_levels=4)


def _cross(G, rows, every=7):
    """Send every ``every``-th point of the given rows through the long axis."""
    G = G.copy()
    for r in rows:
        G[r, ::every, :2] *= -1
    return G


def test_bad_rings_finds_crossing_point_and_reversed_rings():
    G, axis, e1, e2, mu = synthetic_lv()
    assert bad_rings(G, axis, e1, e2) == []
    assert bad_rings(_cross(G, [38, 39]), axis, e1, e2) == [38, 39]
    assert bad_rings(_cross(G, [11, 12, 13]), axis, e1, e2) == [11, 12, 13]
    H = G.copy(); H[38:, :28, :2] *= -1                           # half the ring reflected: winds twice
    assert bad_rings(H, axis, e1, e2) == [38, 39]
    H = G.copy(); H[20] = H[20].mean(0)                           # a ring collapsed to a point
    assert bad_rings(H, axis, e1, e2) == [20]
    H = G.copy(); H[25] = H[25][::-1]                             # one ring running the other way
    assert bad_rings(H, axis, e1, e2) == [25]
    H = G.copy(); H[30, :, 0] += 40.0                             # a shifted ring still winds once
    assert bad_rings(H, axis, e1, e2) == []
    t = axial_fractions(G, axis)
    assert t[0] == 0 and np.isclose(t[-1], 1.0) and np.all(np.diff(t) > 0)


def test_auto_trim_moves_base_cut_to_the_last_good_row():
    grids = _phases()
    grids[1] = _cross(grids[1], [38, 39])
    axis, e1, e2 = np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    res = build_lv_motion(grids, [0, .2, .4, .6], cfg=ConformalLVConfig(**SMALL), period=.8, verbose=False)
    want = axial_fractions(grids[1], axis)[37]
    assert res["valid"] and np.isclose(res["cfg"].base_cut, want, atol=1e-3)
    assert len(res["adjustments"]) == 2 and res["adjustments"][0].startswith("base_cut 1.000 ->")
    assert res["adjustments"][1].startswith("the trims remove") and "trim_removed_mL" in res["quality"][0]
    assert "phase(s) 1" in res["adjustments"][0] and res["filled"] == []
    assert all(r["status"] == "ok" for r in res["quality"])
    # the same surfaces with the screen off keep base_cut 1.0
    res = build_lv_motion(grids, [0, .2, .4, .6], cfg=ConformalLVConfig(**SMALL), period=.8, verbose=False,
                          auto_trim=False)
    assert res["cfg"].base_cut == 1.0 and res["adjustments"] == []


def test_auto_trim_moves_apex_trim_past_crossing_apex_rows():
    grids = _phases()
    grids[2] = _cross(grids[2], [1, 2, 3])
    res = build_lv_motion(grids, [0, .2, .4, .6], cfg=ConformalLVConfig(**SMALL), period=.8, verbose=False)
    want = axial_fractions(grids[2], np.array([0, 0, 1.0]))[4]
    assert res["valid"] and np.isclose(res["cfg"].apex_trim, want, atol=1e-3)
    assert res["adjustments"][0].startswith("apex_trim 0.020 ->") and "phase(s) 2" in res["adjustments"][0]


def test_degenerate_phase_is_filled_from_its_neighbours(tmp_path):
    grids = _phases()
    grids[3][20] = grids[3][20].mean(0)                           # a point ring in the middle of phase 3
    cfg = ConformalLVConfig(**SMALL)
    res = build_lv_motion(grids, [0, .2, .4, .6], cfg=cfg, period=.8, verbose=False, out=tmp_path / "mov")
    q = res["quality"]
    assert res["valid"] and res["filled"] == [3] and res["adjustments"] == []
    assert q[3]["status"] == "filled" and "between phases 2 and 0 (weight 0.50)" in q[3]["note"]
    assert "rows 20-20 of 40 do not wind once" in q[3]["note"]
    assert "the surface is the problem" in q[3]["note"]            # its cells passed; the surface did not
    v = res["volumes_mL"]
    assert np.isclose(v[3], 0.5 * (v[2] + v[0]), rtol=0.05)       # the fill is the mean of its neighbours
    assert np.allclose(res["points"][3], 0.5 * (res["points"][2] + res["points"][0]))
    txt = (tmp_path / "mov" / "motion_quality.csv").read_text().splitlines()
    assert txt[0].endswith(",status,note") and txt[4].split(",")[-2] == "filled"
    assert not (tmp_path / "mov" / "motion_adjustments.txt").exists()
    # with the fill off the phase stays failed and the case is invalid
    res = build_lv_motion(grids, [0, .2, .4, .6], cfg=cfg, period=.8, verbose=False, fill_failed=False)
    assert not res["valid"] and res["quality"][3]["status"] == "failed" and res["filled"] == []
    # without a period the last phase has no later neighbour: it is copied from the previous one
    res = build_lv_motion(grids, [0, .2, .4, .6], cfg=cfg, verbose=False)
    assert res["valid"] and "copied from phase 2" in res["quality"][3]["note"]


def _fake_builder(fail_when, span, n_bad=3):
    """Stand-in for _build_phase: fails with ``n_bad`` cells at ``span`` while
    ``fail_when(cfg)`` holds, so the rim-retry logic can be tested on its own."""
    def build(G, frame, builder, cfg):
        v = np.full(10, 100.0)
        bad = n_bad if fail_when(cfg) else 0
        return np.zeros((8, 3)), [[0]], v, None, bad, (span if bad else (np.nan, np.nan)), {}
    return build


def test_rim_retry_lowers_base_cut_until_the_gate_passes(monkeypatch):
    monkeypatch.setattr(motion_mod, "_build_phase", _fake_builder(lambda c: c.base_cut > 0.95, (0.97, 1.0)))
    frame = (np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.zeros(3))
    G, *_ = synthetic_lv()
    b = build_phases([G, G], frame, "conformal", ConformalLVConfig(), verbose=False)
    assert np.isclose(b["cfg"].base_cut, 0.94) and len(b["adjustments"]) == 3      # two cuts + the removed-volume line
    assert all("base_cut" in a and "base rim" in a for a in b["adjustments"][:2])
    assert b["adjustments"][2].startswith("the trims remove")
    assert all(r["status"] == "ok" for r in b["rows"])


def test_rim_retry_raises_apex_trim_and_stops_at_max_retries(monkeypatch):
    monkeypatch.setattr(motion_mod, "_build_phase", _fake_builder(lambda c: c["apex_trim"] < 0.05, (0.0, 0.2)))
    frame = (np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.zeros(3))
    G, *_ = synthetic_lv()
    b = build_phases([G, G], frame, "single", dict(apex_trim=0.01), verbose=False)
    assert np.isclose(b["cfg"]["apex_trim"], 0.05) and len(b["adjustments"]) == 3
    monkeypatch.setattr(motion_mod, "_build_phase", _fake_builder(lambda c: True, (0.97, 1.0)))
    b = build_phases([G, G], frame, "conformal", ConformalLVConfig(), verbose=False, max_retries=3)
    assert len(b["adjustments"]) == 4 and all(r["status"] == "failed" for r in b["rows"])


def test_failures_away_from_a_rim_are_not_retried(monkeypatch):
    monkeypatch.setattr(motion_mod, "_build_phase", _fake_builder(lambda c: True, (0.4, 0.6)))
    frame = (np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.zeros(3))
    G, *_ = synthetic_lv()
    b = build_phases([G, G], frame, "conformal", ConformalLVConfig(), verbose=False)
    assert b["adjustments"] == [] and b["cfg"].base_cut == 1.0
    assert set(b["degenerate"]) == {0, 1} and "not at a rim" in b["rows"][0]["note"]


SMALL4 = dict(NT_c=16, NR_c=3, apex_levels=4, body_levels=6, base_levels=4)


def _read_points(case):
    txt = open(os.path.join(case, "constant", "polyMesh", "points")).read()
    body = txt.split("(\n", 1)[1].split("\n)")[0]
    return np.array([[float(v) for v in ln.strip("()").split()] for ln in body.splitlines() if ln.strip()])


def test_hex_volumes_are_signed():
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    assert np.isclose(hex_volumes(pts, [[0, 1, 2, 3, 4, 5, 6, 7]])[0], 1.0)
    assert np.isclose(hex_volumes(pts, [[4, 5, 6, 7, 0, 1, 2, 3]])[0], -1.0)      # mirrored hex


@pytest.mark.parametrize("builder,cfg", [("conformal", ConformalLVConfig(**SMALL4)),
                                         ("single", dict(NR=5, core_frac=0.35))])
def test_filled_reference_phase_is_written_from_the_filled_points(tmp_path, builder, cfg):
    """When the reference phase itself was filled from its neighbours, the written
    constant/polyMesh must hold the filled points (so polyMesh + displacement is
    every other phase), for both builders; a failed phase is never the reference;
    write_invalid=False writes nothing for an invalid set of phases."""
    grids = _phases()
    grids[2][20] = grids[2][20].mean(0)                           # the smallest (end-systolic) phase is degenerate
    res = build_lv_motion(grids, [0, .2, .4, .6], builder=builder, cfg=cfg, period=.8, reference=2,
                          verbose=False, out=tmp_path / "mov")
    assert res["valid"] and res["filled"] == [2] and res["ref"] == 2
    assert np.allclose(_read_points(tmp_path / "mov"), res["points"][2] * 1e-3, atol=1e-8)
    assert np.abs(_read_field(tmp_path / "mov" / "0.4" / "wantedDisplacement")).max() == 0.0
    res = build_lv_motion(grids, [0, .2, .4, .6], builder=builder, cfg=cfg, period=.8, reference="es",
                          verbose=False, fill_failed=False)
    assert not res["valid"] and res["quality"][2]["status"] == "failed"
    assert res["ref"] == 1                                        # the smallest phase that passed
    res = build_lv_motion(grids, [0, .2, .4, .6], builder=builder, cfg=cfg, period=.8, verbose=False,
                          fill_failed=False, out=tmp_path / "none", write_invalid=False)
    assert res["case"] is None and not (tmp_path / "none").exists()


def test_metrics_rows_carry_the_severe_non_orthogonality_count():
    from hexmb.services.motion import build_phases
    G, axis, e1, e2, mu = synthetic_lv()
    b = build_phases([G], (axis, e1, e2, mu), "conformal", ConformalLVConfig(**SMALL4), verbose=False, metrics=True)
    r = b["rows"][0]
    assert r["n_non_orthogonality_gt70"] == 0 and r["max_non_orthogonality_deg"] < 70


def test_cli_motion_honours_allow_invalid(tmp_path, capsys):
    from hexmb.cli import main
    grids = _phases()
    grids[2][20] = grids[2][20].mean(0)
    np.savez(tmp_path / "g.npz", grids=np.stack(grids))
    flags = ["--NT-c", "16", "--NR-c", "3", "--apex-levels", "4", "--body-levels", "6", "--base-levels", "4",
             "--period", "0.8", "--no-fill"]
    rc = main(["motion", "--surfaces", str(tmp_path / "g.npz"), "-o", str(tmp_path / "a"), *flags])
    out = capsys.readouterr().out
    assert rc == 1 and "ABORT" in out and not (tmp_path / "a").exists()
    rc = main(["motion", "--surfaces", str(tmp_path / "g.npz"), "-o", str(tmp_path / "b"), *flags, "--allow-invalid"])
    assert rc == 1 and (tmp_path / "b" / "constant" / "polyMesh" / "points").exists()
