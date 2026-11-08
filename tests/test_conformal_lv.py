"""Conformal split-hex LV template on the synthetic prolate cavity."""
import re

import numpy as np
import pytest

from _synthetic import synthetic_lv
from hexmb.templates.conformal_lv import ConformalLVConfig, build_conformal_lv, polymesh_quality

SMALL = dict(NT_c=16, NR_c=3, apex_levels=5, body_levels=8, base_levels=5)


def _patches(case):
    txt = (case / "constant" / "polyMesh" / "boundary").read_text()
    return {n: int(k) for n, k in re.findall(r"(\w+)\n\s+\{\n\s+type \w+;\n\s+nFaces (\d+)", txt)}


def test_build_is_valid_and_counts_match():
    G, axis, e1, e2, mu = synthetic_lv()
    cfg = ConformalLVConfig(**SMALL)
    m = build_conformal_lv(G, axis, e1, e2, cfg, mu=mu)
    q = polymesh_quality(m)
    assert q["n_nonpositive"] == 0
    assert q["cells"] == cfg.approx_cells
    assert q["split_hex"] > 0 and q["hex"] + q["split_hex"] == q["cells"]
    assert 80 < q["volume_mL"] < 120                    # the synthetic cavity is ~103 mL
    assert m.frame is not None


def test_write_has_three_patches(tmp_path):
    G, axis, e1, e2, mu = synthetic_lv()
    m = build_conformal_lv(G, axis, e1, e2, ConformalLVConfig(**SMALL), mu=mu)
    m.write(tmp_path / "c")
    p = _patches(tmp_path / "c")
    assert set(p) == {"wall", "inlet", "outlet1"}
    assert p["inlet"] > p["outlet1"] > 0
    assert m.patches == list(p)


@pytest.mark.parametrize("mode", ["fullcap", "centred"])
def test_inlet_rules(tmp_path, mode):
    G, axis, e1, e2, mu = synthetic_lv()
    m = build_conformal_lv(G, axis, e1, e2, ConformalLVConfig(inlet=mode, **SMALL), mu=mu)
    m.write(tmp_path / mode)
    p = _patches(tmp_path / mode)
    assert "outlet1" not in p and p["inlet"] > 0


def test_bad_config_rejected():
    G, axis, e1, e2, mu = synthetic_lv()
    with pytest.raises(ValueError):
        build_conformal_lv(G, axis, e1, e2, ConformalLVConfig(NT_c=18), mu=mu)
    with pytest.raises(ValueError):
        build_conformal_lv(G, axis, e1, e2, ConformalLVConfig(inlet="nonsense", **SMALL), mu=mu)


def test_tilted_base_ring_keeps_whole_cap(tmp_path):
    """The valve-plane patches come from the tagged top faces, so tilting the
    base ring by 20 degrees must not hand the cap to 'wall'."""
    import os
    from hexmb.templates.conformal_lv import ConformalLVConfig, build_conformal_lv
    G, axis, e1, e2, mu = synthetic_lv()
    cfg = ConformalLVConfig(NT_c=16, NR_c=3, apex_levels=5, body_levels=8, base_levels=5)

    def patch_counts(Gx, out):
        m = build_conformal_lv(Gx, axis, e1, e2, cfg, mu=mu)
        m.write(out)
        txt = (out / "constant" / "polyMesh" / "boundary").read_text()
        import re
        return {n: int(k) for n, k in re.findall(r"(\w+)\s*\{\s*type \w+;\s*nFaces (\d+);", txt)}

    flat = patch_counts(G, tmp_path / "flat")
    Gt = G.copy()
    top = Gt[-1]; u = (top - top.mean(0)) @ e1
    Gt[-1] = top + np.outer(u * np.tan(np.radians(20.0)), axis)   # tilt the last ring 20 deg about e2
    tilted = patch_counts(Gt, tmp_path / "tilt")
    n_top = (cfg.NT_c // 4) ** 2 + 4 * (cfg.NR_c - 1) * (cfg.NT_c // 4)
    assert flat["inlet"] + flat["outlet1"] + flat.get("wall", 0) >= n_top
    assert tilted["inlet"] + tilted["outlet1"] == flat["inlet"] + flat["outlet1"]
    assert tilted["inlet"] + tilted["outlet1"] > 0.6 * n_top


def test_valve_classifier_reads_the_current_points():
    """The cap centre is re-read from the mesh points when the boundary is
    classified, so a mesh whose points were replaced after the build (a phase
    filled from its neighbours) gets the sectors of its own cap.  Translating
    the whole mesh must leave the patch sizes unchanged."""
    G, axis, e1, e2, mu = synthetic_lv()
    cfg = ConformalLVConfig(**SMALL)
    m = build_conformal_lv(G, axis, e1, e2, cfg, mu=mu)
    before = {k: v[1] for k, v in m.topology()[3].items()}
    shift = 6.0 * np.asarray(e1, float)                            # the ED -> ES cap shift of the bundled heart
    m.P = [tuple(p + shift) for p in m.points]; m._ft_key = None
    assert {k: v[1] for k, v in m.topology()[3].items()} == before
    stale = build_conformal_lv(G, axis, e1, e2, cfg, mu=mu)        # the old behaviour: centre frozen at build time
    cls = stale._classify; cls.prepare = None
    stale.P = [tuple(p + shift) for p in stale.points]; stale._ft_key = None
    assert {k: v[1] for k, v in stale.topology(cls)[3].items()} != before
