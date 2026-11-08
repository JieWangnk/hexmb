"""Cardiac Atlas reader: exact bicubic Hermite evaluation of the Sunnybrook models."""
import re
import warnings
from importlib import resources

import numpy as np
import pytest

from hexmb.io.cap import cap_endo_surface, find_exelem, parse_exelem, parse_exnode, surface_frame

DATA = resources.files("hexmb") / "data" / "cap"
EXNODE = str(DATA / "SCD0000101" / "SCD0000101_1.model.exnode")


def _endo_ids(exnode):
    """Node ids of the inner (endo) xi3 face, decided as the reader does (mean lambda per face)."""
    nodes, _ = parse_exnode(exnode)
    elems = parse_exelem(find_exelem(exnode))
    lo = np.mean([nodes[n]["lam"] for e, _ in elems for n in e[:4]])
    hi = np.mean([nodes[n]["lam"] for e, _ in elems for n in e[4:]])
    return sorted({n for e, _ in elems for n in (e[:4] if lo < hi else e[4:])})


def _node_xyz(n, focus):
    return np.array([focus * np.cosh(n["lam"]) * np.cos(n["mu"]),
                     focus * np.sinh(n["lam"]) * np.sin(n["mu"]) * np.cos(n["th"]),
                     focus * np.sinh(n["lam"]) * np.sin(n["mu"]) * np.sin(n["th"])])


def test_bundled_template_is_found_and_parses():
    ex = find_exelem(EXNODE)
    assert ex.endswith("GlobalHermiteParam.exelem")
    elems = parse_exelem(ex)
    assert len(elems) == 16 and all(len(e) == 8 for e, _ in elems)
    assert all(np.allclose(sf, 1.0) for _, sf in elems)            # unit scale factors in these models


def test_nodes_are_reproduced_exactly():
    nodes, focus = parse_exnode(EXNODE)
    G = cap_endo_surface(EXNODE, NL=4 * 24 + 1, NT=360)            # rows hit xi2 = 0..4, columns hit 90 deg
    P = G.reshape(-1, 3)
    inner = _endo_ids(EXNODE)
    assert len(inner) == 20
    worst = max(np.linalg.norm(P - _node_xyz(nodes[n], focus), axis=1).min() for n in inner)
    assert worst < 1e-3
    outer = [n for n in nodes if n not in inner]
    far = min(np.linalg.norm(P - _node_xyz(nodes[n], focus), axis=1).min() for n in outer)
    assert far > 1.0                                                # the epi layer is not on the endo surface
    Ge = cap_endo_surface(EXNODE, NL=4 * 24 + 1, NT=360, layer="epi")
    worst_e = max(np.linalg.norm(Ge.reshape(-1, 3) - _node_xyz(nodes[n], focus), axis=1).min() for n in outer)
    assert worst_e < 1e-3


def test_rows_wind_once_about_the_axis_and_apex_is_a_point():
    G = cap_endo_surface(EXNODE, NL=40, NT=56)
    assert G.shape == (40, 56, 3)
    assert np.ptp(G[0], axis=0).max() < 1e-9                        # apex row is one point
    axis, e1, e2, mu = surface_frame(G)
    for L in range(1, 40):                                          # every ring star-shaped about its centre
        d = G[L] - G[L].mean(0)
        cr = np.cross(d, np.roll(d, -1, axis=0)) @ axis
        assert (cr > 0).all() or (cr < 0).all(), f"ring {L} folds"
    r = np.linalg.norm(G[1:4] - G[1:4].mean(1, keepdims=True), axis=2).mean(1)
    assert r[0] < r[1] < r[2] and r[0] < 5.0                        # small, growing apex rings


def test_axis_crossing_model_warns(tmp_path):
    txt = open(EXNODE).read()
    blocks = re.split(r"(?=\n\s*Node:)", txt)
    # push one base-row endo node onto the axis (lambda -> -0.05)
    nodes, focus = parse_exnode(EXNODE)
    inner = _endo_ids(EXNODE)
    base_node = max(inner, key=lambda n: nodes[n]["mu"])
    for i, b in enumerate(blocks):
        if re.match(rf"\n\s*Node:\s*{base_node}\s*\n", b):
            lines = b.split("\n")
            k = next(j for j, ln in enumerate(lines) if ln.strip() and not ln.strip().startswith("Node"))
            v = lines[k].split(); v[0] = "-0.05"
            lines[k] = "  " + "  ".join(v)
            blocks[i] = "\n".join(lines)
    bad = tmp_path / "SCD0000101_1.model.exnode"
    bad.write_text("".join(blocks))
    with pytest.warns(UserWarning, match="crosses its own long axis"):
        cap_endo_surface(str(bad))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cap_endo_surface(EXNODE)                                    # the real model does not warn


def test_wrong_element_file_is_rejected(tmp_path):
    ex = find_exelem(EXNODE)
    txt = open(ex).read().replace("              26           25           22           21            6            5            2            1",
                                  "              26           25           22           21            6            5            2           99")
    (tmp_path / "x.exelem").write_text(txt)
    with pytest.raises(ValueError, match="node"):
        cap_endo_surface(EXNODE, exelem=str(tmp_path / "x.exelem"))


def test_truncated_element_file_is_rejected(tmp_path):
    ex = find_exelem(EXNODE)
    head = open(ex).read().rsplit("Scale factors:", 1)[0]        # the last element loses its scale factors
    (tmp_path / "t.exelem").write_text(head + "\n")
    with pytest.raises(ValueError, match="Scale factors"):
        parse_exelem(str(tmp_path / "t.exelem"))
