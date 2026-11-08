"""Conformal split-hex transition: exactness, guards, one-region telescoping tube."""
from collections import defaultdict

import numpy as np
import pytest

from hexmb.assembly.splithex import (
    PolyMesh, build_telescoping_ogrid, graded_radial, ogrid_section, subdivide_ring,
)


def _circle(n, r=10.0, z=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(a), r * np.sin(a), np.full(n, z)], 1)


def test_subdivide_ring_keeps_coarse_nodes_and_adds_midpoints():
    c = _circle(8)
    f = subdivide_ring(c)
    assert f.shape == (16, 3)
    assert np.allclose(f[0::2], c)
    assert np.allclose(f[1::2], 0.5 * (c + np.roll(c, -1, axis=0)))


def test_section_is_exactly_subdivisible():
    c = _circle(16)
    rad_c = np.linspace(0, 1, 4)
    rad_f = np.zeros(7); rad_f[0::2] = rad_c; rad_f[1::2] = 0.5 * (rad_c[:-1] + rad_c[1:])
    core_c, pet_c = ogrid_section(c, 0.42, rad_c)
    core_f, pet_f = ogrid_section(subdivide_ring(c), 0.42, rad_f)
    assert np.allclose(core_c, core_f[0::2, 0::2])          # coarse core nodes are fine nodes
    for pc, pf in zip(pet_c, pet_f):
        assert np.allclose(pc, pf[0::2, 0::2])              # same for every petal


def test_nt_must_be_divisible_by_four():
    with pytest.raises(ValueError):
        ogrid_section(_circle(10), 0.42, np.linspace(0, 1, 3))


def test_graded_radial_boundary_layer():
    r = graded_radial(6, wall_ratio=0.3)
    d = np.diff(r)
    assert r[0] == 0 and np.isclose(r[-1], 1.0) and np.all(d > 0)
    assert np.isclose(d[-1] / d[0], 0.3)


def _regions(mesh):
    """Connected cell regions through shared faces."""
    owner = defaultdict(list)
    for ci, c in enumerate(mesh.cells):
        for f in c:
            owner[frozenset(f)].append(ci)
    adj = defaultdict(set)
    for cells in owner.values():
        assert len(cells) <= 2, "a face shared by more than two cells"
        if len(cells) == 2:
            a, b = cells
            adj[a].add(b); adj[b].add(a)
    seen, n = set(), 0
    for s in range(len(mesh.cells)):
        if s in seen:
            continue
        n += 1
        stack = [s]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u); stack.extend(adj[u] - seen)
    return n


def test_telescoping_tube_is_one_positive_region():
    m = build_telescoping_ogrid(lambda f: _circle(48, 10.0, 100.0 * f), NT_c=16, NR_c=3,
                                apex=(0.0, 0.3, 4), body=(0.3, 0.7, 6), base=(0.7, 1.0, 4))
    v = m.cell_volumes()
    assert (v > 0).all()
    assert m.n_split == 2 * (16 + 4 * 2 * 4)           # one coarse layer of cells at each join (Q=4)
    assert m.n_hex + m.n_split == len(m.cells)
    assert _regions(m) == 1
    assert np.isclose(v.sum(), np.pi * 10.0 ** 2 * 100.0, rtol=0.03)   # polygonal ring ~ circle


def test_polymesh_dedups_shared_face(tmp_path):
    m = PolyMesh()
    a = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    b = [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    c = [[0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2]]
    from hexmb.assembly.splithex import emit_ogrid_cell
    emit_ogrid_cell(m, a + b); emit_ogrid_cell(m, b + c)
    assert len(m.P) == 12
    nc, ni, npt = m.write(tmp_path / "case")
    assert (nc, ni, npt) == (2, 1, 12)
    assert m.patches == ["wall"]
    assert (tmp_path / "case" / "constant" / "polyMesh" / "boundary").exists()


def test_signed_pyramids_catch_mirrored_and_folded_cells():
    """The gate must see a mirrored cell (negative) and a folded surface (some
    negative pyramids) that an unsigned volume and checkMesh both miss."""
    from hexmb.assembly.splithex import PolyMesh, emit_ogrid_cell
    q = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    t = [(x, y, 1) for x, y, _ in q]
    pm = PolyMesh()
    emit_ogrid_cell(pm, q + t)                 # right-handed
    emit_ogrid_cell(pm, t + q)                 # mirrored
    assert np.allclose(pm.cell_volumes(), [1.0, -1.0])
    mp = pm.min_pyramid_per_cell()
    assert mp[0] > 0 and mp[1] < 0

    from _synthetic import synthetic_lv
    from hexmb.templates.conformal_lv import ConformalLVConfig, build_conformal_lv, polymesh_quality
    G, axis, e1, e2, mu = synthetic_lv()
    cfg = ConformalLVConfig(NT_c=16, NR_c=3, apex_levels=5, body_levels=8, base_levels=5)
    good = polymesh_quality(build_conformal_lv(G, axis, e1, e2, cfg, mu=mu))
    assert good["n_nonpositive"] == 0 and good["min_pyramid_mm3"] > 0
    # a band of rings folded back along the axis is caught before the build: the
    # builder warns, drops the rows from the fold up, and meshes what is left
    Gf = G.copy(); Gf[15:26] = Gf[15:26][::-1]
    with pytest.warns(UserWarning, match="fold back along the axis"):
        cut = polymesh_quality(build_conformal_lv(Gf, axis, e1, e2, cfg, mu=mu))
    assert cut["n_nonpositive"] == 0 and cut["volume_mL"] < 0.6 * good["volume_mL"]
    # a fold the ring centres cannot see (one layer of mesh points pushed back past
    # the layer below it) must be caught by the signed pyramids, while the unsigned
    # volume of every cell stays positive
    m = build_conformal_lv(G, axis, e1, e2, cfg, mu=mu)
    P = m.points.copy(); s = (P - mu) @ axis
    levels = np.unique(np.round(s, 3)); j = len(levels) // 2
    d = levels[j] - levels[j - 1]
    layer = np.abs(s - levels[j]) < 0.25 * d
    P[layer] -= 1.5 * d * axis                                 # push one layer below the one beneath it
    m.P = [tuple(p) for p in P]; m._ft_key = None
    assert (m.min_pyramid_per_cell() <= 0).sum() > 0
    assert (np.abs(m.cell_volumes()) > 0).all()
    # a clockwise contour must give the same valid mesh (direction is normalised)
    rev = polymesh_quality(build_conformal_lv(G[:, ::-1, :], axis, e1, e2, cfg, mu=mu))
    assert rev["n_nonpositive"] == 0 and np.isclose(rev["volume_mL"], good["volume_mL"], rtol=1e-6)


def test_top_faces_tagged_and_zone_order_checked():
    from _synthetic import synthetic_lv
    from hexmb.templates.conformal_lv import ConformalLVConfig, build_conformal_lv
    G, axis, e1, e2, mu = synthetic_lv()
    cfg = ConformalLVConfig(NT_c=16, NR_c=3, apex_levels=5, body_levels=8, base_levels=5)
    m = build_conformal_lv(G, axis, e1, e2, cfg, mu=mu)
    q = cfg.NT_c // 4
    assert len(m.top_faces) == q * q + 4 * (cfg.NR_c - 1) * q      # one tag per cell of the last layer
    with pytest.raises(ValueError):
        build_conformal_lv(G, axis, e1, e2, ConformalLVConfig(base_frac=0.8, base_cut=0.7), mu=mu)
    with pytest.raises(ValueError):                                 # metres, not millimetres
        build_conformal_lv(G * 1e-3, axis, e1, e2, cfg, mu=mu * 1e-3)


def test_one_layer_coarse_zone_splits_both_ends():
    """A coarse zone one layer thick between two finer zones faces a finer
    neighbour across BOTH end planes, so each of its cells is split top and
    bottom (12 faces).  Splitting only one end leaves the other coarse quad
    facing four fine quads, which the topology then reports as boundary faces
    inside the domain."""
    from hexmb.assembly.splithex import build_multilevel_telescoping_ogrid, emit_ogrid_cell
    zones = [(0.0, 0.3, 3, 1), (0.3, 0.6, 2, 0), (0.6, 1.0, 3, 1)]
    m = build_multilevel_telescoping_ogrid(lambda f: _circle(64, 10.0, 100.0 * f), NT_c=16, NR_c=3,
                                           zones=zones, smooth=False)
    faces, owner, neigh, patches = m.topology()
    n_cap = 8 * 8 + 4 * 4 * 8                           # cells of one level-1 cross-section (Q 8, NR 5)
    n_wall = 2 * 32 + 1 * 16 + 2 * 32                   # wall faces per layer = ring size of that level
    assert len(faces) - len(neigh) == 2 * n_cap + n_wall
    assert m.n_split == 16 + 4 * 2 * 4 and all(len(c) == 12 for c in m.cells if len(c) != 6)
    assert (m.min_pyramid_per_cell() > 0).all() and _regions(m) == 1
    assert np.isclose(m.cell_volumes().sum(), np.pi * 10.0 ** 2 * 100.0, rtol=0.03)
    pm = PolyMesh()
    q = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    emit_ogrid_cell(pm, q + [(x, y, 1) for x, y, _ in q], top_split=True, bot_split=True, tag_top="x")
    assert len(pm.cells[0]) == 12 and len(pm.top_faces) == 4 and np.isclose(pm.cell_volumes()[0], 1.0)


def test_zero_area_faces_are_counted_once():
    """A collapsed face shared by two cells appears twice in face_areas() (once
    per cell) and once in n_zero_area_faces()."""
    from hexmb.assembly.splithex import emit_ogrid_cell
    m = PolyMesh()
    a = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    b = [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    c = [[0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2]]
    emit_ogrid_cell(m, a + b); emit_ogrid_cell(m, b + c)
    P = m.points.copy(); mid = P[:, 2] == 1.0
    P[mid] = P[mid].mean(0)                                  # the shared face collapses to a point
    m.P = [tuple(p) for p in P]; m._ft_key = None
    assert int((m.face_areas() <= 1e-9).sum()) == 2 and m.n_zero_area_faces() == 1
