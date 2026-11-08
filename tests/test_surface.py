"""Bring-your-own-surface reader: STL (open and closed), npz, CAP, mask."""
from importlib import resources

import numpy as np
import pytest

from _synthetic import synthetic_lv

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("rtree")

from hexmb.io.stl_surface import write_surface_stl          # noqa: E402
from hexmb.io.surface import (                              # noqa: E402
    auto_landmarks, endo_grid_from_mesh, grid_from_file, load_surface_mesh, write_surface_npz,
)


def _analytic_radius(z, length=80.0, radius=25.0):
    f = z / length
    return np.maximum(radius * np.sqrt(np.maximum(1e-3, 1 - (1 - f) ** 2)), 2.0)


def test_open_stl_roundtrip(tmp_path):
    G, *_ = synthetic_lv(NL=80, NT=96)
    write_surface_stl(G, tmp_path / "lv.stl")
    mesh = load_surface_mesh(tmp_path / "lv.stl")
    apex, base, kind = auto_landmarks(mesh)
    assert kind == "open" and base[2] > apex[2]               # base is the wide top opening
    G2, axis, e1, e2, mu, info = endo_grid_from_mesh(mesh, NL=40, NT=56)
    assert G2.shape == (40, 56, 3)
    assert abs(axis[2]) > 0.99
    r = np.linalg.norm(G2[:, :, :2], axis=2).mean(1)
    err = np.abs(r - _analytic_radius(G2[:, :, 2].mean(1))) / r
    assert err.max() < 0.02
    assert info["interpolated_rays_per_level"].max() == 0


def test_closed_ellipsoid_auto_landmarks():
    sph = trimesh.creation.icosphere(subdivisions=3)
    ell = sph.copy()
    V = sph.vertices * np.array([20.0, 25.0, 45.0]) + np.array([5.0, -3.0, 10.0])
    V[:, 2] = np.minimum(V[:, 2], 40.0)                         # flat base cap at z = 40
    ell.vertices = V
    assert ell.is_watertight
    apex, base, kind = auto_landmarks(ell)
    assert kind == "closed" and apex[2] < base[2]
    G, axis, e1, e2, mu, info = endo_grid_from_mesh(ell, NL=24, NT=32)
    assert abs(axis[2]) > 0.99
    zc = G[:, :, 2].mean(1)
    i = int(np.argmin(np.abs(zc - 10.0)))
    rr = np.linalg.norm(G[i, :, :2] - G[i, :, :2].mean(0), axis=1)
    assert 18.5 < rr.min() < 21.5 and 23.5 < rr.max() < 26.5


def test_grid_from_file_dispatch(tmp_path):
    G, axis, e1, e2, mu = synthetic_lv()
    write_surface_npz(tmp_path / "g.npz", G, axis, e1, e2, mu)
    Gn, *_, info = grid_from_file(tmp_path / "g.npz")
    assert info["source"] == "npz" and np.allclose(Gn, G)
    ex = resources.files("hexmb") / "data" / "cap" / "SCD0000101" / "SCD0000101_1.model.exnode"
    Gc, axis_c, *_, info = grid_from_file(str(ex), NL=20, NT=32)
    assert info["source"] == "cap" and Gc.shape == (20, 32, 3)
    with pytest.raises(ValueError):
        grid_from_file(tmp_path / "x.unknown")


def test_mask_to_grid(tmp_path):
    nib = pytest.importorskip("nibabel")
    pytest.importorskip("skimage")
    from hexmb.io.surface import grid_from_mask
    # an analytic ellipsoid mask on a 1 mm grid, with an oblique affine
    x, y, z = np.mgrid[0:64, 0:64, 0:96].astype(float)
    mask = (((x - 32) / 20.0) ** 2 + ((y - 32) / 24.0) ** 2 + ((z - 45) / 42.0) ** 2) <= 1.0
    mask &= z <= 75                                            # flat base
    aff = np.eye(4); aff[:3, 3] = [-10.0, 5.0, 2.0]
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), aff), str(tmp_path / "lv.nii.gz"))
    G, axis, e1, e2, mu, info = grid_from_mask(tmp_path / "lv.nii.gz", NL=24, NT=32)
    assert G.shape == (24, 32, 3) and abs(axis[2]) > 0.98
    assert abs(info["mask_volume_mL"] - mask.sum() * 1e-3) < 1e-6
    zc = G[:, :, 2].mean(1)
    i = int(np.argmin(np.abs(zc - (45.0 + 2.0))))              # mid-ellipsoid in world z
    rr = np.linalg.norm(G[i, :, :2] - G[i, :, :2].mean(0), axis=1)
    assert 18.0 < rr.min() < 22.0 and 22.0 < rr.max() < 26.0


def test_metre_scale_surface_is_rejected(tmp_path):
    pytest.importorskip("trimesh"); pytest.importorskip("rtree")
    import trimesh
    from hexmb.io.surface import endo_grid_from_mesh
    G, *_ = synthetic_lv()
    f = tmp_path / "metres.stl"
    write_surface_stl(G * 1e-3, f, scale=1.0)
    with pytest.raises(ValueError, match="millimetres"):
        endo_grid_from_mesh(trimesh.load(str(f)), NL=12, NT=16)


def test_landmarks_and_label_are_rejected_where_they_do_not_apply(tmp_path):
    from hexmb.io.surface import grid_from_file, write_surface_npz
    G, axis, e1, e2, mu = synthetic_lv()
    f = tmp_path / "g.npz"; write_surface_npz(f, G, axis, e1, e2, mu)
    with pytest.raises(ValueError):
        grid_from_file(f, apex=np.zeros(3))
    with pytest.raises(ValueError):
        grid_from_file(tmp_path / "missing.stl", label=1)


def _prolate_mask(zooms, shape, a=25.0, b=25.0, c=60.0, base_frac=0.85):
    """A half-prolate blood pool (apex down, flat base) voxelised at ``zooms``;
    returns (mask, affine, exact volume in mL)."""
    i, j, k = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(float)
    x = (i - shape[0] / 2) * zooms[0]; y = (j - shape[1] / 2) * zooms[1]
    z = (k + 0.5) * zooms[2] - 5.0                                 # apex at z = 0, base at z = c*base_frac
    mask = ((x / a) ** 2 + (y / b) ** 2 + ((z - c) / c) ** 2 <= 1.0) & (z <= c * base_frac) & (z >= 0)
    aff = np.diag([zooms[0], zooms[1], zooms[2], 1.0]); aff[:3, 3] = [-shape[0] / 2 * zooms[0], -shape[1] / 2 * zooms[1], 0.5 * zooms[2] - 5.0]
    h = c * base_frac                                              # cap volume of the spheroid from the apex to the cut
    exact = np.pi * a * b / c ** 2 * (h ** 2 * c - h ** 3 / 3)     # integral of pi*a*b*(1-((z-c)/c)^2) dz
    return mask, aff, exact * 1e-3


def test_slice_reader_on_a_thick_short_axis_stack(tmp_path):
    nib = pytest.importorskip("nibabel"); pytest.importorskip("skimage"); pytest.importorskip("scipy")
    from hexmb.io.surface import grid_from_mask
    from hexmb.templates.conformal_lv import bad_rings, grid_volume
    zooms = (1.25, 1.25, 8.0)
    mask, aff, exact = _prolate_mask(zooms, (64, 64, 10))
    f = tmp_path / "sa.nii.gz"
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), aff), str(f))
    G, axis, e1, e2, mu, info = grid_from_mask(f, NL=40, NT=56)              # auto -> slices (8 mm >= 2.5 x 1.25 mm)
    assert info["method"] == "slices" and info["n_slices"] >= 5 and abs(info["slice_spacing_mm"] - 8.0) < 1e-6
    assert G.shape == (40, 56, 3) and axis[2] > 0.99                        # apex (small outline) is at low z
    assert np.allclose(G[0], G[0, 0]) and G[-1, :, 2].mean() > G[1, :, 2].mean()
    assert not bad_rings(G, axis, e1, e2), "every ring must wind once about the axis"
    assert abs(info["mask_volume_mL"] - mask.sum() * np.prod(zooms) * 1e-3) < 1e-6
    v = grid_volume(G) * 1e-3
    assert abs(v / exact - 1) < 0.04, f"loft volume {v:.1f} mL vs exact {exact:.1f} mL"
    # the same stack at isotropic spacing takes the marching-cubes route
    pytest.importorskip("trimesh"); pytest.importorskip("rtree")
    zooms = (1.5, 1.5, 1.5)
    mask2, aff2, _ = _prolate_mask(zooms, (48, 48, 48))
    f2 = tmp_path / "iso.nii.gz"
    nib.save(nib.Nifti1Image(mask2.astype(np.uint8), aff2), str(f2))
    _, _, _, _, _, info2 = grid_from_mask(f2, NL=24, NT=32)
    assert info2["method"] == "marching"
    _, _, _, _, _, info3 = grid_from_mask(f2, NL=24, NT=32, method="slices")   # the route can be forced
    assert info3["method"] == "slices" and info3["n_slices"] > 20
