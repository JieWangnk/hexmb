#!/usr/bin/env python
"""Raw-mask test on the bundled CC0 heart: voxelise the exact Hermite model the
way a scanner and a segmentation network would deliver it, then read the masks
back through hexmb's mask readers and mesh every phase.

Variants (all 20 phases of SCD0000101):
  sa10   short-axis stack, 1.4 x 1.4 mm pixels, 10 mm slices   (cine CMR as acquired)
  sa8    short-axis stack, 1.4 x 1.4 mm pixels,  8 mm slices
  iso1   isotropic 1.0 mm voxels                               (CT-like)

Label 1 is the blood pool (inside the endocardial surface), label 2 the wall
(between the endocardial and epicardial layers of the model), so the masks look
like a network's output and exercise ``--label 1``.  A voxel is labelled by its
centre.  The thick-slice stacks go through the slice reader, the isotropic mask
through marching cubes and the ray cast (``method="auto"`` picks these).

For every variant the script reports, per phase, the exact volume of the model,
the mask's own volume (voxel count: Simpson's rule for the stacks) and the mesh
volume, plus the gate result, and draws examples/img/synthetic_masks.png.

usage: python scripts/synthetic_masks.py OUT_DIR [--phases 20] [--variants sa10 sa8 iso1]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hexmb.io.cap import cap_endo_surface, surface_frame            # noqa: E402
from hexmb.io.surface import grid_from_file                          # noqa: E402
from hexmb.templates.conformal_lv import grid_volume                 # noqa: E402
from hexmb.services.motion import build_lv_motion                    # noqa: E402

VARIANTS = {"sa10": dict(pix=1.4, slab=10.0), "sa8": dict(pix=1.4, slab=8.0), "iso1": dict(pix=1.0, slab=1.0)}
FINE_NL, FINE_NT = 400, 512


def natural(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(p))]


def bundled_phases(n):
    from importlib import resources
    droot = resources.files("hexmb") / "data" / "cap" / "SCD0000101"
    return sorted((str(p) for p in droot.iterdir() if str(p).endswith(".exnode")), key=natural)[:n]


def section_polygon(G, axis, e1, e2, origin, z):
    """In-plane (u, v) polygon of the ring grid's cross-section at axial height z
    (one point per surface column, linear between rows); None outside the surface."""
    s = (G - origin) @ axis                                   # (NL, NT) axial coordinate of every grid point
    smin, smax = s.min(0), s.max(0)                           # per column
    if z < smin.max() or z > smax.min():
        return None
    NL, NT, _ = G.shape
    pts = np.empty((NT, 3))
    for j in range(NT):
        col = s[:, j]
        i = np.searchsorted(col, z)                            # rows are monotone along the axis
        i = min(max(i, 1), NL - 1)
        w = (z - col[i - 1]) / (col[i] - col[i - 1] + 1e-30)
        pts[j] = (1 - w) * G[i - 1, j] + w * G[i, j]
    d = pts - origin
    return np.stack([d @ e1, d @ e2], 1)


def voxelise(G_endo, G_epi, axis, e1, e2, origin, pix, slab, fov=180.0, z0=-12.0, z1=None):
    """Label volume (nx, ny, nz) in a frame whose third axis is the LV axis.
    Voxel (i, j, k) centre = origin + (i - nx/2 + 0.5) pix e1 + (j - ny/2 + 0.5) pix e2 + (z0 + (k + 0.5) slab) axis."""
    from matplotlib.path import Path as MPath
    n = int(round(fov / pix))
    z1 = z1 if z1 is not None else float(((G_epi - origin) @ axis).max()) + slab
    nz = int(np.ceil((z1 - z0) / slab))
    lab = np.zeros((n, n, nz), np.uint8)
    u = (np.arange(n) - n / 2 + 0.5) * pix
    U, V = np.meshgrid(u, u, indexing="ij")
    uv = np.stack([U.ravel(), V.ravel()], 1)
    for k in range(nz):
        z = z0 + (k + 0.5) * slab
        pe = section_polygon(G_epi, axis, e1, e2, origin, z)
        if pe is None:
            continue
        inside_epi = MPath(pe).contains_points(uv)
        pn = section_polygon(G_endo, axis, e1, e2, origin, z)
        inside_endo = MPath(pn).contains_points(uv) if pn is not None else np.zeros(len(uv), bool)
        sl = np.zeros(len(uv), np.uint8)
        sl[inside_epi] = 2
        sl[inside_endo] = 1
        lab[:, :, k] = sl.reshape(n, n)
    affine = np.eye(4)
    affine[:3, 0] = pix * e1; affine[:3, 1] = pix * e2; affine[:3, 2] = slab * axis
    affine[:3, 3] = origin + (-n / 2 + 0.5) * pix * e1 + (-n / 2 + 0.5) * pix * e2 + (z0 + 0.5 * slab) * axis
    return lab, affine


def main():
    import nibabel as nib
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--phases", type=int, default=20)
    ap.add_argument("--variants", nargs="+", default=["sa10", "sa8", "iso1"])
    ap.add_argument("--period", type=float, default=1.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = bundled_phases(a.phases)
    times = a.period * np.arange(len(files)) / len(files)
    print(f"exact surfaces at {FINE_NL} x {FINE_NT} ...")
    endo = [cap_endo_surface(f, NL=FINE_NL, NT=FINE_NT) for f in files]
    epi = [cap_endo_surface(f, NL=FINE_NL, NT=FINE_NT, layer="epi") for f in files]
    axis, e1, e2, mu = surface_frame(endo[0])
    origin = endo[0][0].mean(0)                                   # phase-1 apex: the scanner frame is fixed in time
    exact = np.array([grid_volume(G) * 1e-3 for G in endo])
    res = dict(exact_mL=exact.tolist(), times=times.tolist(), variants={})
    for name in a.variants:
        spec = VARIANTS[name]
        vdir = out / name; vdir.mkdir(exist_ok=True)
        t0 = time.time()
        mask_v, paths = [], []
        for k, (Gn, Gp) in enumerate(zip(endo, epi)):
            lab, aff = voxelise(Gn, Gp, axis, e1, e2, origin, spec["pix"], spec["slab"])
            p = vdir / f"phase_{k:02d}.nii.gz"
            nib.save(nib.Nifti1Image(lab, aff), str(p))
            mask_v.append(float((lab == 1).sum() * spec["pix"] ** 2 * spec["slab"] * 1e-3))
            paths.append(p)
        t_vox = time.time() - t0
        print(f"\n{name}: {len(paths)} masks ({lab.shape}, {spec['pix']} mm pixels, {spec['slab']} mm slices) in {t_vox:.0f} s; "
              f"mask EDV {max(mask_v):.1f} ESV {min(mask_v):.1f} mL (exact {exact.max():.1f} / {exact.min():.1f})")
        t0 = time.time()
        grids, frame, methods = [], None, set()
        for p in paths:
            G, ax, f1, f2, m, info = grid_from_file(str(p), label=1, frame=frame)
            if frame is None:
                frame = (ax, f1, f2)
            grids.append(G); methods.add(info["method"])
        t_read = time.time() - t0
        case = vdir / "lv_moving"
        t0 = time.time()
        r = build_lv_motion(grids, times, period=a.period, out=case, frame=(frame[0], frame[1], frame[2], grids[0].reshape(-1, 3).mean(0)),
                            verbose=False)
        t_mesh = time.time() - t0
        mesh_v = np.asarray(r["volumes_mL"])
        err = 100 * (mesh_v / exact - 1)
        res["variants"][name] = dict(spec=spec, shape=list(lab.shape), mask_mL=mask_v, mesh_mL=mesh_v.tolist(),
                                     method=sorted(methods), valid=bool(r["valid"]),
                                     statuses=[q["status"] for q in r["quality"]], adjustments=r["adjustments"],
                                     seconds=dict(voxelise=t_vox, read=t_read, mesh=t_mesh),
                                     EDV=dict(exact=float(exact.max()), mask=float(max(mask_v)), mesh=float(mesh_v.max())),
                                     ESV=dict(exact=float(exact.min()), mask=float(min(mask_v)), mesh=float(mesh_v.min())),
                                     EF_pct=dict(exact=float(100 * (1 - exact.min() / exact.max())),
                                                 mask=float(100 * (1 - min(mask_v) / max(mask_v))),
                                                 mesh=float(100 * (1 - mesh_v.min() / mesh_v.max()))),
                                     volume_error_pct=dict(mean=float(err.mean()), min=float(err.min()), max=float(err.max())))
        print(f"  reader {sorted(methods)} {t_read:.0f} s, mesh {t_mesh:.0f} s: {'VALID' if r['valid'] else 'INVALID'} "
              f"({sum(q['status'] == 'ok' for q in r['quality'])} ok of {len(grids)}); mesh EDV {mesh_v.max():.1f} ESV {mesh_v.min():.1f} "
              f"EF {100 * (1 - mesh_v.min() / mesh_v.max()):.0f} %; volume error vs exact mean {err.mean():+.1f} % "
              f"(range {err.min():+.1f} to {err.max():+.1f})" + (f"; adjustments: {r['adjustments']}" if r["adjustments"] else ""))
        (out / "results.json").write_text(json.dumps(res, indent=1))
    print(f"\nresults: {out / 'results.json'}")


if __name__ == "__main__":
    main()
