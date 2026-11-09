#!/usr/bin/env python
"""Circumferential drift of the wall nodes through the cycle, node by node.

``quality_through_cycle.py`` records one RMS and one maximum per phase of the
azimuth change of the wall nodes about the long axis.  This script builds one
level again and looks at the distribution: the azimuth change for nodes beyond
several radii from the axis (the angle is ill-conditioned near it), the
tangential slip in millimetres (radius x angle), the total node displacement,
where the worst node sits, and the apex region against the body.  Everything is
measured against the end-systolic phase, in the fixed frame of phase 0 that the
template uses.

usage: python scripts/wall_node_drift.py OUT.json [--level 2] [--phases all | 0,9,18]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from quality_through_cycle import (LEVELS, ConformalLVConfig, _build_phase, _screen_rings, bundled_phases,  # noqa: E402
                                   cap_endo_surface, grid_volume, surface_frame)


def stats(a):
    a = np.abs(np.asarray(a, float))
    if a.size == 0:
        return None
    return dict(n=int(a.size), max=float(a.max()), rms=float(np.sqrt((a ** 2).mean())),
                p50=float(np.percentile(a, 50)), p99=float(np.percentile(a, 99)), p999=float(np.percentile(a, 99.9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--phases", default="all", help="'all' or a comma-separated list of phase indices")
    ap.add_argument("--n-phases", type=int, default=20)
    a = ap.parse_args()
    spec = LEVELS[a.level]
    files = bundled_phases(a.n_phases)
    grids = [cap_endo_surface(f, NL=spec["NL"], NT=spec["NT"]) for f in files]
    frame = surface_frame(grids[0])
    axis, e1, e2, mu = (np.asarray(v, float) for v in frame)
    cfg = ConformalLVConfig(**spec["cfg"])
    cfg, adjustments, _ = _screen_rings(grids, frame, cfg)
    ref = int(np.argmin([grid_volume(G) for G in grids]))
    print(f"level {a.level}: reference (smallest surface volume) phase {ref}; adjustments {adjustments or 'none'}", flush=True)

    def cyl(P):
        r = P - mu
        x, y = r @ e1, r @ e2
        return np.hypot(x, y), np.degrees(np.arctan2(y, x)), r @ axis

    t0 = time.time()
    pts, _, _, m, n_bad, span, extra = _build_phase(grids[ref], frame, "conformal", cfg)
    faces, owner, neigh, patches = m.topology()
    w0, wn = patches["wall"]
    wall_pts = np.unique(np.concatenate([np.asarray(f, int) for f in faces[w0:w0 + wn]]))
    P_ref = np.asarray(pts, float)[wall_pts]
    rad_ref, ang_ref, z_ref = cyl(P_ref)
    h_ref = z_ref - (grids[ref][0].mean(0) - mu) @ axis       # height above the apex of the reference surface
    print(f"  reference built in {time.time() - t0:.0f} s: {len(wall_pts):,} wall nodes, {len(m.cells):,} cells, "
          f"wall radius {rad_ref.min():.2f}-{rad_ref.max():.1f} mm, height above apex {h_ref.min():.1f}-{h_ref.max():.1f} mm",
          flush=True)
    del m, pts
    res = dict(level=a.level, reference=ref, n_wall=int(len(wall_pts)), phases={})
    phases = range(len(grids)) if a.phases == "all" else [int(x) for x in a.phases.split(",")]
    for k in phases:
        if k == ref:
            continue
        t0 = time.time()
        pts, _, _, m, n_bad, span, extra = _build_phase(grids[k], frame, "conformal", cfg)
        P = np.asarray(pts, float)[wall_pts]
        del m, pts
        rad, ang, z = cyl(P)
        d = (ang - ang_ref + 180.0) % 360.0 - 180.0          # azimuth change, deg
        slip = np.radians(d) * 0.5 * (rad + rad_ref)          # tangential slip, mm
        disp = np.linalg.norm(P - P_ref, axis=1)              # total node displacement, mm
        row = dict(build_seconds=time.time() - t0, gate_ok=bool(n_bad == 0))
        for rmin in (2.0, 5.0, 10.0, 15.0):
            mk = (rad > rmin) & (rad_ref > rmin)
            row[f"deg_r_gt_{rmin:g}mm"] = stats(d[mk])
        mk2 = (rad > 2.0) & (rad_ref > 2.0)
        row["slip_mm_r_gt_2mm"] = stats(slip[mk2])
        row["slip_mm_all"] = stats(slip)
        row["disp_mm_all"] = stats(disp)
        for name, mk in (("apex_h_lt_10mm", mk2 & (h_ref < 10.0)), ("body_h_ge_10mm", mk2 & (h_ref >= 10.0))):
            row["deg_" + name] = stats(d[mk])
            row["slip_mm_" + name] = stats(slip[mk])
        i = int(np.argmax(np.where(mk2, np.abs(d), 0.0)))
        row["worst_node_r_gt_2mm"] = dict(drift_deg=float(d[i]), radius_ref_mm=float(rad_ref[i]), radius_mm=float(rad[i]),
                                          height_above_apex_mm=float(h_ref[i]), slip_mm=float(slip[i]), disp_mm=float(disp[i]))
        row["frac_r_gt_2mm_drift_gt_5deg"] = float((np.abs(d[mk2]) > 5.0).mean())
        row["frac_r_gt_2mm_slip_gt_1mm"] = float((np.abs(slip[mk2]) > 1.0).mean())
        res["phases"][str(k)] = row
        s2, s10, sl, w = row["deg_r_gt_2mm"], row["deg_r_gt_10mm"], row["slip_mm_r_gt_2mm"], row["worst_node_r_gt_2mm"]
        print(f"  phase {k:2d} ({row['build_seconds']:.0f} s): azimuth change r>2mm max {s2['max']:.1f} rms {s2['rms']:.2f} "
              f"p99 {s2['p99']:.2f} deg | r>10mm max {s10['max']:.2f} rms {s10['rms']:.2f} deg | slip max {sl['max']:.2f} "
              f"rms {sl['rms']:.3f} p99 {sl['p99']:.2f} mm | worst node: r_ref {w['radius_ref_mm']:.1f} mm, "
              f"{w['height_above_apex_mm']:.1f} mm above apex, slip {w['slip_mm']:.2f} mm, moved {w['disp_mm']:.1f} mm",
              flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    print("done:", a.out)


if __name__ == "__main__":
    main()
