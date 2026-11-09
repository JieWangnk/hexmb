#!/usr/bin/env python
"""Mesh quality through the whole cardiac cycle at large cell counts.

The question (left open by the Karlsruhe KaHMo group in 2009, who could not hold
the quality of a block-structured hexahedral LV mesh above a few hundred
thousand cells and saw twist between phases): can a body-fitted hexahedral mesh
keep its quality through a full contraction without a lifting equation and
without remeshing?

This script extends the three-level study of ``verification_study.py`` (14,688 /
54,027 / 202,860 cells) upwards on the same bundled CC0 heart, with the same
rule (every direction refined together, nothing set per level or per phase):

    level  NL x NT surface  NT_c  NR_c  levels (apex/body/base)   cells
    L4     135 x 192        84    14    33 / 102 / 41             729,708
    L5     160 x 232        104   17    40 / 125 / 50           1,366,560
    L6     200 x 288        128   21    50 / 153 / 61           2,569,728

One phase is in memory at a time.  For every phase of every level it records
the signed-pyramid gate (min pyramid, cells with a non-positive pyramid,
zero-area faces), the checkMesh-style metrics with OpenFOAM's definitions (max
and average non-orthogonality, faces above 70 deg, max skewness, faces above 4,
max aspect ratio, percentiles of the per-face non-orthogonality and skewness),
the cavity volume, the build time and the peak memory, and the circumferential
drift (twist) of the wall nodes with respect to the reference phase.  The face
topology is taken from the first build (the end-systolic reference); the cell,
point, hexahedron and split-hex counts are compared at every phase, and at
end-diastole the cell-to-vertex lists and every internal face with its owner and
neighbour are compared element by element, and the boundary faces as a set.
(The base-cap faces are labelled wall / inlet / outlet per phase by the valve
classifier, so a few faces change patch between phases; how many, and the patch
sizes at both phases, are recorded; the moving-mesh case takes the labelling of
its reference phase.)  With ``--checkmesh`` the
end-systolic mesh of each level is also written as a polyMesh (with minimal
system dictionaries) and passed through OpenFOAM's ``checkMesh``.

Results go to OUT/results.json (resumable: finished phases are not rebuilt);
``quality_through_cycle_figures.py`` draws the figure and prints the table.

usage: python scripts/quality_through_cycle.py OUT [--levels 4 5 6] [--phases 20]
                                               [--prior PATH/results.json] [--only-phase K] [--checkmesh]
"""
import argparse
import gc
import json
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from hexmb.io.cap import cap_endo_surface, surface_frame                                 # noqa: E402
from hexmb.templates.conformal_lv import ConformalLVConfig, grid_volume, polymesh_quality  # noqa: E402
from hexmb.services.motion import _build_phase, _screen_rings, _write_reference           # noqa: E402
from hexmb.services.quality import polymesh_metrics                                       # noqa: E402
from verification_study import LEVELS, bundled_phases, checkmesh                          # noqa: E402

LEVELS = dict(LEVELS)
LEVELS.update({
    4: dict(NL=135, NT=192, cfg=dict(NT_c=84, NR_c=14, apex_levels=33, body_levels=102, base_levels=41)),
    5: dict(NL=160, NT=232, cfg=dict(NT_c=104, NR_c=17, apex_levels=40, body_levels=125, base_levels=50)),
    6: dict(NL=200, NT=288, cfg=dict(NT_c=128, NR_c=21, apex_levels=50, body_levels=153, base_levels=61)),
})
MIN_RADIUS_MM = 2.0      # nodes closer than this to the long axis have no meaningful angle


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def angles(P, frame, mask_r=MIN_RADIUS_MM):
    """Angle (deg) of every point about the long axis in the (e1, e2) frame, and the
    mask of points far enough from the axis for the angle to mean something."""
    axis, e1, e2, mu = (np.asarray(v, float) for v in frame)
    r = P - mu
    x, y = r @ e1, r @ e2
    return np.degrees(np.arctan2(y, x)), np.hypot(x, y) > mask_r


def drift_deg(ang, ang_ref, mask):
    d = (ang - ang_ref + 180.0) % 360.0 - 180.0
    d = d[mask]
    return float(np.abs(d).max()), float(np.sqrt((d ** 2).mean()))


def percentiles(a, ps=(50, 90, 99, 99.9)):
    a = np.asarray(a, float)
    return {str(p): float(np.percentile(a, p)) for p in ps}


def _same_ids(a, b):
    """Element-by-element equality of two nested sequences of vertex ids (a face list, a
    cell list, one polyhedral cell): one array comparison when the shape is regular,
    element by element when it is ragged (hexahedra next to split-hex polyhedra)."""
    try:
        return bool(np.array_equal(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)))
    except (ValueError, TypeError):
        return len(a) == len(b) and all(_same_ids(x, y) for x, y in zip(a, b))


def _face_key(f):
    return tuple(sorted(int(v) for v in f))


def _patch_labels(faces, patches):
    """{face key: patch name} for the boundary faces."""
    return {_face_key(f): name for name, (s, n) in patches.items() for f in faces[s:s + n]}


def _boundary_relabelled(faces_a, patches_a, faces_b, patches_b):
    """Number of boundary faces whose patch label differs between two builds (the valve
    classifier labels the base-cap faces per phase), or None when the two boundary
    face sets differ."""
    la, lb = _patch_labels(faces_a, patches_a), _patch_labels(faces_b, patches_b)
    if la.keys() != lb.keys():
        return None
    return sum(1 for key, name in la.items() if lb[key] != name)


def write_checkmesh_dicts(case):
    """The polyMesh writer leaves only ``constant/``; ``checkMesh`` also needs a
    ``system/controlDict`` and reads ``fvSchemes`` / ``fvSolution``.  Minimal ones."""
    sysd = Path(case) / "system"
    sysd.mkdir(parents=True, exist_ok=True)

    def header(obj):
        return ("FoamFile\n{\n    format      ascii;\n    class       dictionary;\n"
                f"    location    \"system\";\n    object      {obj};\n}}\n")

    (sysd / "controlDict").write_text(header("controlDict") + "application     foamRun;\nstartFrom       startTime;\n"
                                      "startTime       0;\nstopAt          endTime;\nendTime         1;\ndeltaT          1;\n"
                                      "writeControl    timeStep;\nwriteInterval   1;\n")
    (sysd / "fvSchemes").write_text(header("fvSchemes") + "ddtSchemes { default steadyState; }\n"
                                    "gradSchemes { default Gauss linear; }\ndivSchemes { default none; }\n"
                                    "laplacianSchemes { default Gauss linear corrected; }\n"
                                    "interpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n")
    (sysd / "fvSolution").write_text(header("fvSolution") + "solvers {}\n")


def run_level(L, files, exact, out, res, only_phase=None, do_checkmesh=False):
    spec = LEVELS[L]
    key = str(L)
    lev = res["levels"].setdefault(key, dict(spec=spec, phases={}, started=time.strftime("%Y-%m-%d %H:%M:%S")))
    print(f"\n=== level {L}: surface {spec['NL']} x {spec['NT']}, template {spec['cfg']}", flush=True)
    t0 = time.time()
    grids = [cap_endo_surface(f, NL=spec["NL"], NT=spec["NT"]) for f in files]
    frame = surface_frame(grids[0])
    surf_vol = [grid_volume(G) * 1e-3 for G in grids]
    cfg = ConformalLVConfig(**spec["cfg"])
    cfg, adjustments, degenerate = _screen_rings(grids, frame, cfg)
    lev.update(surface_seconds=time.time() - t0, adjustments=adjustments,
               degenerate={str(k): v for k, v in degenerate.items()}, approx_cells=cfg.approx_cells,
               cfg_used={k: getattr(cfg, k) for k in ("NT_c", "NR_c", "apex_levels", "body_levels", "base_levels",
                                                        "apex_trim", "base_cut", "core_frac")})
    print(f"  surfaces {lev['surface_seconds']:.0f} s; expected cells {cfg.approx_cells:,}; "
          f"adjustments {adjustments or 'none'}; degenerate {degenerate or 'none'}", flush=True)

    # material drift of the SURFACE samples (what the Cardiac Atlas model itself does)
    ref = int(np.argmin(exact))                      # end-systole of the exact model
    ed = int(np.argmax(exact))                       # end-diastole: full topology comparison there
    ang_ref, mask_ref = angles(grids[ref].reshape(-1, 3), frame)
    lev["reference_phase"] = ref
    lev["topology_compared_in_full_at"] = []
    lev["surface_drift_deg"] = {}
    for k, G in enumerate(grids):
        a, m = angles(G.reshape(-1, 3), frame)
        lev["surface_drift_deg"][str(k)] = drift_deg(a, ang_ref, mask_ref & m)

    topo = None                                       # (faces, owner, neigh, wall point ids), from the first phase
    topo_sig = None
    wall_ang_ref = None
    phases = list(range(len(grids))) if only_phase is None else [only_phase]
    # the reference phase first, so the wall-node drift is measured against it
    phases = [ref] + [k for k in phases if k != ref] if only_phase is None else phases
    for k in phases:
        if str(k) in lev["phases"] and lev["phases"][str(k)].get("done"):
            continue
        t0 = time.time()
        pts, _, vols, m, n_bad, span, extra = _build_phase(grids[k], frame, "conformal", cfg)
        t_build = time.time() - t0
        n_cells = len(m.cells)
        t0 = time.time()
        if topo is None:
            faces, owner, neigh, patches = m.topology()
            wall = patches.get("wall")
            wall_pts = np.unique(np.concatenate([np.asarray(f, int) for f in faces[wall[0]:wall[0] + wall[1]]]))
            topo = (faces, owner, neigh, wall_pts)
            topo_cells, topo_patches = m.cells, patches
            topo_sig = (n_cells, len(m.P), m.n_hex, m.n_split, len(faces), int(owner.sum()), int(neigh.sum()))
            lev.update(cells=n_cells, n_points=len(m.P), n_hex=m.n_hex, n_split=m.n_split, n_faces=len(faces),
                       n_internal_faces=int(len(neigh)), topology_from_phase=k,
                       patches={str(k): {name: int(n) for name, (s, n) in patches.items()}})
            same_topology = True
        else:
            same_topology = (n_cells, len(m.P), m.n_hex, m.n_split) == topo_sig[:4]
            if k == ed or (only_phase is not None):
                f2, o2, n2, p2 = m.topology()
                n_int = len(topo[2])
                relabelled = _boundary_relabelled(topo[0], topo_patches, f2, p2)
                same_topology = same_topology and _same_ids(m.cells, topo_cells) \
                    and len(f2) == len(topo[0]) and np.array_equal(n2, topo[2]) \
                    and np.array_equal(o2[:n_int], topo[1][:n_int]) and _same_ids(f2[:n_int], topo[0][:n_int]) \
                    and relabelled is not None
                lev["topology_compared_in_full_at"].append(k)
                lev["patches"][str(k)] = {name: int(n) for name, (s, n) in p2.items()}
                lev.setdefault("boundary_faces_relabelled", {})[str(k)] = relabelled
                print(f"  phase {k}: boundary faces whose patch label differs from phase {lev['topology_from_phase']}: "
                      f"{relabelled} of {len(f2) - n_int:,} (patches {lev['patches'][str(k)]})", flush=True)
                del f2, o2, n2, p2
        faces, owner, neigh, wall_pts = topo
        P = np.asarray(pts, float)
        full = polymesh_metrics(P, faces, owner, neigh, n_cells=n_cells)
        sm = full["summary"]
        q = polymesh_quality(m)
        ang, mk = angles(P[wall_pts], frame)
        if k == ref or wall_ang_ref is None:
            wall_ang_ref, wall_mask_ref = ang, mk
        wd = drift_deg(ang, wall_ang_ref, wall_mask_ref & mk)
        row = dict(phase=k, status="ok" if n_bad == 0 else "failed", n_bad=int(n_bad), bad_span=list(span),
                   cells=n_cells, same_topology=bool(same_topology),
                   volume_mL=q["volume_mL"], surface_volume_mL=surf_vol[k], exact_volume_mL=exact[k],
                   min_volume_mm3=q["min_volume_mm3"], median_volume_mm3=float(np.median(vols)),
                   min_pyramid_mm3=q["min_pyramid_mm3"], n_nonpositive=q["n_nonpositive"],
                   n_zero_area_faces=q["n_zero_area_faces"],
                   max_non_orthogonality_deg=sm["max_non_orthogonality_deg"],
                   average_non_orthogonality_deg=sm["average_non_orthogonality_deg"],
                   n_non_orthogonality_gt70=sm["n_non_orthogonality_gt70"],
                   max_skewness=sm["max_skewness"], n_skewness_gt4=sm["n_skewness_gt4"],
                   max_aspect_ratio=sm["max_aspect_ratio"],
                   wall_drift_max_deg=wd[0], wall_drift_rms_deg=wd[1],
                   build_seconds=t_build, metrics_seconds=time.time() - t0, peak_rss_gb=rss_gb(), done=True)
        for name in ("non_orthogonality_deg", "skewness", "aspect_ratio"):
            if name in full:
                row["p_" + name] = percentiles(full[name])
        lev["phases"][str(k)] = row
        print(f"  phase {k:2d}: {n_cells:,} cells, build {t_build:.0f} s, metrics {row['metrics_seconds']:.0f} s, "
              f"RSS {row['peak_rss_gb']:.1f} GB | gate {'OK' if n_bad == 0 else f'FAIL {n_bad}'} "
              f"min pyr {q['min_pyramid_mm3']:.2e} mm3 | non-orth max {sm['max_non_orthogonality_deg']:.1f} "
              f"(avg {sm['average_non_orthogonality_deg']:.1f}, >70: {sm['n_non_orthogonality_gt70']}) | "
              f"skew max {sm['max_skewness']:.2f} (>4: {sm['n_skewness_gt4']}) | aspect {sm['max_aspect_ratio']:.1f} | "
              f"vol {q['volume_mL']:.2f} mL (exact {exact[k]:.2f}) | wall drift max {wd[0]:.3f} deg | "
              f"topology {'same' if same_topology else 'DIFFERENT'}", flush=True)
        if not same_topology:
            print(f"  CHECK: phase {k} does not share the topology of phase {lev['topology_from_phase']}", flush=True)
        del m, pts, vols, P, full
        gc.collect()
        (out / "results.json").write_text(json.dumps(res, indent=1))

    if do_checkmesh and only_phase is None:
        case = out / f"L{L}_es"
        if case.exists():
            shutil.rmtree(case)
        t0 = time.time()
        pts, _, vols, m, n_bad, span, extra = _build_phase(grids[ref], frame, "conformal", cfg)
        _write_reference(m, "conformal", case)
        write_checkmesh_dicts(case)
        t_write = time.time() - t0
        del m, pts, vols
        gc.collect()
        cm = checkmesh(case)
        lev["checkmesh_es"] = dict(case=str(case), write_seconds=t_write, result=cm)
        if cm is None:
            msg = "OpenFOAM not found, skipped"
        elif not cm["ran"]:
            msg = "checkMesh did not finish (see log.checkMesh in the case)"
        else:
            msg = ("Mesh OK" if cm["mesh_ok"] else f"{cm['failed_checks']} failed checks") + \
                  f"; max non-orth {cm.get('max_non_orthogonality_deg', float('nan')):.2f} deg, " \
                  f"max skew {cm.get('max_skewness', float('nan')):.2f}, max aspect {cm.get('max_aspect_ratio', float('nan')):.2f}"
        print(f"  checkMesh on the end-systolic mesh ({case.name}, written in {t_write:.0f} s): {msg}", flush=True)
        (out / "results.json").write_text(json.dumps(res, indent=1))
    lev["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "results.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--levels", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--phases", type=int, default=20)
    ap.add_argument("--prior", default=None, help="results.json of verification_study.py (levels 1-3) to carry over")
    ap.add_argument("--only-phase", type=int, default=None, help="build one phase only (calibration)")
    ap.add_argument("--checkmesh", action="store_true", help="write the end-systolic mesh of each level and run checkMesh")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rpath = out / "results.json"
    res = json.loads(rpath.read_text()) if rpath.exists() else dict(levels={}, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    files = bundled_phases(a.phases)
    if a.prior:
        prior = json.loads(Path(a.prior).read_text())
        res["exact"] = prior["exact"]
        for L, lev in prior["levels"].items():
            if L not in res["levels"]:
                res["levels"][L] = dict(spec=lev["spec"], cells=lev["cells"], n_points=lev["n_points"],
                                        n_hex=lev["n_hex"], n_split=lev["n_split"], adjustments=lev["adjustments"],
                                        build_seconds_total=lev["build_seconds"], carried_from=str(a.prior),
                                        phases={str(p["phase"]): dict(p, done=True) for p in lev["phases"]})
        print(f"carried levels {sorted(prior['levels'])} from {a.prior}")
    if "exact" not in res:
        t0 = time.time()
        res["exact"] = dict(volumes_mL=[grid_volume(cap_endo_surface(f, NL=400, NT=512)) * 1e-3 for f in files],
                            NL=400, NT=512, seconds=time.time() - t0)
    exact = res["exact"]["volumes_mL"]
    rpath.write_text(json.dumps(res, indent=1))
    for L in a.levels:
        run_level(L, files, exact, out, res, only_phase=a.only_phase, do_checkmesh=a.checkmesh)
    print("\ndone:", rpath)


if __name__ == "__main__":
    main()
