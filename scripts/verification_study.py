#!/usr/bin/env python
"""Mesh-refinement (calculation verification) study on the bundled CC0 heart.

Three meshes of the 20-phase Cardiac Atlas SCD0000101 model, refined together
in every direction by about 1.5 (the surface sampling, the rings, the radial
layers and the axial levels), from the default template upwards:

    level  NL x NT surface  NT_c  NR_c  levels (apex/body/base)   cells (phase 1)
    L1     40 x 56          24    4     10 / 30 / 12              ~15 k   (the default)
    L2     60 x 84          36    6     15 / 45 / 18              ~54 k
    L3     90 x 128         56    9     22 / 68 / 27              ~203 k

For each level the script
  1. builds all phases once, records build time, the gate and the checkMesh-style
     metrics (built in, and checkMesh itself when OpenFOAM is present);
  2. compares the mesh's cavity volume per phase with the exact volume of the
     Hermite model (sampled at 400 x 512) and with the volume of the sampled
     surface it was built from;
  3. writes two moving-mesh cases (reference end-systole and end-diastole) and
     the wall-driven flow cases for the filling (diastole) and ejection
     (systole) windows, then runs them in turn on --procs cores;
  4. reads the flow-rate, peak-speed and kinetic-energy histories and the
     moving-mesh checkMesh output of every run.

Richardson extrapolation and the grid-convergence index (Celik et al. 2008,
ASME J. Fluids Eng. 130, 078001) are applied to the end-diastolic and
end-systolic volumes, the peak valve flow rates, the peak speed and the peak
kinetic energy.  Everything goes to OUT/results.json; scripts/verification_figures.py
draws the figures and prints the tables for docs/VERIFICATION.md.

usage: python scripts/verification_study.py OUT [--procs 4] [--no-flow] [--levels 1 2 3] [--phases 20]
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hexmb.io.cap import cap_endo_surface, surface_frame                      # noqa: E402
from hexmb.templates.conformal_lv import ConformalLVConfig, grid_volume, polymesh_quality  # noqa: E402
from hexmb.services.motion import build_phases, build_lv_motion               # noqa: E402
from hexmb.services.flow import write_flow_case, controldict_value                               # noqa: E402

LEVELS = {
    1: dict(NL=40, NT=56, cfg=dict(NT_c=24, NR_c=4, apex_levels=10, body_levels=30, base_levels=12)),
    2: dict(NL=60, NT=84, cfg=dict(NT_c=36, NR_c=6, apex_levels=15, body_levels=45, base_levels=18)),
    3: dict(NL=90, NT=128, cfg=dict(NT_c=56, NR_c=9, apex_levels=22, body_levels=68, base_levels=27)),
}
EXACT_NL, EXACT_NT = 400, 512
OF_BASHRC = "/opt/openfoam12/etc/bashrc"


def natural(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(p))]


def bundled_phases(n):
    from importlib import resources
    droot = resources.files("hexmb") / "data" / "cap" / "SCD0000101"
    files = sorted((str(p) for p in droot.iterdir() if str(p).endswith(".exnode")), key=natural)
    return files[:n]


def checkmesh(case_dir):
    """Max non-orthogonality, max skewness, max aspect ratio and the final "Mesh OK" / "Failed" line of ``checkMesh -constant``."""
    if not Path(OF_BASHRC).exists():
        return None
    cmd = f"source {OF_BASHRC} 2>/dev/null; cd '{case_dir}' && checkMesh -constant"
    r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    out = {}
    m = re.search(r"non-orthogonality Max:\s*([\d.e+-]+)\s*average:\s*([\d.e+-]+)", r.stdout)
    if m:
        out["max_non_orthogonality_deg"], out["average_non_orthogonality_deg"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"Max skewness = ([\d.e+-]+)", r.stdout)
    if m:
        out["max_skewness"] = float(m.group(1))
    m = re.search(r"Max aspect ratio = ([\d.e+-]+)", r.stdout)
    if m:
        out["max_aspect_ratio"] = float(m.group(1))
    m = re.search(r"Total volume = ([\d.e+-]+)", r.stdout)
    if m:
        out["total_volume_m3"] = float(m.group(1).rstrip("."))
    out["mesh_ok"] = "Mesh OK" in r.stdout
    out["failed_checks"] = int(re.search(r"Failed (\d+) mesh checks", r.stdout).group(1)) if "Failed" in r.stdout else 0
    return out


def read_dat(path):
    rows = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return np.array([[float(v) for v in r[:2]] for r in rows])


def parse_log(log):
    """Time steps, dt range, Courant max, wall time of a foamRun log, plus the checkMesh blocks."""
    txt = Path(log).read_text()
    dts = [float(v) for v in re.findall(r"^deltaT = ([\d.e+-]+)", txt, re.M)]
    cos = [float(v) for v in re.findall(r"Courant Number mean: [\d.e+-]+ max: ([\d.e+-]+)", txt)]
    t_exec = [float(v) for v in re.findall(r"ExecutionTime = ([\d.e+-]+) s", txt)]
    checks = []
    for blk in re.finditer(r"^Time = ([\d.e+-]+)s?\n(.*?)(?=^Time = |\Z)", txt, re.M | re.S):
        body = blk.group(2)
        if "Checking geometry" in body:
            c = {"time": float(blk.group(1))}
            m = re.search(r"non-orthogonality Max:\s*([\d.e+-]+)", body); c["max_non_orthogonality_deg"] = float(m.group(1)) if m else None
            m = re.search(r"Max skewness = ([\d.e+-]+)", body); c["max_skewness"] = float(m.group(1)) if m else None
            m = re.search(r"Max aspect ratio = ([\d.e+-]+)", body); c["max_aspect_ratio"] = float(m.group(1)) if m else None
            m = re.search(r"Min volume = ([\d.e+-]+)", body); c["min_volume_m3"] = float(m.group(1).rstrip(".")) if m else None
            c["mesh_ok"] = "Mesh OK" in body
            c["failed_checks"] = int(re.search(r"Failed (\d+) mesh checks", body).group(1)) if "Failed" in body else 0
            checks.append(c)
    return dict(n_steps=len(dts), dt_min=min(dts) if dts else None, dt_max=max(dts) if dts else None,
                co_max=max(cos) if cos else None, wall_s=t_exec[-1] if t_exec else None,
                fatal="FOAM FATAL" in txt, finished="End" in txt.splitlines()[-3:] or txt.rstrip().endswith("End"),
                checkmesh=checks)


def clean_case(case_dir):
    """Strip the output of an earlier run from a hexmb flow case, keeping the mesh,
    the motion tables (every folder with ``wantedDisplacement``) and the start
    fields, so the case can be rerun from scratch."""
    case_dir = Path(case_dir)
    start = controldict_value(case_dir, "startTime")
    for d in list(case_dir.glob("processor*")) + [case_dir / "postProcessing"]:
        if d.exists():
            shutil.rmtree(d)
    for f in case_dir.glob("log.*"):
        f.unlink()
    for d in case_dir.iterdir():
        if not d.is_dir() or d.name in ("constant", "system"):
            continue
        try:
            t = float(d.name)
        except ValueError:
            continue
        if not (d / "wantedDisplacement").exists():
            shutil.rmtree(d)
            continue
        keep = {"wantedDisplacement"} | ({"U", "p"} if abs(t - start) < 1e-9 else set())
        for e in d.iterdir():
            if e.name not in keep:
                shutil.rmtree(e) if e.is_dir() else e.unlink()


def run_flow(case_dir, procs):
    t = time.time()
    r = subprocess.run(["bash", str(Path(case_dir) / "Allrun"), str(procs)], capture_output=True, text=True)
    info = parse_log(Path(case_dir) / "log.foamRun")
    info.update(wall_clock_s=time.time() - t, allrun_rc=r.returncode, allrun_tail=r.stdout.strip().splitlines()[-2:])
    # a run only counts when foamRun finished, nothing was fatal and the mesh actually moved
    info["ok"] = bool(r.returncode == 0 and not info["fatal"] and info["co_max"] and info["co_max"] > 0)
    pp = Path(case_dir) / "postProcessing"
    hist = {}
    for name, fn in (("Qin", "surfaceFieldValue.dat"), ("Qout", "surfaceFieldValue.dat"),
                     ("Umax", "volFieldValue.dat"), ("KE2", "volFieldValue.dat")):
        fs = sorted((pp / name).glob(f"*/{fn}")) if (pp / name).exists() else []
        if fs:
            arr = np.vstack([read_dat(f) for f in fs])
            hist[name] = arr[np.argsort(arr[:, 0])].tolist()
    info["history"] = hist
    return info


def gci(phi, N, safety=1.25):
    """Celik et al. 2008 procedure for three meshes.  ``phi`` = [coarse, medium, fine]
    values, ``N`` = their cell counts.  Returns the observed order p, the
    extrapolated value, the relative errors and GCIs of the fine and of the coarse
    mesh, or a note when the three values do not converge monotonically."""
    phi = np.asarray(phi, float); N = np.asarray(N, float)
    h = N ** (-1.0 / 3.0)                              # coarse, medium, fine representative sizes
    r32, r21 = h[0] / h[1], h[1] / h[2]
    e32, e21 = phi[1] - phi[0], phi[2] - phi[1]        # (medium - coarse), (fine - medium)
    out = dict(phi=phi.tolist(), N=N.tolist(), r21=float(r21), r32=float(r32))
    if e21 == 0 or e32 == 0:
        out["note"] = "two levels give the same value: the quantity is resolved to the printed precision"
        out.update(p=None, phi_ext=float(phi[2]), e_fine_rel=0.0, gci_fine=0.0, gci_coarse=0.0,
                   e_coarse_rel=float(abs(phi[0] - phi[2]) / abs(phi[2])) if phi[2] else 0.0)
        return out
    ratio = e32 / e21
    if ratio < 0:
        out["note"] = "oscillatory convergence (the sign of the change flips between levels): no order estimated"
        out.update(p=None, phi_ext=None, e_fine_rel=float(abs(e21 / phi[2])) if phi[2] else None,
                   gci_fine=None, gci_coarse=None, e_coarse_rel=float(abs(phi[0] - phi[2]) / abs(phi[2])) if phi[2] else None)
        return out
    s = 1.0
    p = abs(np.log(abs(ratio))) / np.log(r21)
    for _ in range(50):                                # fixed-point iteration of Celik's eq. for p
        q = np.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(np.log(abs(ratio)) + q) / np.log(r21)
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    phi_ext = (r21 ** p * phi[2] - phi[1]) / (r21 ** p - 1.0)
    e_a21 = abs((phi[2] - phi[1]) / phi[2]) if phi[2] else np.nan
    e_a32 = abs((phi[1] - phi[0]) / phi[1]) if phi[1] else np.nan
    out.update(p=float(p), phi_ext=float(phi_ext),
               e_fine_rel=float(abs((phi_ext - phi[2]) / phi_ext)) if phi_ext else None,
               gci_fine=float(safety * e_a21 / (r21 ** p - 1.0)),
               gci_coarse=float(safety * e_a32 * r32 ** p / (r32 ** p - 1.0)),
               e_coarse_rel=float(abs((phi_ext - phi[0]) / phi_ext)) if phi_ext else None)
    if p > 4 or p < 0.5:
        out["note"] = f"observed order {p:.2f} outside 0.5-4: the levels are not in the asymptotic range for this quantity"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--no-flow", action="store_true")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--phases", type=int, default=20)
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--flow-only", action="store_true",
                    help="reuse OUT/results.json and the OUT/L*_* cases from an earlier run: clean them, rewrite the "
                         "flow set-up, rerun the flow and recompute the GCI tables")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.flow_only:
        res = json.loads((out / "results.json").read_text())
        a.levels = sorted(int(k) for k in res["levels"])
        a.no_flow = False
        res["flow_rerun_started"] = time.strftime("%Y-%m-%d %H:%M:%S")
        for L in a.levels:
            for window in ("diastole", "systole"):
                c = res["levels"][str(L)]["cases"][window]
                case = Path(c["path"])
                clean_case(case)
                c["flow"] = write_flow_case(case, window, n_procs=a.procs, verbose=False)
                c["run"] = None
                print(f"  L{L} {window}: cleaned and flow set-up rewritten ({case})")
        (out / "results.json").write_text(json.dumps(res, indent=1))
    else:
        res = build_all(a, out)
    run_and_summarise(a, out, res)


def build_all(a, out):
    files = bundled_phases(a.phases)
    times = a.period * np.arange(len(files)) / len(files)
    res = dict(levels={}, exact={}, times=times.tolist(), started=time.strftime("%Y-%m-%d %H:%M:%S"))

    print(f"exact volumes: Hermite model at {EXACT_NL} x {EXACT_NT}")
    t0 = time.time()
    exact = [grid_volume(cap_endo_surface(f, NL=EXACT_NL, NT=EXACT_NT)) * 1e-3 for f in files]
    res["exact"] = dict(volumes_mL=exact, NL=EXACT_NL, NT=EXACT_NT, seconds=time.time() - t0,
                        EDV_mL=float(max(exact)), ESV_mL=float(min(exact)),
                        ED_phase=int(np.argmax(exact)), ES_phase=int(np.argmin(exact)))
    print(f"  EDV {max(exact):.3f} mL (phase {np.argmax(exact)}), ESV {min(exact):.3f} mL (phase {np.argmin(exact)}), "
          f"{time.time() - t0:.1f} s")

    for L in a.levels:
        spec = LEVELS[L]
        print(f"\n=== level {L}: surface {spec['NL']} x {spec['NT']}, template {spec['cfg']}")
        t0 = time.time()
        grids = [cap_endo_surface(f, NL=spec["NL"], NT=spec["NT"]) for f in files]
        frame = surface_frame(grids[0])
        surf_vol = [grid_volume(G) * 1e-3 for G in grids]
        t_surf = time.time() - t0
        cfg = ConformalLVConfig(**spec["cfg"])
        t0 = time.time()
        b = build_phases(grids, frame, "conformal", cfg, auto_trim=True, verbose=False, metrics=True)
        t_build = time.time() - t0
        rows = []
        for k, m in enumerate(b["meshes"]):
            q = polymesh_quality(m); br = b["rows"][k]
            rows.append(dict(phase=k, status=br["status"], cells=q["hex"] + q["split_hex"], volume_mL=q["volume_mL"],
                             surface_volume_mL=surf_vol[k], exact_volume_mL=exact[k],
                             min_pyramid_mm3=float(m.min_pyramid_per_cell().min()),
                             max_non_orthogonality_deg=br["max_non_orthogonality_deg"],
                             average_non_orthogonality_deg=br["average_non_orthogonality_deg"],
                             n_non_orthogonality_gt70=br["n_non_orthogonality_gt70"],
                             max_skewness=br["max_skewness"], n_skewness_gt4=br["n_skewness_gt4"],
                             max_aspect_ratio=br["max_aspect_ratio"]))
        vols = np.array([r["volume_mL"] for r in rows])
        lev = dict(spec=spec, cells=rows[0]["cells"], n_points=len(b["meshes"][0].P),
                   n_hex=b["meshes"][0].n_hex, n_split=b["meshes"][0].n_split,
                   build_seconds=t_build, surface_seconds=t_surf, adjustments=b["adjustments"],
                   phases=rows, EDV_mL=float(vols.max()), ESV_mL=float(vols.min()),
                   ED_phase=int(vols.argmax()), ES_phase=int(vols.argmin()), cases={})
        print(f"  {rows[0]['cells']} cells, build {t_build:.0f} s; EDV {vols.max():.3f} ESV {vols.min():.3f} mL "
              f"(exact {max(exact):.3f} / {min(exact):.3f}); max non-orth {max(r['max_non_orthogonality_deg'] for r in rows):.1f} deg, "
              f"max skew {max(r['max_skewness'] for r in rows):.2f}; every phase {'ok' if all(r['status'] == 'ok' for r in rows) else 'NOT ok'}")
        for ref, window in (("es", "diastole"), ("ed", "systole")):
            case = out / f"L{L}_{window}"
            if case.exists():
                shutil.rmtree(case)
            t0 = time.time()
            r = build_lv_motion(grids, times, cfg=cfg, reference=ref, period=a.period, out=case, frame=frame,
                                verbose=False, prebuilt=b)
            t_write = time.time() - t0
            cm = checkmesh(case)
            fl = write_flow_case(case, window, n_procs=a.procs, verbose=False)
            lev["cases"][window] = dict(path=str(case), reference=ref, ref_phase=r["ref"], write_seconds=t_write,
                                        checkmesh=cm, flow=fl, run=None)
            print(f"  wrote {case.name}: reference phase {r['ref']}, window {fl['t0']:.3g}-{fl['t1']:.3g} s, "
                  f"checkMesh {'OK' if cm and cm['mesh_ok'] else cm and cm['failed_checks']} ({t_write:.0f} s)")
        res["levels"][str(L)] = lev
        (out / "results.json").write_text(json.dumps(res, indent=1))
    return res


def run_and_summarise(a, out, res):
    if not a.no_flow:
        for L in a.levels:
            for window in ("diastole", "systole"):
                case = Path(res["levels"][str(L)]["cases"][window]["path"])
                print(f"\n--- running L{L} {window} on {a.procs} cores ...", flush=True)
                info = run_flow(case, a.procs)
                res["levels"][str(L)]["cases"][window]["run"] = info
                h = info["history"]
                q = "Qin" if window == "diastole" else "Qout"
                pk = max(abs(v) for _, v in h[q]) * 1e6 if q in h else float("nan")
                um = max(v for _, v in h["Umax"]) if "Umax" in h else float("nan")
                print(f"    {info['n_steps']} steps, dt {info['dt_min']:.2e}-{info['dt_max']:.2e} s, Co max {info['co_max']}, "
                      f"{info['wall_clock_s'] / 60:.1f} min; peak |{q}| {pk:.1f} mL/s, Umax {um:.3f} m/s; "
                      f"fatal {info['fatal']}; moving-mesh checks {sum(c['failed_checks'] for c in info['checkmesh'])} failed")
                if not info["ok"]:
                    print(f"    CHECK: L{L} {window} FAILED (exit {info['allrun_rc']}, Co max {info['co_max']}, fatal {info['fatal']}): "
                          + " | ".join(info["allrun_tail"]), flush=True)
                (out / "results.json").write_text(json.dumps(res, indent=1))

    # --- GCI tables ---------------------------------------------------------- #
    Ls = sorted(int(k) for k in res["levels"])
    N = [res["levels"][str(L)]["cells"] for L in Ls]
    g = {}
    if len(Ls) == 3:
        g["EDV_mL"] = gci([res["levels"][str(L)]["EDV_mL"] for L in Ls], N); g["EDV_mL"]["exact"] = res["exact"]["EDV_mL"]
        g["ESV_mL"] = gci([res["levels"][str(L)]["ESV_mL"] for L in Ls], N); g["ESV_mL"]["exact"] = res["exact"]["ESV_mL"]
        if not a.no_flow:
            def peak(L, window, name, absval=True):
                run = res["levels"][str(L)]["cases"][window]["run"]
                h = run["history"].get(name) if run and run.get("ok") else None
                if not h:
                    return None
                v = np.array(h)[:, 1]
                return float(np.abs(v).max() if absval else v.max())
            for window, name, key, scale in (("diastole", "Qin", "peak_inflow_mL_s", 1e6), ("systole", "Qout", "peak_outflow_mL_s", 1e6),
                                             ("diastole", "Umax", "peak_speed_diastole_m_s", 1.0), ("systole", "Umax", "peak_speed_systole_m_s", 1.0),
                                             ("diastole", "KE2", "peak_KE2_diastole_m5_s2", 1.0), ("systole", "KE2", "peak_KE2_systole_m5_s2", 1.0)):
                vals = [peak(L, window, name) for L in Ls]
                if all(v is not None for v in vals):
                    g[key] = gci([v * scale for v in vals], N)
    res["gci"] = g
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print("\nGCI:")
    for k, v in g.items():
        print(f"  {k}: " + ", ".join(f"{x:.4g}" for x in v["phi"]) + (f"  p {v['p']:.2f}  ext {v['phi_ext']:.4g}  "
              f"error of default mesh {100 * v['e_coarse_rel']:.2f} %  GCI coarse {100 * v['gci_coarse']:.2f} %"
              if v.get("p") is not None else f"  {v.get('note')}") + (f"  exact {v['exact']:.4g}" if "exact" in v else ""))
    print(f"\nresults: {out / 'results.json'}")


if __name__ == "__main__":
    main()
