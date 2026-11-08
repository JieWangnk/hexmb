#!/usr/bin/env python
"""Figures and markdown tables from scripts/verification_study.py's results.json.

usage: python scripts/verification_figures.py RESULTS_DIR [--png examples/img/verification_refinement.png]
Prints the tables for docs/VERIFICATION.md to stdout.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    rdir = Path(a.results_dir)
    r = json.loads((rdir / "results.json").read_text())
    Ls = sorted(int(k) for k in r["levels"])
    lev = {L: r["levels"][str(L)] for L in Ls}
    ex = r["exact"]
    times = np.array(r["times"])

    # ---- tables --------------------------------------------------------------- #
    print("### Meshes\n")
    print("| level | surface NL x NT | ring NT_c / NR_c | levels apex/body/base | cells | points | build (20 phases) | "
          "max non-orth. | faces > 70 deg | max skewness | faces > 4 | max aspect |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for L in Ls:
        v = lev[L]; sp = v["spec"]; ph = v["phases"]
        print(f"| L{L} | {sp['NL']} x {sp['NT']} | {sp['cfg']['NT_c']} / {sp['cfg']['NR_c']} | "
              f"{sp['cfg']['apex_levels']}/{sp['cfg']['body_levels']}/{sp['cfg']['base_levels']} | {v['cells']:,} | {v['n_points']:,} | "
              f"{v['build_seconds']:.0f} s | {max(p['max_non_orthogonality_deg'] for p in ph):.1f} deg | "
              f"{max(p['n_non_orthogonality_gt70'] for p in ph)} | {max(p['max_skewness'] for p in ph):.2f} | "
              f"{max(p['n_skewness_gt4'] for p in ph)} | {max(p['max_aspect_ratio'] for p in ph):.1f} |")
    print("\n### Cavity volume against the exact model\n")
    print("| level | EDV (mL) | error | ESV (mL) | error | mean error over 20 phases | sampled-surface EDV |")
    print("|---|---|---|---|---|---|---|")
    for L in Ls:
        v = lev[L]; ph = v["phases"]
        err = np.array([100 * (p["volume_mL"] / p["exact_volume_mL"] - 1) for p in ph])
        print(f"| L{L} | {v['EDV_mL']:.2f} | {100 * (v['EDV_mL'] / ex['EDV_mL'] - 1):+.2f} % | {v['ESV_mL']:.2f} | "
              f"{100 * (v['ESV_mL'] / ex['ESV_mL'] - 1):+.2f} % | {err.mean():+.2f} % | {max(p['surface_volume_mL'] for p in ph):.2f} |")
    print(f"| exact | {ex['EDV_mL']:.2f} | | {ex['ESV_mL']:.2f} | | | |")
    LABEL = {"EDV_mL": "EDV (mL)", "ESV_mL": "ESV (mL)", "peak_inflow_mL_s": "peak mitral inflow (mL/s)",
             "peak_outflow_mL_s": "peak aortic outflow (mL/s)", "peak_speed_diastole_m_s": "peak speed, diastole (m/s)",
             "peak_speed_systole_m_s": "peak speed, systole (m/s)", "peak_KE2_diastole_m5_s2": "peak 2KE/rho, diastole (m^5/s^2)",
             "peak_KE2_systole_m5_s2": "peak 2KE/rho, systole (m^5/s^2)"}
    for title, keys in (("Richardson extrapolation and grid-convergence index", ("EDV_mL", "ESV_mL")),
                        ("Flow quantities: grid-convergence index", tuple(k for k in r.get("gci", {}) if k not in ("EDV_mL", "ESV_mL")))):
        if not r.get("gci") or not any(k in r["gci"] for k in keys):
            continue
        print(f"\n### {title}\n")
        print("| quantity | L1 (default) | L2 | L3 | observed order p | extrapolated | exact | error of L1 vs extrapolated | GCI of L1 | GCI of L3 | note |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for k in keys:
            if k not in r["gci"]:
                continue
            g = r["gci"][k]; k = LABEL.get(k, k)
            phi = g["phi"]
            fmt = (lambda x: f"{x:.3f}") if abs(phi[0]) > 10 else (lambda x: f"{x:.4g}")
            if g.get("p") is None:
                print(f"| {k} | {fmt(phi[0])} | {fmt(phi[1])} | {fmt(phi[2])} | - | - | {fmt(g['exact']) if 'exact' in g else '-'} | "
                      f"{100 * g['e_coarse_rel']:.2f} % (vs L3) | - | - | {g.get('note', '')} |")
            else:
                print(f"| {k} | {fmt(phi[0])} | {fmt(phi[1])} | {fmt(phi[2])} | {g['p']:.2f} | {fmt(g['phi_ext'])} | "
                      f"{fmt(g['exact']) if 'exact' in g else '-'} | {100 * g['e_coarse_rel']:.2f} % | {100 * g['gci_coarse']:.2f} % | "
                      f"{100 * g['gci_fine']:.2f} % | {g.get('note', '')} |")
    runs = {(L, w): lev[L]["cases"][w].get("run") for L in Ls for w in ("diastole", "systole")}
    if any(runs.values()):
        print("\n### Flow runs (4 cores)\n")
        print("| level | window | time steps | dt (ms) | Courant max | wall time | peak flow (mL/s) | peak speed (m/s) | peak 2KE/rho (m^5/s^2) | moving-mesh checks failed | run |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for (L, w), run in runs.items():
            if not run:
                continue
            h = run["history"]; q = "Qin" if w == "diastole" else "Qout"
            pk = max(abs(v) for _, v in h[q]) * 1e6 if q in h else float("nan")
            um = max(v for _, v in h["Umax"]) if "Umax" in h else float("nan")
            ke = max(v for _, v in h["KE2"]) if "KE2" in h else float("nan")
            print(f"| L{L} | {w} | {run['n_steps']} | {1e3 * run['dt_min']:.2f}-{1e3 * run['dt_max']:.2f} | {run['co_max']:.2f} | "
                  f"{run['wall_clock_s'] / 60:.1f} min | {pk:.1f} | {um:.3f} | {ke:.3e} | "
                  f"{sum(c['failed_checks'] for c in run['checkmesh'])} of {len(run['checkmesh'])} | "
                  f"{'ok' if run.get('ok') else 'FAILED'} |")

    # ---- figure --------------------------------------------------------------- #
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = {1: "#7b3294", 2: "#e66101", 3: "#1b7837"}
        fig, axs = plt.subplots(2, 2, figsize=(12, 8.4))
        ax = axs[0, 0]
        ex_v = np.array(ex["volumes_mL"])
        ax.plot(times, ex_v, "k-", lw=2, label="exact model")
        for L in Ls:
            v = np.array([p["volume_mL"] for p in lev[L]["phases"]])
            ax.plot(times, v, "o-", ms=3, color=cols[L], label=f"L{L}: {lev[L]['cells']:,} cells")
        ax.set_xlabel("time (s)"); ax.set_ylabel("cavity volume (mL)"); ax.set_title("mesh volume per phase"); ax.legend(fontsize=8)
        ax = axs[0, 1]
        N = np.array([lev[L]["cells"] for L in Ls])
        for key, lab, mk in (("EDV_mL", "EDV", "o"), ("ESV_mL", "ESV", "s")):
            e = np.array([abs(lev[L][key] / ex[key] - 1) * 100 for L in Ls])
            ax.loglog(N, e, mk + "-", label=f"{lab} error vs exact")
        h = N ** (-1 / 3)
        ax.loglog(N, e[0] * (h / h[0]) ** 2, "k:", label="slope of 2nd order")
        ax.set_xlabel("cells"); ax.set_ylabel("|volume error| (%)"); ax.set_title("volume error against cell count"); ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        for ax, w, q, lab in ((axs[1, 0], "diastole", "Qin", "mitral inflow"), (axs[1, 1], "systole", "Qout", "aortic outflow")):
            drew = False
            for L in Ls:
                run = runs.get((L, w))
                if run and q in run["history"]:
                    arr = np.array(run["history"][q])
                    ax.plot(arr[:, 0], np.abs(arr[:, 1]) * 1e6, color=cols[L], lw=1.2, label=f"L{L} {lab}")
                    drew = True
                if run and "Umax" in run["history"]:
                    arr = np.array(run["history"]["Umax"])
                    ax.plot(arr[:, 0], arr[:, 1] * 100, "--", color=cols[L], lw=0.9, label=f"L{L} peak speed x100 (m/s)")
            ax.set_xlabel("time (s)"); ax.set_ylabel("flow rate (mL/s)  /  speed x 100 (m/s)")
            ax.set_title(f"{w}: valve flow and peak speed" + ("" if drew else " (no run)"))
            if drew:
                ax.legend(fontsize=7, ncol=2)
        fig.suptitle("Three meshes of the bundled heart, refined 1.5x per level in every direction", fontsize=12)
        fig.tight_layout()
        fig.savefig(a.png, dpi=130)
        print(f"\nwrote {a.png}")


if __name__ == "__main__":
    main()
