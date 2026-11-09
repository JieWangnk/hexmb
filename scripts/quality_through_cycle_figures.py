#!/usr/bin/env python
"""Figure and table for ``quality_through_cycle.py``: mesh quality at every
phase of the cycle for every level, and a markdown table with one row per level.

usage: python scripts/quality_through_cycle_figures.py OUT [--png examples/img/quality_through_cycle.png]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

COLUMNS = ["level", "cells", "points", "build s / phase", "peak RSS GB", "worst max non-orth. deg (phase)",
           "faces > 70 deg", "worst max skewness", "faces > 4", "worst max aspect", "min pyramid / mean cell volume",
           "wall-node drift RMS (max) deg", "all phases pass", "same topology", "checkMesh at end-systole"]


def rows_of(lev):
    ph = lev["phases"]
    rows = [ph[k] for k in ph if ph[k].get("done", True)]
    return sorted(rows, key=lambda r: r["phase"])


def mean_cell_volume_mm3(r):
    return r["volume_mL"] * 1e3 / r["cells"]


def checkmesh_text(lev):
    cm = (lev.get("checkmesh_es") or {}).get("result")
    if cm is None:
        return "-"
    if not cm.get("ran", cm.get("mesh_ok") or cm.get("failed_checks", 0) > 0):
        return "not run"
    s = "Mesh OK" if cm["mesh_ok"] else f"{cm['failed_checks']} failed"
    if "max_non_orthogonality_deg" in cm:
        s += f" ({cm['max_non_orthogonality_deg']:.1f} deg, skew {cm.get('max_skewness', float('nan')):.2f})"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    res = json.loads((Path(a.out) / "results.json").read_text())
    levels = sorted(res["levels"].items(), key=lambda kv: kv[1].get("cells", 0))
    exact = res.get("exact", {}).get("volumes_mL")

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for i, (L, lev) in enumerate(levels):
        rows = rows_of(lev)
        if not rows:
            continue
        col = cmap(i / max(1, len(levels) - 1))
        ph = [r["phase"] for r in rows]
        cells = lev.get("cells", rows[0]["cells"])
        lab = f"L{L}: {cells:,} cells"
        no = [r["max_non_orthogonality_deg"] for r in rows]
        sk = [r["max_skewness"] for r in rows]
        ratio = [r["min_pyramid_mm3"] / mean_cell_volume_mm3(r) for r in rows]
        ax[0, 0].plot(ph, no, "o-", ms=3, color=col, label=lab)
        ax[0, 1].plot(ph, sk, "o-", ms=3, color=col, label=lab)
        ax[1, 0].semilogy(ph, ratio, "o-", ms=3, color=col, label=lab)
        ax[1, 1].plot(ph, [r["volume_mL"] for r in rows], "o-", ms=3, color=col, label=lab)
        k_worst = int(np.argmax(no))
        bs = [r["build_seconds"] for r in rows if "build_seconds" in r]
        build = np.mean(bs) if bs else lev.get("build_seconds_total", 0) / len(rows)
        rss = [r["peak_rss_gb"] for r in rows if "peak_rss_gb" in r]
        drift_rms = [r["wall_drift_rms_deg"] for r in rows if "wall_drift_rms_deg" in r]
        drift_max = [r["wall_drift_max_deg"] for r in rows if "wall_drift_max_deg" in r]
        cells_row = [f"L{L}", f"{cells:,}", f"{lev.get('n_points', 0):,}", f"{build:.0f}",
                     f"{max(rss):.1f}" if rss else "-",
                     f"{max(no):.1f} ({ph[k_worst]})", str(max(r["n_non_orthogonality_gt70"] for r in rows)),
                     f"{max(sk):.2f}", str(max(r["n_skewness_gt4"] for r in rows)),
                     f"{max(r['max_aspect_ratio'] for r in rows):.1f}",
                     f"{min(ratio):.3f}",
                     f"{max(drift_rms):.2f} ({max(drift_max):.0f})" if drift_rms else "-",
                     "yes" if all(r["status"] == "ok" for r in rows) else "NO",
                     "yes" if all(r.get("same_topology", True) for r in rows) else "NO",
                     checkmesh_text(lev)]
        print("| " + " | ".join(cells_row) + " |")
    if exact:
        ax[1, 1].plot(range(len(exact)), exact, "k-", lw=1, label="exact (Hermite model)")
    ax[0, 0].axhline(70, color="r", ls="--", lw=1); ax[0, 0].text(0.2, 70.5, "checkMesh warns above 70", color="r", fontsize=8)
    ax[0, 1].axhline(4, color="r", ls="--", lw=1); ax[0, 1].text(0.2, 4.05, "checkMesh warns above 4", color="r", fontsize=8)
    ax[0, 0].set_ylabel("max non-orthogonality (deg)"); ax[0, 1].set_ylabel("max skewness")
    ax[1, 0].set_ylabel("min signed pyramid volume / mean cell volume"); ax[1, 1].set_ylabel("cavity volume (mL)")
    for A in ax.ravel():
        A.set_xlabel("phase"); A.grid(alpha=0.3)
    ax[0, 0].set_ylim(40, 92); ax[0, 1].set_ylim(1.5, 4.5)
    ax[0, 0].legend(fontsize=8, loc="upper left")
    ax[1, 1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Bundled CC0 heart (Cardiac Atlas SCD0000101), six mesh levels, 20 phases each:\n"
                 "the same hexahedral mesh rebuilt at every phase with identical topology, no lifting equation, no remeshing",
                 fontsize=10)
    png = a.png or str(Path(a.out) / "quality_through_cycle.png")
    fig.savefig(png, dpi=150)
    print("\nfigure:", png)


if __name__ == "__main__":
    main()
