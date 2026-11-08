#!/usr/bin/env python
"""Figure + markdown table from scripts/synthetic_masks.py's results.json.

usage: python scripts/synthetic_mask_figures.py SYNTH_DIR --png examples/img/synthetic_masks.png
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("synth_dir")
    ap.add_argument("--png", default=None)
    ap.add_argument("--slice-variant", default="sa10")
    ap.add_argument("--slice-phase", type=int, default=0)
    a = ap.parse_args()
    d = Path(a.synth_dir)
    r = json.loads((d / "results.json").read_text())
    ex = np.array(r["exact_mL"]); t = np.array(r["times"])
    names = list(r["variants"])
    print("| masks | voxels (mm) | shape | reader | reader + mesh time | valid phases | mesh EDV / ESV (mL) | mask (Simpson) EDV / ESV | mesh volume error vs exact: mean (range) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n in names:
        v = r["variants"][n]; sp = v["spec"]; sec = v["seconds"]; e = v["volume_error_pct"]
        ok = sum(s == "ok" for s in v["statuses"])
        print(f"| {n} | {sp['pix']} x {sp['pix']} x {sp['slab']} | "
              f"{'x'.join(str(s) for s in v['shape'])} | {v['method'][0]} | {sec['read']:.0f} + {sec['mesh']:.0f} s | "
              f"{ok} of {len(v['statuses'])} | {v['EDV']['mesh']:.1f} / {v['ESV']['mesh']:.1f} | {v['EDV']['mask']:.1f} / {v['ESV']['mask']:.1f} | "
              f"{e['mean']:+.1f} % ({e['min']:+.1f} to {e['max']:+.1f}) |")
    print(f"| exact | | | | | | {r['variants'][names[0]]['EDV']['exact']:.1f} / {r['variants'][names[0]]['ESV']['exact']:.1f} | | |")
    if not a.png:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"sa10": "#7b3294", "sa8": "#e66101", "iso1": "#1b7837"}
    labels = {"sa10": "short-axis, 10 mm slices", "sa8": "short-axis, 8 mm slices", "iso1": "isotropic 1 mm (CT-like)"}
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1.0])
    ax = fig.add_subplot(gs[0])
    ax.plot(t, ex, "k-", lw=2.2, label="exact surface")
    for n in names:
        v = r["variants"][n]
        ax.plot(t, v["mask_mL"], ":", color=cols[n], lw=1.3, label=f"{labels[n]}: mask (Simpson)")
        ax.plot(t, v["mesh_mL"], "o-", ms=3, color=cols[n], lw=1.2, label=f"{labels[n]}: hexmb mesh")
    ax.set_xlabel("time (s)"); ax.set_ylabel("cavity volume (mL)"); ax.set_title("volume through the cycle")
    ax.legend(fontsize=6.5, loc="upper center", ncol=1, framealpha=0.85)
    ax = fig.add_subplot(gs[1])
    for n in names:
        v = r["variants"][n]
        ax.plot(t, 100 * (np.array(v["mesh_mL"]) / ex - 1), "o-", ms=3, color=cols[n], label=f"{labels[n]}: mesh")
        ax.plot(t, 100 * (np.array(v["mask_mL"]) / ex - 1), ":", color=cols[n], lw=1.2, label=f"{labels[n]}: mask")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("time (s)"); ax.set_ylabel("volume error vs exact (%)"); ax.set_title("error per phase")
    ax.legend(fontsize=6.5, ncol=2, loc="lower center", framealpha=0.85); ax.set_ylim(-8, 4)
    # a short-axis slice of one mask with the ring the reader traced through it
    ax = fig.add_subplot(gs[2])
    try:
        import nibabel as nib
        from hexmb.io.surface import grid_from_mask
        f = d / a.slice_variant / f"phase_{a.slice_phase:02d}.nii.gz"
        img = nib.load(str(f)); m = np.asarray(img.dataobj) > 0
        try:
            G, axis, e1, e2, mu, info = grid_from_mask(f, NL=40, NT=56)
        except ValueError:                                   # pool + wall labels: the pool is label 1
            G, axis, e1, e2, mu, info = grid_from_mask(f, label=1, NL=40, NT=56)
        m = (np.asarray(img.dataobj) == 1) if len(np.unique(np.asarray(img.dataobj))) > 2 else m
        k = int(np.argmax(np.abs(img.affine[:3, :3]).sum(0) * (np.array(img.header.get_zooms()[:3]) == max(img.header.get_zooms()[:3]))))
        ks = info["slice_indices"]; kk = ks[len(ks) // 2]
        sl = np.take(m, kk, axis=k).T
        ax.imshow(sl, cmap="gray_r", origin="lower", alpha=0.35, extent=None)
        # rings whose axial position falls within half a slice of this slice
        A = img.affine; zc = (A @ np.array([*np.roll([m.shape[0] / 2, m.shape[1] / 2, kk], 0), 1]))[:3]
        zs = G @ axis
        zk = (A[:3, :3] @ np.array([m.shape[0] / 2, m.shape[1] / 2, kk]) + A[:3, 3]) @ axis
        rows = np.where(np.abs(zs.mean(1) - zk) <= info["slice_spacing_mm"] / 2)[0]
        inv = np.linalg.inv(A)
        for i in rows:
            P = (inv[:3, :3] @ G[i].T).T + inv[:3, 3]
            ax.plot(np.r_[P[:, 0], P[0, 0]], np.r_[P[:, 1], P[0, 1]], "-", color="#e66101", lw=1.0)
        ax.set_title(f"{labels[a.slice_variant]}: mid slice + traced rings", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    except Exception as e:  # noqa: BLE001
        ax.text(0.5, 0.5, f"slice panel skipped:\n{e}", ha="center", va="center", fontsize=8, transform=ax.transAxes); ax.axis("off")
    fig.suptitle("Synthetic segmentation masks of the bundled heart through the mask readers (exact volume known)", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.png, dpi=130)
    print("wrote", a.png)


if __name__ == "__main__":
    main()
