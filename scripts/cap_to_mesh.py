#!/usr/bin/env python3
"""Worked example: a public Cardiac Atlas Project LV model -> hex mesh with hexmb.

Steps
-----
1. Read a CAP Sunnybrook LV finite-element model (``*.model.exnode``) and build the
   endocardial blood-pool surface (an ordered ring grid).
2. Optionally write that surface as an STL/CAD file (``--stl``).
3. Mesh the cavity with hexmb: the telescoping multi-resolution template by
   default (coarse apex / fine body / coarse base, valid on real geometry); the
   simple single O-grid with ``--single-region``.
4. Report the scaled-Jacobian quality gate and write an OpenFOAM ``polyMesh``.

hexmb is the mesher: it takes a surface/CAD file and produces the all-hex mesh.
The CAP reader here is one *provider* of that surface -- any ordered ring grid or
STL from your own segmentation feeds hexmb the same way.

Get the data (public domain, CC0): see ``docs/DATA.md``.

Usage
-----
    python scripts/cap_to_mesh.py /path/to/SCD0000101_1.model.exnode -o out/lv \
        --stl out/lv_endo.stl
    python scripts/cap_to_mesh.py /path/to/SCD0000101_1.model.exnode -o out/lv --single-region
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hexmb.io.cap import cap_endo_surface, surface_frame
from hexmb.services.quality import scaled_jac


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exnode", help="CAP LV model .exnode file")
    ap.add_argument("-o", "--out", required=True, help="output case directory")
    ap.add_argument("--stl", default=None, help="also write the endo surface as an STL/CAD file")
    ap.add_argument("--single-region", action="store_true", dest="single_region",
                    help="simple single O-grid instead of telescoping (inverts at the valve plane "
                         "on real geometry; numpy-only, no OpenFOAM needed)")
    ap.add_argument("--NL", type=int, default=40)
    ap.add_argument("--NT", type=int, default=56)
    ap.add_argument("--NR", type=int, default=6)
    ap.add_argument("--body-nt", type=int, default=88, dest="body_nt",
                    help="telescoping body circumferential resolution (raise for finer; 340 ~= 1M cells)")
    ap.add_argument("--core-frac", type=float, default=0.35, dest="core_frac")
    a = ap.parse_args(argv)

    print(f"[1] read CAP model: {a.exnode}")
    G = cap_endo_surface(a.exnode, NL=a.NL, NT=a.NT)
    axis, e1, e2, mu = surface_frame(G)
    rad = np.linalg.norm(G[len(G) // 2] - G[len(G) // 2].mean(0), axis=1).mean()
    print(f"    endocardial surface {G.shape}  mid-radius ~{rad:.0f} mm")

    if a.stl:
        from hexmb.io.stl_surface import write_surface_stl
        Path(a.stl).parent.mkdir(parents=True, exist_ok=True)
        ntri = write_surface_stl(G, a.stl, scale=1.0)
        print(f"[2] wrote CAD surface: {a.stl}  ({ntri} triangles, mm)")

    if a.single_region:
        print("[3] mesh with hexmb (single-region O-grid)")
        from hexmb.templates.lv import LVChamberTemplate
        m = LVChamberTemplate(G, axis, e1, e2, NR=a.NR, core_frac=a.core_frac).build()
        sj = m.scaled_jacobian()
        ok = sj.min() > 0
        print(f"    cells {len(m.hexes)}  points {len(m.points)}  "
              f"min scaled-Jac {sj.min():+.4f}  inverted {int((sj <= 0).sum())}  "
              f"({'VALID' if ok else 'has inverted cells at the valve plane -- use telescoping'})")
        info = m.write_polymesh(Path(a.out) / "constant" / "polyMesh")
        print(f"[4] wrote {a.out}/constant/polyMesh  patches {[p[0] for p in info['patches']]}")
        return 0

    print("[3] mesh with hexmb (telescoping: coarse apex / fine body / coarse base)")
    from hexmb.templates.telescoping_lv import (
        TelescopingLVConfig, build_telescoping_lv, assemble_telescoping_lv,
    )
    cfg = TelescopingLVConfig(body_NT=a.body_nt, body_NR=a.NR, body_levels=30)
    zones = build_telescoping_lv(G, axis, e1, e2, cfg, mu=mu)
    all_ok = True
    for name, z in zones.items():
        sj = scaled_jac(z.points, z.hexes)
        all_ok &= sj.min() > 0
        print(f"    zone {name:5s}: {len(z.hexes):>8d} cells  min scaled-Jac {sj.min():+.3f}  "
              f"({'valid' if sj.min() > 0 else 'INVERTED'})")
    import shutil
    if shutil.which("mergeMeshes"):
        assemble_telescoping_lv(zones, a.out, cfg)          # merge + AMI couple (OpenFOAM)
        print(f"[4] wrote telescoping polyMesh under {a.out}  (AMI-coupled)")
    else:
        print("[4] zones built and gated; OpenFOAM not on PATH, so the merge + AMI assembly "
              "was skipped.\n    Source OpenFOAM 12 to assemble, or use --single-region for a "
              "numpy-only mesh.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
