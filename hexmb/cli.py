"""Command-line interface for hexmb.

    hexmb pipe   --length 100 --radius 10 -o case_pipe
    hexmb elbow  --radius 10 --bend-radius 35 --angle 180 -o case_ubend
    hexmb cap    --exnode SCD0000101_1.model.exnode -o case_lv --stl lv.stl   # CAP LV model -> mesh
    hexmb lv     --surface surface.npz -o case_lv                       # any ordered surface grid
    hexmb verify -o case_pipe          # scaled-Jacobian + checkMesh (if OpenFOAM on PATH)

Each build command writes an OpenFOAM constant/polyMesh and reports the
scaled-Jacobian gate. A mesh with any non-positive cell aborts unless
--allow-invalid is given.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


_MIN_CONTROLDICT = """FoamFile
{
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     foamRun;
solver          movingMesh;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
"""


def _write_min_system(out):
    sysd = Path(out) / "system"
    sysd.mkdir(parents=True, exist_ok=True)
    cd = sysd / "controlDict"
    if not cd.exists():
        cd.write_text(_MIN_CONTROLDICT)


def _report_and_write(m, out, allow_invalid):
    sj = m.scaled_jacobian()
    ok = sj.min() > 0
    print(f"  cells {len(m.hexes)}  points {len(m.points)}  "
          f"min scaled-Jac {sj.min():+.4f}  median {np.median(sj):+.3f}  "
          f"inverted {int((sj <= 0).sum())}  ({'VALID' if ok else 'INVALID'})")
    if not ok and not allow_invalid:
        print("  ABORT: mesh has non-positive cells (use --allow-invalid to write anyway)")
        return 1
    info = m.write_polymesh(Path(out) / "constant" / "polyMesh")
    _write_min_system(out)
    print(f"  wrote {out}/constant/polyMesh  "
          f"(nFaces {info['nFaces']}, patches {[p[0] for p in info['patches']]})")
    return 0


def cmd_pipe(a):
    from .templates.tube import PipeTemplate
    print(f"pipe: length {a.length} radius {a.radius}")
    m = PipeTemplate(a.length, a.radius, a.n_axial, a.NT, a.NR, a.core_frac, a.wall_ratio).build()
    return _report_and_write(m, a.out, a.allow_invalid)


def cmd_elbow(a):
    from .templates.tube import ElbowTemplate
    print(f"elbow: radius {a.radius} bend {a.bend_radius} angle {a.angle}")
    m = ElbowTemplate(a.radius, a.bend_radius, a.angle, a.leg,
                      NT=a.NT, NR=a.NR, core_frac=a.core_frac, wall_ratio=a.wall_ratio).build()
    return _report_and_write(m, a.out, a.allow_invalid)


def cmd_lv(a):
    from .templates.lv import LVChamberTemplate
    fs = np.load(a.surface)
    print(f"lv: surface {a.surface}  grid {fs['grid'].shape}")
    m = LVChamberTemplate(fs["grid"].astype(float), fs["axis"], fs["e1"], fs["e2"],
                          NR=a.NR, core_frac=a.core_frac).build()
    return _report_and_write(m, a.out, a.allow_invalid)


def cmd_cap(a):
    """Cardiac Atlas Project LV model (.exnode) -> endocardial surface -> hex mesh.

    Telescoping (coarse apex / fine body / coarse base) by default -- valid on real
    geometry; --single-region uses one O-grid (inverts at the valve plane). Optionally
    writes the endocardial surface as an STL/CAD file first (--stl).
    """
    import shutil
    from .io.cap import cap_endo_surface, surface_frame
    print(f"cap: {a.exnode}")
    G = cap_endo_surface(a.exnode, NL=a.NL, NT=a.NT)
    axis, e1, e2, mu = surface_frame(G)
    print(f"  endocardial surface {G.shape}  (NL levels x NT circumferential)")
    if a.stl:
        from .io.stl_surface import write_surface_stl
        ntri = write_surface_stl(G, a.stl, scale=1.0)
        print(f"  wrote CAD surface {a.stl}  ({ntri} triangles, mm)")
    if a.single_region:
        from .templates.lv import LVChamberTemplate
        m = LVChamberTemplate(G, axis, e1, e2, NR=a.NR, core_frac=a.core_frac).build()
        return _report_and_write(m, a.out, a.allow_invalid)
    from .templates.telescoping_lv import (
        TelescopingLVConfig, build_telescoping_lv, assemble_telescoping_lv,
    )
    from .services.quality import scaled_jac
    cfg = TelescopingLVConfig(body_NT=a.body_nt, body_NR=a.NR, body_levels=30)
    zones = build_telescoping_lv(G, axis, e1, e2, cfg, mu=mu)
    ok = True
    for name, z in zones.items():
        sjmin = scaled_jac(z.points, z.hexes).min()
        ok &= sjmin > 0
        print(f"  zone {name:5s}: {len(z.hexes):>8d} cells  min scaled-Jac {sjmin:+.3f}")
    if shutil.which("mergeMeshes"):
        assemble_telescoping_lv(zones, a.out, cfg)
        print(f"  wrote telescoping polyMesh under {a.out} (AMI-coupled)")
    else:
        print("  zones gated; OpenFOAM not on PATH, merge + AMI skipped "
              "(source OpenFOAM 12, or use --single-region)")
    return 0 if ok else 1


def cmd_stl(a):
    import glob
    from pathlib import Path as _P
    from .templates.stl import TubeSTLTemplate, NotSingleLumen
    print(f"stl: {a.stl}")
    if a.skeleton:
        outlets = a.outlets or sorted(glob.glob(str(_P(a.stl).parent / "outlet*.stl")))
        if not a.inlet or not outlets:
            print("  --skeleton needs --inlet and outlet*.stl (or --outlets)"); return 1
        print(f"  skeleton centreline from inlet + {len(outlets)} outlets (pitch {a.pitch})")
        t = TubeSTLTemplate.from_skeleton(
            a.stl, a.inlet, outlets, pitch=a.pitch, ns=a.ns, NT=a.NT, NR=a.NR,
            core_frac=a.core_frac, wall_ratio=a.wall_ratio, allow_branches=a.allow_branches)
    else:
        t = TubeSTLTemplate(a.stl, ns=a.ns, NT=a.NT, NR=a.NR,
                            core_frac=a.core_frac, wall_ratio=a.wall_ratio,
                            inlet_stl=a.inlet, step=a.step, allow_branches=a.allow_branches)
    try:
        m = t.build()
    except NotSingleLumen as e:
        print(f"  OUT OF SCOPE: {e}")
        return 2
    d = t.diag
    print(f"  centreline {len(d['centreline'])} stations, "
          f"branch stations {d['branch_stations']}, missed rays (max/station) {d['max_miss']}")
    return _report_and_write(m, a.out, a.allow_invalid)


def cmd_verify(a):
    sys.path.insert(0, str(Path.home() / "GitHub" / "betaFlow" / "tools"))
    pm = Path(a.out) / "constant" / "polyMesh"
    if not (pm / "points").exists():
        print(f"no polyMesh at {pm}"); return 1
    try:
        import foam_mesh as fm
        pts = fm.read_points(pm / "points")
        faces = fm.read_faces(pm / "faces")
        owner = fm.read_labels(pm / "owner")
        neigh = fm.read_labels(pm / "neighbour")
        assert np.all(owner[: len(neigh)] < neigh)
        sf, _ = fm.face_area_vectors(pts, faces)
        cell = np.zeros((int(owner.max()) + 1, 3))
        np.add.at(cell, owner, sf)
        np.add.at(cell, neigh, -sf[: len(neigh)])
        print(f"  readback: {len(pts)} pts, owner<neighbour OK, "
              f"max |sum Sf| per cell {np.abs(cell).max():.2e}")
    except Exception as e:
        print(f"  readback check skipped: {e}")
    if shutil.which("checkMesh") or Path("/opt/openfoam12/etc/bashrc").exists():
        cmd = f"source /opt/openfoam12/etc/bashrc && cd {a.out} && checkMesh -constant"
        r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
        for ln in r.stdout.splitlines():
            if any(s in ln for s in ("hexahedra:", "Min volume", "Max skewness",
                                     "non-orthogonality Max", "Mesh OK", "Failed", "negative")):
                print("  [checkMesh]", ln.strip())
    else:
        print("  OpenFOAM not found; skipped checkMesh")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="hexmb", description="Structured O-grid hex mesher.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("-o", "--out", required=True, help="output case directory")
        sp.add_argument("--NT", type=int, default=32, help="circumferential points (÷4)")
        sp.add_argument("--NR", type=int, default=6, help="radial petal layers")
        sp.add_argument("--core-frac", type=float, default=0.42, dest="core_frac")
        sp.add_argument("--wall-ratio", type=float, default=1.0, dest="wall_ratio",
                        help="wall-cell/core-cell radial size ratio; <1 = boundary layer")
        sp.add_argument("--allow-invalid", action="store_true")

    sp = sub.add_parser("pipe"); common(sp)
    sp.add_argument("--length", type=float, default=100.0)
    sp.add_argument("--radius", type=float, default=10.0)
    sp.add_argument("--n-axial", type=int, default=40, dest="n_axial")
    sp.set_defaults(func=cmd_pipe)

    sp = sub.add_parser("elbow"); common(sp)
    sp.add_argument("--radius", type=float, default=10.0)
    sp.add_argument("--bend-radius", type=float, default=40.0, dest="bend_radius")
    sp.add_argument("--angle", type=float, default=180.0)
    sp.add_argument("--leg", type=float, default=40.0)
    sp.set_defaults(func=cmd_elbow)

    sp = sub.add_parser("lv")
    sp.add_argument("-o", "--out", required=True)
    sp.add_argument("--surface", required=True, help="npz with grid,axis,e1,e2")
    sp.add_argument("--NR", type=int, default=6)
    sp.add_argument("--core-frac", type=float, default=0.35, dest="core_frac")
    sp.add_argument("--allow-invalid", action="store_true")
    sp.set_defaults(func=cmd_lv)

    sp = sub.add_parser("cap")
    sp.add_argument("-o", "--out", required=True)
    sp.add_argument("--exnode", required=True, help="CAP Sunnybrook LV model .exnode file")
    sp.add_argument("--stl", default=None, help="also write the endocardial surface as an STL/CAD file")
    sp.add_argument("--single-region", action="store_true", dest="single_region",
                    help="one O-grid instead of telescoping (inverts at the valve plane)")
    sp.add_argument("--NL", type=int, default=40, help="surface levels apex->base")
    sp.add_argument("--NT", type=int, default=56, help="circumferential points (÷4)")
    sp.add_argument("--NR", type=int, default=6)
    sp.add_argument("--body-nt", type=int, default=88, dest="body_nt",
                    help="telescoping body circumferential resolution (340 ~= 1M cells)")
    sp.add_argument("--core-frac", type=float, default=0.35, dest="core_frac")
    sp.add_argument("--allow-invalid", action="store_true")
    sp.set_defaults(func=cmd_cap)

    sp = sub.add_parser("stl"); common(sp)
    sp.add_argument("--stl", required=True, help="path to a single-lumen tubular STL")
    sp.add_argument("--ns", type=int, default=40, help="centreline stations")
    sp.add_argument("--inlet", default=None,
                    help="inlet cap STL (seeds marching, or anchors the skeleton centreline)")
    sp.add_argument("--step", type=float, default=2.0, help="marching centreline step (mm)")
    sp.add_argument("--skeleton", action="store_true",
                    help="use a voxel-skeleton centreline (handles arches); needs --inlet + outlet*.stl")
    sp.add_argument("--outlets", nargs="+", default=None, help="outlet cap STLs (skeleton mode)")
    sp.add_argument("--pitch", type=float, default=1.2, help="skeleton voxel size (mm)")
    sp.add_argument("--allow-branches", action="store_true",
                    help="mesh the main trunk even if branches are detected")
    sp.set_defaults(func=cmd_stl, core_frac=0.42)

    sp = sub.add_parser("verify")
    sp.add_argument("-o", "--out", required=True)
    sp.set_defaults(func=cmd_verify)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
