"""Command-line interface for hexmb.

    hexmb pipe    --length 100 --radius 10 -o case_pipe
    hexmb elbow   --radius 10 --bend-radius 35 --angle 180 -o case_ubend
    hexmb lv      --surface lv_endo.stl -o case_lv            # any LV surface -> conformal LV mesh
    hexmb lv      --surface lv.nii.gz --label 1 -o case_lv    # segmentation mask -> mesh
    hexmb motion  --surfaces phase_*.stl --period 1.0 -o case_moving   # per-phase surfaces -> moving mesh
    hexmb cap     --exnode SCD0000101_1.model.exnode -o case_lv --stl lv.stl   # CAP model -> mesh
    hexmb demo    -o out_demo                                 # bundled CC0 heart, all 20 phases
    hexmb verify  -o case_pipe          # readback, built-in checkMesh-style metrics, checkMesh
    hexmb export  -o case_lv --inp lv.inp   # case -> VTK (.vtu/.vtp/.pvd) and/or Abaqus .inp
    hexmb flow    -o lv_moving          # moving-mesh case -> runnable wall-driven flow case
    hexmb render  -o case_moving --gif motion.gif   # long-axis cut of a moving-mesh case through the cycle

Surfaces: .stl .obj .ply .off (trimesh), .vtk .vtp .vtu (meshio), .nii/.nii.gz
masks (nibabel + scikit-image), Cardiac Atlas .exnode, or a saved .npz grid.

Each build command writes an OpenFOAM constant/polyMesh and reports the quality
gate.  A mesh with any non-positive cell aborts unless --allow-invalid is given.
"""
from __future__ import annotations

import argparse
import re
import shlex
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

TEMPLATES = ("conformal", "single", "telescoping")


def _write_min_system(out):
    sysd = Path(out) / "system"
    sysd.mkdir(parents=True, exist_ok=True)
    cd = sysd / "controlDict"
    if not cd.exists():
        cd.write_text(_MIN_CONTROLDICT)
    foam = Path(out) / "case.foam"
    if not foam.exists():
        foam.write_text("")


def _report_and_write(m, out, allow_invalid):
    """All-hex MultiBlockMesh: scaled-Jacobian gate, then write."""
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


def _report_and_write_poly(m, out, allow_invalid):
    """Conformal split-hex PolyMesh: signed-volume gate, then write."""
    from .templates.conformal_lv import polymesh_quality
    q = polymesh_quality(m)
    ok = q["n_nonpositive"] == 0 and q["n_zero_area_faces"] == 0
    print(f"  cells {q['cells']} ({q['hex']} hex + {q['split_hex']} split-hex)  points {q['points']}  "
          f"min pyramid {q['min_pyramid_mm3']:+.3e} mm3  failing cells {q['n_nonpositive']}  "
          f"zero-area faces {q['n_zero_area_faces']}  cavity {q['volume_mL']:.1f} mL  "
          f"({'VALID' if ok else 'INVALID'})")
    if not ok:
        from .templates.conformal_lv import bad_cell_hint
        if m.frame is not None:
            print(f"  {bad_cell_hint(m, m.frame[0], m.frame[3])}")
        if not allow_invalid:
            print("  ABORT: mesh has inverted or folded cells (use --allow-invalid to write anyway)")
            return 1
    nc, ni, npt = m.write(Path(out))
    _write_min_system(out)
    print(f"  wrote {out}/constant/polyMesh  (one region, {ni} internal faces, patches {m.patches})")
    return 0


# --------------------------------------------------------------------------- #
#  tubes                                                                       #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
#  left ventricle                                                              #
# --------------------------------------------------------------------------- #
def _xyz(s):
    if s is None:
        return None
    v = [float(x) for x in re.split(r"[,\s]+", s.strip()) if x]
    if len(v) != 3:
        raise ValueError(f"--apex/--base: expected x,y,z, got {s!r}")
    return np.array(v)


def _conformal_cfg(a):
    from .templates.conformal_lv import ConformalLVConfig
    cfg = ConformalLVConfig()
    for name in ("NT_c", "NR_c", "apex_levels", "body_levels", "base_levels", "apex_frac",
                 "base_frac", "apex_trim", "base_cut", "core_frac", "wall_ratio", "inlet",
                 "inlet_area_cm2"):
        v = getattr(a, name, None)
        if v is not None:
            setattr(cfg, name, v)
    return cfg


def _single_kwargs(a):
    """Single O-grid template keywords from the shared CLI options (same valve
    sectors and wall ratio as the conformal template, so the two agree)."""
    c = _conformal_cfg(a)
    return dict(NR=a.NR, core_frac=a.core_frac if a.core_frac is not None else 0.35,
                wall_ratio=c.wall_ratio, mitral_deg=c.mitral_deg, aortic_deg=c.aortic_deg,
                apex_trim=c.apex_trim, base_cut=c.base_cut)


def _load_grid(a, path=None, frame=None):
    """Any surface source -> (G, axis, e1, e2, mu, info), with a one-line report."""
    from .io.surface import grid_from_file
    path = path or a.surface
    kw = {}
    if getattr(a, "label", None) is not None:
        kw["label"] = a.label
    if getattr(a, "mask_method", "auto") != "auto" and str(path).lower().endswith((".nii", ".nii.gz")):
        kw["method"] = a.mask_method
    G, axis, e1, e2, mu, info = grid_from_file(path, NL=a.NL, NT=a.NT, apex=_xyz(getattr(a, "apex", None)),
                                               base=_xyz(getattr(a, "base", None)), frame=frame, **kw)
    src = info.get("source", "mesh")
    line = f"  surface {Path(path).name}: {src} -> grid {G.shape[0]} x {G.shape[1]}"
    if "landmarks" in info:
        ap, ba = info["apex"], info["base"]
        line += (f", landmarks {info['landmarks']}: apex ({ap[0]:.1f} {ap[1]:.1f} {ap[2]:.1f}) "
                 f"base ({ba[0]:.1f} {ba[1]:.1f} {ba[2]:.1f}), length {info['axis_length']:.1f} mm, "
                 f"interpolated rays max {int(info['interpolated_rays_per_level'].max())}/level")
    if info.get("method") == "slices":
        line += (f", {info['n_slices']} slices of {info['slice_spacing_mm']:.1f} mm read slice by slice, "
                 f"length {info['axis_length']:.1f} mm")
    if "mask_volume_mL" in info:
        line += f", mask volume {info['mask_volume_mL']:.1f} mL"
    print(line)
    return G, axis, e1, e2, mu, info


def _maybe_vtk(a, rc, case_dir=None):
    """Write the VTK copy of a case when --vtk was given and the case was written."""
    if getattr(a, "vtk", False) and rc == 0:
        from .io.vtk import case_to_vtk
        case_to_vtk(case_dir or a.out)
    return rc


def _surface_failed(b, allow_invalid):
    """Print the ring screen's note for a surface that :func:`build_phases` marked
    failed (rings that cross the long axis away from the rims).  Returns True when
    the command must stop: the cells may pass the gate, but the mesh would follow
    a surface that is not a surface there."""
    row = b["rows"][0]
    if row["status"] != "failed" or not row.get("note"):
        return False
    print(f"  surface: {row['note']}")
    if allow_invalid:
        return False
    print("  ABORT: the surface cannot be meshed as it is (use --allow-invalid to write anyway)")
    return True


def _mesh_from_grid(G, axis, e1, e2, mu, a):
    """Build + gate + write with the chosen template.  Returns an exit code."""
    from .services.motion import build_phases
    tpl = a.template
    if tpl == "conformal":
        cfg = _conformal_cfg(a)
        print(f"  template conformal: coarse ring NT {cfg.NT_c} NR {cfg.NR_c}, body 2x, "
              f"levels {cfg.apex_levels}/{cfg.body_levels}/{cfg.base_levels} -> {cfg.approx_cells} cells, "
              f"inlet rule '{cfg.inlet}'")
        b = build_phases([G], (axis, e1, e2, mu), "conformal", cfg, auto_trim=a.auto_trim, verbose=False,
                         metrics=a.metrics)
        for line in b["adjustments"]:
            print(f"  auto_trim: {line}")
        if _surface_failed(b, a.allow_invalid):
            return 1
        rc = _report_and_write_poly(b["meshes"][0], a.out, a.allow_invalid)
        if a.metrics:
            r = b["rows"][0]
            print(f"  metrics:  non-orthogonality max {r['max_non_orthogonality_deg']:.1f} deg (average "
                  f"{r['average_non_orthogonality_deg']:.1f}), skewness max {r['max_skewness']:.2f} "
                  f"({r['n_skewness_gt4']} faces > 4), aspect ratio max {r['max_aspect_ratio']:.1f}")
        return _maybe_vtk(a, rc)
    if tpl == "single":
        kw = _single_kwargs(a)
        print(f"  template single O-grid: NR {kw['NR']} core_frac {kw['core_frac']} wall_ratio {kw['wall_ratio']}")
        b = build_phases([G], (axis, e1, e2, mu), "single", kw, auto_trim=a.auto_trim, verbose=False)
        for line in b["adjustments"]:
            print(f"  auto_trim: {line}")
        if _surface_failed(b, a.allow_invalid):
            return 1
        rc = _report_and_write(b["meshes"][0], a.out, a.allow_invalid)
        return _maybe_vtk(a, rc)
    # telescoping (AMI-coupled zones)
    from .templates.telescoping_lv import (
        TelescopingLVConfig, build_telescoping_lv, assemble_telescoping_lv,
    )
    from .services.quality import scaled_jac
    cfg = TelescopingLVConfig(body_NT=a.body_nt, body_NR=a.NR, body_levels=30)
    print(f"  template telescoping (AMI): body NT {cfg.body_NT} NR {cfg.body_NR}")
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
        assemble_telescoping_lv(zones, a.out, cfg, couple=False)
        print(f"  wrote the three zone meshes under {a.out}/{{apex,body,base}}; OpenFOAM not on PATH, "
              "merge + AMI coupling skipped (source OpenFOAM 12 and rerun, or use --template conformal)")
    return 0 if ok else 1


def cmd_lv(a):
    print(f"lv: {a.surface}")
    G, axis, e1, e2, mu, info = _load_grid(a)
    if a.stl:
        from .io.stl_surface import write_surface_stl
        ntri = write_surface_stl(G, a.stl, scale=1.0)
        print(f"  wrote CAD surface {a.stl}  ({ntri} triangles, mm)")
    if a.save_grid:
        from .io.surface import write_surface_npz
        write_surface_npz(a.save_grid, G, axis, e1, e2, mu)
        print(f"  saved grid {a.save_grid}")
    return _mesh_from_grid(G, axis, e1, e2, mu, a)


def cmd_cap(a):
    """Cardiac Atlas Project LV model (.exnode) -> endocardial surface -> hex mesh
    (telescoping AMI zones by default, as in v0.1; --template conformal for the
    one-region split-hex mesh)."""
    print(f"cap: {a.exnode}")
    a.surface = a.exnode
    if a.single_region:
        a.template = "single"
    G, axis, e1, e2, mu, info = _load_grid(a)
    if a.stl:
        from .io.stl_surface import write_surface_stl
        ntri = write_surface_stl(G, a.stl, scale=1.0)
        print(f"  wrote CAD surface {a.stl}  ({ntri} triangles, mm)")
    return _mesh_from_grid(G, axis, e1, e2, mu, a)


def cmd_export(a):
    """A written case -> VTK (.vtu per phase + .vtp per patch + .pvd) and/or an
    Abaqus/CalculiX .inp (all-hex meshes)."""
    from .io.vtk import case_to_vtk, abaqus_from_case
    if not (Path(a.out) / "constant" / "polyMesh" / "points").exists():
        print(f"no polyMesh under {a.out}"); return 1
    if not a.inp or a.vtk_dir or a.vtk:
        case_to_vtk(a.out, a.vtk_dir, binary=not a.ascii, scale=a.scale)
    if a.inp:
        abaqus_from_case(a.out, a.inp, scale=a.scale)
        print(f"  wrote {a.inp} (Abaqus/CalculiX C3D8, node sets per patch)")
    return 0


def cmd_flow(a):
    """A moving-mesh case -> runnable wall-driven OpenFOAM 12 flow case (in place)."""
    from .services.flow import write_flow_case
    write_flow_case(a.out, window=a.window, nu=a.nu, max_co=a.max_co, max_dt=a.max_dt,
                    write_interval=a.write_interval, n_outer=a.n_outer, n_corr=a.n_corr,
                    n_nonorth=a.n_nonorth, n_procs=a.procs, t_end=a.t_end)
    return 0


def _natural_key(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(p))]


def _load_phase_grids(a, files):
    """Per-phase surfaces -> list of corresponding grids + the reference frame."""
    files = sorted(files, key=_natural_key)
    if len(files) == 1 and files[0].lower().endswith(".npz"):
        fs = np.load(files[0])
        if "grids" in fs:
            grids = [np.asarray(g, float) for g in fs["grids"]]
            print(f"  {files[0]}: {len(grids)} phase grids {grids[0].shape}")
            return grids, None
    grids, frame = [], None
    for k, f in enumerate(files):
        G, axis, e1, e2, mu, info = _load_grid(a, path=f, frame=frame)
        if k == 0:
            frame = (axis, e1, e2)
        grids.append(G)
    return grids, (frame[0], frame[1], frame[2], grids[0].reshape(-1, 3).mean(0))


def _run_motion(a, grids, frame, times, out):
    from .services.motion import build_lv_motion
    cfg = _conformal_cfg(a) if a.template == "conformal" else _single_kwargs(a)
    if a.template == "conformal":
        print(f"  template conformal: {cfg.approx_cells} cells per phase, inlet rule '{cfg.inlet}'")
    ref = a.reference if a.reference in ("es", "ed") else int(a.reference)
    res = build_lv_motion(grids, times, builder=a.template, cfg=cfg, reference=ref,
                          period=a.period, out=out, frame=frame,
                          auto_trim=a.auto_trim, fill_failed=a.fill_failed, metrics=a.metrics,
                          interpolation=a.table_interpolation, write_invalid=a.allow_invalid)
    if not res["valid"] and not res["case"]:
        print("  ABORT: a phase failed the gate and the case was not written (use --allow-invalid to write it anyway)")
    if a.metrics and a.template == "conformal":
        q = res["quality"]
        print(f"  metrics over the phases: non-orthogonality max {max(r['max_non_orthogonality_deg'] for r in q):.1f} deg, "
              f"skewness max {max(r['max_skewness'] for r in q):.2f} ({max(r['n_skewness_gt4'] for r in q)} faces > 4 at worst), "
              f"aspect ratio max {max(r['max_aspect_ratio'] for r in q):.1f}  (per phase in motion_quality.csv)")
    if res["case"]:
        _maybe_vtk(a, 0 if res["valid"] else 1, res["case"])
    return res


def cmd_motion(a):
    if a.template == "telescoping":
        print("motion: the AMI telescoping template is not a moving-mesh route; use conformal or single")
        return 2
    if a.apex or a.base:
        print("motion: --apex/--base are detected per phase here (the base moves through the cycle); "
              "they apply to `hexmb lv` only")
        return 2
    print(f"motion: {len(a.surfaces)} surface file(s)")
    grids, frame = _load_phase_grids(a, a.surfaces)
    n = len(grids)
    if a.times:
        times = np.array([float(x) for x in a.times.split(",")])
        if len(times) != n:
            print(f"  --times has {len(times)} entries for {n} phases"); return 1
    else:
        times = a.period * np.arange(n) / n
    res = _run_motion(a, grids, frame, times, a.out)
    return 0 if res["valid"] else 1


def cmd_demo(a):
    """The bundled CC0 Cardiac Atlas heart (SCD0000101, 20 phases): mesh every
    phase with the conformal template, gate it, and write the moving-mesh case.
    Needs numpy + scipy only."""
    from importlib import resources
    from .io.cap import cap_endo_surface, surface_frame
    from .io.stl_surface import write_surface_stl
    if a.template == "telescoping":
        print("demo: the AMI telescoping template is not a moving-mesh route; use conformal or single")
        return 2
    droot = resources.files("hexmb") / "data" / "cap" / "SCD0000101"
    files = sorted((str(p) for p in droot.iterdir() if str(p).endswith(".exnode")), key=_natural_key)
    files = files[: a.phases]
    out = Path(a.out)
    print(f"demo: Cardiac Atlas SCD0000101, {len(files)} phases (CC0), template {a.template}")
    grids = [cap_endo_surface(f, NL=a.NL, NT=a.NT) for f in files]
    axis, e1, e2, mu = surface_frame(grids[0])
    times = a.period * np.arange(len(grids)) / len(grids)
    out.mkdir(parents=True, exist_ok=True)
    ntri = write_surface_stl(grids[0], out / "lv_endo_phase1.stl", scale=1.0)
    print(f"  wrote {out / 'lv_endo_phase1.stl'} ({ntri} triangles, mm)")
    res = _run_motion(a, grids, (axis, e1, e2, mu), times, out / "lv_moving")
    if a.render:
        try:
            from .services.render import render_motion
            png = render_motion(grids, res, times, out / "demo.png")
            print(f"  wrote {png}")
        except ImportError as e:
            print(f"  render skipped ({e}); pip install matplotlib")
        except Exception as e:                       # a figure must not fail a written case
            print(f"  render skipped ({type(e).__name__}: {e})")
    print(f"  done: {out}/lv_moving is an OpenFOAM 12 case (foamRun -solver movingMesh runs the cycle); "
          f"quality table in {out}/lv_moving/motion_quality.csv")
    return 0 if res["valid"] else 1


# --------------------------------------------------------------------------- #
#  tubes from STL, verify                                                      #
# --------------------------------------------------------------------------- #
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


def cmd_render(a):
    from .services.render import render_case_slices
    gif = a.gif or (None if a.png else str(Path(a.out) / "motion_slice.gif"))
    rows = render_case_slices(a.out, out_gif=gif, out_png=a.png, n_panels=a.panels)
    print(f"render: {len(rows)} frames from t {rows[0][0]:.4g} to {rows[-1][0]:.4g} s, cavity "
          f"{min(r[1] for r in rows):.1f}-{max(r[1] for r in rows):.1f} mL")
    for f in (gif, a.png):
        if f:
            print(f"  wrote {f}")
    return 0


def cmd_verify(a):
    from .io.foam import read_polymesh
    pm = Path(a.out) / "constant" / "polyMesh"
    if not (pm / "points").exists():
        print(f"no polyMesh at {pm}"); return 1
    try:
        pts, faces, owner, neigh = read_polymesh(pm)
        if not np.all(owner[: len(neigh)] < neigh):
            raise ValueError("owner >= neighbour on an internal face")
        cell = np.zeros((int(owner.max()) + 1, 3))
        for fi, f in enumerate(faces):
            q = pts[f]; fc = q.mean(0); s = np.zeros(3)
            for i in range(len(f)):
                s += np.cross(q[i] - fc, q[(i + 1) % len(f)] - fc)
            s *= 0.5
            cell[owner[fi]] += s
            if fi < len(neigh):
                cell[neigh[fi]] -= s
        print(f"  readback: {len(pts)} pts, {len(faces)} faces ({len(neigh)} internal), owner<neighbour OK, "
              f"max |sum Sf| per cell {np.abs(cell).max():.2e}")
        from .services.quality import polymesh_metrics, metrics_line
        metrics = polymesh_metrics(pts, faces, owner, neigh)
        q = metrics["summary"]
        print("  metrics:  " + metrics_line(q))
        if (not np.isfinite(pts).all() or not all(np.isfinite(v) for v in q.values())
                or q["n_negative_volume"] or q["max_non_orthogonality_deg"] >= 90
                or q["max_openness"] > 1e-6
                or np.any(np.linalg.norm(metrics["face_areas"], axis=1) <= 0)):
            print("  FAILED: invalid geometry in the written mesh")
            return 1
    except Exception as e:
        print(f"  FAILED: readback check: {e}")
        return 1
    check_mesh = shutil.which("checkMesh")
    if check_mesh or Path("/opt/openfoam12/etc/bashrc").exists():
        if check_mesh:
            command = [check_mesh, "-case", str(Path(a.out).resolve()), "-constant"]
        else:
            cmd = ("source /opt/openfoam12/etc/bashrc && "
                   f"checkMesh -case {shlex.quote(str(Path(a.out).resolve()))} -constant")
            command = ["bash", "-c", cmd]
        r = subprocess.run(command, capture_output=True, text=True)
        shown = 0
        for ln in r.stdout.splitlines():
            if any(s in ln for s in ("hexahedra:", "polyhedra:", "Number of regions", "Min volume",
                                     "Max skewness", "non-orthogonality Max", "Mesh OK", "Failed", "negative")):
                print("  [checkMesh]", ln.strip()); shown += 1
        if not shown:
            tail = [ln for ln in (r.stderr + r.stdout).splitlines() if ln.strip()]
            why = ("OpenFOAM refuses a case path that contains a space (fileName::stripInvalid)"
                   if " " in str(a.out) else (tail[-1].strip() if tail else "no output"))
            print(f"  [checkMesh] no report (exit {r.returncode}): {why}")
        if r.returncode != 0 or re.search(r"\bFailed\s+\d+\s+mesh checks", r.stdout + r.stderr, re.I):
            print("  FAILED: checkMesh reported failure")
            return 1
        if "Mesh OK" not in r.stdout:
            print("  FAILED: checkMesh did not confirm Mesh OK")
            return 1
    else:
        print("  OpenFOAM not found; skipped checkMesh")
    return 0


# --------------------------------------------------------------------------- #
#  parser                                                                      #
# --------------------------------------------------------------------------- #
def _lv_common(sp, default_template="conformal"):
    sp.add_argument("-o", "--out", required=True, help="output case directory")
    sp.add_argument("--template", choices=TEMPLATES, default=default_template,
                    help="conformal: one-region split-hex telescoping LV (default); "
                         "single: one O-grid; telescoping: AMI-coupled zones")
    sp.add_argument("--NL", type=int, default=40, help="surface levels apex->base (ray-cast / CAP)")
    sp.add_argument("--NT", type=int, default=56, help="surface points around (÷4)")
    sp.add_argument("--apex", default=None, help="apex x,y,z (mm); default: detected")
    sp.add_argument("--base", default=None, help="valve-plane centre x,y,z (mm); default: detected")
    sp.add_argument("--label", type=int, default=None, help="mask label to use (.nii input)")
    sp.add_argument("--mask-method", choices=("auto", "slices", "marching"), default="auto", dest="mask_method",
                    help="mask reader: slices (one ring per thick slice), marching (marching cubes + rays), "
                         "auto (slices when the slice spacing is >= 2.5 x the pixel)")
    # conformal
    sp.add_argument("--NT-c", type=int, default=None, dest="NT_c", help="coarse ring size (body = 2x)")
    sp.add_argument("--NR-c", type=int, default=None, dest="NR_c", help="coarse radial layers")
    sp.add_argument("--apex-levels", type=int, default=None, dest="apex_levels")
    sp.add_argument("--body-levels", type=int, default=None, dest="body_levels",
                    help="axial sections in the fine body (refine here)")
    sp.add_argument("--base-levels", type=int, default=None, dest="base_levels")
    sp.add_argument("--apex-frac", type=float, default=None, dest="apex_frac")
    sp.add_argument("--base-frac", type=float, default=None, dest="base_frac")
    sp.add_argument("--apex-trim", type=float, default=None, dest="apex_trim")
    sp.add_argument("--base-cut", type=float, default=None, dest="base_cut")
    sp.add_argument("--wall-ratio", type=float, default=None, dest="wall_ratio")
    sp.add_argument("--inlet", choices=("sectors", "fullcap", "centred"), default=None,
                    help="valve-plane patch rule (conformal)")
    sp.add_argument("--inlet-area-cm2", type=float, default=None, dest="inlet_area_cm2")
    # single / telescoping
    sp.add_argument("--NR", type=int, default=6, help="radial layers (single / telescoping body)")
    sp.add_argument("--core-frac", type=float, default=None, dest="core_frac",
                    help="O-grid core size (default 0.42 conformal, 0.35 single)")
    sp.add_argument("--body-nt", type=int, default=88, dest="body_nt",
                    help="telescoping body circumferential resolution (340 ~= 1M cells)")
    sp.add_argument("--allow-invalid", action="store_true")
    sp.add_argument("--no-auto-trim", action="store_false", dest="auto_trim",
                    help="do not move base_cut / apex_trim when rings cross the axis or cells fail at a rim")
    sp.add_argument("--no-fill", action="store_false", dest="fill_failed",
                    help="motion/demo: keep a failing phase instead of filling it from its neighbours")
    sp.add_argument("--metrics", action="store_true",
                    help="add checkMesh-style metrics (non-orthogonality, skewness, aspect ratio) per phase, "
                         "computed without OpenFOAM (conformal template; ~0.3 s per phase at the default size)")
    sp.add_argument("--vtk", action="store_true",
                    help="also write the case as VTK (<case>/vtk: .vtu per phase, .vtp per patch, motion.pvd)")


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

    from .services.cavity import cmd_cavity
    sp = sub.add_parser("cavity", help="static box or smooth single cavity, with explicit units and quality report")
    source = sp.add_mutually_exclusive_group(required=True)
    source.add_argument("--size", nargs=3, type=float, metavar=("X", "Y", "Z"), help="box lengths")
    source.add_argument("--surface", help="smooth sac STL/OBJ/PLY/OFF/VTK or ordered ring .npz")
    sp.add_argument("-o", "--out", required=True, help="new or empty output directory")
    sp.add_argument("--units", choices=("mm", "cm", "m"), required=True, help="source coordinate units; output is metres")
    sp.add_argument("--cells", nargs=3, type=int, default=[24, 24, 24], help="box cell counts NX NY NZ")
    sp.add_argument("--stretch", nargs=3, type=float, default=[0., 0., 0.], help="box wall clustering X Y Z, each 0..5")
    sp.add_argument("--two-d", action="store_true", help="box only: empty front/back patches, requires NZ=1")
    sp.add_argument("--ends", choices=("closed", "open", "through"), default="closed", help="ring end-cap boundary rule")
    sp.add_argument("--start", type=float, default=0., help="explicit retained start fraction (e.g. 0.02 for a pointed sac)")
    sp.add_argument("--end", type=float, default=1., help="explicit retained end fraction")
    sp.add_argument("--apex", help="surface start landmark x,y,z, in source units")
    sp.add_argument("--base", help="surface end landmark x,y,z, in source units")
    sp.add_argument("--NL", type=int, default=40)
    sp.add_argument("--NT", type=int, default=32)
    sp.add_argument("--NR", type=int, default=6)
    sp.add_argument("--core-frac", type=float, default=0.35)
    sp.add_argument("--wall-ratio", type=float, default=1.)
    sp.add_argument("--max-nonorth", type=float, default=70., help="acceptance limit in degrees, must be <90")
    sp.add_argument("--max-skew", type=float, default=4.)
    sp.add_argument("--vtk", action="store_true")
    sp.set_defaults(func=cmd_cavity)

    sp = sub.add_parser("pipe", help="straight pipe O-grid"); common(sp)
    sp.add_argument("--length", type=float, default=100.0)
    sp.add_argument("--radius", type=float, default=10.0)
    sp.add_argument("--n-axial", type=int, default=40, dest="n_axial")
    sp.set_defaults(func=cmd_pipe)

    sp = sub.add_parser("elbow", help="U-bend / elbow O-grid"); common(sp)
    sp.add_argument("--radius", type=float, default=10.0)
    sp.add_argument("--bend-radius", type=float, default=40.0, dest="bend_radius")
    sp.add_argument("--angle", type=float, default=180.0)
    sp.add_argument("--leg", type=float, default=40.0)
    sp.set_defaults(func=cmd_elbow)

    sp = sub.add_parser("lv", help="left ventricle from any surface (STL/OBJ/PLY/VTK, mask, CAP, npz)")
    sp.add_argument("--surface", required=True, help="surface file (see formats above)")
    sp.add_argument("--stl", default=None, help="also write the ring-grid surface as an STL (mm)")
    sp.add_argument("--save-grid", default=None, dest="save_grid", help="also save the grid as .npz")
    _lv_common(sp)
    sp.set_defaults(func=cmd_lv)

    sp = sub.add_parser("cap", help="Cardiac Atlas Project .exnode LV model -> mesh")
    sp.add_argument("--exnode", required=True, help="CAP Sunnybrook LV model .exnode file")
    sp.add_argument("--stl", default=None, help="also write the endocardial surface as an STL/CAD file")
    sp.add_argument("--single-region", action="store_true", dest="single_region",
                    help="one O-grid instead of telescoping (inverts at the valve plane)")
    _lv_common(sp, default_template="telescoping")
    sp.set_defaults(func=cmd_cap)

    sp = sub.add_parser("motion", help="per-phase surfaces -> moving-mesh case (interpolator mover)")
    sp.add_argument("--surfaces", nargs="+", required=True,
                    help="one surface file per phase (sorted naturally), or one .npz with 'grids'")
    sp.add_argument("--times", default=None, help="comma-separated phase times (s); default uniform")
    sp.add_argument("--period", type=float, default=1.0, help="cycle length (s)")
    sp.add_argument("--reference", default="es", help="reference phase: es | ed | index")
    sp.add_argument("--table-interpolation", choices=("spline", "linear"), default="spline",
                    help="mover interpolation of the displacement table between phases (default spline)")
    _lv_common(sp)
    sp.set_defaults(func=cmd_motion)

    sp = sub.add_parser("demo", help="bundled CC0 Cardiac Atlas heart: 20 phases -> moving mesh")
    sp.add_argument("--period", type=float, default=1.0, help="cycle length (s) for the 20 phases")
    sp.add_argument("--phases", type=int, default=20, help="use the first N phases")
    sp.add_argument("--reference", default="es")
    sp.add_argument("--table-interpolation", choices=("spline", "linear"), default="spline",
                    help="mover interpolation of the displacement table between phases (default spline)")
    sp.add_argument("--render", action="store_true", help="write demo.png (needs matplotlib)")
    _lv_common(sp)
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser("stl", help="single-lumen tubular STL -> O-grid"); common(sp)
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

    sp = sub.add_parser("export", help="a written case -> VTK (.vtu/.vtp/.pvd) and/or Abaqus .inp")
    sp.add_argument("-o", "--out", required=True, help="the case directory")
    sp.add_argument("--vtk", action="store_true", help="write VTK (default when --inp is not given)")
    sp.add_argument("--vtk-dir", default=None, dest="vtk_dir", help="VTK output directory (default <case>/vtk)")
    sp.add_argument("--ascii", action="store_true", help="ASCII VTK arrays instead of base64 binary")
    sp.add_argument("--inp", default=None, help="Abaqus/CalculiX .inp to write (all-hex meshes only)")
    sp.add_argument("--scale", type=float, default=1.0, help="coordinate factor (polyMesh is in metres; 1000 -> mm)")
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("flow", help="moving-mesh case -> wall-driven OpenFOAM 12 flow case (in place)")
    sp.add_argument("-o", "--out", required=True, help="the moving-mesh case directory")
    sp.add_argument("--window", choices=("diastole", "systole"), default="diastole",
                    help="diastole: ES -> ED, mitral open (case reference es); systole: ED -> ES, aortic open "
                         "(case built with --reference ed)")
    sp.add_argument("--nu", type=float, default=0.004 / 1060.0, help="kinematic viscosity m2/s (blood 3.77e-6)")
    sp.add_argument("--max-co", type=float, default=1.0, dest="max_co")
    sp.add_argument("--max-dt", type=float, default=None, dest="max_dt", help="max time step s (default min(1e-3, window/100))")
    sp.add_argument("--write-interval", type=float, default=None, dest="write_interval", help="default window/20")
    sp.add_argument("--n-outer", type=int, default=3, dest="n_outer")
    sp.add_argument("--n-corr", type=int, default=2, dest="n_corr")
    sp.add_argument("--n-nonorth", type=int, default=2, dest="n_nonorth")
    sp.add_argument("--procs", type=int, default=4, help="cores for Allrun (decomposePar + mpirun)")
    sp.add_argument("--t-end", type=float, default=None, dest="t_end", help="override the window end time")
    sp.set_defaults(func=cmd_flow)

    sp = sub.add_parser("verify", help="read a case back and run checkMesh if OpenFOAM is present")
    sp.add_argument("-o", "--out", required=True)
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("render", help="long-axis cut of a moving-mesh case through the cycle (GIF/PNG)")
    sp.add_argument("-o", "--out", required=True, help="the moving-mesh case directory")
    sp.add_argument("--gif", default=None, help="output GIF (default <case>/motion_slice.gif)")
    sp.add_argument("--png", default=None, help="also a PNG strip of --panels frames")
    sp.add_argument("--panels", type=int, default=4)
    sp.set_defaults(func=cmd_render)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except (ValueError, RuntimeError, FileExistsError, FileNotFoundError, KeyError) as e:
        print(f"hexmb {args.cmd}: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
