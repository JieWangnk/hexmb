"""Moving mesh from per-phase surfaces: build the same mesh at every phase, gate
its quality, and write an OpenFOAM 12 case for the ``interpolator`` mesh mover.

The recipe is the one validated on the beating left ventricle: the mesh is
REBUILT from each phase's own surface (iso-topological resurfacing).  Every
phase then has the same connectivity and its own point positions; the case stores the reference
phase's ``constant/polyMesh`` and one ``wantedDisplacement`` point field per
phase time (displacement from the reference, in metres).  OpenFOAM interpolates
between the phase times at run time.

Usage::

    from hexmb.services.motion import build_lv_motion
    res = build_lv_motion(grids, times, builder="conformal", period=1.0, out="case_moving")
    # res["quality"] is one row per phase; res["ref"] the reference phase index;
    # res["adjustments"] what auto_trim changed; res["filled"] the phases filled in

``grids`` are ``(NL, NT, 3)`` surfaces (mm) that correspond point-for-point
across phases (cast with one frame, see :func:`hexmb.io.surface.endo_grid_from_mesh`).
The reference phase is end-systole by default (the smallest cavity): a mesh
built on the contracted shape and expanded morphs more safely than one built on
the dilated shape and squeezed.

Two things the builder does by itself (both can be switched off):

* ``auto_trim``: a row whose ring does not wind once about its own centre (the
  surface passes through its own long axis there, as some Cardiac Atlas fits do
  at end-systole) cannot be meshed.  A block of such rows at the base moves
  ``base_cut`` down to the last good row and a block at the apex moves
  ``apex_trim`` up past it, for every phase alike (the valve plane must sit at
  the same fraction on every phase).  If cells still fail the gate at a rim, the
  rim is trimmed by one step (``base_cut`` 0.03, ``apex_trim`` 0.02) and every
  phase is rebuilt, at most three times.  Every change is reported.
* ``fill_failed``: a phase that still fails (rows crossing the axis in the
  middle of the ventricle, or cells failing away from the rims) is replaced by
  the linear interpolation in time between its nearest valid neighbours, and
  that mesh is gated like any other.  The quality table marks it ``filled``.
"""
from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import numpy as np

from ..templates.conformal_lv import (ConformalLVConfig, build_conformal_lv, bad_cell_span,
                                      bad_rings, axial_fractions, resample_grid_axial, grid_volume)

__all__ = ["build_lv_motion", "build_phases", "write_point_field", "hex_volumes", "FOAM_HEADER"]

FOAM_HEADER = (
    "FoamFile\n{{\n    version     2.0;\n    format      ascii;\n    class       {cls};\n"
    "    location    \"{loc}\";\n    object      {obj};\n}}\n"
)

BASE_STEP, BASE_FLOOR = 0.03, 0.02       # base_cut moves down by BASE_STEP, never within FLOOR of base_frac
APEX_STEP, APEX_CAP = 0.02, 0.02         # apex_trim moves up by APEX_STEP, never within CAP of apex_frac


def _tname(t: float) -> str:
    return f"{t:.6g}"


def hex_volumes(points, hexes) -> np.ndarray:
    """Signed volumes of hexahedra (vertex order as in hexmb's blocks: bottom quad
    then top quad, right-handed), by the divergence theorem over the six quad
    faces in their stored orientation.  A mirrored (inverted) hex is negative; a
    hex folded past its centre is smaller than its convex hull."""
    P = np.asarray(points, float)[np.asarray(hexes, int)]       # (n, 8, 3)
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    cc = P.mean(1)
    vol = np.zeros(len(P))
    for f in faces:
        q = P[:, f, :]
        fc = q.mean(1)
        s = np.zeros((len(P), 3))
        for i in range(4):
            s += np.cross(q[:, i] - fc, q[:, (i + 1) % 4] - fc)
        vol += np.einsum("ij,ij->i", fc - cc, 0.5 * s)
    return vol / 3.0


def write_point_field(case_dir, time_name, disp_m, patch_names, obj="wantedDisplacement"):
    """Write a ``pointVectorField`` (``disp_m``: ``(n_points, 3)`` metres) into
    ``case_dir/time_name/obj`` with ``calculated`` boundary entries."""
    d = Path(case_dir) / time_name
    d.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"({v[0]:.8g} {v[1]:.8g} {v[2]:.8g})" for v in np.asarray(disp_m, float))
    bf = "".join(f"    {p}\n    {{\n        type calculated;\n    }}\n" for p in patch_names)
    (d / obj).write_text(
        FOAM_HEADER.format(cls="pointVectorField", loc=time_name, obj=obj)
        + f"dimensions      [0 1 0 0 0 0 0];\n\ninternalField   nonuniform List<vector>\n"
        f"{len(disp_m)}\n(\n{body}\n);\n\nboundaryField\n{{\n{bf}}}\n")


# --------------------------------------------------------------------------- #
#  per-phase build + gate                                                      #
# --------------------------------------------------------------------------- #
def _cfg_get(cfg, key, default):
    return cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)


def _cfg_with(cfg, **kw):
    if isinstance(cfg, dict):
        c = dict(cfg); c.update(kw)
        return c
    return dataclasses.replace(cfg, **kw)


def _single_gate(pts, hexes, axis, mu):
    """``(volumes, n_bad, (lo, hi), min_sj)`` for the single O-grid: a cell fails on a
    non-positive scaled Jacobian or volume."""
    from .quality import scaled_jac
    pts = np.asarray(pts, float); hexes = np.asarray(hexes, int)
    v = hex_volumes(pts, hexes)
    sj = scaled_jac(pts, hexes)
    bad = (sj <= 0) | (v <= 0)
    span = (float("nan"), float("nan"))
    if bad.any():
        s = (pts[hexes].mean(1) - np.asarray(mu, float)) @ (np.asarray(axis, float) / np.linalg.norm(axis))
        f = (s[bad] - s.min()) / (np.ptp(s) + 1e-30)
        span = (float(f.min()), float(f.max()))
    return v, int(bad.sum()), span, float(sj.min())


def _build_phase(G, frame, builder, cfg):
    """Return ``(points_mm, cells_signature, volumes_mm3, mesh, n_bad, span, extra)``
    for one phase.  ``n_bad`` counts cells failing the hard gate: a non-positive
    signed pyramid or a zero-area face (conformal), a non-positive scaled
    Jacobian or volume (single).  ``span`` is the axial range (0 apex, 1 base) of
    the failing cells' centres."""
    axis, e1, e2, mu = frame
    if builder == "conformal":
        m = build_conformal_lv(G, axis, e1, e2, cfg, mu=mu)
        v = m.cell_volumes()
        n_bad, lo, hi = bad_cell_span(m, axis, mu)
        n_zero = m.n_zero_area_faces(1e-9)
        n_bad += n_zero
        extra = {"min_pyramid_mm3": float(m.min_pyramid_per_cell().min()),
                 "n_zero_area_faces": n_zero}
        return m.points, m.cells, v, m, n_bad, (lo, hi), extra
    if builder == "single":
        from ..templates.lv import LVChamberTemplate
        kw = dict(cfg or {})
        trim = float(kw.pop("apex_trim", 0.02))        # a point apex inverts the first ring of hexes
        cut = float(kw.pop("base_cut", 1.0))
        G = resample_grid_axial(G, trim, cut, axis=axis)   # levels uniform along the axis, tip trimmed
        m = LVChamberTemplate(G, axis, e1, e2, mu=mu, **kw).build()
        pts = np.asarray(m.points, float)
        v, n_bad, span, sj_min = _single_gate(pts, m.hexes, axis, mu)
        extra = {"min_scaled_jac": sj_min, "n_inverted": n_bad}
        return pts, np.asarray(m.hexes).tolist(), v, m, n_bad, span, extra
    raise ValueError(f"builder must be 'conformal' or 'single', got {builder!r}")


def _screen_rings(grids, frame, cfg):
    """Rows whose ring does not wind once about its centre cannot be meshed.  A
    block of them at the base moves ``base_cut`` to the last good row, a block at
    the apex moves ``apex_trim`` past them (the smallest cut / largest trim over
    the phases, so every phase shares one valve plane); rows away from the rims
    mark the phase as degenerate.  Returns ``(cfg, adjustments, degenerate)``."""
    axis, e1, e2, _ = frame
    NL = grids[0].shape[0]
    base_cut = _cfg_get(cfg, "base_cut", 1.0); apex_trim = _cfg_get(cfg, "apex_trim", 0.02)
    base_frac = _cfg_get(cfg, "base_frac", 0.80); apex_frac = _cfg_get(cfg, "apex_frac", 0.24)
    cuts, trims, degenerate, where = {}, {}, {}, {}
    for k, G in enumerate(grids):
        bad = bad_rings(G, axis, e1, e2)
        if not bad:
            continue
        t = axial_fractions(G, axis)
        lo, hi = min(bad), max(bad)
        if set(bad) == set(range(lo, NL)) and lo >= 2:
            cuts[k] = float(t[lo - 1]); where[k] = f"rows {lo}-{NL - 1} of {NL}"
        elif set(bad) == set(range(1, hi + 1)) and hi + 2 < NL:
            trims[k] = float(t[hi + 1]); where[k] = f"rows 1-{hi} of {NL}"
        else:
            degenerate[k] = (f"rows {lo}-{hi} of {NL} do not wind once about the axis (the surface "
                             "crosses itself away from the rims): no trim can fix this phase")
    adjustments = []
    if cuts:
        worst = min(cuts, key=cuts.get)
        new = round(cuts[worst], 4)
        if new <= base_frac + BASE_FLOOR:
            for k in cuts:
                degenerate[k] = (f"{where[k]} do not wind once about the axis and the cut needed "
                                 f"({cuts[k]:.2f}) reaches base_frac {base_frac:.2f}: no trim can fix this phase")
        elif new < base_cut:
            adjustments.append(
                f"base_cut {base_cut:.3f} -> {new:.3f}: {where[worst]} do not wind once about the axis "
                f"on phase(s) {' '.join(map(str, sorted(cuts)))} (the surface crosses itself at the base "
                "rim); the base now stops at the last good row")
            cfg = _cfg_with(cfg, base_cut=new)
    if trims:
        worst = max(trims, key=trims.get)
        new = round(trims[worst], 4)
        if new >= apex_frac - APEX_CAP:
            for k in trims:
                degenerate[k] = (f"{where[k]} do not wind once about the axis and the trim needed "
                                 f"({trims[k]:.2f}) reaches apex_frac {apex_frac:.2f}: no trim can fix this phase")
        elif new > apex_trim:
            adjustments.append(
                f"apex_trim {apex_trim:.3f} -> {new:.3f}: {where[worst]} do not wind once about the axis "
                f"on phase(s) {' '.join(map(str, sorted(trims)))} (the surface crosses itself at the apex); "
                "the apex zone now starts above them")
            cfg = _cfg_with(cfg, apex_trim=new)
    return cfg, adjustments, degenerate


def _add_metrics(meshes, rows, which=None):
    """checkMesh-style face metrics per phase (conformal meshes): the face
    topology is built once (all phases share it) and evaluated on each phase's
    points.  Adds ``max_skewness``, ``n_skewness_gt4``,
    ``max_non_orthogonality_deg``, ``average_non_orthogonality_deg`` and
    ``max_aspect_ratio`` to the rows."""
    from .quality import polymesh_metrics
    first = next((m for m in meshes if m is not None and hasattr(m, "topology")), None)
    if first is None:
        return
    faces, owner, neigh, _ = first.topology()
    for k, m in enumerate(meshes):
        if (which is not None and k not in which) or m is None or not hasattr(m, "topology"):
            continue
        sm = polymesh_metrics(m.points, faces, owner, neigh, n_cells=len(m.cells))["summary"]
        rows[k].update({"max_skewness": round(sm["max_skewness"], 3), "n_skewness_gt4": sm["n_skewness_gt4"],
                        "max_non_orthogonality_deg": round(sm["max_non_orthogonality_deg"], 2),
                        "n_non_orthogonality_gt70": sm["n_non_orthogonality_gt70"],
                        "average_non_orthogonality_deg": round(sm["average_non_orthogonality_deg"], 2),
                        "max_aspect_ratio": round(sm["max_aspect_ratio"], 2)})


def build_phases(grids, frame, builder="conformal", cfg=None, auto_trim=True, verbose=True,
                 max_retries=3, metrics=False):
    """Build and gate every grid with ONE template configuration.

    ``metrics=True`` adds the checkMesh-style face metrics (max / average
    non-orthogonality, max skewness and the count of faces above 4, max aspect
    ratio) to every phase's row, computed by :mod:`hexmb.services.quality` with
    OpenFOAM's definitions; the face topology is built once and reused, so the
    cost is about 0.3 s per phase at the default size.

    With ``auto_trim`` the rings are screened first (see :func:`bad_rings`) and,
    when cells fail the gate at a rim, the rim is trimmed by one step and every
    phase is rebuilt (at most ``max_retries`` times).  Phases whose failure is
    not at a rim are left as they are (``degenerate``); a phase whose rings
    cross the axis away from the rims is marked ``failed`` even when its cells
    pass the gate, because the surface itself is not a surface there.

    Returns a dict: ``points`` (list of ``(n, 3)`` mm), ``cells`` (topology
    signature), ``volumes`` (list of per-cell mm^3), ``meshes``, ``rows`` (one
    quality dict per phase with ``status`` ``ok`` | ``failed``), ``cfg`` (the
    configuration used), ``adjustments`` (strings) and ``degenerate``
    (phase -> reason)."""
    grids = [np.asarray(g, float) for g in grids]
    if cfg is None:
        cfg = ConformalLVConfig() if builder == "conformal" else {}
    adjustments, degenerate = [], {}
    if auto_trim:
        cfg, adjustments, degenerate = _screen_rings(grids, frame, cfg)
        if verbose:
            for a in adjustments:
                print(f"  auto_trim: {a}")
            for k, why in degenerate.items():
                print(f"  phase {k}: {why}")
    axis, _, _, mu = frame
    for attempt in range(max_retries + 1):
        pts, cells0, vols, meshes, rows, spans = [], None, [], [], [], []
        for k, G in enumerate(grids):
            P, cells, v, m, n_bad, span, extra = _build_phase(G, frame, builder, cfg)
            if cells0 is None:
                cells0 = cells
            elif cells != cells0 or len(P) != len(pts[0]):
                raise RuntimeError(f"phase {k} produced a different topology: the surfaces do not "
                                   "correspond (same NL/NT and the same frame on every phase?)")
            pts.append(P); vols.append(v); meshes.append(m); spans.append(span)
            row = {"phase": k, "volume_mL": float(v.sum() * 1e-3), "min_cell_volume_mm3": float(v.min()),
                   "n_nonpositive": n_bad}
            row.update(extra)
            # a phase whose surface crosses itself away from the rims is failed even when every
            # cell happens to pass the gate: the mesh would follow a surface that is not a surface
            row["status"] = "ok" if n_bad == 0 and k not in degenerate else "failed"
            row["note"] = degenerate.get(k, "")
            if n_bad and not row["note"]:
                row["note"] = f"{n_bad} cells fail at axial fraction {span[0]:.2f}-{span[1]:.2f} (0 apex; 1 base)"
            elif n_bad == 0 and row["note"]:
                row["note"] += " (every cell passes the gate; the surface is the problem)"
            rows.append(row)
            if verbose:
                ex = f"  min SJ {extra['min_scaled_jac']:+.3f}" if "min_scaled_jac" in extra else ""
                print(f"  phase {k:2d}  volume {row['volume_mL']:6.1f} mL  min cell vol {v.min():+.3e} mm3  "
                      f"failing cells {n_bad}{ex}")
        failing = [k for k, r in enumerate(rows) if r["n_nonpositive"] and k not in degenerate]
        if not failing or not auto_trim or attempt == max_retries:
            break
        lo = min(spans[k][0] for k in failing); hi = max(spans[k][1] for k in failing)
        n_bad = sum(rows[k]["n_nonpositive"] for k in failing)
        base_cut = _cfg_get(cfg, "base_cut", 1.0); apex_trim = _cfg_get(cfg, "apex_trim", 0.02)
        base_frac = _cfg_get(cfg, "base_frac", 0.80); apex_frac = _cfg_get(cfg, "apex_frac", 0.24)
        phases = " ".join(map(str, failing))
        if lo > 0.9 and round(base_cut - BASE_STEP, 4) > base_frac + BASE_FLOOR:
            new = round(base_cut - BASE_STEP, 4)
            a = (f"base_cut {base_cut:.3f} -> {new:.3f}: {n_bad} cells at the base rim (axial fraction "
                 f"{lo:.2f}-{hi:.2f}) fail on phase(s) {phases}; rebuilding every phase")
            cfg = _cfg_with(cfg, base_cut=new)
        elif hi < 0.3 and round(apex_trim + APEX_STEP, 4) < apex_frac - APEX_CAP:
            new = round(apex_trim + APEX_STEP, 4)
            a = (f"apex_trim {apex_trim:.3f} -> {new:.3f}: {n_bad} cells in the apex zone (axial fraction "
                 f"{lo:.2f}-{hi:.2f}) fail on phase(s) {phases}; rebuilding every phase")
            cfg = _cfg_with(cfg, apex_trim=new)
        else:
            for k in failing:
                degenerate[k] = (f"{rows[k]['n_nonpositive']} cells fail at axial fraction "
                                 f"{spans[k][0]:.2f}-{spans[k][1]:.2f} (0 apex; 1 base): not at a rim; "
                                 "no trim can fix this phase")
                rows[k]["note"] = degenerate[k]
            break
        adjustments.append(a)
        if verbose:
            print(f"  auto_trim: {a}")
    if adjustments:
        # what the trims took away: the cavity volume of each surface above base_cut and below
        # apex_trim, so a cohort table can carry the cut and its cost side by side
        base_cut = _cfg_get(cfg, "base_cut", 1.0); apex_trim = _cfg_get(cfg, "apex_trim", 0.02)
        removed = []
        for k, G in enumerate(grids):
            full = grid_volume(G)
            kept = grid_volume(resample_grid_axial(G, apex_trim, base_cut, 200, axis=axis))
            removed.append(max(0.0, full - kept) * 1e-3)
            rows[k]["trim_removed_mL"] = round(removed[k], 3)
        vol = [r["volume_mL"] for r in rows]
        kmax, kmin = int(np.argmax(vol)), int(np.argmin(vol))
        a = (f"the trims remove {removed[kmax]:.1f} mL ({100 * removed[kmax] / (vol[kmax] + removed[kmax]):.1f} % of the "
             f"untrimmed cavity) at the largest phase ({kmax}) and {removed[kmin]:.1f} mL "
             f"({100 * removed[kmin] / (vol[kmin] + removed[kmin]):.1f} %) at the smallest ({kmin}); the volumes in the "
             "quality table are those of the trimmed cavity (column trim_removed_mL)")
        adjustments.append(a)
        if verbose:
            print(f"  auto_trim: {a}")
    if metrics and builder == "conformal":
        _add_metrics(meshes, rows)
    return {"points": pts, "cells": cells0, "volumes": vols, "meshes": meshes, "rows": rows,
            "cfg": cfg, "adjustments": adjustments, "degenerate": degenerate}


def _fill_phase(k, rows, pts, times, period, meshes, builder, frame):
    """Replace phase ``k`` by the linear interpolation in time between its nearest
    valid neighbours (cyclic when ``period`` is given) and gate the result.
    Returns ``(points, volumes, n_bad, note)`` or ``None`` when no valid phase exists."""
    n = len(rows)
    valid = [j for j in range(n) if rows[j]["status"] == "ok"]
    if not valid:
        return None
    before = [j for j in valid if j < k]; after = [j for j in valid if j > k]
    if period is not None:
        prev = before[-1] if before else valid[-1]
        nxt = after[0] if after else valid[0]
        tp = times[prev] - (period if not before else 0.0)
        tn = times[nxt] + (period if not after else 0.0)
    else:
        prev = before[-1] if before else after[0]
        nxt = after[0] if after else before[-1]
        tp, tn = times[prev], times[nxt]
    w = 0.0 if prev == nxt else float((times[k] - tp) / (tn - tp))
    P = (1.0 - w) * pts[prev] + w * pts[nxt]
    axis, _, _, mu = frame
    if builder == "conformal":
        m = meshes[k]
        m.P = [tuple(p) for p in P]
        v = m.cell_volumes()
        n_bad, _, _ = bad_cell_span(m, axis, mu)
        n_bad += m.n_zero_area_faces(1e-9)
    else:
        v, n_bad, _, _ = _single_gate(P, meshes[k].hexes, axis, mu)
    how = f"copied from phase {prev}" if prev == nxt else f"interpolated between phases {prev} and {nxt} (weight {w:.2f})"
    return P, v, n_bad, how


# --------------------------------------------------------------------------- #
#  case writer                                                                 #
# --------------------------------------------------------------------------- #
def _write_reference(mesh, builder, case_dir, scale=1e-3):
    case_dir = Path(case_dir)
    if builder == "conformal":
        mesh.write(case_dir, scale=scale)
        return list(mesh.patches)
    if scale != 1e-3:
        raise ValueError("the single builder writes the polyMesh in metres from mm (scale 1e-3) only")
    info = mesh.write_polymesh(case_dir / "constant" / "polyMesh")
    return [p[0] for p in info["patches"]]


def _write_case_dicts(case_dir, t_start, t_end, dt, write_interval, interpolation="spline"):
    case_dir = Path(case_dir)
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    (case_dir / "constant").mkdir(parents=True, exist_ok=True)
    H = FOAM_HEADER
    (case_dir / "constant" / "dynamicMeshDict").write_text(
        H.format(cls="dictionary", loc="constant", obj="dynamicMeshDict")
        + "mover\n{\n    type                interpolator;\n"
        "    libs                (\"libfvMeshMoversInterpolator.so\");\n"
        "    field               wantedDisplacement;\n    displacement        yes;\n"
        # spline = Catmull-Rom through the table: continuous wall velocity.  linear gives a
        # piecewise-constant wall velocity that jumps at every table time (stair-step valve
        # flow, speed spikes); see docs/VERIFICATION.md.
        f"    interpolationScheme {interpolation};\n}}\n")
    (case_dir / "system" / "controlDict").write_text(
        H.format(cls="dictionary", loc="system", obj="controlDict")
        + f"application     foamRun;\nsolver          movingMesh;\nstartFrom       startTime;\n"
        f"startTime       {_tname(t_start)};\nstopAt          endTime;\nendTime         {_tname(t_end)};\n"
        f"deltaT          {dt:.6g};\nwriteControl    adjustableRunTime;\nwriteInterval   {write_interval:.6g};\n"
        "writeFormat     ascii;\nwritePrecision  7;\nrunTimeModifiable no;\n")
    (case_dir / "system" / "fvSchemes").write_text(
        H.format(cls="dictionary", loc="system", obj="fvSchemes")
        + "ddtSchemes\n{\n    default         Euler;\n}\ngradSchemes\n{\n    default         Gauss linear;\n}\n"
        "divSchemes\n{\n    default         none;\n}\nlaplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n"
        "interpolationSchemes\n{\n    default         linear;\n}\nsnGradSchemes\n{\n    default         corrected;\n}\n")
    (case_dir / "system" / "fvSolution").write_text(
        H.format(cls="dictionary", loc="system", obj="fvSolution")
        + "solvers\n{\n}\n\nPIMPLE\n{\n    nOuterCorrectors 1;\n    nCorrectors      1;\n"
        "    nNonOrthogonalCorrectors 0;\n}\n")


def build_lv_motion(grids, times, builder="conformal", cfg=None, reference="es", period=None,
                    out=None, frame=None, overwrite=True, scale=1e-3, verbose=True,
                    auto_trim=True, fill_failed=True, prebuilt=None, metrics=False, interpolation="spline",
                    write_invalid=True):
    """Build one LV mesh per phase, gate every phase, and (optionally) write the
    moving-mesh case.

    Parameters
    ----------
    grids : sequence of (NL, NT, 3) arrays
        One endocardial surface per phase (mm), point-for-point corresponding.
    times : sequence of float
        Phase times in seconds, ascending.
    builder : "conformal" | "single"
        Conformal split-hex telescoping LV (default; one conservative region) or
        the single O-grid chamber (:class:`~hexmb.templates.lv.LVChamberTemplate`).
    cfg : ConformalLVConfig | dict | None
        Template settings (a dict of ``LVChamberTemplate`` keywords for "single",
        plus ``apex_trim`` / ``base_cut``).
    reference : "es" | "ed" | int
        Reference phase: smallest cavity, largest cavity, or an explicit index.
    interpolation : "spline" | "linear"
        How the mover interpolates the displacement table between phases.
        ``spline`` (Catmull-Rom, default) gives a continuous wall velocity;
        ``linear`` holds the velocity constant between phases, so the valve
        flow steps at every table time (see docs/VERIFICATION.md).
    write_invalid : bool
        With ``False`` the case is written only when every phase passed (or was
        filled); the result then carries ``valid=False`` and ``case=None``.
        The CLI passes ``--allow-invalid`` here.
    period : float, optional
        Cycle length (s).  When given, the phases up to the reference one are
        repeated one period later, so the table covers a whole cycle from the
        reference time (``startTime``) to ``startTime + period`` (``endTime``).
    out : path, optional
        Case directory to write.  Omit to only build and gate.
    frame : (axis, e1, e2, mu), optional
        Frame for the valve classifier; default from the reference grid.
    auto_trim : bool
        Screen the rings and trim a rim that fails the gate (module docstring).
    fill_failed : bool
        Replace a phase that still fails by the interpolation of its valid
        neighbours in time (module docstring).
    prebuilt : dict, optional
        The result of :func:`build_phases` for these grids, to write several
        cases (say one per reference phase) from one build.  Not modified.

    Returns a dict with ``points`` (``(NP, n_points, 3)`` mm), ``quality`` (one dict
    per phase, with ``status`` ``ok`` | ``filled`` | ``failed`` and a ``note``),
    ``volumes_mL``, ``ref``, ``valid`` (every phase ok or filled), ``cfg`` (the
    configuration used), ``adjustments``, ``filled`` (phase indices), ``patches``
    and ``case``.
    """
    grids = [np.asarray(g, float) for g in grids]
    times = np.asarray(times, float)
    if len(grids) < 2:
        raise ValueError("a moving mesh needs at least two phases")
    if len(grids) != len(times):
        raise ValueError("one time per grid")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    if len({g.shape for g in grids}) != 1:
        raise ValueError("all grids must have the same (NL, NT, 3) shape")
    if period is not None and period <= times[-1]:
        raise ValueError("period must exceed the last phase time")

    if frame is None:
        from ..io.cap import surface_frame
        axis, e1, e2, mu = surface_frame(grids[0])
        frame = (axis, e1, e2, mu)

    if prebuilt is None:
        b = build_phases(grids, frame, builder, cfg, auto_trim=auto_trim, verbose=verbose, metrics=metrics)
    else:
        if len(prebuilt["rows"]) != len(grids):
            raise ValueError(f"prebuilt holds {len(prebuilt['rows'])} phases for {len(grids)} grids")
        b = dict(prebuilt, rows=[dict(r) for r in prebuilt["rows"]], points=list(prebuilt["points"]),
                 volumes=list(prebuilt["volumes"]), adjustments=list(prebuilt["adjustments"]))
    pts, vols, meshes, rows = b["points"], b["volumes"], b["meshes"], b["rows"]
    for k, r in enumerate(rows):
        r["time_s"] = float(times[k])
    filled = []
    if fill_failed:
        for k in [k for k, r in enumerate(rows) if r["status"] == "failed"]:
            f = _fill_phase(k, rows, pts, times, period, meshes, builder, frame)
            if f is None:
                break
            P, v, n_bad, how = f
            if n_bad:
                rows[k]["note"] += f"; {how} still fails the gate ({n_bad} cells)"
                continue
            pts[k], vols[k] = P, v
            rows[k].update({"volume_mL": float(v.sum() * 1e-3), "min_cell_volume_mm3": float(v.min()),
                            "n_nonpositive": 0, "status": "filled",
                            "note": f"{how}; the surface of this phase: {rows[k]['note']}"})
            if "min_pyramid_mm3" in rows[k]:
                rows[k]["min_pyramid_mm3"] = float(meshes[k].min_pyramid_per_cell().min())
                rows[k]["n_zero_area_faces"] = 0
            if "min_scaled_jac" in rows[k]:
                from .quality import scaled_jac
                rows[k]["min_scaled_jac"] = float(scaled_jac(P, np.asarray(meshes[k].hexes, int)).min())
                rows[k]["n_inverted"] = 0
            if "max_skewness" in rows[k]:
                _add_metrics(meshes, rows, which=[k])
            filled.append(k)
            if verbose:
                print(f"  phase {k:2d}: {how}  -> volume {rows[k]['volume_mL']:6.1f} mL, every cell passes")
    pts = np.stack(pts)
    volumes = np.array([r["volume_mL"] for r in rows])
    # the reference mesh becomes constant/polyMesh: choose it among the phases that passed
    # (or were filled); a failed phase is the smallest when its inverted cells subtract
    good = [k for k, r in enumerate(rows) if r["status"] in ("ok", "filled")] or list(range(len(rows)))
    if reference == "es":
        ref = good[int(volumes[good].argmin())]
    elif reference == "ed":
        ref = good[int(volumes[good].argmax())]
    else:
        ref = int(reference)
    ok = all(r["status"] in ("ok", "filled") for r in rows)
    res = {"points": pts, "quality": rows, "volumes_mL": volumes, "ref": ref, "valid": ok,
           "patches": None, "case": None, "frame": frame, "cfg": b["cfg"],
           "adjustments": b["adjustments"], "filled": filled}
    if verbose:
        failed = [r["phase"] for r in rows if r["status"] == "failed"]
        tail = (f"; {len(filled)} phase(s) filled from neighbours ({' '.join(map(str, filled))})" if filled else "")
        tail += f"; FAILED phase(s) {' '.join(map(str, failed))}" if failed else ""
        print(f"  reference phase {ref} (t {times[ref]:.4f} s, {volumes[ref]:.1f} mL); "
              f"EDV {volumes.max():.1f}  ESV {volumes.min():.1f} mL  "
              f"EF {100 * (1 - volumes.min() / volumes.max()):.0f} %  -> {'VALID' if ok else 'INVALID'}{tail}")
    if out is None or (not ok and not write_invalid):
        return res

    case_dir = Path(out)
    if case_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{case_dir} exists (overwrite=False)")
        is_case = ((case_dir / "constant").exists() or (case_dir / "case.foam").exists()
                   or not any(case_dir.iterdir()))
        if not is_case:
            raise ValueError(f"{case_dir} exists and is not an OpenFOAM case directory (no constant/ or "
                             "case.foam); refusing to delete it. Choose an empty or case directory.")
        shutil.rmtree(case_dir)
    if ref in filled:                                      # the writers read the mesh's own points
        if builder == "conformal":
            meshes[ref].P = [tuple(p) for p in pts[ref]]; meshes[ref]._ft_key = None
        else:
            meshes[ref].points = np.asarray(pts[ref], float)
    patches = _write_reference(meshes[ref], builder, case_dir, scale)
    ref_pts = pts[ref]
    for k in range(len(grids)):
        write_point_field(case_dir, _tname(times[k]), (pts[k] - ref_pts) * scale, patches)
    t_end = float(times[-1])
    n_written = len(grids)
    if period is not None:
        # wrap: repeat the phases up to (and including) the reference one period later, so the
        # table covers a whole cycle starting at the reference time
        for k in range(len(grids)):
            if times[k] <= times[ref]:
                write_point_field(case_dir, _tname(times[k] + period), (pts[k] - ref_pts) * scale, patches)
                n_written += 1
        t_end = float(times[ref] + period)
    dt = float(np.diff(times).min()) / 5.0
    _write_case_dicts(case_dir, float(times[ref]), t_end, dt, float(np.diff(times).min()), interpolation=interpolation)
    keys = ["phase", "time_s"] + [k for k in rows[0] if k not in ("phase", "time_s", "status", "note")] + ["status", "note"]
    with open(case_dir / "motion_quality.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]).replace(",", ";") for k in keys) + "\n")
    if b["adjustments"]:
        (case_dir / "motion_adjustments.txt").write_text("\n".join(b["adjustments"]) + "\n")
    (case_dir / "case.foam").write_text("")
    res.update({"patches": patches, "case": str(case_dir)})
    if verbose:
        print(f"  wrote {case_dir}: constant/polyMesh at phase {ref}, "
              f"{n_written} wantedDisplacement time folders (t {_tname(times[0])} to {_tname(t_end)}), "
              "motion_quality.csv" + (", motion_adjustments.txt" if b["adjustments"] else ""))
    return res
