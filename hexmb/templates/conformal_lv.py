"""Conformal split-hex telescoping left ventricle: one conservative region.

The LV application of :mod:`hexmb.assembly.splithex`.  A coarse apex cap, a
2x-finer body and a coarse valve/base cap, joined by conformal split-hex
transitions, so the whole cavity is ONE region with no interpolated coupling.
This is the mesh route used for the validated moving-ventricle CFD runs: the
AMI-coupled telescoping route (:mod:`hexmb.templates.telescoping_lv`) stalls
the diastolic filling solve because its interpolated interface does not
conserve flux; this one runs the full cycle.

Two calls::

    cfg  = ConformalLVConfig()                       # validated small defaults (~15k cells)
    mesh = build_conformal_lv(surface, axis, e1, e2, cfg)   # PolyMesh (hex + split-hex)
    mesh.write("case")                               # constant/polyMesh with wall/inlet/outlet1

``surface`` is an ``(NL, NT, 3)`` endocardial grid in millimetres with the apex
at index 0 and the base (valve plane) at index ``NL-1``; ``axis`` the apex->base
unit vector; ``e1``/``e2`` the base-plane frame (see :func:`hexmb.io.cap.surface_frame`
or :func:`hexmb.io.surface.endo_grid_from_mesh`).

Resolution is set by the coarse ring (``NT_c``, ``NR_c``): the body is
``2*NT_c`` circumferential and ``2*NR_c-1`` radial.  Keep the coarse ring modest
(24 to 44): it is also the apex ring, which pinches on a contracting ventricle.
Refine the body axially (``body_levels``) for more cells; that is skew-safe
through the whole cycle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..assembly.splithex import PolyMesh, build_telescoping_ogrid

__all__ = ["ConformalLVConfig", "build_conformal_lv", "lv_valve_classifier", "polymesh_quality",
           "bad_cell_hint", "bad_cell_span", "bad_rings", "axial_fractions", "resample_grid_axial",
           "grid_volume"]


@dataclass
class ConformalLVConfig:
    """Resolution, axial zoning and valve-plane patching for the conformal LV."""

    # -- resolution (coarse ring; the body is 2x finer) --------------------- #
    NT_c: int = 24             # coarse circumferential (divisible by 4); body = 2*NT_c
    NR_c: int = 4              # coarse radial layers; body = 2*NR_c - 1
    apex_levels: int = 10      # axial sections in the apex cap
    body_levels: int = 30      # axial sections in the fine body (refine HERE for more cells)
    base_levels: int = 12      # axial sections in the valve cap

    # -- axial split + O-grid shape ---------------------------------------- #
    # Fractions below are of the long axis: the distance along ``axis`` from the first
    # ring's centre (0) to the last ring's centre (1).  The builder re-spaces the
    # surface's rows uniformly along that axis first, so a surface whose rows crowd
    # the apex (a dome sampled at uniform polar angle, as the Cardiac Atlas models
    # are) gets the same layer thickness as one sampled uniformly.
    apex_frac: float = 0.24    # apex|body cut
    base_frac: float = 0.80    # body|base cut
    apex_trim: float = 0.02    # start the apex zone this far up the axis: a surface whose first
    #                            ring is a point would otherwise give coincident points and
    #                            zero-area faces at the tip, and a dome gives flat wedge cells
    #                            there; 2 % is ~2 mm and caps the tip with a disc about as wide
    #                            as the first axial layer is thick (at 1 % the disc was narrower
    #                            than the layer, its rim cells were flat and 12 wall faces of the
    #                            bundled heart had skewness > 4 at every phase; 2 % gives 2.3-2.5
    #                            for 0.08 % less cavity)
    base_cut: float = 1.0      # cap the base at this fraction (<1 drops a noisy flared rim)
    core_frac: float = 0.42    # O-grid core size
    wall_ratio: float = 1.0    # <1 clusters cells at the wall (boundary layer). Keep 1.0 on a
    #                            MOVING ventricle: thin wall cells invert under the morph.
    resample_levels: int = 200 # the surface is resampled to this many levels, uniform along the axis
    smooth: bool = True        # relax the transition skew (radially constrained Laplacian)
    smooth_iters: int = 3

    # -- valve-plane patches ----------------------------------------------- #
    inlet: str = "sectors"     # "sectors": mitral + aortic angular sectors about the cap centre
    #                            "fullcap": the whole base cap is the inlet (diastolic window only)
    #                            "centred": an ellipse of inlet_area_cm2 at the cap centre, rest wall
    inlet_area_cm2: float = 5.5
    mitral_deg: tuple = (150.0, 340.0)   # sectors in degrees about the cap centre, in the e1/e2 frame
    aortic_deg: tuple = (0.0, 125.0)     # (a sector may wrap through 0)
    cap_depth_mm: float = 3.0            # faces within this depth of the top plane are "on the cap"

    @property
    def approx_cells(self) -> int:
        """Exact cell count for these settings (the build is deterministic)."""
        q = self.NT_c // 4
        coarse = q * q + 4 * (self.NR_c - 1) * q
        qf, nrf = 2 * q, 2 * self.NR_c - 1
        fine = qf * qf + 4 * (nrf - 1) * qf
        return coarse * (self.apex_levels - 1 + self.base_levels - 1) + fine * (self.body_levels - 1)


def _in_sector(ang, sec):
    lo, hi = sec
    return (lo <= ang <= hi) if lo <= hi else (ang >= lo or ang <= hi)


def lv_valve_classifier(mesh: PolyMesh, axis, e1, e2, mu, cfg: ConformalLVConfig):
    """Boundary-face classifier: faces on the base (valve) plane become ``inlet``
    (mitral) / ``outlet1`` (aortic) / ``wall`` by the rule in ``cfg.inlet``;
    every other boundary face is ``wall``.

    "On the valve plane" means tagged as a top face of the last axial section
    when the mesh was built (``mesh.top_faces``), so a tilted or curved base ring
    keeps its whole cap.  A mesh without tags falls back to the depth rule
    (within ``cfg.cap_depth_mm`` of the highest point, facing out).

    Sector angles are taken about the cap's OWN centre (the mean of the
    top-plane points).  At end-systole the cap sits several mm off the long
    axis, and angles about the cavity centroid would hand most of the cap to
    one valve.

    The cap geometry (top-plane height, cap centre, the ellipse of the
    ``centred`` rule) is read from ``mesh.points`` by ``classify.prepare()``,
    which :meth:`PolyMesh.topology` calls before classifying, so a mesh whose
    points were replaced after the build (a phase filled from its neighbours)
    is classified on its current points.
    """
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    e1 = np.asarray(e1, float); e2 = np.asarray(e2, float); mu = np.asarray(mu, float)
    mode = cfg.inlet.lower()
    if mode not in ("sectors", "fullcap", "centred"):
        raise ValueError(f"cfg.inlet must be 'sectors', 'fullcap' or 'centred', got {cfg.inlet!r}")
    g = {}                                   # cap geometry, refreshed by prepare()

    def prepare():
        P = mesh.points
        s_all = (P - mu) @ axis
        g["smax"] = float(s_all.max())
        g["tagged"] = bool(mesh.top_faces)
        if g["tagged"]:
            cap = np.array([P[list(k)].mean(0) for k in mesh.top_faces])
        else:
            cap = P[s_all > g["smax"] - cfg.cap_depth_mm]
        g["centre"] = cap.mean(0)
        if mode == "centred":
            uv = np.stack([(cap - g["centre"]) @ e1, (cap - g["centre"]) @ e2], 1)
            w, vec = np.linalg.eigh(np.cov(uv.T))
            q = float(np.sqrt(max(w[1], 1e-12) / max(w[0], 1e-12)))
            g["a"] = float(np.sqrt(cfg.inlet_area_cm2 * 100.0 * q / np.pi)); g["b"] = g["a"] / q
            g["ulong"], g["ushort"] = vec[:, 1], vec[:, 0]

    prepare()

    def classify(fc, nrm, on_top=False):
        if g["tagged"]:
            on_cap = on_top
        else:
            on_cap = float((fc - mu) @ axis) > g["smax"] - cfg.cap_depth_mm and float(nrm @ axis) > 0.4
        if on_cap:
            if mode == "fullcap":
                return "inlet"
            d = fc - g["centre"]
            if mode == "centred":
                u, v = float(d @ e1), float(d @ e2)
                ulong, ushort, a, b = g["ulong"], g["ushort"], g["a"], g["b"]
                rho = ((u * ulong[0] + v * ulong[1]) / a) ** 2 + ((u * ushort[0] + v * ushort[1]) / b) ** 2
                return "inlet" if rho <= 1.0 else "wall"
            ang = float(np.degrees(np.arctan2(d @ e2, d @ e1))) % 360.0
            if _in_sector(ang, cfg.aortic_deg):
                return "outlet1"
            if _in_sector(ang, cfg.mitral_deg):
                return "inlet"
        return "wall"

    classify.uses_top_tag = True
    classify.prepare = prepare
    return classify


def build_conformal_lv(surface, axis, e1, e2, cfg: ConformalLVConfig | None = None, mu=None) -> PolyMesh:
    """Build the conformal split-hex telescoping LV from an endocardial surface.

    Returns a :class:`~hexmb.assembly.splithex.PolyMesh` whose ``write(case_dir)``
    applies the valve classifier automatically.  ``mesh.frame`` holds
    ``(axis, e1, e2, mu)`` for later use (motion tables, patch checks).
    """
    from scipy.interpolate import interp1d

    cfg = cfg or ConformalLVConfig()
    if cfg.NT_c % 4:
        raise ValueError(f"NT_c={cfg.NT_c} must be divisible by 4")
    G = np.asarray(surface, float)
    if G.ndim != 3 or G.shape[2] != 3:
        raise ValueError(f"surface must be (NL, NT, 3), got {G.shape}")
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    mu = G.reshape(-1, 3).mean(0) if mu is None else np.asarray(mu, float)
    extent = float(np.ptp((G.reshape(-1, 3) - mu) @ axis))
    if extent < 5.0:
        raise ValueError(f"surface extent along the axis is {extent:.3g}: hexmb works in millimetres "
                         "(a ventricle is 60-120 mm long); scale the surface by 1000 if it is in metres")
    if not (0.0 <= cfg.apex_trim < cfg.apex_frac < cfg.base_frac < cfg.base_cut <= 1.0):
        raise ValueError(f"need 0 <= apex_trim ({cfg.apex_trim}) < apex_frac ({cfg.apex_frac}) < "
                         f"base_frac ({cfg.base_frac}) < base_cut ({cfg.base_cut}) <= 1")

    G1 = resample_grid_axial(G, 0.0, 1.0, cfg.resample_levels, axis=axis)
    interp = interp1d(np.linspace(0, 1, cfg.resample_levels), G1, axis=0)
    mesh = build_telescoping_ogrid(
        lambda f: interp(f), NT_c=cfg.NT_c, NR_c=cfg.NR_c, core_frac=cfg.core_frac,
        wall_ratio=cfg.wall_ratio,
        apex=(cfg.apex_trim, cfg.apex_frac, cfg.apex_levels),
        body=(cfg.apex_frac, cfg.base_frac, cfg.body_levels),
        base=(cfg.base_frac, cfg.base_cut, cfg.base_levels),
        smooth=cfg.smooth, smooth_iters=cfg.smooth_iters)
    mesh._classify = lv_valve_classifier(mesh, axis, e1, e2, mu, cfg)
    mesh.frame = (axis, e1, e2, mu)
    return mesh


def polymesh_quality(mesh: PolyMesh) -> dict:
    """Quality summary of a :class:`PolyMesh`.  The hard gate is
    ``n_nonpositive == 0 and n_zero_area_faces == 0``, where ``n_nonpositive``
    counts cells with any non-positive signed pyramid volume (inverted OR folded
    cells; an unsigned volume misses folds, and so does checkMesh once the
    writer has turned every face outward).  Also the signed cavity volume in mL
    (mm^3 input) and cell counts."""
    vols = mesh.cell_volumes()
    mp = mesh.min_pyramid_per_cell()
    areas = mesh.face_areas()
    return {
        "cells": len(mesh.cells),
        "hex": mesh.n_hex,
        "split_hex": mesh.n_split,
        "points": len(mesh.P),
        "min_volume_mm3": float(vols.min()),
        "min_pyramid_mm3": float(mp.min()),
        "n_nonpositive": int((mp <= 0).sum()),
        "min_face_area_mm2": float(areas.min()),
        "n_zero_area_faces": mesh.n_zero_area_faces(1e-9),
        "volume_mL": float(vols.sum() * 1e-3),
    }


def bad_cell_span(mesh: PolyMesh, axis, mu):
    """``(n_bad, lo, hi)``: how many cells fail the signed-pyramid gate and the
    range of their centres along the long axis as a fraction of the mesh extent
    (0 apex, 1 base).  ``(0, nan, nan)`` when every cell passes."""
    mp = mesh.min_pyramid_per_cell()
    bad = mp <= 0
    if not bad.any():
        return 0, float("nan"), float("nan")
    cc = mesh._cell_centres()
    s = (cc - np.asarray(mu, float)) @ (np.asarray(axis, float) / np.linalg.norm(axis))
    f = (s[bad] - s.min()) / (np.ptp(s) + 1e-30)
    return int(bad.sum()), float(f.min()), float(f.max())


def bad_cell_hint(mesh: PolyMesh, axis, mu) -> str:
    """Where the failing cells sit along the long axis, with the config knob that
    usually fixes it.  Empty string when every cell passes.  (The motion service
    applies these adjustments itself when ``auto_trim`` is on.)"""
    n, lo, hi = bad_cell_span(mesh, axis, mu)
    if n == 0:
        return ""
    hint = ""
    if lo > 0.9:
        hint = "all at the base rim: try base_cut 0.97 (--base-cut 0.97) to drop the folded top ring"
    elif hi < 0.3:
        hint = "all in the apex zone: try apex_trim 0.05 or NT_c 16 (--apex-trim / --NT-c)"
    return f"{n} bad cell(s) at axial fraction {lo:.2f}-{hi:.2f} (0 apex, 1 base); {hint}".rstrip("; ")


def axial_fractions(G, axis):
    """Axial position of every row's ring centre as a fraction of the distance
    from the first ring centre to the farthest one along ``axis``."""
    G = np.asarray(G, float)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    s = (G.mean(1) - G[0].mean(0)) @ axis
    return s / (s.max() + 1e-30)


def bad_rings(G, axis, e1, e2, min_radius=1e-6):
    """Rows of an ``(NL, NT, 3)`` grid whose ring does not wind once about its own
    centre in the plane normal to ``axis``: the polar angle of the points about the
    ring centre must advance in one direction and total one turn.  A ring that
    passes through the long axis (a Cardiac Atlas model with ``lambda <= 0``), a
    ring that overlaps itself, or a ring whose centre lies outside it fails; so
    does a ring collapsed to a point.  Row 0 is skipped (a point apex is normal).
    Such rows cannot be meshed: the builder drops them at a rim or marks the phase."""
    G = np.asarray(G, float)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    e1 = np.asarray(e1, float); e1 = e1 - (e1 @ axis) * axis; e1 /= np.linalg.norm(e1)
    e2 = np.asarray(e2, float); e2 = e2 - (e2 @ axis) * axis; e2 /= np.linalg.norm(e2)
    bad, turn = [], np.zeros(G.shape[0])
    for k in range(1, G.shape[0]):
        R = G[k] - G[k].mean(0)
        if np.linalg.norm(R, axis=1).max() < min_radius:
            bad.append(k)
            continue
        a = np.arctan2(R @ e2, R @ e1)
        d = np.diff(np.append(a, a[0]))
        d = (d + np.pi) % (2 * np.pi) - np.pi            # turning increments in (-pi, pi]
        total = d.sum()                                   # a multiple of 2 pi; one turn is wanted
        if not np.isclose(abs(total), 2 * np.pi, atol=1e-6) or (np.sign(d) != np.sign(total)).any():
            bad.append(k)
        else:
            turn[k] = np.sign(total)
    # every ring must turn the same way as the majority (a reversed ring folds the cells between)
    majority = np.sign(turn.sum()) or 1.0
    bad += [k for k in range(1, G.shape[0]) if turn[k] and turn[k] != majority]
    return sorted(bad)


def resample_grid_axial(G, lo: float = 0.0, hi: float = 1.0, levels: int | None = None, axis=None):
    """Resample an ``(NL, NT, 3)`` grid onto the fraction range ``[lo, hi]`` of its
    length (``lo > 0`` trims a point apex; ``hi < 1`` drops a folded base rim).

    With ``axis`` given, the fraction is measured along that unit vector between
    the first and last ring centres and the new levels sit at uniform axial
    spacing, whatever the spacing of the input rows (a dome sampled at uniform
    polar angle crowds its rows at the tip).  Rows beyond the point where the
    ring centres stop advancing along the axis (a rim folded back on itself) are
    dropped with a warning.  Without ``axis`` the fraction is the row index."""
    import warnings
    from scipy.interpolate import interp1d
    G = np.asarray(G, float)
    NL = G.shape[0]
    levels = levels or NL
    if axis is None:
        return interp1d(np.linspace(0, 1, NL), G, axis=0)(np.linspace(lo, hi, levels))
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    s = (G.mean(1) - G[0].mean(0)) @ axis
    top = NL
    for k in range(1, NL):
        if s[k] <= s[k - 1]:
            top = k
            break
    if top < 3:
        raise ValueError("the ring centres do not advance along the axis from the apex: "
                         "check the apex/base landmarks and the axis direction")
    if top < NL:
        warnings.warn(f"rows {top}-{NL - 1} of {NL} fold back along the axis (a rim folded on itself); "
                      f"they are dropped, keeping {s[top - 1] / s.max():.3f} of the axial extent", stacklevel=2)
        s, G = s[:top], G[:top]
    t = s / s[-1]
    return interp1d(t, G, axis=0)(np.linspace(lo, hi, levels))


def grid_volume(G):
    """Volume (mm^3) enclosed by a ring grid closed by its two end caps: the
    bilinear surface split into triangles plus a fan over each end ring from
    that ring's centre (the apex ring may be a single point).  Divergence
    theorem with one consistent orientation, so the result does not depend on
    which way the rings wind.  This is the volume the mesh of this grid
    converges to, and the volume the base trim removes when it is compared
    between a grid and its trimmed copy."""
    G = np.asarray(G, float)
    if G.ndim != 3 or G.shape[2] != 3 or G.shape[0] < 2:
        raise ValueError(f"grid must be (NL >= 2, NT, 3), got {G.shape}")

    def tri(a, b, c):
        return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0

    tot = 0.0
    for i in range(G.shape[0] - 1):
        a, b = G[i], G[i + 1]
        a2, b2 = np.roll(a, -1, axis=0), np.roll(b, -1, axis=0)
        tot += tri(a, a2, b2) + tri(a, b2, b)
    r0, rN = G[0], G[-1]
    c0, cN = np.repeat(r0.mean(0)[None], len(r0), 0), np.repeat(rN.mean(0)[None], len(rN), 0)
    tot += tri(c0, np.roll(r0, -1, axis=0), r0) + tri(cN, rN, np.roll(rN, -1, axis=0))
    return abs(float(tot))
