"""Telescoping O-grid left ventricle --- one parameterized entry point.

A single morphing O-grid over the whole LV cavity cannot be refined: its apex
ring pinches to nothing at end-systole and its valve-plane cap folds, so the mesh
inverts at any size. The telescoping design splits the cavity into three axial
zones -- a **coarse apex cap**, a **fine body** (where the jet and vortex live),
and a **coarse valve/base cap** -- each an all-hex O-grid, joined by OpenFOAM
non-conformal (AMI-style) couplings. The body then scales past a million moving
cells at constant quality while the caps stay coarse.

This module turns the previously hand-edited build into two calls::

    cfg   = TelescopingLVConfig(body_NT=340, body_NR=24)      # ~1M; defaults are validated
    zones = build_telescoping_lv(surface, axis, e1, e2, cfg)  # {'apex','body','base'}
    counts = assemble_telescoping_lv(zones, case_dir, cfg)    # write + merge + couple

``surface`` is an ``(NL, NT, 3)`` endocardial grid in millimetres, ``axis`` the
long-axis unit vector, ``e1``/``e2`` the base-plane frame.

The per-subject tuning that used to be edited by hand --- the valve-plane
de-spiking, the apex trim, the base cut below the flared tip --- are now named
fields on :class:`TelescopingLVConfig` with the validated defaults, so a new
subject is a config, not a code edit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample

from ..assembly.ogrid import build_lv_ogrid
from ..core.topology import FaceKey
from ..assembly.ami import HybridAssembler

_BLOCKS = ("core", "p0", "p1", "p2", "p3")


@dataclass
class TelescopingLVConfig:
    """Resolution and per-subject tuning for the telescoping LV.  Defaults are the
    configuration validated to ~1M moving cells (worst-cycle scaled-Jacobian > 0
    across the whole cardiac cycle)."""

    # -- resolution --------------------------------------------------------- #
    body_NT: int = 340         # body ring count (2*NR + NT/4 "across"); drives the cell total
    body_NR: int = 24          # body radial layers
    body_levels: int = 64      # body axial levels
    apex_NT: int = 44          # apex cap ring (keep coarse: >~56 pinches at end-systole)
    apex_NR: int = 6
    apex_levels: int = 40
    base_NT: int = 88          # valve cap ring (finer than apex -> smoother inlet)
    base_NR: int = 8
    base_levels: int = 30

    # -- axial split + O-grid shape ---------------------------------------- #
    apex_frac: float = 0.22    # apex|body cut (fraction of the long axis, 0=apex tip)
    base_frac: float = 0.82    # body|base cut
    base_cut: float = 0.98     # cap the base HERE, not at 1.0: the extreme flared tip
    #                            (the last ~2%) is segmentation noise and inverts.
    core_frac: float = 0.42    # O-grid core size (bigger is worse at the caps -- keep 0.42)
    apex_trim: int = 2         # drop the n most-apical near-zero-radius rings, cap flat

    # -- per-subject surface tuning ---------------------------------------- #
    valve_despike: bool = True  # Fourier low-pass the valve-plane rings (remove annulus spikes)
    valve_start: float = 0.45   # taper the de-spike from this fraction up to the valve plane
    valve_modes: int = 8        # circumferential Fourier modes kept at the valve plane
    surf_smooth: int = 4        # light structured surface smoothing per zone (4 clears the
    #                             valve-plane transition skew -> checkMesh Mesh OK; 2 leaves 1 face ~4.05)

    # -- valve orifice sectors (degrees, in the e1/e2 base frame) ---------- #
    mitral_deg: tuple = (150.0, 340.0)
    aortic_deg: tuple = (0.0, 125.0)

    @property
    def approx_cells(self) -> int:
        """Rough total cell count (dominated by the body)."""
        q = self.body_NT // 4
        per = q * q + 4 * (self.body_NR - 1) * q
        return per * (self.body_levels - 1)


# --------------------------------------------------------------------------- #
#  surface helpers                                                            #
# --------------------------------------------------------------------------- #
def _sm(g, it):
    """Light structured smoother (circumferential + axial, ends held)."""
    g = g.copy()
    for _ in range(it):
        up = np.roll(g, -1, 1); dn = np.roll(g, 1, 1)
        lo = np.empty_like(g); hi = np.empty_like(g)
        lo[1:] = g[:-1]; lo[0] = g[0]; hi[:-1] = g[1:]; hi[-1] = g[-1]
        g[1:-1] = ((up + dn + lo + hi) / 4.0)[1:-1]
    return g


def _band(surface, flo, fhi, nlev, NT, smooth, trim=0):
    """Resample the surface between axial fractions [flo, fhi] to nlev x NT."""
    G1 = interp1d(np.linspace(0, 1, surface.shape[0]), surface, axis=0)(np.linspace(flo, fhi, nlev))
    Gg = _sm(resample(G1, NT, axis=1), smooth)
    return Gg[trim:] if (trim and flo == 0.0) else Gg


def valve_despike(Gg, start=0.45, modes=8):
    """Fourier low-pass the top rings (tapered from ``start`` to the valve plane):
    keep the low circumferential modes (the true, gently D-shaped annulus) and drop
    the high ones (the segmentation spikes).  Preserves the annulus shape AND size
    -- unlike a Laplacian smooth, which would shrink the inlet."""
    Gg = Gg.copy()
    NLs = Gg.shape[0]
    for i in range(NLs):
        f = i / (NLs - 1)
        if f <= start:
            continue
        K = int(modes + (1 - ((f - start) / (1 - start))) * 24)   # many modes low, ``modes`` at top
        Fc = np.fft.rfft(Gg[i], axis=0)
        Fc[max(K, modes) + 1:] = 0
        Gg[i] = np.fft.irfft(Fc, n=Gg.shape[1], axis=0)
    return Gg


# --------------------------------------------------------------------------- #
#  zone build                                                                 #
# --------------------------------------------------------------------------- #
def _base_patch_fn(axis, e1, e2, mu, cfg):
    ao, mi = cfg.aortic_deg, cfg.mitral_deg

    def patch(xyz):
        d = xyz - mu
        ang = np.degrees(np.arctan2(d @ e2, d @ e1)) % 360.0
        if ao[0] <= ang <= ao[1]:
            return "outlet1"
        if mi[0] <= ang <= mi[1]:
            return "inlet"
        return "wall"
    return patch


def _build_zone(Gg, NR, kind, axis, e1, e2, mu, cfg):
    m = build_lv_ogrid(Gg, axis, e1, e2, NR=NR, core_frac=cfg.core_frac)
    for k in range(4):
        m.blocks[f"p{k}"].set_face(FaceKey.I1, patch="wall")        # petal outer = endocardial wall
    if kind == "apex":
        for nm in _BLOCKS:
            m.blocks[nm].set_face(FaceKey.K0, patch="wall")          # closed apex cap
            m.blocks[nm].set_face(FaceKey.K1, patch="apexTop")       # interface to body
    elif kind == "body":
        for nm in _BLOCKS:
            m.blocks[nm].set_face(FaceKey.K0, patch="bodyBot")
            m.blocks[nm].set_face(FaceKey.K1, patch="bodyTop")
    else:  # base / valve cap
        for nm in _BLOCKS:
            m.blocks[nm].set_face(FaceKey.K0, patch="baseBot")
        fn = _base_patch_fn(axis, e1, e2, mu, cfg)
        for nm in _BLOCKS:
            m.set_predicate_face(nm, FaceKey.K1, fn)                 # valve plane -> inlet/outlet1/wall
    m.assemble()
    return m


def build_telescoping_lv(surface, axis, e1, e2, cfg: TelescopingLVConfig | None = None, mu=None):
    """Build the three telescoping O-grid zones (apex / body / base) of the LV.

    Returns ``{'apex': mesh, 'body': mesh, 'base': mesh}`` -- each an assembled,
    patch-tagged all-hex :class:`MultiBlockMesh`.  Write them and couple
    ``apexTop<->bodyBot`` and ``bodyTop<->baseBot`` with
    :func:`assemble_telescoping_lv` (or ``HybridAssembler`` directly)."""
    cfg = cfg or TelescopingLVConfig()
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    if mu is None:
        mu = surface.reshape(-1, 3).mean(0)

    apex_G = _band(surface, 0.0, cfg.apex_frac, cfg.apex_levels, cfg.apex_NT, cfg.surf_smooth, trim=cfg.apex_trim)
    body_G = _band(surface, cfg.apex_frac, cfg.base_frac, cfg.body_levels, cfg.body_NT, cfg.surf_smooth)
    base_G = _band(surface, cfg.base_frac, cfg.base_cut, cfg.base_levels, cfg.base_NT, cfg.surf_smooth)
    if cfg.valve_despike:
        base_G = valve_despike(base_G, cfg.valve_start, cfg.valve_modes)

    return {
        "apex": _build_zone(apex_G, cfg.apex_NR, "apex", axis, e1, e2, mu, cfg),
        "body": _build_zone(body_G, cfg.body_NR, "body", axis, e1, e2, mu, cfg),
        "base": _build_zone(base_G, cfg.base_NR, "base", axis, e1, e2, mu, cfg),
    }


def assemble_telescoping_lv(zones, case_dir, cfg: TelescopingLVConfig | None = None, couple=True):
    """Write the three zones under ``case_dir`` and (if ``couple``) merge + AMI-couple
    them into one domain in ``case_dir/apex``.  Returns the coupling counts (or the
    written cell counts if ``couple`` is False)."""
    case_dir = Path(case_dir).resolve()          # absolute: mergeMeshes cd's into the master case
    counts = {}
    for name, m in zones.items():
        c = case_dir / name
        m.write_polymesh(c / "constant" / "polyMesh", scale=1e-3)
        (c / "system").mkdir(parents=True, exist_ok=True)
        (c / "system" / "controlDict").write_text(
            'FoamFile{version 2.0;format ascii;class dictionary;object controlDict;}\n'
            'application checkMesh;startFrom startTime;startTime 0;stopAt endTime;'
            'endTime 1;deltaT 1;writeControl timeStep;writeInterval 1;\n')
        counts[name] = len(m.hexes)
    if not couple:
        return counts
    a = HybridAssembler(case_dir / "apex")
    a.merge([case_dir / "body", case_dir / "base"])
    return a.couple([("apexTop", "bodyBot"), ("bodyTop", "baseBot")])
