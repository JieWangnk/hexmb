"""Turn a single-lumen tubular STL into O-grid rings.

VMTK-free pipeline using trimesh:
  1. centreline: PCA principal axis -> slice-centroid march -> perpendicular
     refinement -> smoothing.
  2. rings: at each centreline station, cast NT rays outward in the
     rotation-minimizing plane and hit the wall. This gives NT boundary points
     in consistent angular order AND directly tests the star-shaped condition
     (a ray that misses = the section is not star-shaped there / a branch).

Requires trimesh. Only handles a single lumen; a station whose section is not
star-shaped about the centreline (branch, strong concavity) is reported.
"""
from __future__ import annotations

import numpy as np

from ..assembly.ogrid import build_ogrid_from_rings, graded_radial
from ..assembly.sweep import parallel_transport_frames


def _load(path):
    import trimesh

    m = trimesh.load(path, process=True)
    if hasattr(m, "to_geometry"):
        m = m.to_geometry()
    elif isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    return m


def _smooth_curve(cl, iters=5):
    cl = cl.copy()
    for _ in range(iters):
        c = cl.copy()
        cl[1:-1] = 0.5 * c[1:-1] + 0.25 * (c[:-2] + c[2:])
    return cl


def extract_centreline(mesh, ns=40, trim=0.03, refine=True):
    """Return an (ns,3) centreline for a tubular mesh."""
    V = mesh.vertices
    c0 = V.mean(0)
    _, _, vt = np.linalg.svd(V - c0, full_matrices=False)
    axis = vt[0]
    proj = (V - c0) @ axis
    lo, hi = proj.min(), proj.max()
    hs = np.linspace(lo + trim * (hi - lo), hi - trim * (hi - lo), ns)
    pts = []
    for h in hs:
        sec = mesh.section(plane_origin=c0 + h * axis, plane_normal=axis)
        if sec is None or len(sec.discrete) == 0:
            continue
        loop = max(sec.discrete, key=lambda L: len(L))
        pts.append(loop.mean(0))
    if len(pts) < 4:
        raise ValueError("could not trace a centreline (too few valid sections)")
    cl = _smooth_curve(np.array(pts))
    if not refine:
        return cl
    # one perpendicular-reslice pass (fixes oblique slices through a bend)
    _, e1s, e2s = parallel_transport_frames(cl)
    t = np.gradient(cl, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    pts2 = []
    for i in range(len(cl)):
        sec = mesh.section(plane_origin=cl[i], plane_normal=t[i])
        if sec is None or len(sec.discrete) == 0:
            pts2.append(cl[i]); continue
        loop = min(sec.discrete, key=lambda L: np.linalg.norm(L.mean(0) - cl[i]))
        pts2.append(loop.mean(0))
    return _smooth_curve(np.array(pts2))


def inlet_seed(inlet_path, wall_mesh):
    """Return (point, direction-into-vessel) from an inlet cap STL."""
    cap = _load(inlet_path)
    p = cap.vertices.mean(0)
    n = cap.face_normals.mean(0)
    n /= np.linalg.norm(n) + 1e-12
    # orient into the vessel (toward the wall centroid)
    if n @ (wall_mesh.vertices.mean(0) - p) < 0:
        n = -n
    return p, n


def _recentre(mesh, p, d):
    """Section perpendicular to d at p; return the centroid of the lumen loop
    nearest p (recentres p onto the vessel axis), the local radius, or None."""
    sec = mesh.section(plane_origin=p, plane_normal=d)
    if sec is None or len(sec.discrete) == 0:
        return None, None
    loop = min(sec.discrete, key=lambda L: np.linalg.norm(L.mean(0) - p))
    c = loop.mean(0)
    r = np.median(np.linalg.norm(loop - c, axis=1))
    return c, r


def march_centreline(mesh, seed_point, seed_dir, step=1.5, max_steps=600):
    """Follow the lumen from an inlet seed. Each step: recentre onto the lumen
    axis (section centroid), advance, and reject a jump to another limb (a loop
    whose centroid is more than ~2 local radii away). Handles moderate bends;
    very tight arches still need a Voronoi (VMTK-quality) centreline."""
    p = np.asarray(seed_point, float)
    d = np.asarray(seed_dir, float)
    d /= np.linalg.norm(d) + 1e-12
    c0, r0 = _recentre(mesh, p, d)
    if c0 is not None:
        p = c0
    pts = [p.copy()]
    r_prev = r0 or step
    for _ in range(max_steps):
        p_try = p + step * d
        c, r = _recentre(mesh, p_try, d)
        if c is None:
            break
        if np.linalg.norm(c - p_try) > 2.0 * r_prev:   # jumped to another limb
            c = p_try                                   # keep straight this step
            r = r_prev
        nd = c - p
        if np.linalg.norm(nd) < 1e-6:
            break
        nd /= np.linalg.norm(nd)
        d = 0.3 * d + 0.7 * nd                          # follow the lumen quickly
        d /= np.linalg.norm(d) + 1e-12
        pts.append(c.copy())
        p = c
        r_prev = r
        if not mesh.contains([p + 0.5 * step * d])[0]:  # about to exit the lumen
            break
    return _smooth_curve(np.array(pts))


def _resample(cl, ns):
    """Resample a polyline to ns points uniformly by arclength."""
    seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    snew = np.linspace(0, s[-1], ns)
    return np.stack([np.interp(snew, s, cl[:, k]) for k in range(3)], axis=1)


def rings_by_raycast(mesh, centreline, NT=32, frames=None, clamp=2.5):
    """Cast NT rays outward per station. Returns (rings (ns,NT,3), miss (ns,)
    count of rays that missed OR were rejected). `clamp` rejects a ray whose hit
    distance exceeds `clamp` x the station's median radius -- these are rays that
    cross into another lumen at a junction (the cause of arch/branch spikes);
    they are replaced by angular interpolation from the good rays."""
    ns = len(centreline)
    if frames is None:
        _, e1s, e2s = parallel_transport_frames(centreline)
    else:
        _, e1s, e2s = frames
    ang = 2 * np.pi * np.arange(NT) / NT
    origins = np.repeat(centreline, NT, axis=0)
    dirs = np.zeros((ns * NT, 3))
    for s in range(ns):
        dirs[s * NT:(s + 1) * NT] = (
            np.outer(np.cos(ang), e1s[s]) + np.outer(np.sin(ang), e2s[s])
        )
    locs, ray_idx, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=True)
    rings = np.full((ns * NT, 3), np.nan)
    dist = np.full(ns * NT, np.inf)
    d = np.linalg.norm(locs - origins[ray_idx], axis=1)
    for k, ri in enumerate(ray_idx):
        if d[k] < dist[ri] and d[k] > 1e-6:  # nearest wall hit outward
            dist[ri] = d[k]; rings[ri] = locs[k]
    rings = rings.reshape(ns, NT, 3)
    dist = dist.reshape(ns, NT)
    # reject cross-lumen outliers: hit distance >> the station's median radius
    if clamp:
        for s in range(ns):
            good = np.isfinite(dist[s])
            if good.sum() >= 3:
                med = np.median(dist[s, good])
                out = good & (dist[s] > clamp * med)
                rings[s, out] = np.nan
    miss = np.isnan(rings[:, :, 0]).sum(axis=1)
    # fill missed/rejected rays by angular interpolation within the station
    for s in range(ns):
        bad = np.isnan(rings[s, :, 0])
        if bad.any() and not bad.all():
            good = ~bad
            for c in range(3):
                rings[s, bad, c] = np.interp(
                    np.where(bad)[0], np.where(good)[0], rings[s, good, c], period=NT
                )
    return rings, miss


def ogrid_from_stl(path, ns=40, NT=32, NR=6, core_frac=0.42, wall_ratio=1.0,
                   inlet_stl=None, step=2.0, centreline=None, return_diag=False):
    """Full STL -> O-grid MultiBlockMesh (not assembled, no patches).
    Centreline source, in priority order: an explicit `centreline` array (e.g.
    from the skeleton backend); else marching from `inlet_stl`; else a PCA-axis
    centreline (gentle single tubes only)."""
    mesh = _load(path)
    if centreline is not None:
        cl = _resample(np.asarray(centreline, float), ns)
    elif inlet_stl is not None:
        sp, sd = inlet_seed(inlet_stl, mesh)
        cl = _resample(march_centreline(mesh, sp, sd, step=step), ns)
    else:
        cl = extract_centreline(mesh, ns=ns)
    frames = parallel_transport_frames(cl)
    rings, miss = rings_by_raycast(mesh, cl, NT=NT, frames=frames)
    # branch detector: a perpendicular section with >1 substantial loop = a junction
    t = np.gradient(cl, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    branch = 0
    for i in range(len(cl)):
        sec = mesh.section(plane_origin=cl[i], plane_normal=t[i])
        if sec is not None:
            big = [L for L in sec.discrete if len(L) > 3]
            if len(big) > 1:
                branch += 1
    diag = {"centreline": cl, "rings": rings, "miss": miss,
            "n_bad_stations": int((miss > 0).sum()), "max_miss": int(miss.max()),
            "branch_stations": branch, "ns": ns}
    _, e1s, e2s = frames
    rd = graded_radial(NR, wall_ratio)
    m = build_ogrid_from_rings(rings, e1s, e2s, NR=NR, core_frac=core_frac, radial_dist=rd)
    if return_diag:
        return m, diag
    return m
