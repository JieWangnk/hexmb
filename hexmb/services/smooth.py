"""Interior mesh smoothing on the assembled point/hex mesh.

Laplacian (Jacobi) smoothing built from hex edges, holding boundary nodes fixed.
By default only in-plane edges (within a k-layer) are used, so nodes stay on
their cross-section level -- full-3D smoothing collapses a strongly tapered
cavity (the same failure mode as Laplacian mesh morphing).
"""
from __future__ import annotations

import numpy as np

from ..core.topology import HEX_EDGES, HEX_INPLANE_EDGES


def _adjacency(n_points: int, hexes: np.ndarray, edge_template):
    edges = set()
    for hx in hexes:
        for a, b in edge_template:
            edges.add((min(int(hx[a]), int(hx[b])), max(int(hx[a]), int(hx[b]))))
    E = np.array(sorted(edges))
    rows = np.r_[E[:, 0], E[:, 1]]
    cols = np.r_[E[:, 1], E[:, 0]]
    return rows, cols


def laplacian_smooth(points, hexes, fixed_mask, iters: int = 50, in_plane: bool = True):
    """Return smoothed points; nodes with fixed_mask True are held. When
    `in_plane`, only within-layer edges are used (recommended for LV-like
    tapered cavities)."""
    points = np.array(points, float, copy=True)
    n = len(points)
    template = HEX_INPLANE_EDGES if in_plane else HEX_EDGES
    rows, cols = _adjacency(n, np.asarray(hexes), template)
    deg = np.bincount(rows, minlength=n).astype(float)
    deg[deg == 0] = 1.0
    free = ~fixed_mask
    for _ in range(iters):
        acc = np.zeros_like(points)
        np.add.at(acc, rows, points[cols])
        avg = acc / deg[:, None]
        points[free] = avg[free]
    return points


def _node_cells(n_points, hexes):
    from collections import defaultdict
    nc = defaultdict(list)
    for ci, h in enumerate(hexes):
        for n in h:
            nc[int(n)].append(ci)
    return nc


def optimize_smooth(points, hexes, fixed_mask, iters=10, nsamp=6, focus_below=1e9):
    """Quality-guarded ('smart Laplacian' + line search) smoother. For each free
    node it samples positions from the node toward its Laplacian target and keeps
    the one that MAXIMISES the minimum scaled-Jacobian of the node's incident
    cells. It therefore never inverts a cell and can untangle poor ones -- unlike
    plain Laplacian, which diffuses and can fold cells.

    `focus_below`: only move nodes that touch a cell with min scaled-Jac below
    this (set small, e.g. 0.2, to only repair the bad regions and leave good ones)."""
    from .quality import scaled_jac
    points = np.array(points, float, copy=True)
    hexes = np.asarray(hexes)
    n = len(points)
    rows, cols = _adjacency(n, hexes, HEX_EDGES)
    deg = np.bincount(rows, minlength=n).astype(float); deg[deg == 0] = 1.0
    nc = _node_cells(n, hexes)
    free = np.where(~fixed_mask)[0]
    alphas = np.linspace(0.0, 1.0, nsamp + 1)[1:]
    for _ in range(iters):
        # Laplacian targets for all nodes at once
        acc = np.zeros_like(points)
        np.add.at(acc, rows, points[cols])
        target = acc / deg[:, None]
        for node in free:
            cells = nc.get(node)
            if not cells:
                continue
            hc = hexes[cells]
            cur = points[node].copy()
            best_q = scaled_jac(points, hc).min()
            if best_q >= focus_below:      # region already good enough; skip
                continue
            best_pos = cur
            d = target[node] - cur
            for a in alphas:
                points[node] = cur + a * d
                q = scaled_jac(points, hc).min()
                if q > best_q:
                    best_q = q; best_pos = points[node].copy()
            points[node] = best_pos
    return points


def optimize_smooth_project(points, hexes, free_mask, wall_mask, wall_mesh,
                            iters=10, nsamp=6, focus_below=1e9):
    """Quality-guarded smoother that also lets WALL nodes slide along the STL:
    a wall node's candidate positions are projected back onto `wall_mesh`
    (closest point), so the surface is preserved but boundary-pinned poor cells
    can untangle. `free_mask` = interior nodes that move freely; `wall_mask` =
    wall nodes that move tangentially (projected). Everything else is fixed."""
    import trimesh
    from .quality import scaled_jac
    points = np.array(points, float, copy=True)
    hexes = np.asarray(hexes)
    n = len(points)
    rows, cols = _adjacency(n, hexes, HEX_EDGES)
    deg = np.bincount(rows, minlength=n).astype(float); deg[deg == 0] = 1.0
    nc = _node_cells(n, hexes)
    movable = np.where(free_mask | wall_mask)[0]
    alphas = np.linspace(0.0, 1.0, nsamp + 1)[1:]

    def proj(p):
        cp, _, _ = trimesh.proximity.closest_point(wall_mesh, p[None, :])
        return cp[0]

    for _ in range(iters):
        acc = np.zeros_like(points)
        np.add.at(acc, rows, points[cols])
        target = acc / deg[:, None]
        for node in movable:
            cells = nc.get(node)
            if not cells:
                continue
            hc = hexes[cells]
            cur = points[node].copy()
            best_q = scaled_jac(points, hc).min()
            if best_q >= focus_below:
                continue
            best_pos = cur
            d = target[node] - cur
            is_wall = wall_mask[node]
            for a in alphas:
                cand = cur + a * d
                if is_wall:
                    cand = proj(cand)     # keep on the STL surface
                points[node] = cand
                q = scaled_jac(points, hc).min()
                if q > best_q:
                    best_q = q; best_pos = points[node].copy()
            points[node] = best_pos
    return points
