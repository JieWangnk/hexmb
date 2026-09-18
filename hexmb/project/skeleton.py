"""Voxel-skeleton centreline backend (pure Python: skimage + scipy + networkx).

Handles arches and branches that the marching centreline cannot. Pipeline:
  1. make the lumen watertight (wall + cap STLs), voxelize the interior,
  2. 3D skeletonize -> medial-axis voxels,
  3. build a 26-connected graph,
  4. anchor endpoints to the inlet/outlet caps and take the graph shortest path
     inlet -> outlet (shortest path avoids spurs), then smooth.

Needs the caps (inlet + outlets) to anchor and to close the surface. Without
caps it falls back to the longest geodesic path.
"""
from __future__ import annotations

import numpy as np

from .stl_tube import _load, _smooth_curve, _resample


def _watertight(wall_path, cap_paths):
    import trimesh

    parts = [_load(wall_path)] + [_load(c) for c in cap_paths]
    m = trimesh.util.concatenate(parts)
    m.merge_vertices()
    return m


def _skeleton_graph(solid_mesh, pitch):
    import networkx as nx
    from skimage.morphology import skeletonize

    vg = solid_mesh.voxelized(pitch).fill()
    M = np.asarray(vg.matrix)
    skel = skeletonize(M)
    vox = np.argwhere(skel)
    index = {tuple(v): i for i, v in enumerate(vox)}
    G = nx.Graph()
    G.add_nodes_from(range(len(vox)))
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            if (dx, dy, dz) != (0, 0, 0)]
    for v in vox:
        i = index[tuple(v)]
        for o in offs:
            j = index.get((v[0] + o[0], v[1] + o[1], v[2] + o[2]))
            if j is not None and j > i:
                G.add_edge(i, j, w=float(np.linalg.norm(o)))
    world = vg.indices_to_points(vox.astype(float))
    return G, world


def _nearest_node(world, target):
    return int(np.argmin(np.linalg.norm(world - target, axis=1)))


def skeleton_centrelines(wall_path, inlet_path=None, outlet_paths=(), pitch=1.2,
                         smooth_iters=12):
    """Return a list of centreline paths (world mm), main path first.

    With inlet + outlets: one path per outlet = graph shortest path inlet->outlet
    (the main aorta path over the arch is the longest of these). Without caps:
    the single longest geodesic path."""
    import networkx as nx

    caps = ([inlet_path] if inlet_path else []) + list(outlet_paths)
    solid = _watertight(wall_path, caps) if caps else _load(wall_path)
    G, world = _skeleton_graph(solid, pitch)
    comp = max(nx.connected_components(G), key=len)
    Gc = G.subgraph(comp).copy()

    paths = []
    if inlet_path and outlet_paths:
        inlet_c = _load(inlet_path).vertices.mean(0)
        src = _nearest_node(world, inlet_c)
        if src not in Gc:
            src = _nearest_node(world[list(Gc.nodes)], inlet_c)
            src = list(Gc.nodes)[src]
        for op in outlet_paths:
            oc = _load(op).vertices.mean(0)
            dst = _nearest_node(world, oc)
            if dst not in Gc:
                continue
            try:
                nodes = nx.dijkstra_path(Gc, src, dst, weight="w")
            except nx.NetworkXNoPath:
                continue
            w = world[nodes]
            paths.append((np.linalg.norm(np.diff(w, axis=0), axis=1).sum(), w))
        paths.sort(key=lambda t: -t[0])
        out = [_smooth_curve(w, smooth_iters) for _, w in paths]
    else:
        def far(s):
            d = nx.single_source_dijkstra_path_length(Gc, s, weight="w")
            n = max(d, key=d.get)
            return n, d[n]
        a, _ = far(next(iter(comp)))
        b, _ = far(a)
        nodes = nx.dijkstra_path(Gc, a, b, weight="w")
        out = [_smooth_curve(world[nodes], smooth_iters)]
    return out


def skeleton_segments(wall_path, inlet_path, outlet_paths, pitch=1.2, min_len=8.0,
                      smooth_iters=8, tol=6.0):
    """Decompose the vessel tree into single-lumen SEGMENTS between junctions and
    terminals (inlet/outlets). Returns a list of dicts:
        {"path": (n,3) world mm, "ends": (type0, type1)}
    where each type is "inlet", "outlet<k>", or "junction". Segments shorter than
    min_len are dropped (spurs)."""
    import networkx as nx

    caps = [inlet_path] + list(outlet_paths)
    solid = _watertight(wall_path, caps)
    G, world = _skeleton_graph(solid, pitch)
    comp = max(nx.connected_components(G), key=len)
    Gc = G.subgraph(comp).copy()

    # terminal nodes = nearest skeleton node to each cap centroid
    term = {}
    inlet_c = _load(inlet_path).vertices.mean(0)
    term[_nearest_node(world, inlet_c)] = "inlet"
    for k, op in enumerate(outlet_paths, 1):
        term[_nearest_node(world, _load(op).vertices.mean(0))] = f"outlet{k}"

    # used subtree = union of shortest paths inlet -> each outlet
    src = _nearest_node(world, inlet_c)
    used = nx.Graph()
    for dst in list(term):
        if term[dst] == "inlet" or dst not in Gc:
            continue
        try:
            nodes = nx.dijkstra_path(Gc, src, dst, weight="w")
        except nx.NetworkXNoPath:
            continue
        nx.add_path(used, nodes)

    def kind(n):
        if n in term:
            return term[n]
        return "junction"

    # walk degree-2 chains between nodes of degree != 2
    branch = [n for n in used if used.degree(n) != 2]
    segs = []
    seen = set()
    for b in branch:
        for nb in used.neighbors(b):
            e = tuple(sorted((b, nb)))
            if e in seen:
                continue
            path = [b, nb]
            seen.add(e)
            prev, cur = b, nb
            while used.degree(cur) == 2:
                nxt = [x for x in used.neighbors(cur) if x != prev][0]
                seen.add(tuple(sorted((cur, nxt))))
                path.append(nxt)
                prev, cur = cur, nxt
            w = _smooth_curve(world[path], smooth_iters)
            L = np.linalg.norm(np.diff(w, axis=0), axis=1).sum()
            if L >= min_len:
                segs.append({"path": w, "ends": (kind(path[0]), kind(path[-1])),
                             "nodes": (int(path[0]), int(path[-1])), "len": L})
    return segs


def skeleton_main_centreline(wall_path, inlet_path=None, outlet_paths=(), pitch=1.2,
                             ns=60, smooth_iters=12):
    """The main (longest) centreline path, resampled to ns points."""
    paths = skeleton_centrelines(wall_path, inlet_path, outlet_paths, pitch, smooth_iters)
    if not paths:
        raise ValueError("skeleton produced no path")
    return _resample(paths[0], ns)
