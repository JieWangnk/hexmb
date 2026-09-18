"""Write an all-hex mesh to an OpenFOAM `constant/polyMesh` directory.

The mesh is given as `points` (nPoint,3) and `hexes` (nHex,8) in the fixed
HEX_OFFSETS vertex order, plus `patch_of` mapping each boundary face
(tuple(sorted(4 global ids))) to a patch name. Every face is oriented
geometrically outward from its cell centroid, so correctness does not depend on
a block's index handedness.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.topology import HEX_FACES

_HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  12
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       {cls};
{note}    location    "constant/polyMesh";
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def _header(cls: str, obj: str, note: str | None = None) -> str:
    note_line = f'    note        "{note}";\n' if note else ""
    return _HEADER.format(cls=cls, obj=obj, note=note_line)


def _face_normal(pts: np.ndarray) -> np.ndarray:
    """Newell-ish area vector of one polygon (n,3)."""
    pbar = pts.mean(axis=0)
    q = pts - pbar
    return 0.5 * np.cross(q, np.roll(q, -1, axis=0)).sum(axis=0)


def build_faces(points: np.ndarray, hexes: np.ndarray):
    """Return (faces_list, owner, neighbour, boundary_keys) where faces are
    oriented owner->neighbour (internal) or outward (boundary), and
    boundary_keys[i] is tuple(sorted(ids)) for boundary face i (aligned with the
    boundary portion of faces_list)."""
    hexes = np.asarray(hexes)
    nH = len(hexes)
    cell_cent = points[hexes].mean(axis=1)  # (nH,3)

    # collect every candidate face oriented outward from its own cell
    faces_by_key: dict[tuple, list] = {}
    for fi, loop in enumerate(HEX_FACES):
        vids = hexes[:, loop]  # (nH,4) global ids
        fp = points[vids]  # (nH,4,3)
        pbar = fp.mean(axis=1)
        q = fp - pbar[:, None, :]
        nrm = 0.5 * np.cross(q, np.roll(q, -1, axis=1)).sum(axis=1)  # (nH,3)
        outward = np.einsum("ij,ij->i", nrm, pbar - cell_cent) >= 0
        for c in range(nH):
            loopids = vids[c] if outward[c] else vids[c][::-1]
            key = tuple(sorted(int(x) for x in loopids))
            faces_by_key.setdefault(key, []).append((int(c), tuple(int(x) for x in loopids)))

    internal = []  # (owner, neighbour, loop)
    boundary = []  # (owner, loop, key)
    for key, occ in faces_by_key.items():
        if len(occ) == 1:
            c, loop = occ[0]
            boundary.append((c, loop, key))
        elif len(occ) == 2:
            (ca, la), (cb, lb) = occ
            if ca < cb:
                owner, neigh, loop = ca, cb, la  # la is outward from ca -> toward cb
            else:
                owner, neigh, loop = cb, ca, lb
            internal.append((owner, neigh, loop))
        else:
            raise ValueError(f"face {key} shared by {len(occ)} cells (non-manifold)")

    internal.sort(key=lambda t: (t[0], t[1]))
    faces_list = [t[2] for t in internal]
    owner = [t[0] for t in internal]
    neighbour = [t[1] for t in internal]

    boundary_faces = [(t[0], t[1], t[2]) for t in boundary]
    return faces_list, owner, neighbour, boundary_faces


def write_polymesh(case_dir, points_mm, hexes, patch_of, patch_types=None, scale=1e-3):
    """Write points/faces/owner/neighbour/boundary. `patch_of` maps
    tuple(sorted(ids)) -> patch name for every boundary face. `patch_types`
    optionally maps patch name -> OF type (default 'wall' if name contains
    'wall' else 'patch')."""
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    points_mm = np.asarray(points_mm, float)
    hexes = np.asarray(hexes)

    faces_list, owner, neighbour, boundary_faces = build_faces(points_mm, hexes)
    n_internal = len(faces_list)

    # group boundary faces by patch, in first-seen order
    order = []
    grouped: dict[str, list] = {}
    for own, loop, key in boundary_faces:
        if key not in patch_of:
            raise ValueError(f"boundary face {key} has no patch assignment (orphan)")
        pname = patch_of[key]
        if pname not in grouped:
            grouped[pname] = []
            order.append(pname)
        grouped[pname].append((own, loop))

    # append boundary faces patch by patch
    patch_info = []
    start = n_internal
    for pname in order:
        flist = grouped[pname]
        for own, loop in flist:
            faces_list.append(loop)
            owner.append(own)
        patch_info.append((pname, len(flist), start))
        start += len(flist)

    n_faces = len(faces_list)
    n_points = len(points_mm)
    n_cells = len(hexes)
    note = (
        f"nPoints:{n_points}  nCells:{n_cells}  "
        f"nFaces:{n_faces}  nInternalFaces:{n_internal}"
    )

    pm = case_dir
    _write_points(pm / "points", points_mm * scale)
    _write_faces(pm / "faces", faces_list)
    _write_labels(pm / "owner", owner, "owner", note, n_cells)
    _write_labels(pm / "neighbour", neighbour, "neighbour", note, n_cells)
    _write_boundary(pm / "boundary", patch_info, patch_types)
    return {
        "nPoints": n_points,
        "nCells": n_cells,
        "nFaces": n_faces,
        "nInternalFaces": n_internal,
        "patches": patch_info,
    }


def _write_points(path, pts):
    with open(path, "w") as f:
        f.write(_header("vectorField", "points"))
        f.write(f"\n\n{len(pts)}\n(\n")
        for p in pts:
            f.write(f"({p[0]:.10g} {p[1]:.10g} {p[2]:.10g})\n")
        f.write(")\n")


def _write_faces(path, faces):
    with open(path, "w") as f:
        f.write(_header("faceList", "faces"))
        f.write(f"\n\n{len(faces)}\n(\n")
        for loop in faces:
            f.write(f"{len(loop)}({' '.join(str(int(x)) for x in loop)})\n")
        f.write(")\n")


def _write_labels(path, labels, obj, note, n_cells):
    with open(path, "w") as f:
        f.write(_header("labelList", obj, note=note))
        f.write(f"\n\n{len(labels)}\n(\n")
        f.write("\n".join(str(int(x)) for x in labels))
        f.write("\n)\n")


def _write_boundary(path, patch_info, patch_types):
    patch_types = patch_types or {}
    with open(path, "w") as f:
        f.write(_header("polyBoundaryMesh", "boundary"))
        f.write(f"\n\n{len(patch_info)}\n(\n")
        for name, nfaces, start in patch_info:
            ptype = patch_types.get(name, "wall" if "wall" in name.lower() else "patch")
            f.write(f"    {name}\n    {{\n")
            f.write(f"        type            {ptype};\n")
            if ptype == "wall":
                f.write(f"        inGroups        List<word> 1({name});\n")
            f.write(f"        nFaces          {nfaces};\n")
            f.write(f"        startFace       {start};\n")
            f.write("    }\n")
        f.write(")\n")
