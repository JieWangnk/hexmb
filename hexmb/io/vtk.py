"""VTK XML writers with no VTK library: ``.vtu`` unstructured grids of
hexahedra and general polyhedra, ``.vtp`` boundary patches, ``.pvd`` time
series, and an Abaqus/CalculiX ``.inp`` for all-hex meshes.

``case_to_vtk`` converts a case hexmb wrote (``constant/polyMesh`` plus the
``wantedDisplacement`` time folders of a moving mesh) into one ``.vtu`` per
phase with the displacement as point data, one ``.vtp`` per boundary patch, and
a ``.pvd`` that ParaView opens as an animation.  Hexahedra keep VTK's hexahedron
type (12); the split-hex transition cells are VTK polyhedra (42).
"""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

__all__ = ["write_vtu", "write_vtp", "write_pvd", "write_abaqus_inp", "case_to_vtk",
           "abaqus_from_case", "cells_from_polymesh", "hex_corners"]

VTK_POLYGON, VTK_HEXAHEDRON, VTK_POLYHEDRON = 7, 12, 42


# --------------------------------------------------------------------------- #
#  XML helpers                                                                 #
# --------------------------------------------------------------------------- #
_VTK_TYPES = {np.dtype("float64"): "Float64", np.dtype("float32"): "Float32",
              np.dtype("int64"): "Int64", np.dtype("int32"): "Int32", np.dtype("uint8"): "UInt8"}


def _data_array(name, arr, binary, ncomp=None):
    arr = np.ascontiguousarray(arr)
    if arr.dtype.kind in "iu" and arr.dtype != np.int32 and arr.dtype != np.int64:
        arr = arr.astype(np.int64)
    if arr.dtype.kind == "f" and arr.dtype != np.float32:
        arr = arr.astype(np.float64)
    ncomp = ncomp or (arr.shape[1] if arr.ndim == 2 else 1)
    attrs = f'type="{_VTK_TYPES[arr.dtype]}" Name="{name}" NumberOfComponents="{ncomp}"'
    if binary:
        raw = arr.tobytes()
        payload = base64.b64encode(np.array([len(raw)], dtype=np.uint64).tobytes() + raw).decode()
        return f'<DataArray {attrs} format="binary">{payload}</DataArray>\n'
    flat = arr.ravel()
    txt = " ".join(f"{v:.9g}" for v in flat) if arr.dtype.kind == "f" else " ".join(map(str, flat))
    return f'<DataArray {attrs} format="ascii">{txt}</DataArray>\n'


def _header():
    return '<?xml version="1.0"?>\n<VTKFile type="{kind}" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n'


def _fields(block, data, binary):
    if not data:
        return ""
    out = f"<{block}>\n"
    for name, arr in data.items():
        out += _data_array(name, np.asarray(arr), binary)
    return out + f"</{block}>\n"


# --------------------------------------------------------------------------- #
#  cell shapes                                                                 #
# --------------------------------------------------------------------------- #
def cells_from_polymesh(faces, owner, neighbour, n_cells=None):
    """Per-cell lists of face ids from an OpenFOAM face table."""
    owner = np.asarray(owner, int); neighbour = np.asarray(neighbour, int)
    if n_cells is None:
        n_cells = int(max(owner.max(), neighbour.max() if len(neighbour) else -1)) + 1
    cells = [[] for _ in range(n_cells)]
    for fi, c in enumerate(owner):
        cells[c].append(fi)
    for fi, c in enumerate(neighbour):
        cells[c].append(fi)
    return cells


def hex_corners(cell_faces, points):
    """The 8 corner point ids of a cell made of 6 quads, in VTK / OpenFOAM
    hexahedron order (bottom quad, then the top quad above it, the bottom quad
    turning so that its right-hand normal points into the cell).  Returns
    ``None`` when the cell is not such a hex."""
    if len(cell_faces) != 6 or any(len(f) != 4 for f in cell_faces):
        return None
    bottom = list(cell_faces[0])
    sb = set(bottom)
    top_face = next((f for f in cell_faces[1:] if not (set(f) & sb)), None)
    if top_face is None:
        return None
    st = set(top_face)
    sides = [f for f in cell_faces[1:] if f is not top_face]
    top = []
    for v in bottom:
        partner = None
        for f in sides:
            if v in f:
                k = list(f).index(v)
                for nb in (f[(k + 1) % 4], f[(k - 1) % 4]):
                    if nb in st:
                        partner = nb
                        break
            if partner is not None:
                break
        if partner is None:
            return None
        top.append(partner)
    P = np.asarray(points, float)
    b = P[bottom]
    nrm = np.cross(b[1] - b[0], b[2] - b[0]) + np.cross(b[2] - b[0], b[3] - b[0])
    if nrm @ (P[top].mean(0) - b.mean(0)) < 0:          # right-hand normal must point to the top
        bottom = [bottom[0], bottom[3], bottom[2], bottom[1]]
        top = [top[0], top[3], top[2], top[1]]
    return bottom + top


# --------------------------------------------------------------------------- #
#  writers                                                                     #
# --------------------------------------------------------------------------- #
def write_vtu(path, points, cells, point_data=None, cell_data=None, binary=True):
    """Write an unstructured grid.

    ``cells``: a list of cells, each a list of faces (point-id lists).  A cell of
    six quads is written as a VTK hexahedron, any other as a VTK polyhedron.
    An ``(n, 8)`` integer array of hex corner ids (VTK order) is accepted too.
    ``point_data`` / ``cell_data``: ``{name: array}``.  Returns the path.
    """
    path = Path(path)
    P = np.asarray(points, float)
    if isinstance(cells, np.ndarray) and cells.ndim == 2 and cells.shape[1] == 8:
        conn = [list(map(int, h)) for h in cells]
        types = np.full(len(cells), VTK_HEXAHEDRON, np.uint8)
        faces_arr, faceoff = np.zeros(0, np.int64), np.full(len(cells), -1, np.int64)
    else:
        conn, types, faces_arr, faceoff = [], [], [], []
        fpos = 0
        for c in cells:
            hc = hex_corners(c, P)
            if hc is not None:
                conn.append(hc); types.append(VTK_HEXAHEDRON); faceoff.append(-1)
                continue
            ids = sorted({int(v) for f in c for v in f})
            conn.append(ids); types.append(VTK_POLYHEDRON)
            blk = [len(c)]
            for f in c:
                blk.append(len(f)); blk.extend(int(v) for v in f)
            faces_arr.extend(blk); fpos += len(blk); faceoff.append(fpos)
        types = np.asarray(types, np.uint8)
        faces_arr = np.asarray(faces_arr, np.int64); faceoff = np.asarray(faceoff, np.int64)
    offsets = np.cumsum([len(c) for c in conn]).astype(np.int64)
    connectivity = np.asarray([v for c in conn for v in c], np.int64)

    out = _header().format(kind="UnstructuredGrid")
    out += f'<UnstructuredGrid>\n<Piece NumberOfPoints="{len(P)}" NumberOfCells="{len(conn)}">\n'
    out += "<Points>\n" + _data_array("Points", P, binary, 3) + "</Points>\n"
    out += "<Cells>\n" + _data_array("connectivity", connectivity, binary) + _data_array("offsets", offsets, binary)
    out += _data_array("types", types, binary)
    if (types == VTK_POLYHEDRON).any():
        out += _data_array("faces", faces_arr, binary) + _data_array("faceoffsets", faceoff, binary)
    out += "</Cells>\n"
    out += _fields("PointData", point_data, binary) + _fields("CellData", cell_data, binary)
    out += "</Piece>\n</UnstructuredGrid>\n</VTKFile>\n"
    path.write_text(out)
    return path


def write_vtp(path, points, polys, cell_data=None, point_data=None, binary=True):
    """Write polygons (a list of point-id lists) as PolyData.  Returns the path."""
    path = Path(path)
    P = np.asarray(points, float)
    connectivity = np.asarray([v for f in polys for v in f], np.int64)
    offsets = np.cumsum([len(f) for f in polys]).astype(np.int64)
    out = _header().format(kind="PolyData")
    out += (f'<PolyData>\n<Piece NumberOfPoints="{len(P)}" NumberOfVerts="0" NumberOfLines="0" '
            f'NumberOfStrips="0" NumberOfPolys="{len(polys)}">\n')
    out += "<Points>\n" + _data_array("Points", P, binary, 3) + "</Points>\n"
    out += "<Polys>\n" + _data_array("connectivity", connectivity, binary) + _data_array("offsets", offsets, binary) + "</Polys>\n"
    out += _fields("PointData", point_data, binary) + _fields("CellData", cell_data, binary)
    out += "</Piece>\n</PolyData>\n</VTKFile>\n"
    path.write_text(out)
    return path


def write_pvd(path, entries):
    """``entries``: ``[(time, file_path_relative_to_the_pvd), ...]``.  Returns the path."""
    path = Path(path)
    out = '<?xml version="1.0"?>\n<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">\n<Collection>\n'
    for t, f in entries:
        out += f'<DataSet timestep="{t:.9g}" group="" part="0" file="{f}"/>\n'
    out += "</Collection>\n</VTKFile>\n"
    path.write_text(out)
    return path


def write_abaqus_inp(path, points, hexes, node_sets=None, element_type="C3D8", scale=1.0, part="LV"):
    """All-hex mesh -> Abaqus / CalculiX input deck: ``*NODE`` (1-based, coordinates
    times ``scale``), ``*ELEMENT, TYPE=C3D8`` in the same corner order as VTK and
    OpenFOAM, and one ``*NSET`` per entry of ``node_sets`` (``{name: point ids}``,
    the boundary patches as node sets).  Split-hex polyhedra cannot be written as
    C3D8; a conformal LV mesh goes to ``.vtu`` instead.  Returns the path."""
    path = Path(path)
    P = np.asarray(points, float) * scale
    H = np.asarray(hexes, int)
    if H.ndim != 2 or H.shape[1] != 8:
        raise ValueError(f"hexes must be (n, 8) corner ids, got {H.shape}")
    lines = [f"*HEADING\nhexmb mesh {part}: {len(H)} hexahedra, {len(P)} nodes", "*NODE"]
    lines += [f"{i + 1}, {p[0]:.9g}, {p[1]:.9g}, {p[2]:.9g}" for i, p in enumerate(P)]
    lines.append(f"*ELEMENT, TYPE={element_type}, ELSET=ALL")
    lines += [f"{i + 1}, " + ", ".join(str(int(v) + 1) for v in h) for i, h in enumerate(H)]
    for name, ids in (node_sets or {}).items():
        ids = sorted({int(v) + 1 for v in ids})
        lines.append(f"*NSET, NSET={name}")
        for k in range(0, len(ids), 16):
            lines.append(", ".join(map(str, ids[k:k + 16])))
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- #
#  a hexmb case -> VTK                                                         #
# --------------------------------------------------------------------------- #
def _read_boundary(pm):
    import re
    txt = (Path(pm) / "boundary").read_text()
    pats = []
    for m in re.finditer(r"(\w+)\s*\{\s*type\s+(\w+);[^}]*?nFaces\s+(\d+);\s*startFace\s+(\d+);", txt):
        pats.append((m.group(1), m.group(2), int(m.group(3)), int(m.group(4))))
    return pats


def case_to_vtk(case_dir, out_dir=None, field="wantedDisplacement", binary=True, scale=1.0, verbose=True):
    """Convert a case hexmb wrote into VTK files ParaView opens directly.

    * ``mesh_<time>.vtu`` for every time folder holding ``field`` (the reference
      ``constant/polyMesh`` moved by that displacement), with ``displacement``
      as point data and ``volume`` as cell data; a case without time folders
      gives one ``mesh.vtu``;
    * ``boundary_<patch>.vtp`` for every patch of the reference mesh, with the
      patch's faces;
    * ``motion.pvd`` listing the ``.vtu`` files by time.

    ``scale`` multiplies the coordinates (the polyMesh is in metres).  Returns
    the output directory.
    """
    from .foam import read_polymesh, read_point_field, case_time_dirs
    from ..services.quality import polymesh_metrics
    case_dir = Path(case_dir)
    out = Path(out_dir) if out_dir else case_dir / "vtk"
    out.mkdir(parents=True, exist_ok=True)
    pm = case_dir / "constant" / "polyMesh"
    pts, faces, owner, neigh = read_polymesh(pm)
    pts = pts * scale
    cells_f = cells_from_polymesh(faces, owner, neigh)
    cells = [[faces[fi] for fi in c] for c in cells_f]
    patches = _read_boundary(pm)
    for name, _typ, n, start in patches:
        write_vtp(out / f"boundary_{name}.vtp", pts, faces[start:start + n],
                  cell_data={"face": np.arange(start, start + n)}, binary=binary)
    tdirs = case_time_dirs(case_dir, field)
    entries = []
    if not tdirs:
        vol = polymesh_metrics(pts, faces, owner, neigh, n_cells=len(cells))["volume"]
        write_vtu(out / "mesh.vtu", pts, cells, cell_data={"volume": vol}, binary=binary)
        entries.append((0.0, "mesh.vtu"))
    for d in tdirs:
        disp = read_point_field(d / field) * scale
        P = pts + disp
        vol = polymesh_metrics(P, faces, owner, neigh, n_cells=len(cells))["volume"]
        name = f"mesh_{d.name}.vtu"
        write_vtu(out / name, P, cells, point_data={"displacement": disp}, cell_data={"volume": vol}, binary=binary)
        entries.append((float(d.name), name))
    write_pvd(out / "motion.pvd", entries)
    if verbose:
        print(f"  wrote {out}: {len(entries)} .vtu ({len(cells)} cells), {len(patches)} patch .vtp, motion.pvd")
    return out


def abaqus_from_case(case_dir, path, scale=1.0, element_type="C3D8"):
    """The ``constant/polyMesh`` of an all-hex case -> Abaqus/CalculiX ``.inp``,
    with one node set per boundary patch.  Raises for a mesh with any cell that
    is not six quads (the conformal LV's split-hex cells): use ``case_to_vtk``."""
    from .foam import read_polymesh
    case_dir = Path(case_dir)
    pm = case_dir / "constant" / "polyMesh"
    pts, faces, owner, neigh = read_polymesh(pm)
    cells = [[faces[fi] for fi in c] for c in cells_from_polymesh(faces, owner, neigh)]
    hexes = []
    for ci, c in enumerate(cells):
        hc = hex_corners(c, pts)
        if hc is None:
            raise ValueError(f"cell {ci} has {len(c)} faces: not a hexahedron; this mesh cannot be written as "
                             f"{element_type} elements (hexmb export --vtk writes it as VTK polyhedra)")
        hexes.append(hc)
    sets = {}
    for name, _typ, n, start in _read_boundary(pm):
        sets[name] = sorted({v for f in faces[start:start + n] for v in f})
    return write_abaqus_inp(path, pts, np.asarray(hexes, int), node_sets=sets, element_type=element_type,
                            scale=scale, part=case_dir.name)
