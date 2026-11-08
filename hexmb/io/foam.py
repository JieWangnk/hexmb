"""Minimal readers for what hexmb itself writes: an ASCII ``constant/polyMesh``,
a ``pointVectorField`` time folder and a ``controlDict`` entry.  No OpenFOAM
installation needed (used by ``hexmb verify`` and ``hexmb render``)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

__all__ = ["read_polymesh", "read_point_field", "controldict_value", "case_time_dirs"]


def _body(path):
    txt = Path(path).read_text()
    i = txt.index("(", txt.index("\n", txt.index("}")))      # first list after the FoamFile header
    return txt[i + 1: txt.rindex(")")]


def read_polymesh(pm):
    """``(points, faces, owner, neighbour)`` from ``constant/polyMesh`` (ASCII).
    Points come back in the file's units (metres as written by hexmb)."""
    pm = Path(pm)
    pts = np.array([[float(v) for v in ln.strip("() ").split()]
                    for ln in _body(pm / "points").strip().splitlines() if ln.strip()])
    faces = [[int(v) for v in ln[ln.index("(") + 1: ln.rindex(")")].split()]
             for ln in _body(pm / "faces").strip().splitlines() if ln.strip()]
    owner = np.array([int(v) for v in _body(pm / "owner").split()])
    neigh = np.array([int(v) for v in _body(pm / "neighbour").split()])
    return pts, faces, owner, neigh


def read_point_field(path):
    """The ``internalField nonuniform List<vector>`` of a point field file -> (n, 3)."""
    txt = Path(path).read_text()
    i = txt.index("(\n", txt.index("internalField"))
    j = txt.index("\n);", i)
    return np.array([[float(v) for v in ln.strip("() ").split()]
                     for ln in txt[i + 2:j].splitlines() if ln.strip()])


def controldict_value(case_dir, key):
    """A numeric ``key value;`` entry of ``system/controlDict`` as float."""
    txt = (Path(case_dir) / "system" / "controlDict").read_text()
    m = re.search(rf"^\s*{key}\s+([0-9.eE+-]+)\s*;", txt, re.M)
    if not m:
        raise KeyError(f"{key} not found in {case_dir}/system/controlDict")
    return float(m.group(1))


def case_time_dirs(case_dir, field="wantedDisplacement"):
    """Time folders of a case holding ``field``, sorted by time value."""
    case_dir = Path(case_dir)
    dirs = [d for d in case_dir.iterdir()
            if d.is_dir() and re.fullmatch(r"[0-9.eE+-]+", d.name) and (d / field).exists()]
    return sorted(dirs, key=lambda d: float(d.name))
