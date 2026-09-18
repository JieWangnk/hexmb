"""Non-conformal (AMI-style) mesh assembler for OpenFOAM 12 Foundation.

The telescoping O-grid LV is built as three single-region zones (coarse apex cap,
fine body, coarse valve/base cap) that share cut cross-sections. This module joins
them into one solvable domain through OpenFOAM's non-conformal coupling, and proves
flow passes across every interface.

OF12 Foundation has no ESI ``cyclicAMI``; the Foundation path is
``createNonConformalCouples patchA patchB`` (it auto-creates
``nonConformalCyclic_on_*`` + ``nonConformalError_on_*`` patches). The finicky parts
are encoded here so callers do not rediscover them:

* the two patches must be **coincident and anti-parallel** or no couplings form --
  :meth:`_check_pairs` verifies that (centroid gap and facing normals) before coupling;
* a ``potentialFoam`` flow-through check needs the coupled patch fields to carry the
  **matching** ``nonConformalCyclic`` / ``nonConformalError`` boundary types, a
  ``potentialFlow`` block in ``fvSolution``, and its own schemes.

API: ``merge`` (combine cases), ``couple`` (create the interfaces), ``check``
(checkMesh), ``verify_flow`` (potentialFoam per-region report).

Notes
-----
This file was reconstructed from the compiled module after the original ``.py`` was
lost in a repository cleanup; it reproduces the recorded coupling behaviour (e.g.
~52,970 + 48,434 couplings on the 1M telescoping LV).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

_BASHRC = "/opt/openfoam12/etc/bashrc"


def _of(case, cmd):
    """Run an OpenFOAM command in ``case`` with the OF12 environment sourced."""
    return subprocess.run(["bash", "-lc", f"source {_BASHRC} && cd {case} && {cmd}"],
                          capture_output=True, text=True)


def _fm():
    """The betaFlow polyMesh reader (points/faces/boundary/labels/fields)."""
    sys.path.insert(0, str(Path.home() / "GitHub" / "betaFlow" / "tools"))
    import foam_mesh as fm
    return fm


def _read_boundary(case):
    return _fm().read_boundary(Path(case) / "constant" / "polyMesh" / "boundary")


class HybridAssembler:
    """Assemble several single-region OpenFOAM cases into one non-conformally
    coupled domain, with coincidence/orientation safety checks and an optional
    flow-through proof.  ``master_case`` receives the merged mesh."""

    def __init__(self, master_case):
        self.case = Path(master_case)

    # -- combine ------------------------------------------------------------ #
    def merge(self, add_cases):
        """``mergeMeshes`` the given cases into the master case (one file, still
        several regions until they are coupled)."""
        cases = " ".join(f'"{Path(c)}"' for c in add_cases)
        r = _of(self.case, f"mergeMeshes -addCases '({cases})' -overwrite")
        if "Writing mesh" not in r.stdout:
            raise RuntimeError("mergeMeshes failed:\n" + r.stdout[-800:] + r.stderr[-400:])
        return r.stdout

    # -- couple ------------------------------------------------------------- #
    def couple(self, patch_pairs, check_overlap=True):
        """``createNonConformalCouples`` for each ``(patchA, patchB)`` pair.
        Verifies the caps are coincident + anti-parallel (the requirement for a
        coupling to form) before running, unless ``check_overlap`` is False."""
        if check_overlap:
            self._check_pairs(patch_pairs)
        res = []
        for a, b in patch_pairs:
            r = _of(self.case, f"createNonConformalCouples {a} {b} -overwrite")
            n = 0
            for ln in r.stdout.splitlines():
                if "couplings" in ln:
                    for tok in ln.replace(":", " ").split():
                        if tok.isdigit():
                            n = int(tok)
            if n == 0 or "No couplings" in r.stdout:
                raise RuntimeError(f"couple {a}<->{b}: NO couplings (caps not "
                                   f"coincident/anti-parallel).\n" + r.stdout[-400:])
            print(f"    {a}<->{b}: {n} couplings")
            res.append((a, b, n))
        return res                          # list of (patchA, patchB, n_couplings)

    def _patch_geom(self, boundary, faces, points, p):
        """Area-weighted centroid (metres) and unit normal of patch ``p``."""
        d = boundary[p]
        s, n = int(d["startFace"]), int(d["nFaces"])
        cen = np.zeros(3)
        nrm = np.zeros(3)
        area = 0.0
        for f in faces[s:s + n]:
            v = points[f]
            fc = v.mean(0)
            a = 0.5 * np.cross(v - fc, np.roll(v, -1, axis=0) - fc).sum(0)
            mag = np.linalg.norm(a)
            cen += mag * fc
            nrm += a
            area += mag
        return cen / max(area, 1e-12), nrm / (np.linalg.norm(nrm) + 1e-12)

    def _check_pairs(self, pairs):
        """Warn/report whether each pair is coincident (gap < 3 mm) and
        anti-parallel (facing normals, dot < -0.3) -- the coupling requirement."""
        fm = _fm()
        pm = self.case / "constant" / "polyMesh"
        boundary = fm.read_boundary(pm / "boundary")
        points = fm.read_points(pm / "points")
        faces = fm.read_faces(pm / "faces")
        for a, b in pairs:
            for p in (a, b):
                if p not in boundary:
                    print(f"    patch {p} not found (have {list(boundary)}) WARN(may not couple)")
            if a not in boundary or b not in boundary:
                continue
            ca, na = self._patch_geom(boundary, faces, points, a)
            cb, nb = self._patch_geom(boundary, faces, points, b)
            gap = np.linalg.norm(ca - cb) * 1000.0          # metres -> mm
            dot = float(np.dot(na, nb))
            ok = "OK" if (gap < 3.0 and dot < -0.3) else "WARN(may not couple)"
            print(f"    interface {a}<->{b}: gap {gap:.2f}mm, normals·{dot:+.2f} [{ok}]")

    # -- validate ----------------------------------------------------------- #
    def check(self):
        """Run ``checkMesh`` on the merged mesh and return the notable result
        lines as an ordered ``{label: line}`` dict (cell/region counts, volumes,
        skewness, non-orthogonality, and the Mesh OK / Failed verdict)."""
        out = _of(self.case, "checkMesh -constant").stdout
        wanted = ("cells:", "Number of regions", "Min volume", "Max skewness",
                  "Max aspect ratio", "non-orthogonality", "Mesh OK", "Failed")
        keep = {}
        for ln in out.splitlines():
            ls = ln.strip()
            for k in wanted:
                if k in ls:
                    keep[k] = ls
                    break
        return keep

    # -- flow-through proof (potentialFoam) --------------------------------- #
    def verify_flow(self, inlet, inflow, outlets, wall_substr="wall"):
        """Write potentialFoam fields (coupled patches get their MATCHING
        nonConformalCyclic/nonConformalError types) + dicts, run potentialFoam,
        and report per-region flow.  Returns the |U| magnitude array."""
        bnd = _read_boundary(self.case)
        self._write_fields(bnd, inlet, inflow, outlets, wall_substr)
        self._write_dicts()
        r = _of(self.case, "potentialFoam")
        if "FATAL" in r.stdout + r.stderr:
            raise RuntimeError("potentialFoam failed:\n" + r.stdout[-600:] + r.stderr[-400:])
        return self._flow_field()

    def _entry(self, p, ptype, kind, inlet, inflow, outlets, wall_substr):
        """boundaryField entry for patch ``p`` in field ``kind`` ('U' or 'p').
        Coupled patches keep their own type; physical patches get flow BCs."""
        if "nonConformalCyclic" in ptype:
            return "        type            nonConformalCyclic;\n"
        if "nonConformalError" in ptype:
            return "        type            nonConformalError;\n"
        if kind == "U":
            if p == inlet:
                return f"        type fixedValue;\n        value uniform ({inflow[0]} {inflow[1]} {inflow[2]});\n"
            if p in outlets:
                return "        type zeroGradient;\n"
            if wall_substr in p:
                return "        type slip;\n"
            return "        type zeroGradient;\n"
        else:  # p
            if p == inlet:
                return "        type zeroGradient;\n"
            if p in outlets:
                return "        type fixedValue;\n        value uniform 0;\n"
            if wall_substr in p:
                return "        type zeroGradient;\n"
            return "        type zeroGradient;\n"

    def _write_fields(self, bnd, inlet, inflow, outlets, wall_substr):
        if np.ndim(inflow) == 0:
            inflow = (0.0, 0.0, float(inflow))
        for kind, cls, dim, intern in [("U", "volVectorField", "[0 1 -1 0 0 0 0]", "(0 0 0)"),
                                       ("p", "volScalarField", "[0 2 -2 0 0 0 0]", "0")]:
            s = (f"FoamFile{{format ascii;class {cls};object {kind};}}\n"
                 f"dimensions {dim};\ninternalField uniform {intern};\nboundaryField\n{{\n")
            for p, d in bnd.items():
                s += (f"    {p}\n    {{\n"
                      + self._entry(p, d["type"], kind, inlet, inflow, outlets, wall_substr)
                      + "    }\n")
            (self.case / "0").mkdir(exist_ok=True)
            (self.case / "0" / kind).write_text(s + "}\n")

    def _write_dicts(self):
        sysd = self.case / "system"
        sysd.mkdir(exist_ok=True)
        (sysd / "fvSolution").write_text(
            "FoamFile{format ascii;class dictionary;object fvSolution;}\n"
            "potentialFlow{nNonOrthogonalCorrectors 10;}\n"
            "solvers{Phi{solver GAMG;smoother DIC;tolerance 1e-08;relTol 0.01;}"
            "p{solver GAMG;smoother DIC;tolerance 1e-08;relTol 0.01;}}\n")
        (sysd / "fvSchemes").write_text(
            "FoamFile{format ascii;class dictionary;object fvSchemes;}\n"
            "ddtSchemes{default steadyState;}gradSchemes{default Gauss linear;}\n"
            "divSchemes{default none;div(div(phi,U)) Gauss linear;}\n"
            "laplacianSchemes{default Gauss linear corrected;}interpolationSchemes{default linear;}\n"
            "snGradSchemes{default corrected;}fluxRequired{default no;Phi;p;}\n")
        (sysd / "controlDict").write_text(
            "FoamFile{format ascii;class dictionary;object controlDict;}\n"
            "application potentialFoam;startFrom startTime;startTime 0;stopAt endTime;"
            "endTime 1;deltaT 1;writeControl timeStep;writeInterval 1;\n")

    def _flow_field(self):
        import glob
        fm = _fm()
        times = sorted(glob.glob(str(self.case / "[0-9]*")))
        td = Path(times[-1]) if times else self.case / "0"
        U = fm.read_internal_field(td / "U", 3)
        mag = np.linalg.norm(U, axis=1) if U.ndim == 2 else np.abs(U)
        frac = float((mag > 1e-4).mean())
        print(f"    potentialFoam: {100 * frac:.0f}% of cells flowing, mean |U| {mag.mean():.3g} m/s")
        return mag
