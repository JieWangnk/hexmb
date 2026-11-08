"""Turn a hexmb moving-mesh case into a runnable OpenFOAM 12 flow case.

The flow is driven by the wall motion alone: the cavity volume change pushes
blood through whichever valve is open.  No valve leaflets are modelled; an
open valve is a zero-total-pressure opening (``totalPressure p0 0`` with
``pressureInletOutletVelocity``) and a closed valve is a moving wall.  That is
the usual prescribed-motion LV CFD model (wall-driven, pressure-free openings).

Two windows are written, each starting at the case's reference phase (the
``constant/polyMesh``), because the mesh mover needs zero displacement at the
start time:

* ``diastole``: from end-systole (reference ``es``, the default of
  ``hexmb motion``) to the next end-diastole -- mitral ``inlet`` open, aortic
  ``outlet1`` closed;
* ``systole``: from end-diastole (a case built with ``--reference ed``) to the
  next end-systole -- aortic open, mitral closed.

Numerics: ``foamRun -solver incompressibleFluid`` (PIMPLE with ``correctPhi``
and ``correctMeshPhi`` for the moving mesh), laminar, blood ``nu`` 3.77e-6
m^2/s, Euler in time with a Courant-number limited step, second-order
``linearUpwind grad(U)`` convection, limited-corrected Laplacians for the
split-hex transition (up to about 64 degrees of non-orthogonality).

Function objects log the valve flow rates, the peak speed, the kinetic energy
integral, the wall shear stress at write times and ``checkMesh`` of the moving
mesh at write times, so a refinement study can compare them across meshes
(see ``scripts/verification_study.py`` and ``docs/VERIFICATION.md``).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from ..io.foam import read_polymesh, read_point_field, case_time_dirs, controldict_value
from .motion import FOAM_HEADER, _tname
from .quality import polymesh_metrics

__all__ = ["write_flow_case", "case_phase_volumes", "find_window", "NU_BLOOD"]

NU_BLOOD = 0.004 / 1060.0        # kinematic viscosity of blood, m^2/s (4 mPa s, 1060 kg/m^3)


def _patches(case_dir):
    txt = (Path(case_dir) / "constant" / "polyMesh" / "boundary").read_text()
    return [(m.group(1), m.group(2)) for m in
            re.finditer(r"(\w+)\s*\{\s*type\s+(\w+);[^}]*?nFaces\s+\d+;\s*startFace\s+\d+;", txt)]


def case_phase_volumes(case_dir, field="wantedDisplacement"):
    """``(times, volumes_m3)`` of the moving mesh at every time folder holding
    ``field``: the reference ``constant/polyMesh`` moved by that displacement.
    Reads ``motion_quality.csv`` when it is there and complete, else computes."""
    case_dir = Path(case_dir)
    tdirs = case_time_dirs(case_dir, field)
    if not tdirs:
        raise FileNotFoundError(f"{case_dir}: no time folders with {field}; this is not a hexmb moving-mesh case")
    times = np.array([float(d.name) for d in tdirs])
    pts, faces, owner, neigh = read_polymesh(case_dir / "constant" / "polyMesh")
    vols = []
    for d in tdirs:
        P = pts + read_point_field(d / field)
        vols.append(polymesh_metrics(P, faces, owner, neigh)["summary"]["total_volume"])
    return times, np.array(vols)


def find_window(times, volumes, window, t_start=None):
    """``(t0, t1)`` of the requested window in a volume-versus-time table.

    The window starts at ``t_start`` (default: the first time), which must be
    the reference phase of the case: the volume minimum (end-systole) for
    ``diastole``, the maximum (end-diastole) for ``systole``; it ends at the
    next maximum / minimum after it.  Raises with the fix when the start is the
    wrong phase (rebuild the case with ``--reference es`` or ``ed``)."""
    v = np.asarray(volumes, float); t = np.asarray(times, float)
    if window not in ("diastole", "systole"):
        raise ValueError(f"window must be 'diastole' or 'systole', got {window!r}")
    t0 = float(t[0]) if t_start is None else float(t_start)
    sel = t >= t0 - 1e-9
    t, v = t[sel], v[sel]
    if not len(t) or abs(t[0] - t0) > 1e-9:
        raise ValueError(f"no table entry at the start time {t0}")
    k0 = int(v.argmin()) if window == "diastole" else int(v.argmax())
    if k0 != 0:
        want = "es" if window == "diastole" else "ed"
        raise ValueError(f"the {window} window must start at the reference phase: this case starts at {t0:.4g} s "
                         f"with {v[0] * 1e6:.1f} mL while the {'minimum' if window == 'diastole' else 'maximum'} "
                         f"is at {t[k0]:.4g} s ({v[k0] * 1e6:.1f} mL); rebuild the case with --reference {want}")
    if len(t) < 2:
        raise ValueError("the table has no phase after the start time; build the case with --period so it wraps")
    rest = v[1:]
    k1 = 1 + (int(rest.argmax()) if window == "diastole" else int(rest.argmin()))
    return t0, float(t[k1])


def write_flow_case(case_dir, window="diastole", nu=NU_BLOOD, max_co=1.0, max_dt=None, write_interval=None,
                    n_outer=3, n_corr=2, n_nonorth=2, convection="linearUpwind grad(U)", n_procs=4,
                    t_end=None, verbose=True):
    """Add the flow set-up to a hexmb moving-mesh case, in place.

    Writes ``U`` and ``p`` in the start-time folder, ``constant/physicalProperties``
    and ``momentumTransport``, ``system/controlDict`` (``incompressibleFluid`` over
    the window, with function objects), ``fvSchemes``, ``fvSolution``,
    ``decomposeParDict`` and an ``Allrun`` script (decompose all time folders with
    ``-time '0:'`` so the processors hold the motion tables, run on ``n_procs``
    cores, reconstruct; exit 2 with a CHECK line when the final Courant number is 0).  The mesh-motion dictionaries are kept.  ``t_end``
    overrides the window's end.  Returns a dict with the window, patches and
    the files written.
    """
    case_dir = Path(case_dir)
    times, vols = case_phase_volumes(case_dir)
    t_start = controldict_value(case_dir, "startTime")
    t0, t1 = find_window(times, vols, window, t_start)
    if t_end is not None:
        t1 = float(t_end)
    names = dict(_patches(case_dir))
    if "wall" not in names or "inlet" not in names:
        raise ValueError(f"{case_dir}: expected patches 'wall' and 'inlet', found {sorted(names)}")
    has_aortic = "outlet1" in names
    if window == "systole" and not has_aortic:
        raise ValueError("the systole window needs an aortic patch 'outlet1'; this case was built with an "
                         "inlet rule that has none (fullcap / centred)")
    open_patch = "inlet" if window == "diastole" else "outlet1"
    closed = [p for p in ("inlet", "outlet1") if p in names and p != open_patch]

    H = FOAM_HEADER
    tdir = case_dir / _tname(t0)
    # -- fields ------------------------------------------------------------- #
    def bc(kind):
        blk = ""
        for p in ("wall",) + tuple(closed):
            if kind == "U":
                blk += f"    {p}\n    {{\n        type            movingWallVelocity;\n        value           uniform (0 0 0);\n    }}\n"
            else:
                blk += f"    {p}\n    {{\n        type            fixedFluxPressure;\n        value           uniform 0;\n    }}\n"
        if kind == "U":
            blk += (f"    {open_patch}\n    {{\n        type            pressureInletOutletVelocity;\n"
                    "        value           uniform (0 0 0);\n    }\n")
        else:
            blk += (f"    {open_patch}\n    {{\n        type            totalPressure;\n        p0              uniform 0;\n"
                    "        value           uniform 0;\n    }\n")
        return blk
    (tdir / "U").write_text(H.format(cls="volVectorField", loc=_tname(t0), obj="U")
                            + "dimensions      [0 1 -1 0 0 0 0];\ninternalField   uniform (0 0 0);\n\nboundaryField\n{\n"
                            + bc("U") + "}\n")
    (tdir / "p").write_text(H.format(cls="volScalarField", loc=_tname(t0), obj="p")
                            + "dimensions      [0 2 -2 0 0 0 0];\ninternalField   uniform 0;\n\nboundaryField\n{\n"
                            + bc("p") + "}\n")
    # -- constant ----------------------------------------------------------- #
    (case_dir / "constant" / "physicalProperties").write_text(
        H.format(cls="dictionary", loc="constant", obj="physicalProperties")
        + f"viscosityModel  constant;\nnu              [0 2 -1 0 0 0 0] {nu:.6g};\n")
    (case_dir / "constant" / "momentumTransport").write_text(
        H.format(cls="dictionary", loc="constant", obj="momentumTransport") + "simulationType  laminar;\n")
    # -- system ------------------------------------------------------------- #
    span = t1 - t0
    wint = write_interval if write_interval else span / 20.0
    mdt = max_dt if max_dt else min(1e-3, span / 100.0)
    fo = ("functions\n{\n"
          "    Qin\n    {\n        type surfaceFieldValue; libs (\"libfieldFunctionObjects.so\");\n"
          "        select patch; patch inlet; operation sum; fields (phi);\n"
          "        writeControl timeStep; log no; writeFields no;\n    }\n")
    if has_aortic:
        fo += ("    Qout\n    {\n        type surfaceFieldValue; libs (\"libfieldFunctionObjects.so\");\n"
               "        select patch; patch outlet1; operation sum; fields (phi);\n"
               "        writeControl timeStep; log no; writeFields no;\n    }\n")
    fo += ("    magU\n    {\n        type mag; libs (\"libfieldFunctionObjects.so\"); field U;\n"
           "        executeControl timeStep; writeControl writeTime;\n    }\n"
           "    Umax\n    {\n        type volFieldValue; libs (\"libfieldFunctionObjects.so\");\n"
           "        select all; operation max; fields (mag(U));\n"
           "        writeControl timeStep; log no; writeFields no;\n    }\n"
           "    magSqrU\n    {\n        type magSqr; libs (\"libfieldFunctionObjects.so\"); field U;\n"
           "        executeControl timeStep; writeControl writeTime;\n    }\n"
           "    KE2\n    {\n        // twice the kinetic energy per unit density: integral of |U|^2 (m^5/s^2)\n"
           "        type volFieldValue; libs (\"libfieldFunctionObjects.so\");\n"
           "        select all; operation volIntegrate; fields (magSqr(U));\n"
           "        writeControl timeStep; log no; writeFields no;\n    }\n"
           "    wallShearStress\n    {\n        type wallShearStress; libs (\"libfieldFunctionObjects.so\");\n"
           "        executeControl writeTime; writeControl writeTime;\n    }\n"
           "    checkMesh\n    {\n        type checkMesh; libs (\"libutilityFunctionObjects.so\");\n"
           "        executeControl writeTime;\n    }\n"
           "}\n")
    (case_dir / "system" / "controlDict").write_text(
        H.format(cls="dictionary", loc="system", obj="controlDict")
        + f"application     foamRun;\nsolver          incompressibleFluid;\nstartFrom       startTime;\n"
        f"startTime       {_tname(t0)};\nstopAt          endTime;\nendTime         {_tname(t1)};\n"
        f"deltaT          1e-05;\nwriteControl    adjustableRunTime;\nwriteInterval   {wint:.6g};\n"
        f"writeFormat     ascii;\nwritePrecision  7;\ntimePrecision   8;\nrunTimeModifiable no;\n"
        f"adjustTimeStep  yes;\nmaxCo           {max_co:.6g};\nmaxDeltaT       {mdt:.6g};\n\n" + fo)
    (case_dir / "system" / "fvSchemes").write_text(
        H.format(cls="dictionary", loc="system", obj="fvSchemes")
        + "ddtSchemes\n{\n    default         Euler;\n}\n"
        "gradSchemes\n{\n    default         Gauss linear;\n    grad(U)         cellLimited Gauss linear 1;\n}\n"
        f"divSchemes\n{{\n    default         none;\n    div(phi,U)      Gauss {convection};\n"
        "    div((nuEff*dev2(T(grad(U))))) Gauss linear;\n}\n"
        "laplacianSchemes\n{\n    default         Gauss linear limited corrected 0.33;\n}\n"
        "interpolationSchemes\n{\n    default         linear;\n}\n"
        "snGradSchemes\n{\n    default         limited corrected 0.33;\n}\n")
    (case_dir / "system" / "fvSolution").write_text(
        H.format(cls="dictionary", loc="system", obj="fvSolution")
        + "solvers\n{\n"
        "    p\n    {\n        solver          GAMG;\n        smoother        DICGaussSeidel;\n"
        "        tolerance       1e-06;\n        relTol          0.05;\n    }\n"
        "    pFinal\n    {\n        $p;\n        relTol          0;\n    }\n"
        "    \"(Phi|MeshPhi|pcorr)\"\n    {\n        $p;\n        relTol          0.01;\n    }\n"
        "    pcorrFinal\n    {\n        $p;\n        relTol          0;\n    }\n"
        "    U\n    {\n        solver          smoothSolver;\n        smoother        symGaussSeidel;\n"
        "        tolerance       1e-07;\n        relTol          0.05;\n    }\n"
        "    UFinal\n    {\n        $U;\n        relTol          0;\n    }\n"
        "}\n\n"
        f"PIMPLE\n{{\n    correctPhi      yes;\n    correctMeshPhi  yes;\n    momentumPredictor yes;\n"
        f"    nOuterCorrectors {n_outer};\n    nCorrectors     {n_corr};\n    nNonOrthogonalCorrectors {n_nonorth};\n}}\n\n"
        "relaxationFactors\n{\n    equations\n    {\n        U               0.8;\n        UFinal          1;\n    }\n}\n")
    (case_dir / "system" / "decomposeParDict").write_text(
        H.format(cls="dictionary", loc="system", obj="decomposeParDict")
        + f"numberOfSubdomains {n_procs};\nmethod          scotch;\n")
    run = (case_dir / "Allrun")
    run.write_text(
        "#!/bin/bash\n# wall-driven LV flow: decompose, run foamRun (incompressibleFluid + interpolator mesh mover), reconstruct\n"
        "cd \"$(dirname \"$0\")\" || exit 1\n"
        "source /opt/openfoam12/etc/bashrc 2>/dev/null || { echo 'OpenFOAM 12 not found at /opt/openfoam12'; exit 1; }\n"
        f"NP=${{1:-{n_procs}}}\n"
        "if [ \"$NP\" -gt 1 ]; then\n"
        "    # -time '0:' decomposes EVERY time folder: the interpolator mover reads wantedDisplacement\n"
        "    # from each table time, and a plain decomposePar copies only the start time, so the\n"
        "    # processors would hold no later tables and the mesh would never move (Courant 0).\n"
        "    decomposePar -force -time '0:' > log.decomposePar 2>&1 || { echo 'decomposePar failed (see log.decomposePar)'; exit 1; }\n"
        "    nt=$(ls -d processor0/[0-9]* 2>/dev/null | wc -l)\n"
        "    [ \"$nt\" -lt 2 ] && { echo \"CHECK: processor0 holds $nt time folder(s); the motion tables were not decomposed\"; exit 1; }\n"
        "    mpirun -np \"$NP\" foamRun -parallel > log.foamRun 2>&1\n"
        "    rc=$?\n"
        "    reconstructPar -newTimes > log.reconstructPar 2>&1\n"
        "else\n"
        "    foamRun > log.foamRun 2>&1\n"
        "    rc=$?\n"
        "fi\n"
        "co=$(grep 'Courant Number mean' log.foamRun | tail -n1 | awk '{print $NF}')\n"
        "echo \"foamRun exit $rc; last Courant max $co\"\n"
        "if [ \"$co\" = 0 ] || [ -z \"$co\" ]; then\n"
        "    echo 'CHECK: Courant max is 0, the mesh did not move (missing wantedDisplacement tables, wrong reference time?)'\n"
        "    [ \"$rc\" -eq 0 ] && rc=2\n"
        "fi\n"
        "exit $rc\n")
    run.chmod(0o755)
    (case_dir / "flow_setup.txt").write_text(
        f"window {window}: t {t0:.6g} -> {t1:.6g} s (cavity {vols[np.argmin(np.abs(times - t0))] * 1e6:.1f} -> "
        f"{vols[np.argmin(np.abs(times - t1))] * 1e6:.1f} mL)\n"
        f"open valve: {open_patch} (totalPressure p0 0 / pressureInletOutletVelocity); closed: {' '.join(closed) or 'none'} "
        f"(movingWallVelocity / fixedFluxPressure); wall movingWallVelocity / fixedFluxPressure\n"
        f"nu {nu:.6g} m2/s laminar; maxCo {max_co}; maxDeltaT {mdt:.6g}; writeInterval {wint:.6g}; "
        f"div(phi,U) {convection}; PIMPLE {n_outer} outer / {n_corr} corr / {n_nonorth} non-orth; {n_procs} procs\n")
    if verbose:
        print(f"  flow case ({window}): t {t0:.4g} -> {t1:.4g} s, open valve {open_patch}, closed "
              f"{' '.join(closed) or 'none'}; nu {nu:.3g} m2/s, maxCo {max_co}, maxDeltaT {mdt:.2g} s; "
              f"run with {case_dir / 'Allrun'} [n_procs]")
    return dict(window=window, t0=t0, t1=t1, open=open_patch, closed=closed, n_procs=n_procs,
                files=["U", "p", "physicalProperties", "momentumTransport", "controlDict", "fvSchemes",
                       "fvSolution", "decomposeParDict", "Allrun", "flow_setup.txt"])
