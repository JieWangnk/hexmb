# Engineering review: cavity readiness

Review date: 12 November 2026. Baseline: `6fb5ddce276b1d5d85805dcd9447a5b510c755b9`.
Scope: source inspection, baseline suite, targeted regression tests and static
mesh export/readback. No independent OpenFOAM solver run was available here.

## Assessment

The engine is a useful specialised mesher: structured blocks, topological joins,
ordered ring reconstruction, conformal resolution transitions and moving-LV
support already exist. The original 74 tests pass in this environment. Its
original public interface is strongly cardiac-specific. Calling it a general
arbitrary-cavity mesher would exceed both its topology and the available evidence.

## Findings addressed in this change

| Priority | Finding and consequence | Change |
|---|---|---|
| High | `cmd_verify` swallowed readback exceptions and always returned 0, including after external failures. Automation could accept a broken case. | Readback/geometry failures return 1; external nonzero exit, failed-check report or missing Mesh OK returns 1. Regression tests simulate external outcomes. |
| Medium | `scaled_jac` added a fixed dimensional epsilon to a product of lengths. The score changed with coordinate scale, even for identical cubic cells. | Normalise each edge before the determinant. Tests span scaling factors 1e-12 to 1e6 and cover collapse/NaN. |
| Medium | `graded_radial` accepted non-positive or non-finite ratios. | Reject those inputs with an actionable error. |
| Medium | `polymesh_metrics` could not consume tuple faces emitted by `build_faces`. | Explicit integer-array indexing; new cavity tests exercise that in-memory path. |
| Capability | Cavity users had to inherit cardiac patch conventions and implicit millimetres. | Separate box/ring cavity templates and CLI with explicit units, end-cap choices, wall/empty patch types, quality limits and JSON provenance. |

The new cavity workflow refuses non-empty output directories. This protection
is not retrofitted to all existing commands. Existing LV trimming and phase
filling defaults are also unchanged.

## Evidence and remaining work

The final suite passes: **97 passed** (Python 3.12 with surface/image extras).
The single warning is an environment pytest-plugin deprecation.

Tests cover exact rectangular and polygonal-prism volumes, face coverage,
source-unit equivalence, 2D empty patches, grading, rejected folded/collapsed
rings, refusal to overwrite, quality-limit failures and a metre-input STL sac
through VTK export. The 32×32×1 box smoke test has 1,024 cells, scaled Jacobian 1,
zero non-orthogonality to printed precision, and no non-positive volumes.
The new path uses only ordinary hexahedra; the existing conformal LV path still
contains split-hex polyhedra and should not be described as strictly hex-only.

Before claiming broader production readiness:

1. Run the new cases through supported OpenFOAM versions and a lid-driven cavity
   benchmark, comparing velocity profiles and mesh refinement. The added tests
   verify meshing and command behaviour, not flow accuracy.
2. Add signed distance and enclosed-volume error budgets against the original
   surface, with explicit limits on ray interpolation, smoothing and truncation.
   A high-quality mesh can represent the wrong cavity.
3. Strengthen general surface topology/self-intersection screening and provide
   explicit local opening selection. The current generic ring caps are whole
   end sections; they are not arbitrary ports.
4. For moving cavities, add generic patch correspondence and intermediate-time
   validity checks. Passing sampled phases alone does not prove validity of
   spline-interpolated motion between them.
5. Add resource estimates and large-case export benchmarks. Assembly is
   vectorised, but face construction and text export still contain Python loops.

This release makes static box and suitable smooth-cavity meshing directly
usable. It does not solve general automatic hexahedral decomposition.
