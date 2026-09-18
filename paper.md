---
title: 'hexmb: a structured O-grid hexahedral mesh engine for morphing cardiovascular geometries'
tags:
  - Python
  - computational fluid dynamics
  - mesh generation
  - hexahedral meshing
  - OpenFOAM
  - cardiovascular
authors:
  - name: Jie Wang
    orcid: 0000-0000-0000-0000   # TODO: replace with your ORCID
    affiliation: 1
affiliations:
  - name: "TODO: your institution, City, Country"
    index: 1
date: 18 September 2026
bibliography: paper.bib
---

# Summary

`hexmb` builds all-hexahedral **O-grid (butterfly)** meshes and writes them
directly as an OpenFOAM `constant/polyMesh` [@OpenFOAM1998]. An O-grid fills each cross-section
with a square "H" core block surrounded by four boundary-fitted petal blocks:
all cells are hexahedra, the boundary is honoured, and there is no point where
the mesh converges to a singular axis. Because of that, a single O-grid mesh can
be morphed through large deformation --- such as a left ventricle contracting by
more than half its volume every heartbeat --- as one iso-topological mesh, with
no remeshing between phases.

The engine is one operation applied to different inputs: sweeping a butterfly
cross-section along a stack of rings builds a straight pipe, a bend, a tapered
vessel, or a heart chamber. A *telescoping* variant joins coarse apex and valve
caps to a fine body through non-conformal interfaces, so the body can carry the
cell budget while the closing regions stay coarse enough to survive the
contraction. `hexmb` ships with a reader for public Cardiac Atlas Project
left-ventricle finite-element models [@Fonseca2011; @Radau2009], so the whole
path from a public, CC0-licensed heart geometry to a checked hexahedral mesh is
reproducible with no private data.

# Statement of need

Moving-boundary computational fluid dynamics (CFD) of the cardiac left ventricle
(LV) needs a volume mesh that stays valid as the cavity deforms through the beat.
The natural structured mesh --- rings converging to the apex, like lines of
longitude on a globe --- has a polar-axis singularity: cells at the apex have
almost no volume and invert once the ventricle contracts. Cut-cell meshers such
as OpenFOAM's `snappyHexMesh` avoid the singularity but need fragile per-phase
remeshing and give up the near-wall hexahedral layers that gradient-based
quantities (for example wall shear stress) depend on. O-grid / butterfly
topologies are standard in structured grid generation [@Thompson1999], but
general, open tooling that *constructs and morphs* them for a deforming
biological chamber and exports straight to OpenFOAM has not been readily
available.

`hexmb` fills that gap. Its numpy-only core assembles the block topology once,
deduplicates shared nodes by union-find, gates each cell on the scaled Jacobian
[@Knupp2001], writes a valid OpenFOAM `polyMesh`, and morphs the mesh across the
cycle by radial-basis-function interpolation of the moving wall [@deBoer2007].
The telescoping variant couples its coarse and fine regions through the
non-conformal arbitrary mesh interface [@FarrellMaddison2011], the same
interpolation OpenFOAM's coupling is built on. The software targets the growing
interest in image-driven, prescribed-motion LV hemodynamics
[@Peighambari2026] by supplying the mesh half of that pipeline as open, tested,
reproducible software, and it is written to generalise beyond the heart to any
single-lumen, star-shaped, genus-0 geometry (pipes, bends, arteries, single
chambers).

# Functionality

- **Templates**: pipe, elbow, single-lumen STL tube, single LV chamber, and a
  telescoping multi-resolution LV.
- **Direct OpenFOAM export**: `constant/polyMesh` with patch tagging
  (`wall` / `inlet` / `outlet`), no intermediate format.
- **Quality and verification**: per-cell scaled Jacobian gate, plus a
  `checkMesh`-based `verify` command.
- **Data input**: a reader for Cardiac Atlas Project `.exnode` LV models and an
  STL surface exporter, so a public heart geometry can be meshed end to end.
- **Interfaces**: a command-line tool (`hexmb pipe|elbow|cap|lv|stl|verify`) and
  a small Python API.

The repository includes a worked example (`scripts/cap_to_mesh.py`), an
`examples/` directory with a ready-to-open pipe case and a CAP endocardial
surface, and a self-contained test suite.

# Limitations

`hexmb` applies to a single connected lumen describable as a stack of
star-shaped cross-sections along a centreline; branches, junctions, and
multi-lumen trees are out of scope. Assembling the telescoping non-conformal
coupling requires OpenFOAM; the numpy-only build and quality gate do not.

# Acknowledgements

The left-ventricle demonstration uses the Cardiac Atlas Project's Sunnybrook
Cardiac Data, released into the public domain (CC0) [@Fonseca2011; @Radau2009].

# References
