# Open data for left-ventricle geometry through the cardiac cycle

What hexmb needs for a moving mesh is, per subject, one endocardial surface (or
one 3D mask) per cardiac phase.  This page lists the open sources that were
checked for that on 6 November 2026 by reading each landing page (no files were
downloaded for the check; counts marked "paper" come from the dataset paper,
where the page gives none).  Licences are quoted as the pages state them; check them
again before you redistribute anything derived.

## The short answer

Only one fully open set ships 3D LV geometry on every phase: the **Cardiac Atlas
Project's Sunnybrook models** (45 subjects, 20 phases each, CC0).  hexmb reads
them directly (`.exnode`), one subject is bundled in the package, and the whole
set runs through `hexmb motion` (see the cohort table at the end).  Every other
open source gives either cine images with masks on two frames (ED and ES), 2D
contours, a static end-diastolic mesh, or a shape model with no motion.  To go
beyond 45 subjects you therefore segment the cine yourself; the open tool that
does that for all frames and writes per-frame surfaces is **biv-me** (Auckland,
Apache-2.0), which runs on the Sunnybrook cine, the cMAC 2011 volunteers and
any other cine set whose licence allows it.

## Sources checked

| Dataset | Licence (as stated) | Subjects | Per phase 3D LV geometry? | What you get | Access |
|---|---|---|---|---|---|
| Cardiac Atlas: Sunnybrook Cardiac Data + "Left ventricular models" (scd_capmodels.zip) | CC0 1.0 | 45 | **Yes**: 20 `.exnode` LV models per subject (the zip is not documented on the page; this is what the files hold) | cine DICOM, manual contours, CAP finite-element LV models | direct download |
| Cardiac Atlas: LV Segmentation Challenge 2011 (DETERMINE) | CAP data-use agreement; use limited to the consensus project | 100 + 100 | no (2D myocardium masks) | SA+LA cine, masks on the training set | request, approval, login |
| Cardiac Atlas: LV Statistical Shape Challenge 2015 | none stated | 200 + 200 | ED and ES point sets only | corresponding LV point sets, no images | request form |
| Cardiac Atlas: Motion Tracking Challenge 2011 (cMAC) | "open access policy by attribution" | 15 volunteers + phantom | not stated (cine, tagged MRI, 3D US; ground-truth landmarks) | DICOM, VTK, NIfTI | direct download |
| Cardiac Atlas: MITEA (3D echo) | CC BY-NC-SA 4.0, research only | 134 | ED and ES only | 3D echo with CMR-derived LV labels | request form |
| Cardiac Atlas: biv-me rescans | CC BY-NC-SA 4.0 | 10, scanned twice | likely per frame (biv-me output), page silent | cine + biventricular surface meshes (.vtk) | request form |
| Cardiac Atlas: biventricular PCA modes (UK Biobank) | none stated | 630 / 4 329 | ED and ES shape model only | mean shape + 200 modes (HDF5) | direct download |
| ACDC (Bernard 2018) | CC BY-NC-SA 4.0, non-commercial research only | 150 | ED and ES 3D masks only | SA cine NIfTI, LV/myo/RV labels at ED and ES | accept terms, download |
| M&Ms (2020), M&Ms-2 (2021) | research-use agreement, no derivative works, no redistribution | 345 / 360 (paper) | ED and ES only | SA (+LA) cine, labels at ED and ES | request |
| CMRxMotion 2022, CMRxRecon 2023/24 | none on page | 40 x 4 / 300 / 330 | ED and ES (2022); 2023 label frames not stated; 2024 no labels | cine, k-space (CMRxRecon) | Synapse |
| York cardiac MRI (Andreopoulos & Tsotsos 2008) | not seen (page down on the day; a mirror says non-commercial) | 33 | 2D endo+epi contours on all 20 frames, stackable to 3D | 8-15 slices x 20 frames, MATLAB | direct when the page is up |
| Kaggle Data Science Bowl 2016 | Kaggle rules | about 1 140 (unverified) | no (EDV/ESV targets, no masks) | cine DICOM | Kaggle account |
| Zenodo 5573921 (Gerach 2021, KIT four-chamber model) | CC BY-NC 4.0 | 1 | no: one diastasis mesh + time-resolved volume/thickening curves | VTU, STL | direct |
| Zenodo 3890034 (Strocchi 2020), 4590294 (Rodero 2021), 4593739 (Rodero SSM) | CC BY 4.0 | 24 / 20 / 39 | no, end-diastolic CT only | four-chamber tetrahedral meshes | direct |
| Zenodo 4506463 (Schuler & Loewe, biventricular SSM from Bai 2015) | CC BY 4.0 | shape model | no, one phase | mean surface + 100 modes + 100 instances (VTP/VTU) | direct |
| Zenodo 15649643 (Ugurlu 2025, UK Biobank representative meshes, CEMRG) | CC0 1.0 | 1 423 meshes | no, end-diastolic frames | biventricular tet meshes + fibres (113 GB) | direct |
| MeshHeart / CHeart (Qiao, Bai; Imperial) | MIT label on the repo; no release | trained on 38 309 UK Biobank | would be 4D, but no model or meshes released | training code only | none |
| UK Digital Heart atlas (Bai 2015); 2026 four-chamber UK Biobank SSM preprint (arXiv 2603.28711); MCSI-Net (Xia 2022) | no download or licence found | 1 000+ / 95 695 / UKB | not released ("we will release" in the 2026 preprint) | n/a | n/a |

Notes from the check:

- Zenodo 4593739 and 4593738 are the same record.  Strocchi's 24 CT meshes are
  record 3890034, not 4590294 (that one is Rodero's 20 healthy hearts).
- The M&Ms agreement bars derivative works, so meshes built from it cannot be
  shared.  The LVSC 2011 terms limit use to the consensus-segmentation project.
- The Cardiac Atlas shape-model and challenge pages carry no licence statement
  at all; only Sunnybrook is CC0.
- Some old Cardiac Atlas URLs 404; the current pages sit under cardiacatlas.org.

## Routes to more subjects

1. **Sunnybrook models, as they are** (45 subjects, CC0): `hexmb motion` on the
   20 `.exnode` files of each subject.  No segmentation step.  This is the cohort
   below.
2. **Cine + biv-me**: the biv-me pipeline (github.com/UOA-Heart-Mechanics-Research/biv-me,
   Apache-2.0; Dillon et al., Med Image Anal 2026, 10.1016/j.media.2026.104252)
   segments CMR DICOM with nnU-Net and fits a biventricular model frame by frame,
   writing per-frame meshes (.txt vertices, optional .obj/.vtk).  Its LV
   endocardial surface per frame is exactly what `hexmb motion --surfaces` reads.
   It needs Python 3.11, PyTorch (GPU recommended) and its Git-LFS model weights.
   Open cine to run it on: Sunnybrook (45, CC0), cMAC 2011 (15, attribution),
   ACDC (150, non-commercial), York (33, licence to confirm).
3. **Cine + any short-axis segmentation** (nnU-Net, the UK Biobank
   segmentations, a vendor tool): one NIfTI blood-pool mask per phase is read
   directly, slice by slice, by `hexmb motion --surfaces phase_*.nii.gz --label 1`
   (thick slices are the normal case; the reader's volume error against a
   known surface is in `VERIFICATION.md`).  No surface fitting step, so this is
   the route to a cohort of thousands once the segmentations exist; what it
   does not give is a smooth fitted model to compare with.
4. **ED/ES-only sets** (ACDC, M&Ms, MITEA, the Atlas PCA modes): two anchor
   shapes per subject.  hexmb can mesh both, but motion between them would be
   interpolated, so these suit shape studies, not wall-motion CFD.
5. **Generative 4D models** (MeshHeart, CHeart, the 2026 four-chamber SSM):
   nothing released yet; worth asking the authors for sample meshes.

## The Sunnybrook cohort through hexmb 0.2

All 45 subjects (20 phases each) were run through `hexmb`'s moving-mesh builder
at the default settings (last on 8 November 2026, with the per-phase quality
metrics on): 40 pass the gate on every phase
exactly as given, 4 more after the builder trimmed the base rim by itself
(2-8 % of the long axis, where the fitted model folds through its own axis or
narrows past the ring below at end-systole), and the last one, a model that
crosses its own long axis in the body of the ventricle on one phase, has that
phase filled from its neighbours and reported.  All 45 cases are valid.
Mesh-derived EF spans 13-77 %; no face of any mesh reaches OpenFOAM's
skewness warning level, and non-orthogonality stays under 80 deg.  The
per-subject table with the metrics, the figures and the analysis are in
`DATA.md`.
