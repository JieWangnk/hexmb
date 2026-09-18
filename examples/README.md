# Examples

Ready-made results so you can see what hexmb produces without running anything.

## What's here

| Path | What it is | How to look at it |
|---|---|---|
| `pipe/` | An OpenFOAM case — a straight-pipe O-grid mesh (8736 hex, min scaled-Jac +0.71) | `paraview pipe/pipe.foam` (or `paraFoam -case pipe`) |
| `lv_endo.stl` | The endocardial (blood-pool) surface of one public CAP left ventricle, as a CAD file (4368 triangles, mm) | any STL viewer / ParaView |
| `img/` | Reference renders of the results below | any image viewer |

## Reference renders (`img/`)

- `pipe.png` — the pipe mesh; note the square butterfly core + petal blocks on the end cap (no polar axis).
- `lv_telescoping_cut.png` — a long-axis cut of the telescoping LV mesh: coarse apex cap, fine body, coarse base cap, butterfly core.
- `lv_moving_mesh.gif` — the **moving mesh**: the telescoping LV rebuilt at each of the 20 CAP cardiac phases, fixed camera, so the same-topology mesh deforms through the beat (contracts at systole, refills). This is what hexmb is for — one iso-topological mesh morphing over the cycle, no remeshing.
- `lv_endo_stl.png` — the `lv_endo.stl` surface.

## Regenerate any of these

```bash
# the pipe case
hexmb pipe --length 100 --radius 10 -o examples/pipe

# the CAP surface + the full telescoping LV mesh (needs a CAP download + OpenFOAM 12)
python scripts/cap_to_mesh.py /path/to/SCD0000101_1.model.exnode -o out/lv --stl examples/lv_endo.stl
```

The full telescoping LV mesh itself is not shipped (it is large and needs OpenFOAM
12 to assemble the AMI couplings); `scripts/cap_to_mesh.py` builds it in a few
seconds once you have a CAP model — see [`../docs/DATA.md`](../docs/DATA.md).
