# pvbatch script: long-axis velocity-magnitude slice of a hexmb flow case at one time.
# usage: pvbatch scripts/flow_slice_render.py CASE_DIR TIME OUT.png [TITLE] [VMAX]
# Reads the reconstructed OpenFOAM case (case.foam), slices it through the cavity centre
# along the long axis (the plane that holds the mitral and aortic patch centres) and
# colours |U| from 0 to VMAX m/s (default 1) with the mesh edges drawn.
import sys
from paraview.simple import *  # noqa: F401,F403

case, t, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
title = sys.argv[4] if len(sys.argv) > 4 else ""
vmax = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0   # top of the colour scale (m/s)
r = OpenFOAMReader(FileName=case + "/case.foam")
r.ListtimestepsaccordingtocontrolDict = 1       # list the fields from the flow's start time, not from the
r.UpdatePipelineInformation()                    # displacement-only folders before it
r.MeshRegions = ["internalMesh"]
avail = list(r.GetProperty("CellArrays").GetAvailable())
have_mag = "mag(U)" in avail
r.CellArrays = ["U", "mag(U)"] if have_mag else ["U"]
r.UpdatePipeline(t)
b = r.GetDataInformation().GetBounds()
c = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
ext = [b[1] - b[0], b[3] - b[2], b[5] - b[4]]
long_axis = ext.index(max(ext))                      # the longest extent is the apex-base direction
normal = [0.0, 0.0, 0.0]; normal[(long_axis + 2) % 3] = 1.0
internal = ExtractBlock(Input=r); internal.Selectors = ["/Root/internalMesh"]
sl = Slice(Input=internal); sl.SliceType = "Plane"; sl.SliceType.Origin = c; sl.SliceType.Normal = normal
try:
    sl.Triangulatetheslice = 0                       # keep the cut polygons, not their triangles
except Exception:  # noqa: BLE001
    pass
mag = Calculator(Input=sl); mag.ResultArrayName = "speed"; mag.AttributeType = "Cell Data"
mag.Function = '"mag(U)"' if have_mag else "mag(U)"     # the function object's own field when it was written
view = GetActiveViewOrCreate("RenderView"); view.ViewSize = [900, 900]
try:
    view.UseColorPaletteForBackground = 0
except Exception:  # noqa: BLE001
    pass
view.Background = [1, 1, 1]
view.OrientationAxesVisibility = 0
d = Show(mag, view); d.SetRepresentationType("Surface With Edges"); d.EdgeColor = [0.25, 0.25, 0.25]; d.LineWidth = 0.4
ColorBy(d, ("CELLS", "speed"))
lut = GetColorTransferFunction("speed")
for preset in ("Viridis (matplotlib)", "Viridis", "Inferno (matplotlib)", "Cool to Warm"):
    try:
        lut.ApplyPreset(preset, True); break
    except Exception:  # noqa: BLE001 - preset names differ between ParaView versions
        continue
lut.RescaleTransferFunction(0.0, vmax)
bar = GetScalarBar(lut, view); bar.Title = "|U| (m/s)"; bar.ComponentTitle = ""; bar.TitleColor = [0, 0, 0]; bar.LabelColor = [0, 0, 0]
bar.Visibility = 1; bar.WindowLocation = "Any Location"; bar.Position = [0.86, 0.25]; bar.ScalarBarLength = 0.5
if title:
    txt = Text(Text=title); td = Show(txt, view); td.Color = [0, 0, 0]; td.FontSize = 20; td.WindowLocation = "Upper Center"
view.CameraPosition = [c[i] + (normal[i] * 0.5) for i in range(3)]
view.CameraFocalPoint = c
up = [0.0, 0.0, 0.0]; up[long_axis] = 1.0
view.CameraViewUp = up
view.CameraParallelProjection = 1
ResetCamera(view)
view.ViewTime = t
Render(view)
SaveScreenshot(out, view, ImageResolution=[900, 900])
print("wrote", out)
