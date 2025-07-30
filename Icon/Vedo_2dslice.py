"""Use sliders to slice a Volume
(click button to change colormap)"""
from vedo import dataurl, Volume, Text2D
from vedo.applications import Slicer3DPlotter
import itk

#vol = Volume(dataurl + "embryo.slc")

itk_img = itk.imread(filename="./MEN_001_0003.nii.gz")
vtk_img = itk.vtk_image_from_image(l_image=itk_img)
vol = Volume(vtk_img)

plt = Slicer3DPlotter(
    vol,
    cmaps=("gist_ncar_r", "jet", "Spectral_r", "hot_r", "bone_r"),
    use_slider3d=False,
    bg="white",
    bg2="blue9",
)

# Can now add any other vedo object to the Plotter scene:
plt += Text2D(__doc__)

plt.show(viewup='z')
plt.close()