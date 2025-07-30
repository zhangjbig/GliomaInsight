from vedo import Plotter, Volume
from .. import Script
from Script.mySlice import 3DPlotter

plt = Plotter(bg2='bisque', size=(1000, 800), interactive=False)

import itk
itk_img = itk.imread("./MEN_001_0003.nii.gz")
vtk_img = itk.vtk_image_from_image(itk_img)

vol = Volume(vtk_img)
vol_mesh = vol.tomesh()  # 转为 Mesh
e1 = vol_mesh.clone().cut_with_plane(normal=[0, 1, 0]).c("green4")

plt.show(vol_mesh, e1, "Volume Mesh and Plane Cutting")

"""Render meshes into inset windows
(which can be dragged)"""
from vedo import *
import itk

plt = Plotter(bg2='bisque', size=(1000,800), interactive=False)
itk_img = itk.imread(filename="./MEN_001_0003.nii.gz")
vtk_img = itk.vtk_image_from_image(l_image=itk_img)
vol = Volume(vtk_img)
print(type(vtk_img))

e = Volume(vol)#.isosurface()
#e.normalize().shift(-2,-1.5,-2).c("gold")

plt.show(e, __doc__, viewup='z')

# make clone copies of the embryo surface and cut them:
e1 = e.clone().cut_with_plane(normal=[0,1,0]).c("green4")
e2 = e.clone().cut_with_plane(normal=[1,0,0]).c("red5")

# add 2 draggable inset windows:
plt.add_inset(e1, pos=(0.9,0.8))
plt.add_inset(e2, pos=(0.9,0.5))

# customised axes can also be inserted:
ax = Axes(
    xrange=(0,1), yrange=(0,1), zrange=(0,1),
    xtitle='front', ytitle='left', ztitle='head',
    yzgrid=False, xtitle_size=0.15, ytitle_size=0.15, ztitle_size=0.15,
    xlabel_size=0, ylabel_size=0, zlabel_size=0, tip_size=0.05,
    axes_linewidth=2, xline_color='dr', yline_color='dg', zline_color='db',
    xtitle_offset=0.05, ytitle_offset=0.05, ztitle_offset=0.05,
)

ex = e.clone().scale(0.25).pos(0,0.1,0.1).alpha(0.1).lighting('off')
plt.add_inset(ax, ex, pos=(0.1,0.1), size=0.15, draggable=False)
plt.interactive().close()

"""Use sliders to slice a Volume
(click button to change colormap)"""
from vedo import dataurl, Volume, Text2D
from vedo.applications import Slicer3DPlotter
import itk

vol = Volume(dataurl + "embryo.slc")
nifti_file_name = "D:/www/xxx/tumorTest/UI/BraTS2021_00495/BraTS2021_00495_flair.nii.gz"
itk_img = itk.imread(filename=nifti_file_name)
vtk_img = itk.vtk_image_from_image(l_image=itk_img)
vol = Volume(vtk_img)

plt = Slicer3DPlotter(
    vol,
    cmaps=("gist_ncar_r", "jet", "Spectral_r", "hot_r", "bone_r"),
    use_slider3d=True,
    bg="white",
    bg2="blue9",
)

# Can now add any other vedo object to the Plotter scene:
plt += Text2D(__doc__)

plt.show(viewup='z')
plt.close()