"""Interactively slice a Volume along a plane.
Middle button + drag to slide the plane along the arrow"""
import itk
from vedo import *
from vedo import Volume, Plotter
#用来做数据处理

normal = [0, 0, 1]
cmap = "gist_stern_r"
cmap2="viridis_r"

itk_img = itk.imread(filename="./MEN_001.nii.gz")
vtk_img = itk.vtk_image_from_image(l_image=itk_img)
vol = Volume(vtk_img).cmap(cmap)
itk_img = itk.imread(filename="./MEN_001_0003.nii.gz")
vtk_img = itk.vtk_image_from_image(l_image=itk_img)
vol2 = Volume(vtk_img).cmap(cmap2)

def func(w, _):
    c, n = pcutter.origin, pcutter.normal
    vslice = vol.slice_plane(c, n, autocrop=True,border=1.0).cmap('bone') #给动态的切割面一个原点和一条法线
    vslice.name = "Slice"
    #vslice2 = vol2.slice_plane(c, n, autocrop=True, border=1.0).cmap(cmap2)  # 给动态的切割面一个原点和一条法线
    #vslice2.name = "Slice2"
    plt.at(1).remove("Slice","Slice2").add(vslice)

vslice = vol.slice_plane(vol.center(), normal).cmap(cmap)
vslice.name = "Slice"

plt = Plotter(axes=8, N=2, bg="k", bg2="bb", interactive=False) #N:desired renderers,可以qt_windows,sharecam,
plt.at(0).show(vol,vol2, __doc__, zoom=1.5)

pcutter = PlaneCutter(
    vslice,
    normal=normal,#平面的法线，此处赋值为平面法线
    alpha=0, #输入网络截止部分的透明度
    c=(0.25,0.25,0.25),
    padding=0,
    can_translate=True,
    can_scale=True, #启用部件的缩放功能
)
pcutter.add_observer("interaction", func)
plt.at(1).add(pcutter)
plt.interactive()

plt.close()