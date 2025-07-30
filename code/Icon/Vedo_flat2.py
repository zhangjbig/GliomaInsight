"""A flag-post style marker"""

#常驻数据（flag）
import itk
from vedo import ParametricShape, precision, color_map, show, Volume, Plotter

#s = ParametricShape("RandomHills").cmap("coolwarm")

itk_img = itk.imread(filename="./MEN_001_0003.nii.gz")
vtk_img = itk.vtk_image_from_image(l_image=itk_img)
vol = Volume(vtk_img).cmap('viridis_r')

vslice = vol.slice_plane(origin=vol.center(), normal=(0,1,1)) #mesh
vslice.lighting('plastic').add_scalarbar('Slice', c='w').cmap('Purples_r') #建立一个实例之后是渲染方法
                                                                                    # lighting: style=[metallic, plastic, shiny, glossy, ambient, off]
arr = vslice.pointdata[0] # retrieve vertex array data

plt = Plotter(axes=0, bg='k', bg2='bb')
#plt.use_depth_peeling()
#plt.add_callback('as my mouse moves please call', func) # be kind to vedo ;)
'''
s = vol.yslice(50).cmap('viridis_r') #返回的是一个mesh对象
pts = s.clone().decimate(n=10).vertices#这个提取的是表面高度点

fss = []
'''
for p in pts:
    col = color_map(p[2], name="coolwarm", vmin=0, vmax=0.7)
    ht = precision(p[2], 3)
    fs = s.flagpost(f"Heigth:\nz={ht}m", p, c=col)
    fss.append(fs)

show(s, *fss, __doc__, bg="bb", axes=1, viewup="z")