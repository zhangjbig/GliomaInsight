import itk
from vedo import  precision, Sphere, Volume, Plotter

#可以拿来当小框切片，加slicer
def func(evt):
    if not evt.actor:
        return
    pid = evt.actor.closest_point(evt.picked3d, return_point_id=True)
    txt = f"Probing:\n{precision(evt.actor.picked3d, 3)}\nvalue = {arr[pid]}"

    pts = evt.actor.points()
    sph = Sphere(pts[pid], c='orange7').pickable(False)
    fp = sph.flagpole(txt, s=7, offset=(-150,15), font=2).follow_camera()
    # remove old and add the two new objects
    plt.remove('Sphere', 'FlagPole').add(sph, fp).render()

itk_img = itk.imread(filename="./MEN_001_0003.nii.gz")
vtk_img = itk.vtk_image_from_image(l_image=itk_img)
vol = Volume(vtk_img)

#vol = Volume(dataurl+'embryo.slc').alpha([0,0,0.8]).c('w').pickable(False)
vslice = vol.slice_plane(origin=vol.center(), normal=(0,1,1)) #mesh
vslice.lighting('plastic').add_scalarbar('Slice', c='w').cmap('Purples_r') #建立一个实例之后是渲染方法
                                                                                    # lighting: style=[metallic, plastic, shiny, glossy, ambient, off]
arr = vslice.pointdata[0] # retrieve vertex array data

plt = Plotter(axes=0, bg='k', bg2='bb')
#plt.use_depth_peeling()
plt.add_callback('as my mouse moves please call', func) # be kind to vedo ;)
plt.show(vslice, __doc__)#少加点就行了...

plt.close()