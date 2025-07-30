import itk
from vedo import Volume
from mySlice import Slicer3DPlotterz

#还是没有实现和感兴趣区域一起切割,但是准备拿来进小框
class Slice_2D():
    def __init__(self,label_path,ori_path):
        self.label_path = label_path
        self.ori_path = ori_path
    def get_volume(self):
        itk_img = itk.imread(filename=self.ori_path)
        data1 = itk.vtk_image_from_image(l_image=itk_img)
        self.vol = Volume(data1)
        itk_img = itk.imread(filename=self.label_path)
        data2 = itk.vtk_image_from_image(l_image=itk_img)
        self.vol2 = Volume(data2)
        self.plt = Slicer3DPlotterz(
            self.vol,
            cmaps=("bone", "bone_r"),
            bg="white",
            bg2="blue9",
        )



plt.show(
    vol2,
    axes=0,
    size=(300,300),
    interactive=True,
    viewup='z')
plt.close()

#image saving
"""
sphere = Sphere().linewidth(1)
plt = show(sphere, interactive=False)
plt.screenshot('image.png')
plt.interactive()
plt.close()
"""