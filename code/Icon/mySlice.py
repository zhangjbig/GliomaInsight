from vedo import Volume, show#, isosurface
import itk
from vedo.applications import RayCastPlotter

# 加载 NIfTI 文件
nifti_file_name = "D:/www/xxx/tumorTest/UI/BraTS2021_00495/BraTS2021_00495_flair.nii.gz"
# Read NIFTI file
itk_img = itk.imread(filename=nifti_file_name)

# Convert itk to vtk
vtk_img = itk.vtk_image_from_image(l_image=itk_img)
vol = Volume(vtk_img)

# 渲染 3D 图像
vol.alpha([0, 0.3, 0.6, 1])  # 控制透明度
vol.cmap("gray")            # 设置灰度图

# 显示交互窗口
show(vol, axes=1, bg="white")

plt = RayCastPlotter(vol, bg='black', bg2='blackboard', axes=7)
plt.show(viewup="z")
plt.close()

from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer, \
    vtkRenderWindow, vtkRenderWindowInteractor


def show_3d_nifti_image(nifti_file_name):

    # Read NIFTI file
    itk_img = itk.imread(filename=nifti_file_name)

    # Convert itk to vtk
    vtk_img = itk.vtk_image_from_image(l_image=itk_img)

    # Extract vtkImageData contour to vtkPolyData
    contour = vtkDiscreteMarchingCubes()
    contour.SetInputData(vtk_img)

    # Define colors, mapper, actor, renderer, renderWindow, renderWindowInteractor
    colors = vtkNamedColors()

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(contour.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)

    renderer = vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(colors.GetColor3d("SteelBlue"))

    renderWindow = vtkRenderWindow()
    renderWindow.AddRenderer(renderer)

    renderWindowInteractor = vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.Initialize()
    renderWindowInteractor.Start()

if __name__ == '__main__':
    show_3d_nifti_image("D:/www/xxx/tumorTest/UI/BraTS2021_00495/BraTS2021_00495_flair.nii.gz")
