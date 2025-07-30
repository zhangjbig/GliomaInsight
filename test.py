from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QSizePolicy, QVBoxLayout, QWidget
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import Plotter, Sphere, Volume
import sys, os
import itk

from Script.mySlice import Slicer3DPlotterx
from Script.mySlice import Slicer3DPlottery
from Script.mySlice import Slicer3DPlotterz

class VTKDemo(QMainWindow):
    def __init__(self):
        super().__init__()

        # 加载 .ui 文件
        uic.loadUi("modal_flair.ui", self)  # 你保存的 .ui 文件

        # 获取 gridLayout_vtk
        layout = self.gridLayout_vtk# 这是 QGridLayout

        #2.3d医学图像显示成功
        self.vtkWidget = QVTKRenderWindowInteractor(parent=self.vtkContainer)
        self.vtkWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vtkWidget.setMinimumSize(400, 400)
        layout.addWidget(self.vtkWidget)

        self.vtkWidget.Initialize()
        self.vtkWidget.Start()
        self.vtkWidget.show()

        # ====== 使用 Vedo 渲染一个 3D Volume ======
        # 1. 读取体积数据
        ori_path = 'exampleSettings/BraTS2021_00495/00000209_brain_flair.nii'  # 你的路径
        image = itk.imread(ori_path)
        vtk_image = itk.vtk_image_from_image(image)
        vol = Volume(vtk_image)

        label_path = 'exampleSettings/BraTS2021_00495/BraTS2021_00495_seg.nii'
        itk_img = itk.imread(filename=label_path)
        data2 = itk.vtk_image_from_image(l_image=itk_img)
        vol2 = Volume(data2).cmap("gist_stern_r")

        # 2. 使用 Plotter 渲染
        plt = Plotter(axes=8, bg="k", bg2="bb", interactive=False,
                      qt_widget=self.vtkWidget)  # N:desired renderers,可以qt_windows,sharecam,
        plt.show(vol, vol2, zoom=1.5)  # self.vol11,self.vol12,self.vol13,
        #plt.screenshot("result/img/a3d_check.png")

        # 强制刷新渲染窗口
        self.vtkWidget.GetRenderWindow().Render()
        self.vtkWidget.update()


        # 创建切片显示用的 vtkWidgets，并添加到 UI 中
        self.slice_widget_x = QVTKRenderWindowInteractor(parent=self.vtkContainerX)  # 或 parent=self.vtkContainerX
        self.gridLayout_vtk_5.addWidget(self.slice_widget_x)

        self.slice_widget_y = QVTKRenderWindowInteractor(parent=self.vtkContainerY)
        self.gridLayout_vtk_6.addWidget(self.slice_widget_y)

        self.slice_widget_z = QVTKRenderWindowInteractor(parent=self.vtkContainerZ)
        self.gridLayout_vtk_7.addWidget(self.slice_widget_z)

        # 初始化 vtkWidgets
        for w in [self.slice_widget_x, self.slice_widget_y, self.slice_widget_z]:
            w.Initialize()
            w.Start()
            w.show()

        self.Vis_2D(vol)

    def Vis_2D(self, vol):
        # xyz方向切片渲染
        plt_2 = Slicer3DPlotterz(
            vol,
            bg="white",
            bg2="blue9",
            qt_widget=self.slice_widget_z
        )
        plt_2.show(axes=0, size=(100,30), interactive=True, viewup='z')

        plt_3 = Slicer3DPlotterx(
            vol,
            bg="white",
            bg2="blue9",
            qt_widget=self.slice_widget_x
        )
        plt_3.show(axes=0, size=(100,30), interactive=True, viewup='x')

        plt_4 = Slicer3DPlottery(
            vol,
            bg="white",
            bg2="blue9",
            qt_widget=self.slice_widget_y
        )
        plt_4.show(axes=0, size=(100,30), interactive=True, viewup='y')

        # 截图保存
        '''plt_2.screenshot("result/img/z方向切片.png")
        plt_3.screenshot("result/img/x方向切片.png")
        plt_4.screenshot("result/img/y方向切片.png")'''


'''class VTKRenderer:
    def __init__(self, parent, layout):
        """
        初始化 VTK 渲染器并将其添加到布局中

        :param parent: 父控件（通常是一个 QWidget）
        :param layout: 用于添加 VTK 控件的布局（QGridLayout）
        """
        self.parent = parent
        self.layout = layout

        # 创建 VTK 渲染控件
        self.vtkWidget = QVTKRenderWindowInteractor(parent=self.parent)
        self.vtkWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vtkWidget.setMinimumSize(400, 400)

        # 将 VTK 控件添加到布局
        self.layout.addWidget(self.vtkWidget)

        # 初始化 VTK 控件
        self.vtkWidget.Initialize()
        self.vtkWidget.Start()
        self.vtkWidget.show()

    def render_3d_volume(self, vol, vol2):
        """使用 Vedo 渲染 3D Volume"""
        plt = Plotter(axes=8, bg="k", bg2="bb", interactive=False, qt_widget=self.vtkWidget)
        plt.show(vol, vol2, zoom=1.5)
        #plt.screenshot("result/img/a3d_check.png")
        self.vtkWidget.GetRenderWindow().Render()
        self.vtkWidget.update()

    def render_2d_slices(self, dia, vol):
        self.dia = dia
        """显示 XYZ 方向切片"""
        #模态1
        self.slice_widget_x = QVTKRenderWindowInteractor(parent=self.dia.vtkContainerX)  # 或 parent=self.vtkContainerX
        self.dia.gridLayout_vtk_5.addWidget(self.slice_widget_x)
        self.slice_widget_y = QVTKRenderWindowInteractor(parent=self.dia.vtkContainerY)
        self.dia.gridLayout_vtk_6.addWidget(self.slice_widget_y)
        self.slice_widget_z = QVTKRenderWindowInteractor(parent=self.dia.vtkContainerZ)
        self.dia.gridLayout_vtk_7.addWidget(self.slice_widget_z)

        # 初始化切片控件
        for w in [self.slice_widget_x, self.slice_widget_y, self.slice_widget_z]:
            w.Initialize()
            w.Start()
            w.show()


        # 渲染切片
        plt_2 = Slicer3DPlotterz(vol, bg="white", bg2="blue9", qt_widget=self.slice_widget_z)
        plt_2.show(axes=0, size=(100,30), interactive=True, viewup='z')

        plt_3 = Slicer3DPlotterx(vol, bg="white", bg2="blue9", qt_widget=self.slice_widget_x)
        plt_3.show(axes=0, size=(100,30), interactive=True, viewup='x')

        plt_4 = Slicer3DPlottery(vol, bg="white", bg2="blue9", qt_widget=self.slice_widget_y)
        plt_4.show(axes=0, size=(100,30), interactive=True, viewup='y')


        # 截图保存
        plt_2.screenshot("result/img/z方向切片.png")
        plt_3.screenshot("result/img/x方向切片.png")
        plt_4.screenshot("result/img/y方向切片.png")


class VTKDemo(QMainWindow):
    def __init__(self, title="医学图像查看器", ori_path=None, label_path=None):
        super().__init__()

        # 加载 .ui 文件
        #uic.loadUi("testui.ui", self)  # 加载UI文件
        self.dia = uic.loadUi("modal_flair.ui", self)
        #title = "医学图像查看器：flair模态"
        self.setWindowTitle(title)
        if not ori_path or not label_path:
            raise ValueError("必须提供 ori_path 和 label_path")

        # 1. 初始化 VTK 渲染器
        self.vtk_renderer = VTKRenderer(self.dia, self.gridLayout_vtk)  # 用self.dia作为UI对象

        # 2. 3D图像渲染
        #ori_path = 'exampleSettings/BraTS2021_00495/00000209_brain_flair.nii'
        #label_path = 'exampleSettings/BraTS2021_00495/BraTS2021_00495_seg.nii'

        image = itk.imread(ori_path)
        vtk_image = itk.vtk_image_from_image(image)
        vol = Volume(vtk_image)

        label = itk.imread(label_path)
        vtk_label = itk.vtk_image_from_image(label)
        vol2 = Volume(vtk_label).cmap("gist_stern_r")

        # 3D
        self.vtk_renderer.render_3d_volume(vol, vol2)

        # 3. 渲染切片
        self.vtk_renderer.render_2d_slices(self.dia, vol)


class VTKDemo2(QMainWindow):
    def __init__(self):
        super().__init__()

        # 加载 .ui 文件
        #uic.loadUi("testui.ui", self)  # 加载UI文件
        self.dia = uic.loadUi("modal_flair.ui", self)
        self.setWindowTitle("医学图像查看器：t1模态")

        # 1. 初始化 VTK 渲染器
        self.vtk_renderer = VTKRenderer(self.dia, self.gridLayout_vtk)  # 用self.dia作为UI对象

        # 2. 3D图像渲染
        ori_path = 'exampleSettings/BraTS2021_00495/00000209_brain_t1.nii'
        label_path = 'exampleSettings/BraTS2021_00495/BraTS2021_00495_seg.nii'

        image = itk.imread(ori_path)
        vtk_image = itk.vtk_image_from_image(image)
        vol = Volume(vtk_image)

        label = itk.imread(label_path)
        vtk_label = itk.vtk_image_from_image(label)
        vol2 = Volume(vtk_label).cmap("gist_stern_r")

        self.vtk_renderer.render_3d_volume(vol, vol2)

        # 3. 渲染切片
        self.vtk_renderer.render_2d_slices(self.dia, vol)


class VTKDemo3(QMainWindow):
    def __init__(self):
        super().__init__()

        # 加载 .ui 文件
        #uic.loadUi("testui.ui", self)  # 加载UI文件
        self.dia = uic.loadUi("modal_flair.ui", self)
        self.setWindowTitle("医学图像查看器：t1ce模态")

        # 1. 初始化 VTK 渲染器
        self.vtk_renderer = VTKRenderer(self.dia, self.gridLayout_vtk)  # 用self.dia作为UI对象

        # 2. 3D图像渲染
        ori_path = 'exampleSettings/BraTS2021_00495/00000209_brain_t1ce.nii'
        label_path = 'exampleSettings/BraTS2021_00495/BraTS2021_00495_seg.nii'

        image = itk.imread(ori_path)
        vtk_image = itk.vtk_image_from_image(image)
        vol = Volume(vtk_image)

        label = itk.imread(label_path)
        vtk_label = itk.vtk_image_from_image(label)
        vol2 = Volume(vtk_label).cmap("gist_stern_r")

        self.vtk_renderer.render_3d_volume(vol, vol2)

        # 3. 渲染切片
        self.vtk_renderer.render_2d_slices(self.dia, vol)


class VTKDemo4(QMainWindow):
    def __init__(self):
        super().__init__()

        # 加载 .ui 文件
        #uic.loadUi("testui.ui", self)  # 加载UI文件
        self.dia = uic.loadUi("modal_flair.ui", self)
        self.setWindowTitle("医学图像查看器：t2模态")

        # 1. 初始化 VTK 渲染器
        self.vtk_renderer = VTKRenderer(self.dia, self.gridLayout_vtk)  # 用self.dia作为UI对象

        # 2. 3D图像渲染
        ori_path = 'exampleSettings/BraTS2021_00495/00000209_brain_t2.nii'
        label_path = 'exampleSettings/BraTS2021_00495/BraTS2021_00495_seg.nii'

        image = itk.imread(ori_path)
        vtk_image = itk.vtk_image_from_image(image)
        vol = Volume(vtk_image)

        label = itk.imread(label_path)
        vtk_label = itk.vtk_image_from_image(label)
        vol2 = Volume(vtk_label).cmap("gist_stern_r")

        self.vtk_renderer.render_3d_volume(vol, vol2)

        # 3. 渲染切片
        self.vtk_renderer.render_2d_slices(self.dia, vol)

'''
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VTKDemo()
    window.show()
    sys.exit(app.exec_())





