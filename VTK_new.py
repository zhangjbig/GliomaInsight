
from Script.mySlice import Slicer3DPlotterx
from Script.mySlice import Slicer3DPlottery
from Script.mySlice import Slicer3DPlotterz


from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QSizePolicy,QApplication
from vedo import Plotter, Sphere, Volume
import itk,os

class SimpleView():
    def __init__(self):
        pass

    def VisNii(self,label_path,ori_path):
        try:
            print(f"label_path:{label_path}")
            print(f"ori_path:{ori_path}")

        except RuntimeError:
            print("VTK——new中get_volume执行出错")



class VTKRenderer:
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
        '''plt_2.screenshot("result/img/z方向切片.png")
        plt_3.screenshot("result/img/x方向切片.png")
        plt_4.screenshot("result/img/y方向切片.png")'''


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



class VTKRendererTam:
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
        '''plt_2.screenshot("result/img/z方向切片.png")
        plt_3.screenshot("result/img/x方向切片.png")
        plt_4.screenshot("result/img/y方向切片.png")'''


class VTKTam(QMainWindow):
    def __init__(self, ori_path=None, label_path=None):
        super().__init__()

        # 加载 .ui 文件
        #uic.loadUi("testui.ui", self)  # 加载UI文件
        self.dia = uic.loadUi("tamUI.ui", self)
        title = "Enrichment Degree of Angio-tams"
        self.setWindowTitle(title)
        if not ori_path or not label_path:
            raise ValueError("必须提供 ori_path 和 label_path")

        # 1. 初始化 VTK 渲染器
        self.vtk_renderer = VTKRendererTam(self.dia, self.gridLayout_vtk)  # 用self.dia作为UI对象

        # 2. 3D图像渲染
        #ori_path = 'exampleSettings/BraTS2021_00495/00000209_brain_flair.nii'
        #label_path = 'exampleSettings/BraTS2021_00495/BraTS2021_00495_seg.nii'

        image = itk.imread(ori_path)
        vtk_image = itk.vtk_image_from_image(image)
        vol = Volume(vtk_image)

        label = itk.imread(label_path)
        #测试
        array = itk.GetArrayFromImage(label)
        print("Label 最大值：", array.max())  # 如果是 0，说明 label 全部是背景

        vtk_label = itk.vtk_image_from_image(label)
        vol2 = Volume(vtk_label).cmap({0: "white", 1: "red"}).alpha([0, 1])  # 0 全透明，1 不透明红
        #vol2 = Volume(vtk_label).cmap("gist_stern_r")
        

        # 3D
        self.vtk_renderer.render_3d_volume(vol, vol2)

        # 3. 2D渲染切片
        #self.vtk_renderer.render_2d_slices(self.dia, vol)