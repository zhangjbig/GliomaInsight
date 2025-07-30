import itk,os
import numpy as np
from vedo import Volume, Plotter, Text2D, BoxCutter
from vedo.applications import Slicer3DPlotter
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from Script.mySlice import Slicer3DPlotterx
from Script.mySlice import Slicer3DPlottery
from Script.mySlice import Slicer3DPlotterz
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor, vtkRenderer, vtkRenderWindow
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import QTimer

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import QSizePolicy
import vtk

#还是没有实现和感兴趣区域一起切割,但是准备拿来进小框
class Slice_2D:
    def __init__(self,dia):
        self.dia = dia

    def get_volume(self, label_path, ori_path):#数据
        print("执行get_volume")
        self.label_path = label_path
        self.ori_path = ori_path
        ori_path = 'exampleSettings/BraTS2021_00495/00000209_brain_flair.nii'
        ori_path_2 = 'exampleSettings/BraTS2021_00495/00000209_brain_t1.nii'
        ori_path_3 = 'exampleSettings/BraTS2021_00495/00000209_brain_t1ce.nii'
        ori_path_4 = 'exampleSettings/BraTS2021_00495/00000209_brain_t2.nii'

        #注意这里改进一下图片读取，允许读错或者不读


        itk_img = itk.imread(filename=ori_path)
        data1 = itk.vtk_image_from_image(l_image=itk_img)
        self.vol = Volume(data1)
        itk_img = itk.imread(filename=ori_path_2)
        data11 = itk.vtk_image_from_image(l_image=itk_img)
        self.vol11 = Volume(data11)
        itk_img = itk.imread(filename=ori_path_3)
        data12 = itk.vtk_image_from_image(l_image=itk_img)
        self.vol12 = Volume(data12)
        itk_img = itk.imread(filename=ori_path_4)
        data13 = itk.vtk_image_from_image(l_image=itk_img)
        self.vol13 = Volume(data13)

        itk_img = itk.imread(filename=label_path)
        data2 = itk.vtk_image_from_image(l_image=itk_img)
        self.vol2 = Volume(data2).cmap("gist_stern_r")

        # 修改贴图位置
        # 注意！在生成灰度分布时要注意QGridLayout的继承容器一定要注明否则会不显示！！
        self.vtkWidget = QVTKRenderWindowInteractor()
        self.dia.gridLayout_00.addWidget(self.vtkWidget)

        self.vtkWidget.setMinimumSize(300, 300)
        self.vtkWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.vtkWidget.Initialize()
        self.vtkWidget.Start()
        self.vtkWidget.show()  # 非常关键！

        # 强制渲染一帧
        self.vtkWidget.GetRenderWindow().Render()

        # 激活布局区域
        parent_widget = self.dia.gridLayout_00.parentWidget()
        parent_widget.setStyleSheet("border: 1px solid red")  # 触发布局
        self.dia.update()
        self.dia.repaint()

        '''self.vtkWidget = QVTKRenderWindowInteractor()
        print("确认 layout 是否真的存在")
        print(f"gridLayout_00能够查找到：{self.dia.gridLayout_00}")
        print("gridLayout_00已绑定成功")
        self.dia.gridLayout_00.addWidget(self.vtkWidget)
        self.vtkWidget.setMinimumSize(300, 300)
        self.vtkWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vtkWidget.Initialize()
        self.vtkWidget.Start()
        self.vtkWidget.setStyleSheet("background-color: red;")
        parent_widget = self.dia.gridLayout_00.parentWidget()
        print(f"parent widget of gridLayout_00: {parent_widget}")
        parent_widget.setStyleSheet("border: 2px solid red;")'''


        # 给 layout 添加一个 QLabel 验证布局是否有效
        '''label = QLabel("Test Label")
        label.setStyleSheet("background-color: yellow; font-size: 18px;")
        self.dia.gridLayout_00.addWidget(label)
        label.show()'''


        print("第一个窗口ok")
        '''self.vtkWidget = QVTKRenderWindowInteractor()
        gridLayout_00 = getattr(self.dia, "gridLayout_00")
        gridLayout_00.addWidget(self.vtkWidget)'''
        '''self.vtkWidget_2 = QVTKRenderWindowInteractor()
        self.dia.gridLayout_02.addWidget(self.vtkWidget_2)
        self.vtkWidget_2.Initialize()
        self.vtkWidget_2.Start()
        print("第2个窗口ok")
        self.vtkWidget_3 = QVTKRenderWindowInteractor()
        self.dia.gridLayout_03.addWidget(self.vtkWidget_3)
        self.vtkWidget_4 = QVTKRenderWindowInteractor()
        self.dia.gridLayout_04.addWidget(self.vtkWidget_4)
        print("第3 4个窗口ok")'''

        '''self.vtkWidget_5 = QVTKRenderWindowInteractor()
        self.gridLayout_05.addWidget(self.vtkWidget_5)
        self.vtkWidget_6 = QVTKRenderWindowInteractor()
        self.gridLayout_06.addWidget(self.vtkWidget_6)
        self.vtkWidget_7 = QVTKRenderWindowInteractor()
        self.gridLayout_07.addWidget(self.vtkWidget_7)
        self.vtkWidget_8 = QVTKRenderWindowInteractor()
        self.gridLayout_08.addWidget(self.vtkWidget_8)

        self.vtkWidget_9 = QVTKRenderWindowInteractor()
        self.gridLayout_09.addWidget(self.vtkWidget_9)
        self.vtkWidget_10 = QVTKRenderWindowInteractor()
        self.gridLayout_010.addWidget(self.vtkWidget_10)
        self.vtkWidget_11 = QVTKRenderWindowInteractor()
        self.gridLayout_011.addWidget(self.vtkWidget_11)
        self.vtkWidget_12 = QVTKRenderWindowInteractor()
        self.gridLayout_012.addWidget(self.vtkWidget_12)

        self.vtkWidget_13 = QVTKRenderWindowInteractor()
        self.gridLayout_013.addWidget(self.vtkWidget_13)
        self.vtkWidget_14 = QVTKRenderWindowInteractor()
        self.gridLayout_014.addWidget(self.vtkWidget_14)
        self.vtkWidget_15 = QVTKRenderWindowInteractor()
        self.gridLayout_015.addWidget(self.vtkWidget_15)
        self.vtkWidget_16 = QVTKRenderWindowInteractor()
        self.gridLayout_016.addWidget(self.vtkWidget_16)'''
        '''#self.plt = Plotter(qt_widget=self.vtkWidget_16)
        # itk_img = itk.imread(filename="Test_data/BraTS-GLI-0009_0003.nii.gz")
        # print("worked!")
        # vtk_img = itk.vtk_image_from_image(l_image=itk_img)
        # # print(vtk_img)
        # # vol = Volume(vtk_img)
        # # mesh = vol.isosurface()
        # # self.cutter = BoxCutter(mesh)
        # # self.plt += [mesh, self.cutter]
        # # self.plt.show()
        # # Extract vtkImageData contour to vtkPolyData
        # contour = vtkDiscreteMarchingCubes()
        # contour.SetInputData(vtk_img)
        #
        # # Define colors, mapper, actor, renderer, renderWindow, renderWindowInteractor
        # colors = vtkNamedColors()
        #
        # mapper = vtkPolyDataMapper()
        # mapper.SetInputConnection(contour.GetOutputPort())
        #
        # actor = vtkActor()
        # actor.SetMapper(mapper)
        #
        # renderer = vtkRenderer()
        # renderer.AddActor(actor)
        # renderer.SetBackground(colors.GetColor3d("SteelBlue"))
        #
        # renderWindow = vtkRenderWindow()
        # renderWindow.AddRenderer(renderer)
        # self.vtkWidget_16.SetRenderWindow(renderWindow)
        # self.vtkWidget_16.Initialize()
        # self.vtkWidget_16.Start()

        # plt = Slicer3DPlotter(
        #     self.vol,
        #     cmaps=("gist_ncar_r", "jet", "Spectral_r", "hot_r", "bone_r"),
        #     use_slider3d=False,
        #     bg="white",
        #     bg2="blue9",
        #     qt_widget=self.vtkWidget_13
        # )
        # print("plt!!!!!!!!!")
        # # Can now add any other vedo object to the Plotter scene:
        # # plt += Text2D(__doc__)
        # plt.show(viewup='z')
        #plt.close()'''

        print("准备执行Vis_3D和Vis_2D")
        Slice_2D.Vis_3D(self,self.vol,self.vtkWidget)
        #Slice_2D.Vis_2D(self,self.vol,self.vtkWidget_2,self.vtkWidget_3,self.vtkWidget_4)


        '''Slice_2D.Vis_3D(self, self.vol11,self.vtkWidget_5)
        Slice_2D.Vis_2D(self, self.vol11,self.vtkWidget_6,self.vtkWidget_7,self.vtkWidget_8)


        Slice_2D.Vis_3D(self, self.vol12, self.vtkWidget_9)
        Slice_2D.Vis_2D(self, self.vol12, self.vtkWidget_10, self.vtkWidget_11, self.vtkWidget_12)


        Slice_2D.Vis_3D(self, self.vol13, self.vtkWidget_13)
        Slice_2D.Vis_2D(self, self.vol13, self.vtkWidget_14, self.vtkWidget_15, self.vtkWidget_16)
'''
        print("get_volume执行结束")


    def Vis_3D(self,vol,vtkWidget):#3D区可视化
        plt = Plotter(axes=8, bg="k", bg2="bb", interactive=False,qt_widget=vtkWidget)  # N:desired renderers,可以qt_windows,sharecam,
        plt.show(vol, self.vol2, __doc__, zoom=1.5)#self.vol11,self.vol12,self.vol13,
        plt.screenshot("result/img/3d_check.png")  # 保存截图
        print("3D 渲染成功！")
        # ???????这里执行出错了？？？



    def Vis_2D(self,vol,vtkWidgetx,vtkWidgety,vtkWidgetz):
        width, height = self.dia.graphicsView_2.width(), self.dia.graphicsView_2.height()
        print("width:",width,"; height:",height)
        #self.vol2.cmap('jet')
        plt_2 = Slicer3DPlotterz(
            vol,
            bg="white",
            bg2="blue9",
            qt_widget=vtkWidgetz
        )
        plt_2.show(
            #self.vol2,
            axes=0,
            size=(width, height),
            interactive=True,
            viewup='z'
        )
        plt_3 = Slicer3DPlotterx(
            vol,
            bg="white",
            bg2="blue9",
            qt_widget=vtkWidgetx
        )
        plt_3.show(
            #self.vol2,
            axes=0,
            size=(width, height),
            interactive=True,
            viewup='z')
        plt_4 = Slicer3DPlottery(
            vol,
            bg="white",
            bg2="blue9",
            qt_widget=vtkWidgety
        )
        plt_4.show(
            #self.vol2,
            axes=0,
            size=(width, height),
            interactive=True,
            viewup='z')
        plt_2.screenshot("result/img/z方向切片.png")
        plt_3.screenshot("result/img/x方向切片.png")
        plt_4.screenshot("result/img/y方向切片.png")


    '''def show11(self):
        """按下按钮时调用这个函数，它会延迟300ms后再执行渲染"""
        print("准备延迟加载 VTK 控件...")
        QTimer.singleShot(300, lambda: self.show11_delayed)  # 延迟300ms调用真正的显示函数

    def show11_delayed(self):
        """这个是真正执行图像渲染和UI绑定的部分"""
        print("开始渲染")

        ori_path = 'exampleSettings/BraTS2021_00495/00000209_brain_flair.nii'
        nii_path = 'exampleSettings/BraTS2021_00495/BraTS2021_00495_seg.nii'


        ori_image = itk.imread(ori_path)
        label_image = itk.imread(nii_path)

        vtk_ori = itk.vtk_image_from_image(ori_image)
        vtk_label = itk.vtk_image_from_image(label_image)

        vol = Volume(vtk_ori)
        vol2 = Volume(vtk_label).cmap("gist_stern_r")

        # 创建控件
        self.vtkWidget = QVTKRenderWindowInteractor()
        self.vtkWidget.setMinimumSize(300, 300)
        self.vtkWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dia.gridLayout_00.addWidget(self.vtkWidget)
        self.vtkWidget.Initialize()
        self.vtkWidget.Start()
        self.vtkWidget.show()

        # 创建 Plotter
        plt = Plotter(axes=8, bg="k", bg2="bb", interactive=False, qt_widget=self.vtkWidget)
        plt.show(vol, vol2, zoom=1.5)

        # 保存截图（用于确认）
        os.makedirs("result/img", exist_ok=True)
        plt.screenshot("result/img/show11delay_3d_check.png")
        print("3D 渲染完成并截图")

        # 强制刷新渲染窗口
        self.vtkWidget.GetRenderWindow().Render()
        self.vtkWidget.update()'''

    def show11(self):
        ori_path = 'exampleSettings/BraTS2021_00495/00000209_brain_flair.nii'
        nii_path = 'exampleSettings/BraTS2021_00495/BraTS2021_00495_seg.nii'

        print("开始渲染")
        ori_image = itk.imread(ori_path)
        label_image = itk.imread(nii_path)

        vtk_ori = itk.vtk_image_from_image(ori_image)
        vtk_label = itk.vtk_image_from_image(label_image)

        vol = Volume(vtk_ori)
        vol2 = Volume(vtk_label).cmap("gist_stern_r")
        plt = Plotter(bg='black', axes=8, interactive=True)
        plt.show(vol, vol2, zoom=1.2)

        plt_2 = Slicer3DPlotterz(
            vol,
            bg="white",
            bg2="blue9"
        )
        plt_2.show(
            # self.vol2,
            axes=0,
            size=(300, 300),
            interactive=True,
            viewup='z'
        )

        '''        # ✅ 显式设置父控件为 page7
        print("显式设置父控件为 page7")
        self.vtkWidget = QVTKRenderWindowInteractor(parent=self.dia.page_7)
        self.vtkWidget.setMinimumSize(300, 300)
        self.vtkWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dia.gridLayout_00.addWidget(self.vtkWidget)

        self.vtkWidget.Initialize()
        self.vtkWidget.Start()
        self.vtkWidget.show()

        plt = Plotter(axes=8, bg="k", bg2="bb", interactive=False, qt_widget=self.vtkWidget)
        plt.show(vol, vol2, zoom=1.5)

        os.makedirs("result/img", exist_ok=True)
        plt.screenshot("result/img/show11_parent_3d_check.png")
        print("渲染成功，图像已保存 result/img/show11_parent_3d_check.png")

        self.vtkWidget.GetRenderWindow().Render()
        self.vtkWidget.update()
        '''
    def show12(self):
        self.vtkWidget = QVTKRenderWindowInteractor()
        self.vtkWidget.setMinimumSize(300, 300)
        self.vtkWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dia.gridLayout_00.addWidget(self.vtkWidget)

        self.vtkWidget.Initialize()
        self.vtkWidget.Start()
        self.vtkWidget.show()



        # 原生 VTK 渲染器 + 红色球体
        renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(renderer)

        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(50)
        sphere.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)  # 红色

        renderer.AddActor(actor)
        renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

        print("✅ VTK 原生红球已尝试渲染")

    def show1label(self):
        label = QLabel("Test Label")
        label.setStyleSheet("background-color: yellow; font-size: 18px;")
        self.dia.gridLayout_00.addWidget(label)
        label.show()
