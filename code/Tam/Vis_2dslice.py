import itk
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


#还是没有实现和感兴趣区域一起切割,但是准备拿来进小框
class Slice_2D:
    def __init__(self,label_path,ori_path):
        pass

    def get_volume(self, label_path, ori_path):#数据
        self.label_path = label_path
        self.ori_path = ori_path

        #注意这里改进一下图片读取，允许读错或者不读

        itk_img = itk.imread(filename=ori_path)
        data1 = itk.vtk_image_from_image(l_image=itk_img)
        self.vol = Volume(data1)
        itk_img = itk.imread(filename=self.ori_path_2)
        data11 = itk.vtk_image_from_image(l_image=itk_img)
        self.vol11 = Volume(data11)
        itk_img = itk.imread(filename=self.ori_path_3)
        data12 = itk.vtk_image_from_image(l_image=itk_img)
        self.vol12 = Volume(data12)
        itk_img = itk.imread(filename=self.ori_path_4)
        data13 = itk.vtk_image_from_image(l_image=itk_img)
        self.vol13 = Volume(data13)

        itk_img = itk.imread(filename=label_path)
        data2 = itk.vtk_image_from_image(l_image=itk_img)
        self.vol2 = Volume(data2).cmap("gist_stern_r")

        # 修改贴图位置
        # 注意！在生成灰度分布时要注意QGridLayout的继承容器一定要注明否则会不显示！！
        self.vtkWidget = QVTKRenderWindowInteractor()
        self.gridLayout_00.addWidget(self.vtkWidget)
        self.vtkWidget_2 = QVTKRenderWindowInteractor()
        self.gridLayout_02.addWidget(self.vtkWidget_2)
        self.vtkWidget_3 = QVTKRenderWindowInteractor()
        self.gridLayout_03.addWidget(self.vtkWidget_3)
        self.vtkWidget_4 = QVTKRenderWindowInteractor()
        self.gridLayout_04.addWidget(self.vtkWidget_4)

        self.vtkWidget_5 = QVTKRenderWindowInteractor()
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
        self.gridLayout_016.addWidget(self.vtkWidget_16)
        #self.plt = Plotter(qt_widget=self.vtkWidget_16)
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
        #plt.close()

        Slice_2D.Vis_3D(self,self.vol,self.vtkWidget)
        Slice_2D.Vis_2D(self,self.vol,self.vtkWidget_2,self.vtkWidget_3,self.vtkWidget_4)

        Slice_2D.Vis_3D(self, self.vol11,self.vtkWidget_5)
        Slice_2D.Vis_2D(self, self.vol11,self.vtkWidget_6,self.vtkWidget_7,self.vtkWidget_8)

        Slice_2D.Vis_3D(self, self.vol12, self.vtkWidget_9)
        Slice_2D.Vis_2D(self, self.vol12, self.vtkWidget_10, self.vtkWidget_11, self.vtkWidget_12)

        Slice_2D.Vis_3D(self, self.vol13, self.vtkWidget_13)
        Slice_2D.Vis_2D(self, self.vol13, self.vtkWidget_14, self.vtkWidget_15, self.vtkWidget_16)


    def Vis_3D(self,vol,vtkWidget):#3D区可视化
        plt = Plotter(axes=8, bg="k", bg2="bb", interactive=False,qt_widget=vtkWidget)  # N:desired renderers,可以qt_windows,sharecam,
        plt.show(vol, self.vol2, __doc__, zoom=1.5)#self.vol11,self.vol12,self.vol13,
        print("Succeed!")

    def Vis_2D(self,vol,vtkWidgetx,vtkWidgety,vtkWidgetz):
        width, height = self.dia.graphicsView_2.width(), self.dia.graphicsView_2.height()
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
