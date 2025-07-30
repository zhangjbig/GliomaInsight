import itk
from vedo import dataurl, precision, Sphere, Volume, Plotter,show

import sys
# from PySide2 import QtWidgets, QtCore
from PyQt5 import Qt
from vedo.applications import Slicer3DPlotter, RayCastPlotter
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import Plotter, Mesh, BoxCutter, dataurl

class MainWindow(Qt.QMainWindow):

    def __init__(self, parent=None):

        Qt.QMainWindow.__init__(self, parent)
        self.frame = Qt.QFrame()
        self.layout = Qt.QGridLayout()
        self.vtkWidget = QVTKRenderWindowInteractor()

        # Create renderer and add the vedo objects and callbacks
        self.plt = Plotter(qt_widget=self.vtkWidget)
        mesh = Mesh(dataurl+'cow.vtk')
        # 导入图片
        itk_img = itk.imread(filename="./MEN_001_0003.nii")
        vtk_img = itk.vtk_image_from_image(l_image=itk_img)
        vol = Volume(vtk_img)
        mesh = vol.isosurface()
        vol.show()

        # self.cutter = BoxCutter(mesh)
        # self.plt += [mesh, self.cutter]
        # self.plt.show()

        plt = Slicer3DPlotter(
            vol,
            cmaps=("gist_ncar_r", "jet", "Spectral_r", "hot_r", "bone_r"),
            use_slider3d=False,
            bg="white",
            bg2="blue9",
            qt_widget=self.vtkWidget
        )
        vol.mode(1).cmap("jet")
        plt = RayCastPlotter(vol, bg='black', bg2='blackboard', axes=7)
        print("plt!!!!!!!!!")
        # Can now add any other vedo object to the Plotter scene:
        # plt += Text2D(__doc__)
        #plt.show(viewup='z')

        box_cutter_button_on = Qt.QPushButton("Start the box cutter")
        box_cutter_button_on.clicked.connect(self.ctool_start)

        box_cutter_button_off = Qt.QPushButton("Stop the box cutter")
        box_cutter_button_off.clicked.connect(self.ctool_stop)

        # Set-up the rest of the Qt window
        self.layout.addWidget(self.vtkWidget)
        self.layout.addWidget(box_cutter_button_on)
        self.layout.addWidget(box_cutter_button_off)
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)
        self.show()

    def ctool_start(self):
        self.cutter.on()

    def ctool_stop(self):
        self.cutter.off()

    def onClose(self):
        #Disable the interactor before closing to prevent it
        #from trying to act on already deleted items
        self.vtkWidget.close()
import itk
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer, \
    vtkRenderWindow, vtkRenderWindowInteractor

from vedo import Axes, Volume, Box, dataurl, settings, show
from vedo.pyplot import histogram

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

if __name__ == "__main__":
    itk_img = itk.imread(filename="./MEN_001_0003.nii")
    vtk_img = itk.vtk_image_from_image(l_image=itk_img)
    vol = Volume(vtk_img)
    print(vol)
    # vol.append(vol, axis='x').show().close()
    vaxes = Axes(vol, xygrid=False)

    slab = vol.slab([45, 55], axis='z', operation='mean')
    slab.cmap('Set1_r', vmin=10, vmax=80).add_scalarbar("intensity")
    # histogram(slab).show().close()  # quickly inspect it

    bbox = slab.metadata["slab_bounding_box"]
    slab.z(-bbox[5] + vol.zbounds()[0])  # move slab to the bottom

    # create a box around the slab for reference
    slab_box = Box(bbox).wireframe().c("black")

    show(__doc__, vol, slab, slab_box, vaxes, axes=14, viewup='z')


    #show_3d_nifti_image("./MEN_001_0003.nii")
    # itk_img = itk.imread(filename="./MEN_001_0003.nii")
    # vtk_img = itk.vtk_image_from_image(l_image=itk_img)
    # vol = Volume(vtk_img)
    # vol.show()
    # app = Qt.QApplication(sys.argv)
    # window = MainWindow()
    # app.aboutToQuit.connect(window.onClose) # <-- connect the onClose event
    # app.exec_()