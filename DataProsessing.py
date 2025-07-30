import pandas as pd
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import *
from qtpy import uic

import matplotlib
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import itk
from vedo import *
from vedo import Volume, Plotter, PlaneCutter, Sphere, precision
#用来做数据处理

matplotlib.use("Qt5Agg")  # 声明使用QT5

class child(QDialog):
    def __init__(self,path,ori_path,ori_path_2,ori_path_3,ori_path_4):
        super().__init__()
        self.openingUI = uic.loadUi("visualization.ui", self)
        self.read(path,ori_path,ori_path_2,ori_path_3,ori_path_4)
    def read(self,path,ori_path,ori_path_2,ori_path_3,ori_path_4):
        readData.read_csv(self.openingUI,path,ori_path,ori_path_2,ori_path_3,ori_path_4)

class readData:
    def __init__(self):
        pass
    def is_number(self,s):
        try:
            float(s)
            return True
        except (TypeError, ValueError):
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False
    def read_csv(self,path,ori_path,ori_path_2,ori_path_3,ori_path_4):
        df = pd.read_csv(path)
        numdata = pd.DataFrame() #数字特征
        for idx, row in df.iterrows():  # 遍历 DataFrame
            if readData.is_number(self,row['Value']):#存储数字结果
                numdata = numdata._append({'Feature': row['Feature'], 'Value': row['Value']}, ignore_index=True)
                #numdata = numdata.append({'Value': row['Value']}, ignore_index=True)
        # 假设我们要将第一行数据作为列标题
        new_columns = numdata.iloc[0]#如果是行，使用[0]
        numdata.columns = new_columns
        #numdata = numdata.set_axis(new_columns, axis='columns')

        # 删除第一行，因为它已经作为列标题了
        numdata = numdata.drop(df.index[0])
        #df.to_csv('./lib/Estimate/demo.csv', index=False)
        #存储供生存分析使用的data
        numdata.to_csv('./lib/Estimate/demo.csv', index=False)


        #df2 = pd.read_csv('./lib/Estimate/demo.csv')
        # 将 DataFrame 写入文本文件
        #df2.to_csv('demo.txt', sep=' ', index=False)
        DataProsess.visData(self,numdata,ori_path,ori_path_2,ori_path_3,ori_path_4)

class DataProsess:
    def __init__(self):
        pass
    def visData(self,numdata,ori_path,ori_path_2,ori_path_3,ori_path_4):
        self.ori_path = ori_path
        self.numdata = numdata

        self.vtkWidget = QVTKRenderWindowInteractor()
        self.gridlayout = QGridLayout(self.graphicsView_2)
        self.gridlayout.addWidget(self.vtkWidget)

        Vis_3D.readData(self,ori_path,ori_path_2,ori_path_3,ori_path_4)

class Vis_3D(QDialog):
    def __init__(self):
        super(Vis_3D, self).__init__()

    def readImage(self,ori_path):
        itk_img = itk.imread(filename=ori_path)
        vtk_img = itk.vtk_image_from_image(l_image=itk_img)
        vol = Volume(vtk_img).cmap("gist_stern_r")
        return vol

    def geneSlice(self,ori_path):
        vol = Vis_3D.readImage(self,ori_path)
        normal = [0, 0, 1]
        vslice = vol.slice_plane(vol.center(), normal).cmap("bone")
        vslice.name = "Slice"
        return vslice,vol

    def genePcutter(self,vslice):
        normal = [0, 0, 1]
        pcutter = PlaneCutter(
            vslice,
            normal=normal,  # 平面的法线，此处赋值为平面法线
            alpha=0,  # 输入网络截止部分的透明度
            c=(0.25, 0.25, 0.25),
            padding=0,
            can_translate=True,
            can_scale=True,  # 启用部件的缩放功能
        )
        return pcutter

    def readData(self,ori_path,ori_path_2,ori_path_3,ori_path_4):
        self.cmap = "gist_stern_r"
        self.cmap2 = "viridis_r"

        def func(w, _):
            c, n = self.pcutter.origin, self.pcutter.normal
            self.vslice = self.vol.slice_plane(c, n, autocrop=True, border=1.0).cmap('bone')  # 给动态的切割面一个原点和一条法线
            self.vslice.name = "Slice"
            self.plt.at(0).remove("Slice").show(self.vslice)
        def func2(w, _):
            c, n = self.pcutter2.origin, self.pcutter2.normal
            self.vslice2 = self.vol2.slice_plane(c, n, autocrop=True, border=1.0).cmap('bone')  # 给动态的切割面一个原点和一条法线
            self.vslice2.name = "Slice2"
            self.plt.at(1).remove("Slice").show(self.vslice2)
        def func3(w, _):
            c, n = self.pcutter3.origin, self.pcutter3.normal
            self.vslice3 = self.vol3.slice_plane(c, n, autocrop=True, border=1.0).cmap('bone')  # 给动态的切割面一个原点和一条法线
            self.vslice3.name = "Slice3"
            self.plt.at(2).remove("Slice3").show(self.vslice3)
        def func4(w, _):
            c, n = self.pcutter4.origin, self.pcutter4.normal
            self.vslice4 = self.vol4.slice_plane(c, n, autocrop=True, border=1.0).cmap('bone')  # 给动态的切割面一个原点和一条法线
            self.vslice4.name = "Slice4"
            self.plt.at(3).remove("Slice4").show(self.vslice4)

        self.vslice,self.vol = Vis_3D.geneSlice(self,ori_path)
        self.vslice2,self.vol2 = Vis_3D.geneSlice(self,ori_path_2)
        self.vslice3,self.vol3 = Vis_3D.geneSlice(self,ori_path_3)
        self.vslice4,self.vol4 = Vis_3D.geneSlice(self,ori_path_4)

        self.plt = Plotter(axes=8, N=4, bg="k", bg2="bb", interactive=True,qt_widget=self.vtkWidget)  # N:desired renderers,可以qt_windows,sharecam,
        self.plt.interactive()
        self.plt.at(0).show(self.vslice, __doc__,  zoom=1.5)#
        #self.plt.at(1).show(self.vslice2, __doc__, zoom=1.5)  #
        #self.plt.at(2).show(self.vslice3, __doc__, zoom=1.5)  #
        #self.plt.at(3).show(self.vslice4, __doc__, zoom=1.5)  #

        #切面模型
        self.pcutter = Vis_3D.genePcutter(self,self.vslice)
        self.pcutter2 = Vis_3D.genePcutter(self,self.vslice2)
        self.pcutter3 = Vis_3D.genePcutter(self,self.vslice3)
        self.pcutter4 = Vis_3D.genePcutter(self,self.vslice4)

        self.pcutter.add_observer("interaction", func)
        self.plt.at(0).add(self.pcutter)

        self.pcutter2.add_observer("interaction", func2)
        self.plt.at(1).add(self.pcutter2)

        self.pcutter3.add_observer("interaction", func3)
        self.plt.at(2).add(self.pcutter3)

        self.pcutter4.add_observer("interaction", func4)
        self.plt.at(3).add(self.pcutter4)
        self.plt.interactive()

        self.arr = self.vslice.pointdata[0]  # retrieve vertex array data
        self.arr2 = self.vslice2.pointdata[0]
        self.arr3 = self.vslice3.pointdata[0]
        self.arr4 = self.vslice4.pointdata[0]


        def flatFunc(evt):
            if not evt.actor:
                return
            try:
                pid = evt.actor.closest_point(evt.picked3d, return_point_id=True)
            except AttributeError:
                pass
            else:
                txt = f"Position:{precision(evt.actor.picked3d, 3)}\noriginal_shape_MeshVolume = {self.arr[pid]}"

                pts = evt.actor.points()
                sph = Sphere(pts[pid], c='orange7').pickable(True)
                fp = sph.flagpole(txt, s=7, offset=(-150, 15), font=2).follow_camera()
                # remove old and add the two new objects
                self.plt.at(0).remove('Sphere', 'FlagPole').add(sph, fp).render()
                #print(pid)

        self.arr = self.vslice.pointdata[0]  # retrieve vertex array data

        def flatFunc_3D(evt):
            if not evt.actor:
                return
            try:
                pid = evt.actor.closest_point(evt.picked2d, return_point_id=True)
            except AttributeError:
                pass
            else:
                #self.i += 1
                #for idx, row in self.numdata.iterrows():
                txt = f"shape of the slice:{self.vslice.metadata['shape']}\n" \
                      f"original bounds of the slice:{self.vslice.metadata['original_bounds']}"

                pts = evt.actor.points()
                sph = Sphere(pts[pid], c='orange7').pickable(True)
                fp = sph.flagpole(txt, s=7, offset=(-150, 15), font=2).follow_camera()
                # remove old and add the two new objects
                self.plt.at(0).remove('Sphere', 'FlagPole').add(sph, fp).render()

        #把at0去掉了
        self.plt.at(0).add_callback('as my mouse moves please call', flatFunc)  # be kind to vedo ;)
        self.plt.at(0).add_callback('LeftButtonPress', flatFunc_3D)

        def flatFunc2(evt):
            if not evt.actor:
                return
            try:
                pid = evt.actor.closest_point(evt.picked3d, return_point_id=True)
            except AttributeError:
                pass
            else:
                txt = f"Position:{precision(evt.actor.picked3d, 3)}\noriginal_shape_MeshVolume = {self.arr2[pid]}"

                pts = evt.actor.points()
                sph = Sphere(pts[pid], c='orange7').pickable(True)
                fp = sph.flagpole(txt, s=7, offset=(-150, 15), font=2).follow_camera()
                # remove old and add the two new objects
                self.plt.at(1).remove('Sphere', 'FlagPole').add(sph, fp).render()
                #print(pid)

        self.arr2 = self.vslice2.pointdata[0]  # retrieve vertex array data

        def flatFunc_3D2(evt):
            if not evt.actor:
                return
            try:
                pid = evt.actor.closest_point(evt.picked2d, return_point_id=True)
            except AttributeError:
                pass
            else:
                txt = f"shape of the slice:{self.vslice2.metadata['shape']}\n" \
                      f"original bounds of the slice:{self.vslice2.metadata['original_bounds']}"

                pts = evt.actor.points()
                sph = Sphere(pts[pid], c='orange7').pickable(True)
                fp = sph.flagpole(txt, s=7, offset=(-150, 15), font=2).follow_camera()
                # remove old and add the two new objects
                self.plt.at(1).remove('Sphere', 'FlagPole').add(sph, fp).render()

        #把at0去掉了
        self.plt.at(1).add_callback('as my mouse moves please call', flatFunc2)  # be kind to vedo ;)
        self.plt.at(1).add_callback('LeftButtonPress', flatFunc_3D2)

        def flatFunc3(evt):
            if not evt.actor:
                return
            try:
                pid = evt.actor.closest_point(evt.picked3d, return_point_id=True)
            except AttributeError:
                pass
            else:
                txt = f"Position:{precision(evt.actor.picked3d, 3)}\noriginal_shape_MeshVolume = {self.arr3[pid]}"

                pts = evt.actor.points()
                sph = Sphere(pts[pid], c='orange7').pickable(True)
                fp = sph.flagpole(txt, s=7, offset=(-150, 15), font=2).follow_camera()
                # remove old and add the two new objects
                self.plt.at(2).remove('Sphere', 'FlagPole').add(sph, fp).render()
                #print(pid)

        self.arr3 = self.vslice3.pointdata[0]  # retrieve vertex array data

        def flatFunc_3D3(evt):
            if not evt.actor:
                return
            try:
                pid = evt.actor.closest_point(evt.picked2d, return_point_id=True)
            except AttributeError:
                pass
            else:
                txt = f"shape of the slice:{self.vslice3.metadata['shape']}\n" \
                      f"original bounds of the slice:{self.vslice3.metadata['original_bounds']}"

                pts = evt.actor.points()
                sph = Sphere(pts[pid], c='orange7').pickable(True)
                fp = sph.flagpole(txt, s=7, offset=(-150, 15), font=2).follow_camera()
                # remove old and add the two new objects
                self.plt.at(2).remove('Sphere', 'FlagPole').add(sph, fp).render()

        #把at0去掉了
        self.plt.at(2).add_callback('as my mouse moves please call', flatFunc3)  # be kind to vedo ;)
        self.plt.at(2).add_callback('LeftButtonPress', flatFunc_3D3)

        def flatFunc4(evt):
            if not evt.actor:
                return
            try:
                pid = evt.actor.closest_point(evt.picked3d, return_point_id=True)
            except AttributeError:
                pass
            else:
                txt = f"Position:{precision(evt.actor.picked3d, 3)}\noriginal_shape_MeshVolume = {self.arr4[pid]}"

                pts = evt.actor.points()
                sph = Sphere(pts[pid], c='orange7').pickable(True)
                fp = sph.flagpole(txt, s=7, offset=(-150, 15), font=2).follow_camera()
                # remove old and add the two new objects
                self.plt.at(3).remove('Sphere', 'FlagPole').add(sph, fp).render()
                #print(pid)

        self.arr4 = self.vslice4.pointdata[0]  # retrieve vertex array data

        def flatFunc_3D4(evt):
            if not evt.actor:
                return
            try:
                pid = evt.actor.closest_point(evt.picked2d, return_point_id=True)
            except AttributeError:
                pass
            else:
                txt = f"shape of the slice:{self.vslice4.metadata['shape']}\n" \
                      f"original bounds of the slice:{self.vslice4.metadata['original_bounds']}"

                pts = evt.actor.points()
                sph = Sphere(pts[pid], c='orange7').pickable(True)
                fp = sph.flagpole(txt, s=7, offset=(-150, 15), font=2).follow_camera()
                # remove old and add the two new objects
                self.plt.at(3).remove('Sphere', 'FlagPole').add(sph, fp).render()

        #把at0去掉了
        self.plt.at(3).add_callback('as my mouse moves please call', flatFunc4)  # be kind to vedo ;)
        self.plt.at(3).add_callback('LeftButtonPress', flatFunc_3D4)

        #self.plt.show(vslice, __doc__)  # 少加点就行了...
