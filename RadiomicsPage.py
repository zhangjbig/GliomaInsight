from __future__ import print_function

import pandas as pd
from PyQt5.QtCore import pyqtSignal
from radiomics import featureextractor as FEE # This module is used for interaction with pyradiomics
import yaml

from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QFileDialog, QApplication

from DataProsessing import child


class RadiomicsFunc(QDialog):
    isChecked = pyqtSignal(int)
    def __init__(self):
        super().__init__()
        self.dia = uic.loadUi("New_UI.ui", self)

    #创建参数文件
    def setupParaFile(self):
        self.para_path = './ParaFile/Param.yml'

    def ImgType_init(self,tag):
        self.para_path = './ParaFile/Param.yml'
        settingData = {

        }
        with open(self.para_path, 'w', encoding='utf-8') as f:
            if tag == 1: #Original
                self.displayOri()
            elif tag == 2:
                self.displayCT()
            elif tag == 3:
                self.displayMR3()
            elif tag == 4:
                self.displayMR5()
            elif tag == 5:
                self.displayMR()
            yaml.dump(data=settingData, stream=f, allow_unicode=True)

    def featureSet(self):#get10PercentileFeatureValue():获取百分之十的特征值
        # 写入的数据类型是字典
        featureData = {

        }
        with open(self.para_path, 'w', encoding='utf-8') as f:#如果没有选择特征计算，那么自动全选，排除不推荐的功能
            if self.checkBox_First.isChecked():
                featureData["featureClass"] = {"firstorder": [] }
                #self.tabWidget.setTabEnabled(1)
            #else:
            if self.checkBox_shape.isChecked():
                featureData["featureClass"] = {"shape": [] }
                #self.tabWidget.setTabEnabled(2)
            if self.checkBox_glcm.isChecked():
                featureData["featureClass"] = {"glcm": [] }
                #self.tabWidget.setTabEnabled(3)
            if self.checkBox_glrlm.isChecked():
                featureData["featureClass"] = {"glrlm": []}
                #self.tabWidget.setTabEnabled(4)
            if self.checkBox_glszm.isChecked():
                featureData["featureClass"] = {"glszm": [] }
                #self.tabWidget.setTabEnabled(5)
            if self.checkBox_gldm.isChecked():
                featureData["featureClass"] = {"gldm": [] }
                #self.tabWidget.setTabEnabled(6)
            if self.checkBox_ngtdm.isChecked():
                 featureData["featureClass"] = {"ngtdm": [] }
                 #self.tabWidget.setTabEnable(7)
            yaml.dump(data=featureData, stream=f, allow_unicode=True)

    def outPuts(self,images,name,nii_path,ifsave,ori_path,label_path,place,ori_path_2,ori_path_3,ori_path_4):
        self.ori_path = ori_path
        self.ori_path_2 = ori_path_2
        self.ori_path_3 = ori_path_3
        self.ori_path_4 = ori_path_4
        self.label_path = label_path
        # 使用配置文件初始化特征抽取器
        settings = {}
        settings['binWidth'] = 25
        settings['sigma'] = [3, 5]
        settings['resampledPixelSpacing'] = [1, 1, 1]
        settings['voxelArrayShift'] = 1000
        settings['normalize'] = True
        settings['normalizeScale'] = 100  # 这些全部按自己需要来设置

        extractor = FEE.RadiomicsFeatureExtractor(**settings)
        extractor.enableAllFeatures()
        extractor.enableAllImageTypes()

        try:
            result = {}
            for modality, file_path in images.items():
                if nii_path:
                    result[modality] = extractor.execute(file_path, nii_path)
                else:
                    result[modality] = extractor.execute(file_path)
            # 输出特征结果
            # for modality, features in result.items():
            #     print(f"Features for {modality}:")
            #     for feature_name, value in features.items():
            #         print(f"\t{feature_name}: {value}")
                    
        except ValueError|TypeError:
            try:
                result = {}
                for modality, file_path in images.items():
                    if nii_path:
                        result[modality] = extractor.execute(file_path, nii_path, label=2)
                    else:
                        result[modality] = extractor.execute(file_path)
                # 输出特征结果
                for modality, features in result.items():
                    print(f"Features for {modality}:")
                    for feature_name, value in features.items():
                        print(f"\t{feature_name}: {value}")
            except ValueError:
                pass

        type(result)
        flag = True
        # 储存数据
        for modality, features in result.items():
            save_path = place + name + '_for_' + modality + '.csv'
            df = pd.DataFrame()
            for key, value in features.items():
                # 如果当前特征是所选特征之一，则将其添加到 DataFrame
                # if key in selected_features:
                df = df._append({'Feature': key, 'Value': value}, ignore_index=True)
            df.to_csv(save_path, index=False)
            if flag:
                #RadiomicsFunc.invokeDialog_1(self, save_path)
                flag = False
        if ifsave == False:
            print("Deleted")

    # 界面弹出
    def invokeDialog_1(self,path):
        dialog = child(path,self.ori_path,self.ori_path_2,self.ori_path_3,self.ori_path_4)
        dialog.exec()

    def displayOri(self):
        with open('./ParaFile/exampleCT.yaml', "rb") as ff:
            content = ff.read()
            with open(self.para_path, "ab") as f2:
                # 将读取的数据写入到新的对象中
                f2.write(content)

    def displayCT(self):
        with open('./ParaFile/exampleCT.yaml', "rb") as ff:
            content = ff.read()
            with open(self.para_path, "ab") as f2:
                # 将读取的数据写入到新的对象中
                f2.write(content)

    def displayMR3(self):
        with open('./ParaFile/exampleMR_3mm.yaml', "rb") as ff:
            content = ff.read()
            with open(self.para_path, "ab") as f2:
                # 将读取的数据写入到新的对象中
                f2.write(content)

    def displayMR5(self):
        with open('./ParaFile/exampleMR_5mm.yaml', "rb") as ff:
            content = ff.read()
            with open(self.para_path, "ab") as f2:
                # 将读取的数据写入到新的对象中
                f2.write(content)

    def displayMR(self):
        with open('./ParaFile/exampleMR_NoResampling.yaml.yaml', "rb") as ff:
            content = ff.read()
            with open(self.para_path, "ab") as f2:
                # 将读取的数据写入到新的对象中
                f2.write(content)

    def Add_Nor(self, state):
            data = {

            }  # 有一个问题是会一直先输出一个空字典，还不知道怎么避免捏
            with open(self.para_path, 'a', encoding='utf-8') as f:
                # print(str(state))
                if state:
                    data["setting"] = {"normalize": "True", "normalizeScale": 500}
                yaml.dump(data=data, stream=f, allow_unicode=True)

    def Add_Mask(self, state):
            data = {

            }  # 有一个问题是会一直先输出一个空字典，还不知道怎么避免捏
            with open(self.para_path, 'a', encoding='utf-8') as f:
                if state:
                    data["setting"] = {"minimumROIDimensions": "2", "minimumROISize": "50"}
                yaml.dump(data=data, stream=f, allow_unicode=True)

    def Add_Bin(self, state):  # 修改为带有BinWidth的输入项
            data = {

            }  # 有一个问题是会一直先输出一个空字典，还不知道怎么避免捏
            with open(self.para_path, 'a', encoding='utf-8') as f:
                if state:  # 可以改成binwidth的输入项
                    data["setting"] = {"minimumROISize": "50"}
                yaml.dump(data=data, stream=f, allow_unicode=True)

    def Add_Misc(self, state):
            data = {

            }  # 有一个问题是会一直先输出一个空字典，还不知道怎么避免捏
            with open(self.para_path, 'a', encoding='utf-8') as f:
                if state:
                    data["setting"] = {"label": 1}
                yaml.dump(data=data, stream=f, allow_unicode=True)

    def Add_FirstSpe(self, state):
            data = {

            }  # 有一个问题是会一直先输出一个空字典，还不知道怎么避免捏
            with open(self.para_path, 'a', encoding='utf-8') as f:
                if state:
                    data["setting"] = {"voxelArrayShift": "1000"}
                yaml.dump(data=data, stream=f, allow_unicode=True)

    def Add_FirstSpe2(self, state):
            data = {

            }  # 有一个问题是会一直先输出一个空字典，还不知道怎么避免捏
            with open(self.para_path, 'a', encoding='utf-8') as f:
                if state:
                    data["setting"] = {"voxelArrayShift": "300"}
                yaml.dump(data=data, stream=f, allow_unicode=True)

    def Add_Resampling(self, state):
            data = {

            }  # 有一个问题是会一直先输出一个空字典，还不知道怎么避免捏
            with open(self.para_path, 'a', encoding='utf-8') as f:
                if state:
                    data["setting"] = {"interpolator ": "'sitkBSpline'", "resampledPixelSpacing": " [2, 2, 2]"}
                yaml.dump(data=data, stream=f, allow_unicode=True)

    def Add_Resampling2(self, state):
            data = {

                }  # 有一个问题是会一直先输出一个空字典，还不知道怎么避免捏
            with open(self.para_path, 'a', encoding='utf-8') as f:
                if state:
                    data["setting"] = {"preCrop ": "true"}
                yaml.dump(data=data, stream=f, allow_unicode=True)


    def Add_2D(self, state):
            data = {

            }  # 有一个问题是会一直先输出一个空字典，还不知道怎么避免捏
            with open(self.para_path, 'a', encoding='utf-8') as f:
                if state:
                    data["setting"] = {"force2D ": "true"}  # 获取一个dimension，输入
                yaml.dump(data=data, stream=f, allow_unicode=True)

    def Add_Voxel(self,state):
        data = {

        }  # 有一个问题是会一直先输出一个空字典，还不知道怎么避免捏
        with open(self.para_path, 'a', encoding='utf-8') as f:
            if state:
                data["voxelSetting"] = {"kernelRadius ": 2,"maskedKernel":"true","initValue":"nan","voxelBatch":10000}  # 获取一个dimension，输入
            yaml.dump(data=data, stream=f, allow_unicode=True)

    def cusParaSet(self,place):
        file_name = QFileDialog.getOpenFileName(None, "Open File", "./", "yaml(*.yml;*.yaml)")
        self.para_path = file_name[0]
        place.setText(self.para_path)
        #读取yaml文件

    """
                          if tag == 1: #Original
                  settingData["imageType"] = {"Original": {}} #写入参数设定文件
                  self.displayOri(settingData)
              elif tag == 2:
                  settingData["imageType"] = {"Original": {}}
                  self.displayCT(settingData)
              elif tag == 3:
                  settingData["imageType"] = {"LoG": {}}
              elif tag == 4:
                  settingData["imageType"] = {"Square": {}}
              elif tag == 5:
                  settingData["imageType"] = {"SquareRoot": {}}
              elif tag == 6:
                  settingData["imageType"] = {"Logarithm": {}}
              elif tag == 7:
                  settingData["imageType"] = {"Exponential": {}}
              elif tag == 8:
                  settingData["imageType"] = {"Gradient": {}}
              elif tag == 9:
                  settingData["imageType"] = {"LocalBinaryPattern2D": {}}
              else :
                  settingData["imageType"] = {"LocalBinaryPattern3D": {}}
              """
