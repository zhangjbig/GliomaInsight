import numpy as np
import nibabel as nib
import pandas as pd
from PyQt5.QtWidgets import QDialog
import SimpleITK as sitk

from lib import HeterogeneityCAD
from FirstOrderStatistics import FirstOrderStatistics
#from HeterogeneityCAD.TextureGLCM import TextureGLCM

class HeterCAD(QDialog):
    def __init__(self):
        pass
        #self.cacData()
    def cacData(self):
        # 加载图像
        image_path = self.ori_path
        image = sitk.ReadImage(image_path)

        # 获取图像数据和大小
        image_array = sitk.GetArrayFromImage(image)
        size = image.GetSize()
        spacing = image.GetSpacing()

        # 计算灰度级别
        gray_levels = np.unique(image_array)

        # 构建参数矩阵和参数矩阵坐标
        parameter_matrix = image_array  # 简单起见，将整个图像用作参数矩阵
        parameter_matrix_coordinates = np.indices(size).reshape(3, -1).T  # 参数矩阵的坐标

        # 计算参数值（这是一个示例，你可能需要根据具体特征进行更复杂的计算）
        parameter_values = np.mean(image_array, axis=(1, 2))  # 例如，使用局部区域的均值作为参数值

        # 确定灰度级别数
        num_gray_levels = len(gray_levels)

        #self.TextureGLCM(gray_levels,num_gray_levels,parameter_matrix,parameter_matrix_coordinates,parameter_values)
        self.FirstOrder(gray_levels, parameter_values)
        # 现在你可以将这些数据传递给你的 TextureGLCM 类的实例化，以计算纹理特征。
    def TextureGLCM(self,grayLevels,numGrayLevels,parameterMatrix,parameterMatrixCoordinates,parameterValues):
        # 你需要提供所需的输入数据
        #grayLevels = None  # 灰度级别
        #numGrayLevels = None  # 灰度级别数
        #parameterMatrix = None  # 参数矩阵
        #parameterMatrixCoordinates = None  # 参数矩阵坐标
        #parameterValues = None  # 参数值
        allKeys = ["Autocorrelation", "Cluster Prominence", "Cluster Shade", ...]  # 所有特征的键列表

        # 实例化 TextureGLCM 类
        textureGLCM = TextureGLCM(grayLevels, numGrayLevels, parameterMatrix, parameterMatrixCoordinates,
                                  parameterValues, allKeys)

        # 计算纹理特征
        texture_features = textureGLCM.EvaluateFeatures()

        # 打印结果
        print(texture_features)

    def prints(self,string):
        self.textBrowser_5.append(string)
    def FirstOrder(self,grayLevels,parameterValues,ori_path= './BraTS-GLI-0009_0000.nii.gz'):
        #ori_path = './BraTS-GLI-0009_0000.nii.gz'
        label_path = './BraTS-GLI-0009.nii.gz'

        data_nii = nib.load(ori_path)
        data1 = data_nii.get_fdata()
        data_nii = nib.load(label_path)
        data2 = data_nii.get_fdata()
        # 假设您有一些参数数组和灰度级列表
        #parameterValues = np.array([1, 2, 3, 4, 5])
        bins = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        #grayLevels = 10
        allKeys = ["Voxel Count", "Gray Levels", "Energy", "Entropy", "Minimum Intensity", "Maximum Intensity",
                   "Mean Intensity", "Median Intensity", "Range", "Mean Deviation", "Root Mean Square",
                   "Standard Deviation", "Skewness", "Kurtosis", "Variance", "Uniformity"]

        # 创建 FirstOrderStatistics 类的实例
        firstOrderStats = FirstOrderStatistics(parameterValues, bins, grayLevels, allKeys)

        # 调用 EvaluateFeatures 方法来评估特征
        features = firstOrderStats.EvaluateFeatures()

        # 打印评估后的特征
        for key, value in features.items():
            print(f"{key}: {value}")
    def saveFile(self,result,filename):
        # 储存数据
        for modality, features in result.items():
            save_path = filename + '_of_' + 'HeterogeneityCAD' +'_for_' +  modality + '.csv'
            df = pd.DataFrame()
            for key, value in features.items():
                # 如果当前特征是所选特征之一，则将其添加到 DataFrame
                # if key in selected_features:
                df = df.append({'Feature': key, 'Value': value}, ignore_index=True)

            # 将 DataFrame 保存为 CSV 文件
            df.to_csv(save_path, index=False)
