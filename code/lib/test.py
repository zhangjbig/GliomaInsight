import SimpleITK as sitk
import numpy as np

# 加载图像
image_path = "BraTS-GLI-0009_0000.nii.gz"
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

# 现在你可以将这些数据传递给你的 TextureGLCM 类的实例化，以计算纹理特征。
