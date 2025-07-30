import os
import SimpleITK as sitk
import glob

# 输入NRRD文件路径
input_nrrd_path = "./Test_data/brain1_label.nrrd"

files = glob.glob(input_nrrd_path)
for filename in files:
    print(filename)

    # 使用SimpleITK读取NRRD数据
    nrrd_image = sitk.ReadImage(filename)

    output_nifti_path = filename.split('.nrrd')[0] + '.nii.gz'
    print(output_nifti_path)

    # 将SimpleITK图像保存为NIfTI格式
    sitk.WriteImage(nrrd_image, output_nifti_path)

print("Conversion completed.")
