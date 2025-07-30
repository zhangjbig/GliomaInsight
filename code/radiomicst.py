from data import RadioMLProgress
# 定义用于多模态分析的图像文件路径
image_files = {
                'T1': './Test_data/BraTS-GLI-0009_0000.nii.gz'
            }
# 读取图像
images = {}
for modality, file_path in image_files.items():
    if file_path:
        images[modality] = file_path
        print(modality + ':' + file_path + '\n')
test = RadioMLProgress()
test.extract(images,'111','./Test_data/BraTS-GLI-0009.nii.gz','./Test_data/BraTS-GLI-0009_0000.nii.gz')