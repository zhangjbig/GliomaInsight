import pandas as pd
from radiomics import featureextractor
import nibabel as nib

path = "D:/www/xxx/tumorTest/UI/BraTS2021_00495/BraTS2021_00495_flair.nii.gz"
nii_path = "D:/www/xxx/tumorTest/UI/BraTS2021_00495/BraTS2021_00495_seg.nii.gz"
data_nii = nib.load(path)
images = {
        'flair': path
}

# 定义特征提取设置
settings = {}
settings['binWidth'] = 25
settings['sigma'] = [3, 5]
settings['resampledPixelSpacing'] = [1, 1, 1]
settings['voxelArrayShift'] = 1000
settings['normalize'] = True
settings['normalizeScale'] = 100  # 这些全部按自己需要来设置

# 实例化特征提取器
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor.enableAllFeatures()
extractor.enableAllImageTypes()

# 结果存储字典
result = []

# 假设 images 是 {modality: file_path} 格式
for modality, file_path in images.items():
    try:
        features = extractor.execute(file_path, nii_path) if nii_path else extractor.execute(file_path)
        # 转化为字典格式，每行包括 modality 信息
        feature_row = {"Modality": modality, **features}
        result.append(feature_row)
    except Exception as e:
        print(f"Error extracting features for {modality}: {e}")

# 转为 DataFrame 并保存到 CSV
df = pd.DataFrame(result)
csv_output_path = "radiomics_features.csv"
df.to_csv(csv_output_path, index=False)
print(f"Feature extraction completed and saved to {csv_output_path}")

