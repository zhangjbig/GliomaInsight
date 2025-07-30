
#读取nii文件
import nibabel as nib
#数据处理
from skimage import transform #图像缩放
from scipy import ndimage #图像旋转
#获取文件位置
import glob
import os, zipfile
#可视化训练过程
from matplotlib import pyplot
import matplotlib as plt
#打包输出结果+基本数据处理
import pandas as pd
import numpy as np
#构建模型
import tensorflow as tf
from tensorflow import keras
from keras import layers
"""
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)  # 设置GPU显存用量按需使用
    tf.config.set_visible_devices([gpus[0]], "GPU")
# 打印显卡信息，确认GPU可用
print(gpus)
print(tf.__version__)
"""
"""
install.packages("devtools")
devtools::install_github("CFWP/FMradio")
"""
#获得数据集,预计可以读取一个表格，获得其中每张MRI患者的存活时长
img_paths = [
    os.path.join(os.getcwd(), "MosMedData/CT-0", x)
    for x in os.listdir("MosMedData/CT-0")
]
suvival_paths = [
]
print("MRI Images: " + str(len(img_paths)))
print("Survival periods: " + str(len(suvival_paths)))


# 数据准备：图像增强、裁剪、噪声过滤
def read_nifti_file(filepath):
    # 读取文件
    scan = nib.load(filepath)
    # 获取数据
    scan = scan.get_fdata()
    return scan
"""归一化"""
def normalize(volume):
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume
"""修改图像大小"""
def resize_volume(img):
    # Set the desired depth
    desired_depth = 256 #64
    desired_width = 256 #128
    desired_height = 64 #128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # 旋转
    img = ndimage.rotate(img, 90, reshape=False)
    # 数据调整
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img
# 定义预处理工作流
def process_scan(path):
    # 读取文件
    volume = read_nifti_file(path)
    # 归一化
    volume = normalize(volume)
    # 调整尺寸 width, height and depth
    volume = resize_volume(volume)
    return volume

img_scans = np.array([process_scan(path) for path in img_paths])  #只是把图像走一遍工作流就可以了
#分离出标签
train_labels = suvival_paths.pop('MPG') #test_labels = suvival_paths.pop('MPG')


#拆分训练数据集和测试数据集
'''构建训练集和验证集'''
# train_dataset = dataset.sample(frac=0.8,random_state=0)
# test_dataset = dataset.drop(train_dataset.index)

#网络设计：确定CNN的网络结构，包括卷积层、池化层、全连接层和输出层等。卷积层通常用于提取特征，池化层则用于降低数据维度和提取主要特征，全连接层和输出层用于产生回归预测结果。
"""
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=torch.nn.Conv2d(63,16,5)
        self.conv2=torch.nn.Conv2d(16,4,5)
        self.pool=torch.nn.MaxPool2d(2)
        self.fc1=torch.nn.Linear(6084,64)
        self.fc2=torch.nn.Linear(64,1)
    def forward(self,x):
        batch_size=x.size(0)
        x=x.to(torch.float32)
        x=torch.relu(self.pool(self.conv1(x)))
        x=torch.relu(self.pool(self.conv2(x)))
        x=x.view(batch_size,-1)
        x=self.fc1((x))
        x=torch.sigmoid(self.fc2(x))
        return x
"""
# 构建3D卷积神经网络模型
def get_model(width=128, height=128, depth=64):
    """构建 3D 卷积神经网络模型"""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # 定义模型
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

# 通过为每个完成的时期打印一个点来显示训练进度
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000
# 构建模型
model = get_model(width=128, height=128, depth=64)


# patience 值用来检查改进 epochs 的数量
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
#使用history对象中存储的统计信息可视化模型的训练进度
EPOCHS = 1000

#训练模型
history = model.fit(img_scans,
                    train_labels,
                    epochs=EPOCHS, #训练迭代次数
                    validation_split = 0.2, #从训练集再拆分验证集，作为早停的衡量指标
                    verbose=0, #是否输出过程
                    callbacks=[early_stop, PrintDot()])
model.summary() #打印模型的相关信息

#模型评估：拟合结果
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()
plot_history(history) #可视化过程
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

"""
# 试用模型
example_batch = normed_train_data[:10] # 从训练数据中抽取十条
example_result = model.predict(example_batch)
example_result
"""

# 设置动态学习率
"""
initial_learning_rate = 1e-4
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=30, decay_rate=0.96, staircase=True
)
# 编译
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)
# 保存模型
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
"""
# 定义早停策略
"""
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

epochs = 100

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)
"""


