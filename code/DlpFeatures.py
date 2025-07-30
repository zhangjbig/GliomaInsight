import shutil, nibabel
import imageio
import numpy as np

import os
import glob
from torchvision import transforms
from PIL import Image

import torchvision.models as models

class DLPExtractor:
    def __init__(self,progressBar):
        self.progressBar = progressBar
    def converts(self,nii_path,outputfile,textB,save_file_name):
        self.save_file_name = save_file_name
        self.textbrowser = textB
        self.outputfile = outputfile
        self.nii_path = str(nii_path)  # 原图像
        print(self.nii_path)

        DLPExtractor.load_progress_bar(self) #1
        DLPExtractor.niito2D(self,self.nii_path)

        #outputfile = ' '  # 输出文件夹

    def niito2D(self,filepath):
        try:
            inputfiles = os.listdir(filepath)  # 遍历文件夹数据
        except FileNotFoundError:
            pass
        else:
            self.prints(str('Input file is ' + str(inputfiles)))
            self.prints(str('Output folder is ' + str(self.outputfile)))
            DLPExtractor.load_progress_bar(self) #2
            for inputfile in inputfiles:
                image_array = nibabel.load(filepath + inputfile).get_fdata()  # 数据读取
                # set destination folder
                if not os.path.exists(self.outputfile):
                    os.makedirs(self.outputfile)  # 不存在输出文件夹则新建
                    self.prints(str('Created ouput directory: ' + str(self.outputfile)))
                    self.prints(str('Reading NIfTI file...'))

                self.total_slices = image_array.shape[2]
                mid_slice = int(self.total_slices / 2)  # 获取中间层，注意可能是浮点数
                slice_counter = mid_slice  # 从第几个切片开始

                # iterate through slices
                # for current_slice in range(mid_slice-slices, mid_slice+slices):
                for current_slice in range(mid_slice-1, mid_slice + 1):
                    # alternate slices
                    if (slice_counter % 1) == 0:
                        data = image_array[:, :, current_slice]  # 保存该切片
                        data = data.astype(np.uint8)
                        # alternate slices and save as png
                        if (slice_counter % 1) == 0:
                            self.prints('Saving image...')
                            # 切片命名
                            image_name = inputfile[:-4] + "{:0>3}".format(str(current_slice + 1)) + ".png"

                            # 创建nii对应的图像的文件夹
                            img_f_path = os.path.join(self.outputfile, inputfile[:-6])
                            if not os.path.exists(img_f_path):
                                os.mkdir(img_f_path)  # 新建文件夹

                            # 保存
                            imageio.imwrite(image_name, data)
                            strr = 'Saved as ' + image_name
                            self.prints(strr)
                            #self.prints()

                            # move images to folder
                            src = image_name
                            try:
                                shutil.move(src, img_f_path)
                                slice_counter += 1
                                self.prints('Moved.')
                            except shutil.Error:
                                pass

            self.prints('Finished converting images')
            DLPExtractor.load_progress_bar(self) #3
            DLPExtractor.runNet(self)

    def netSet(self,n):
        self.n = n
        if n == 1:
            self.net = DLPExtractor.GoogleNet_extractor
        elif n == 2:
            self.net = DLPExtractor.DenseNet121_extractor
        elif n == 3:
            self.net = DLPExtractor.ResNet18_extractor
        elif n == 4:
            self.net = DLPExtractor.vgg16_extractor
        elif n == 5:
            self.net = DLPExtractor.vgg19_extractor

    def saveData(self,feat): #路径准备
        result1 = np.array(feat)
        filename = self.save_file_name + '.txt'
        DLPExtractor.load_progress_bar(self)  # 4
        np.savetxt(filename, result1)

    def runNet(self):
        all_image_dir = self.outputfile
        DLPExtractor.load_progress_bar(self) #5
        if (os.path.exists(all_image_dir)):
            # 获取该目录下的所有文件或文件夹目录路径
            files = glob.glob(all_image_dir + '\\*')
            #print(files)

        # 获取和处理图像
        for file in files:
            self.prints(str(file))  # 各个nii图像切片的文件夹
            image_dir = os.walk(file)
            self.prints(str(image_dir))  # 对文件夹下的每个图像进行遍历
            for path, dir_lst, file_lst in image_dir:
                for file_name in file_lst:
                    image = os.path.join(path, file_name)
                    self.prints(str(image))  # 此处为图片名称
                    for i in range(1, 2):
                        im = Image.open(image).convert('RGB')
                        # img = img.astype(np.uint8)
                        trans_vgg = transforms.Compose([

                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])

                        nimg = trans_vgg(im)
                        nimg.unsqueeze_(dim=0)

                        # 调用特征提取网络
                        if (self.n ==4) | (self.n == 5):
                            self.net(self,nimg)
                        elif self.n == 3:# 对于resnet
                            trans_res = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor()]
                            )
                            nimg_res = trans_res(im)
                            nimg_res.unsqueeze_(dim=0)
                            self.net(self,nimg_res)
            self.load_progress_bar()  # 6

    def prints(self,string):
        self.textbrowser.append(string)

    # 调用该函数时进度条加
    def load_progress_bar(self):
        self.progressBar.setValue(self.progressBar.value() + 25)
        #self.cont_label_title[0] += 1
        #self.cont_label_title[1] += 1
        if self.progressBar.value() >= 100:
            self.prints("Finished!")
            # self.window.close()
            #self.open_table_notice()  # 开一个窗口
            # self.timer.stop()

    def GoogleNet_extractor(self,nimg):
        # 获取vgg16原始模型
        GoogleNet_model = models.googlenet(pretrained=True)
        image_feature_GoogleNet = GoogleNet_model(nimg).data[0]
        self.prints('features of vgg_model_GoogleNet: ')
        self.prints(str( image_feature_GoogleNet))
        self.saveData( image_feature_GoogleNet)

    def DenseNet121_extractor(self,nimg):
        # 获取vgg16原始模型
        DenseNet121_model = models.densenet121(pretrained=True)
        image_feature_DenseNet121 = DenseNet121_model(nimg).data[0]
        self.prints('features of vgg_model_DenseNet: ')
        self.prints(str(image_feature_DenseNet121))
        self.saveData(image_feature_DenseNet121)

    def ResNet18_extractor(self,nimg):
        # 获取vgg16原始模型
        ResNet18_model = models.resnet18(pretrained=True)
        image_feature_ResNet18 = ResNet18_model(nimg).data[0]
        self.prints('features of vgg_model_ResNet18: ')
        self.prints(str(image_feature_ResNet18))
        self.saveData(image_feature_ResNet18)

    def vgg16_extractor(self,nimg):
        # 获取vgg16原始模型
        vgg16_model = models.vgg16(pretrained=True)
        image_feature_vgg16 = vgg16_model(nimg).data[0]
        self.prints('features of vgg_model_vgg16: ')
        self.prints(str(image_feature_vgg16))
        self.saveData(image_feature_vgg16)

    def vgg19_extractor(self,nimg):
        # 获取vgg19原始模型, 输出图像维度是1000.
        vgg_model_1000 = models.vgg19(pretrained=True)

        # 使用原始vgg19得到图像特征.
        print(vgg_model_1000(nimg).data.shape)
        image_feature_1000 = vgg_model_1000(nimg).data[0]
        self.prints('features of vgg_model_vgg19: ')
        self.prints(str(image_feature_1000))
        self.saveData(image_feature_1000)
