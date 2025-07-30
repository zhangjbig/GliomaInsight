import subprocess
import time

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog
from PyQt5.uic.properties import QtCore
from PyQt5 import QtCore

class RadioML(object):
    def __init__(self,feaNums,textBrowser,result_sample_place,result_OS_place):
        self.textBrowser = textBrowser
        self.FeaNum = feaNums + '\n'
        self.scoreNum = "Rscript .\Risk_score.R -f demo_feature_data.txt -p prefix -c perm5." + feaNums + ".feature_coe.txt -m perm5." + feaNums + "\n"
        self.sample_place = result_sample_place
        self.OS_place = result_OS_place
    """  
    def numCheck(self,feaNums):
        if (int)(feaNums)<5 | (int)(feaNums)>20:
            self.prints("Warning:choose the high frequency threshold, and meanwhile about several features(5 ~ 20 ), input the threshold(integer)")
            self.prints("If the number is still invalid, it will be forced to set 20 while caculating.")
            self.FeaNum = '20\n'
            self.scoreNum = 'Rscript .\Risk_score.R -f demo_feature_data.txt -p prefix -c perm5.20.feature_coe.txt -m perm5.20'
    """
    def Features(self):
        # 定义要执行的多个命令
        feaCommands = [
            "cd lib\n",
            "cd Estimate\n",
            "Rscript .\RadioML_features.R -p TIANTAN -t 5 -n 1000 -R prefix_1000_5_training.RData\n",
            self.FeaNum
        ]
        try:
            self.Run(feaCommands)
        except UnicodeDecodeError:
            self.prints("There might be a problem, check your number and try again latter.")

    def Score(self):
        # 定义要执行的多个命令
        feaCommands = [
            "cd Estimate\n",
            self.scoreNum
        ]
        self.showResults()
        #self.Run(feaCommands)

    def Run(self,cmd_command):
        # 启动一个交互式命令行
        self.process = subprocess.Popen(["cmd"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True)
        # 发送多个命令到命令行
        for command in cmd_command:
            self.process.stdin.write(command)
            self.process.stdin.flush()  # 刷新缓冲区以确保命令被发送
            time.sleep(1)  # 等待一秒钟以确保命令有足够的时间执行

        print("Finished")
        # 读取命令的输出
        output_line = self.process.stdout.readline()
        start_time = time.time()
        while output_line:

            self.prints(output_line.strip())
            print(output_line.strip())
            # 设置超时时间为1秒
            #print(time.time() - start_time)
            if time.time() - start_time >= 1:
                break
            output_line = self.process.stdout.readline()

        self.prints("Now you can check your results in ./lib/Estimator")

    def showResults(self):
        #pixmap = QPixmap(r"Estimate\FALSE_Sample_KM.png")
        self.sample_place.setPixmap(QPixmap(r"lib\Estimate\TIANTAN_FALSE_Sample_KM.png"))
        self.OS_place.setPixmap(QPixmap(r"lib\Estimate\TIANTAN_FALSE_OS_KM.png"))
        self.sample_place.setScaledContents(True)

        self.sample_place.show()
        print("done")

    def Train(self):
        files = QFileDialog.getOpenFileName (None, "QFileDialog.getOpenFileNames()", "./",
                                                "txt(*.csv; *txt)")
        train_path = files[0]
        self.train = "Rscript.\RadioML_main.R - f " + train_path + " - p prefix - t 5 - n 1000\n"
        feaCommands = [
            "cd lib\n",
            "cd Estimate\n",
            self.train
        ]
        try:
            self.Run(feaCommands)
            self.prints("Trained successfully!")
        except UnicodeDecodeError:
            self.prints("There might be a problem, check your number and try again latter.")

    def prints(self,string):
        self.textBrowser.append(string)

