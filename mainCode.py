import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import *
from qtpy import uic, QtCore
from UI_settings import Ui_MainDialog

def checkState(Value,timer):
    if Value == 100:
        timer.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    openingUI = uic.loadUi("openingUI.ui")
    #对加载图作出设定并显示
    openingUI.setWindowFlag(Qt.FramelessWindowHint)  # 将界面设置为无框
    openingUI.setAttribute(Qt.WA_TranslucentBackground)  # 将界面属性设置为半透明
    openingUI.shadow = QGraphicsDropShadowEffect()  # 设定一个阴影,半径为10,颜色为#444444,定位为0,0
    openingUI.shadow.setBlurRadius(10)
    openingUI.shadow.setColor(QColor("#444444"))
    openingUI.shadow.setOffset(0, 0)
    openingUI.frame.setGraphicsEffect(openingUI.shadow)  # 为frame设定阴影效果
    openingUI.show()    #显示加载图

    Bar = openingUI.progressBar

    timer = QtCore.QTimer()  # 建立一个计时器
    print(timer.timeout)

    dailog = None
    while True:
        timer.start(100)  # 更新时间间隔, 不知道为啥进度条跑得不好
        timer.timeout.connect(lambda: Bar.setValue(Bar.value() + 10))  # 设置参数
        timer.timeout.connect(lambda: checkState(Bar.value(), timer))  # 检查时间
        Bar.setValue(50)
        Bar.setValue(100)
        app.processEvents()  # 使动画正常播放，不影响主界面构造
        if not dailog:
            dailog = Ui_MainDialog()
            break
    openingUI.close()
    openingUI.progressBar.setValue(100)
    dailog.exec()

    sys.exit(app.exec_())
