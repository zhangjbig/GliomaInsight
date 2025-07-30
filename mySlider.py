from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QSlider


class MySlider(QSlider):  # 继承QSlider
    customSliderClicked = pyqtSignal(str)  # 创建信号

    def __init__(self, parent=None):
        super(QSlider, self).__init__(parent)

    def mousePressEvent(self, QMouseEvent):  # 重写的鼠标点击事件
        super().mousePressEvent(QMouseEvent)
        pos = QMouseEvent.pos().x() / self.width()
        self.setValue(round(pos * (self.maximum() - self.minimum()) + self.minimum()))  # 设定滑动条滑块位置为鼠标点击处
        self.customSliderClicked.emit("mouse Press")  # 发送信号
