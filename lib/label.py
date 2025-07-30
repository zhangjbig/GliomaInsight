from PyQt5.QtGui import *#####需要自己添加的库
from PyQt5 import QtCore, QtGui, QtWidgets
import sys

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(645, 500)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(0, 0, 471, 461))
        self.label.setPixmap(QPixmap(r"Estimate\Sample_KM.png"))# 图片路径

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Form()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
