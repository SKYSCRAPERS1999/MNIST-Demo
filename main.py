import sys, PyQt5, os, torch
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt

from train import Net

class MainWindow(QMainWindow):

    def __init__(self, path = "mnist_cnn.pt"):
        super(MainWindow, self).__init__()
        
        self.model = Net()
        self.model.load_state_dict(torch.load(path))
        
        self.initUI()
        
    def onMyToolBarButtonClick(self, s):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;PNG (*.png)", options=options)
        
        if fileName:
            try:
                self.img = mpimg.imread(fileName) 
                self.img = torch.from_numpy(self.img[:, :, 0]).unsqueeze(0).unsqueeze(0)
                
                pixmap = QPixmap(fileName)
                pixmap = pixmap.scaled(self.img_label.width(), self.img_label.height())
                
                self.img_label.setPixmap(pixmap)
                
                prediction_digit = self.model(self.img).argmax()
                output_str = "Prediction of {}: {}".format(os.path.basename(fileName), prediction_digit)
                print(output_str)
                self.digit_label.setText(output_str)
                
            except Exception as e:
                print("Error uploading the file: {}", e)
            
    def initUI(self):
        """initialize UI settings"""
        
        self.H, self.W = 600, 600
        self.setWindowTitle("Simple Digit Recognition")
        self.setWindowIcon(QIcon("icons/trial.ico"))
        self.resize(self.W, self.H)
        self.moveCenter()
        
        layout = QVBoxLayout()
        
        self.upload_btn = QPushButton(QIcon("icons/filesave.png"), "Upload Image")
        self.upload_btn.clicked.connect(self.onMyToolBarButtonClick)
        self.upload_btn.setFixedHeight(50)
        
        layout.addWidget(self.upload_btn)
        
        self.img_label = QLabel(self)
        self.img_label.setFixedHeight(400)
        layout.addWidget(self.img_label)
            
        self.digit_label = QLabel(self)
        self.digit_label.setFont(QFont('Arial', 32))

        layout.addWidget(self.digit_label)
        
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
    def moveCenter(self):
        """make the window to be located at the center of desktop"""
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    app.exec_()
    
if __name__ == '__main__':
    main()
