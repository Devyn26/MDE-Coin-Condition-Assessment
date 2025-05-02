'''
Updated for F25-06 coin assessment team
Updated by: Eric Morley
Date: 4/25/2025
'''

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox, QPlainTextEdit, QLabel, QFileDialog  #QLabel is for image showing
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap  #pixmap is for image showing
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from . import ImageHSV 
from . import patternMatching
from . import WheatStalkGrader
from . import MorganGrader


obversePath = ""
HSV_Results = "----"
PCA_Results = "----"
FFT_Results = "----"
FeatureMatch_Results = "----"
percFlag = False

class App(QWidget):

    

    def __init__(self):
        super().__init__()
        self.title = "Team Noble Rats"
        self.left = 0
        self.top = 0
        self.width = 1920
        self.height = 1080
        self.initUI()

    def initUI(self):
        grid = QGridLayout()
        grid.setColumnMinimumWidth(0,100)
        grid.setColumnMinimumWidth(4,100)
        self.setWindowTitle(self.title)
        self.setLayout(grid)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setStyleSheet("background-color: #f1c6b6;")


        # ------------- buttons -------------
        obverse_upload = QPushButton('Load Obverse Of Coin', self)
        font = obverse_upload.font()      # lineedit current font
        font.setPointSize(20)               # change it's size
        obverse_upload.setFont(font)
        obverse_upload.setFixedSize(600, 50)
        obverse_upload.setStyleSheet("background-color : #a8937e;"
                                   "border-radius :5px;")
        grid.addWidget(obverse_upload, 2, 1)
        obverse_upload.clicked.connect(self.on_obverse_click)

        reverse_upload = QPushButton('Load Reverse Of Coin', self)
        reverse_upload.setFixedSize(600, 50)
        reverse_upload.setFont(font)
        reverse_upload.setStyleSheet("background-color : #a8937e;"
                                   "border-radius :5px;")
        grid.addWidget(reverse_upload, 2, 3)
        reverse_upload.clicked.connect(self.on_reverse_click)


        calculate_HSV = QPushButton('Calculate HSV', self)
        calculate_HSV.setFont(font)
        calculate_HSV.setFixedSize(600, 50)
        calculate_HSV.setStyleSheet("background-color : #a8937e;"
                                   "border-radius :5px;")
        grid.addWidget(calculate_HSV, 4, 1)
        calculate_HSV.clicked.connect(self.on_calculate_HSV_click)


        calculate_fourier = QPushButton('Calculate FFT', self)
        calculate_fourier.setFont(font)
        calculate_fourier.setFixedSize(600, 50)
        calculate_fourier.setStyleSheet("background-color : #a8937e;"
                                   "border-radius :5px;")
        grid.addWidget(calculate_fourier, 4, 3)
        calculate_fourier.clicked.connect(self.on_calculate_fourier_click)



        calculate_liberty = QPushButton('Calculate Feature Match', self)
        calculate_liberty.setFont(font)
        calculate_liberty.setFixedSize(600, 50)
        calculate_liberty.setStyleSheet("background-color : #a8937e;"
                                   "border-radius :5px;")
        grid.addWidget(calculate_liberty, 4, 2)
        calculate_liberty.clicked.connect(self.on_calculate_liberty_click)

        # ------------- text boxes -------------
        self.title_text_box = QLineEdit("Team Noble Rats: Coin Grading Generator")
        self.title_text_box.setReadOnly(True)
        self.title_text_box.setFont(font)


        self.result_box = QTextEdit("HSV Results: " + HSV_Results)
        self.result_box.append("\n")
        self.result_box.append("PCA: " + PCA_Results)
        self.result_box.append("\n")
        self.result_box.append("FFT Analysis Results: " + FFT_Results)
        self.result_box.append("\n")
        self.result_box.append("Feature Match Results: " + FeatureMatch_Results)
        self.result_box.append("\n")
        self.result_box.setFont(font)
        self.result_box.setFixedWidth(600)
        self.result_box.setReadOnly(True)
        grid.addWidget(self.result_box, 1, 2)


        #------------- upload paths ------------
        self.upload_path_obverse = QLineEdit(self)
        self.upload_path_obverse.setFont(font)
        self.upload_path_obverse.setFixedWidth(600)
        self.upload_path_obverse.setReadOnly(True)
        self.upload_path_obverse.setText("Filepath")
        grid.addWidget(self.upload_path_obverse, 3, 1)
        obversePath = self.upload_path_obverse

        self.upload_path_reverse = QLineEdit(self)
        self.upload_path_reverse.setFont(font)
        self.upload_path_reverse.setFixedWidth(600)
        self.upload_path_reverse.setReadOnly(True)
        self.upload_path_reverse.setText("Filepath")
        grid.addWidget(self.upload_path_reverse, 3, 3)

        #------------- How to box ------------
        self.how_to_box = QPlainTextEdit("To use this software, load the obverse and the reverse of coin you would like characterized. Clicking on the buttons on the bottom of the application will calculate characteristics of the coin. HSV Results represents the redness of the fields of the coin as a percentage. PCA represents the coins color characterization as red, red-brown, brown, or an outlier. Both the FFT results and the Feature Match results represent how worn the coin is on a scale of 0 to 70, where 0 is the most worn and 70 is the least.")
        fontSmall = self.how_to_box.font()      # lineedit current font
        fontSmall.setPointSize(13)               # change it's size
        self.how_to_box.setFont(fontSmall)
        #self.how_to_box.setFixedWidth(600)
        self.how_to_box.setFixedHeight(100)
        self.how_to_box.setReadOnly(True)
        grid.addWidget(self.how_to_box,0,1,1,3)


        #------------- Default LEFT Image -----------
        self.label_obverse = QLabel(self)
        pixmap_obverse = QPixmap("./LincolnCent/Images/Obverse/Red/1928Penny65RD.jpg")
        self.label_obverse.setScaledContents(False)
        self.label_obverse.setPixmap(pixmap_obverse.scaled(600,600, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        grid.addWidget(self.label_obverse, 1, 1)



        #------------- Default RIGHT Image -----------
        self.label_reverse = QLabel(self)
        pixmap_reverse = QPixmap("./LincolnCent/Images/Reverse/1928PennyBack65.jpg")
        self.label_reverse.setScaledContents(False)
        self.label_reverse.setPixmap(pixmap_reverse.scaled(600,600, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        grid.addWidget(self.label_reverse, 1, 3)


        # ------------- show QT Application -------------
        self.show()
    
    @pyqtSlot()
    def openFileNameDialog_obverse(self):
        options = QFileDialog.Options()

        options |= QFileDialog.DontUseNativeDialog
        obverse_fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Image Files (*.py)", options=options)
        if obverse_fileName:
            print(obverse_fileName)
        return obverse_fileName

    @pyqtSlot()
    def openFileNameDialog_reverse(self):
        options = QFileDialog.Options()

        #options.setStyleSheet("background-color: white")

        options |= QFileDialog.DontUseNativeDialog
        reverse_fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Image Files (*.py)", options=options)
        if reverse_fileName:
            print(reverse_fileName)
        return reverse_fileName

    @pyqtSlot()
    def on_obverse_click(self):
        # obverse_fileName = self.openFileNameDialog_obverse()
        # Demo Code, use above when running normally
        obverse_fileName = 'test_images/Lincoln_obv_img_proc.jpg'
        print('Obverse button has been clicked with filepath name: ', obverse_fileName)
        self.upload_path_obverse.setText(obverse_fileName)
        pixmap_obverse = QPixmap(obverse_fileName)
        if pixmap_obverse.isNull():
            print('Obverse filepath does not exist, please try again')
            pixmap_obverse = QPixmap("./LincolnCent/Images/InvalidImage.jpg")
            #self.label_obverse.setPixmap(pixmap_obverse)
            self.label_obverse.setScaledContents(True)
            self.label_obverse.setPixmap(pixmap_obverse.scaled(600,600, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            self.label_obverse.setMaximumSize(600, 600)
        else:
            #self.label_obverse.setPixmap(pixmap_obverse)
            self.label_obverse.setScaledContents(True)
            self.label_obverse.setPixmap(pixmap_obverse.scaled(600,600, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            self.label_obverse.setMaximumSize(600, 600)

        

    @pyqtSlot()
    def on_reverse_click(self):
        # reverse_fileName = self.openFileNameDialog_reverse()
        # Demo Code, use above when running normally
        reverse_fileName = 'test_images/Lincoln_rev_img_proc.jpg'
        print('Reverse button has been clicked with filepath name: ', reverse_fileName)
        self.upload_path_reverse.setText(reverse_fileName)
        pixmap_reverse = QPixmap(reverse_fileName)
        if pixmap_reverse.isNull():
            print('Reverse filepath does not exist, please try again')
            pixmap_reverse = QPixmap("./LincolnCent/Images/InvalidImage.jpg")
            #self.label_reverse.setPixmap(pixmap_reverse)
            self.label_reverse.setScaledContents(True)
            self.label_reverse.setPixmap(pixmap_reverse.scaled(600,600, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            self.label_reverse.setMaximumSize(600, 600)
        else:
            #self.label_reverse.setPixmap(pixmap_reverse)
            self.label_reverse.setScaledContents(True)
            self.label_reverse.setPixmap(pixmap_reverse.scaled(600,600, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            self.label_reverse.setMaximumSize(600, 600)

        

    @pyqtSlot()
    def on_calculate_fourier_click(self):
        global HSV_Results
        global PCA_Results 
        global FFT_Results
        global FeatureMatch_Results
        fft_results_list = []
        print('Calculate FFT button has been clicked')
        if False == patternMatching.imgIsMSD(self.upload_path_reverse.text()):
            fft_results_list = WheatStalkGrader.gradeWheatStalkPenny(self.upload_path_reverse.text())
            lsg = fft_results_list[0] #left stalk grade
            rsg = fft_results_list[1] #right stalk grade
            oss = fft_results_list[2] #overall sheldon scale grade
            oss_string = ""

            #Error checking for feature ranges
            if ((lsg + rsg)/2) < 46.211:
                oss_string = "<8 | Possible Image Error."
            elif ((lsg + rsg)/2) > 118.711:
                oss_string = ">68 | Above Standard Image Range."
            else:
                oss_string = str(round(oss,3))
                

            FFT_Results = "Left Stalk Grade: " + str(round(lsg,3)) + "\n" + "Right Stalk Grade: " + str(round(rsg,3)) + "\n" + "Estimated Sheldon Scale Grade: " + oss_string
        else:
            fft_results_list = MorganGrader.gradeMorganSilverDollar(self.upload_path_reverse.text())
            blw = fft_results_list[0] #bottom left wing
            brw = fft_results_list[1] #bottom right wing
            ulw = fft_results_list[2] #upper left wing
            urw = fft_results_list[3] #upper right wing
            oss = fft_results_list[4] #overall sheldon scale grade
            oss_string = ""

            #Error checking for feature ranges
            if ((blw + brw + ulw + urw)/4) < 26.194:
                oss_string = "<3 | Possible Image Error."
            elif ((blw + brw + ulw + urw)/4) > 87.487:
                oss_string = "70 | Above Standard Image Range."
            else:
                oss_string = str(round(oss,3))

            FFT_Results = "Lower Left Wing: " + str(round(blw,3)) + "\n" + "Lower Right Wing: " + str(round(brw,3)) + "\n" + "Upper Left Wing: " + str(round(ulw,3)) + "\n" + "Upper Right Wing: " + str(round(urw,3)) + "\n" + "Estimated Sheldon Scale Grade: " + oss_string


        
        self.result_box.setText("HSV Results: " + str(HSV_Results) + "% Red")
        self.result_box.append("\n")
        self.result_box.append("PCA: " + str(PCA_Results))
        self.result_box.append("\n")
        self.result_box.append("FFT Analysis Results: ")
        self.result_box.append(str(FFT_Results))
        self.result_box.append("\n")
        self.result_box.append("Feature Match Results: " + str(FeatureMatch_Results))
        self.result_box.append("\n")      

    @pyqtSlot()
    def on_calculate_HSV_click(self):
        global HSV_Results
        global PCA_Results 
        global FFT_Results
        global FeatureMatch_Results
        HSV_Results = ImageHSV.Image_HSV_Region1(self.upload_path_obverse.text())
        PCA_Results = ImageHSV.ONLY_ONE_COIN_INPUT_FOR_COLOR_CLASSIFICATION(self.upload_path_obverse.text())
        self.result_box.setText("HSV Results: " + str(HSV_Results) + "% Red")
        self.result_box.append("\n")
        self.result_box.append("PCA: " + str(PCA_Results[0]) +str(PCA_Results[1]))
        self.result_box.append("\n")
        self.result_box.append("FFT Analysis Results: ") 
        self.result_box.append(str(FFT_Results))
        self.result_box.append("\n")
        self.result_box.append("Feature Match Results: " + str(FeatureMatch_Results))
        self.result_box.append("\n")

    @pyqtSlot()
    def on_calculate_liberty_click(self):
        global HSV_Results
        global PCA_Results 
        global FFT_Results
        global FeatureMatch_Results
        print('Calculate Pattern Match button has been clicked')
            
        if patternMatching.imgIsMSD(self.upload_path_reverse.text()):
            print('Morgan Silver Dollar detected, attempting to grade')
            PCA_Results = "----"
            FeatureMatch_Results = round(patternMatching.gradeCoin(self.upload_path_reverse.text(), True, False))
        else:
            print('Lincoln Head Cent detected, attempting to grade')
            PCA_Results = ImageHSV.ONLY_ONE_COIN_INPUT_FOR_COLOR_CLASSIFICATION(self.upload_path_obverse.text())[0]
            FeatureMatch_Results = round(patternMatching.gradeCoin(self.upload_path_obverse.text(), False, str(PCA_Results) == 'Brown'))
        
        self.result_box.setText("HSV Results: " + str(HSV_Results) + "% Red")
        self.result_box.append("\n")
        self.result_box.append("PCA: " + str(PCA_Results))
        self.result_box.append("\n")
        self.result_box.append("FFT Analysis Results: ")
        self.result_box.append(str(FFT_Results))
        self.result_box.append("\n")
        self.result_box.append("Feature Match Results: " + str(FeatureMatch_Results))
        self.result_box.append("\n")

def runLWCCode():
    print("\nRunning LWC Grader")
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())


if __name__=='__main__':
    print("Hello world")
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())

