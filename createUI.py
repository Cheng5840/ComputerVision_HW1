import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QGroupBox, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QSpinBox, QTextEdit
import os
import numpy as np
import Q1_functions as q1_funcs
import q1, q2, q3, q4


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MainWindow - cvdlHw1.ui")
        self.setGeometry(100, 100, 800, 600)
        self.loadAllFile = ""  # 圖片文件夾路徑
        self.files = []  # 圖片文件列表
        self.width = 11
        self.height = 8
        self.obj_point = np.zeros((self.width * self.height, 3), dtype=np.float32)
        self.obj_point[:, :2] = np.mgrid[0 : self.width, 0 : self.height].T.reshape(
            -1, 2
        )
        self.matrix = None   # ret, intrinsic, distort, r_vecs, t_vecs

        self.initUI()

    def initUI(self):
        # 主 Widget 與佈局
        mainWidget = QWidget()
        mainLayout = QHBoxLayout(mainWidget)

        # 左側 Load Image 區域
        loadImageGroup = QGroupBox("Load Image")
        loadImageLayout = QVBoxLayout()
        loadImageLayout.setSpacing(10)
        
        loadFolderButton = QPushButton("Load folder")
        loadFolderButton.clicked.connect(lambda: q1.load_folder(self))
        loadFolderButton.setFixedWidth(120)

        loadImageLButton = QPushButton("Load Image_L")
        loadImageLButton.setFixedWidth(120)
        
        loadImageRButton = QPushButton("Load Image_R")
        loadImageRButton.setFixedWidth(120)

        loadImageLayout.addWidget(loadFolderButton)
        loadImageLayout.addWidget(loadImageLButton)
        loadImageLayout.addWidget(loadImageRButton)
        loadImageGroup.setLayout(loadImageLayout)
        loadImageGroup.setFixedHeight(300)

        # 中間 Calibration 區域
        calibrationGroup = QGroupBox("1. Calibration")
        calibrationLayout = QVBoxLayout()
        calibrationLayout.setSpacing(10)

        #1.1
        findCornersButton = QPushButton("1.1 Find corners")
        findCornersButton.clicked.connect(lambda: q1.findCorners(self))
        findCornersButton.setFixedWidth(120)
        
        #1.2
        findIntrinsicButton = QPushButton("1.2 Find intrinsic")
        findIntrinsicButton.clicked.connect(lambda: q1.findInstrinsic(self))
        findIntrinsicButton.setFixedWidth(120)
        
        # 1.3 Find extrinsic with SpinBox
        findExtrinsicGroup = QGroupBox("1.3 Find extrinsic")
        extrinsicLayout = QHBoxLayout()
        extrinsicSpinBox = QSpinBox()
        extrinsicSpinBox.setRange(1, 15)  # 設定範圍 1 到 15
        extrinsicSpinBox.setFixedWidth(40)
        
        findExtrinsicButton = QPushButton("1.3 Find extrinsic")
        findExtrinsicButton.clicked.connect(lambda: q1.findExtrinsic(self, extrinsicSpinBox.value() - 1))

        findExtrinsicButton.setFixedWidth(120)       
        extrinsicLayout.addWidget(extrinsicSpinBox)
        extrinsicLayout.addWidget(findExtrinsicButton)
        findExtrinsicGroup.setLayout(extrinsicLayout)

        #1.4
        findDistortionButton = QPushButton("1.4 Find distortion")
        findDistortionButton.clicked.connect(lambda: q1.findDistorsion(self))
        findDistortionButton.setFixedWidth(120)
        
        #1.5
        showResultButton = QPushButton("1.5 Show result")
        showResultButton.clicked.connect(lambda: q1.showResultClick(self))
        showResultButton.setFixedWidth(120)

        calibrationLayout.addWidget(findCornersButton)
        calibrationLayout.addWidget(findIntrinsicButton)
        calibrationLayout.addWidget(findExtrinsicGroup)
        calibrationLayout.addWidget(findDistortionButton)
        calibrationLayout.addWidget(showResultButton)
        calibrationGroup.setLayout(calibrationLayout)
        calibrationGroup.setFixedHeight(300)

        # 中間 Augmented Reality 區域
        # 2.1
        arGroup = QGroupBox("2. Augmented Reality")
        arLayout = QVBoxLayout()
        arLayout.setSpacing(10)
        
        textInput = QTextEdit()
        textInput.setFixedHeight(30)
        
        showWordsOnBoardButton = QPushButton("2.1 show words on board")
        showWordsOnBoardButton.setFixedWidth(120)
        showWordsOnBoardButton.clicked.connect(lambda: q2.horizontallyShow(self,  textInput.toPlainText()))  # 以 "K" 為例

        # 2.2
        showWordsVerticalButton = QPushButton("2.2 show words vertical")
        showWordsVerticalButton.setFixedWidth(120)
        showWordsVerticalButton.clicked.connect(lambda: q2.verticallyShow(self,  textInput.toPlainText()))

        arLayout.addWidget(textInput)
        arLayout.addWidget(showWordsOnBoardButton)
        arLayout.addWidget(showWordsVerticalButton)
        arGroup.setLayout(arLayout)
        arGroup.setFixedHeight(300)


        # 右側 Stereo Disparity Map 區域
        #3.1
        stereoGroup = QGroupBox("3. Stereo disparity map")
        stereoLayout = QVBoxLayout()
        stereoLayout.setSpacing(10)
        
        stereoButton = QPushButton("3.1 stereo disparity map")
        stereoButton.setFixedWidth(120)
        stereoButton.clicked.connect(lambda: q3.disparityMap(self))

        stereoLayout.addWidget(stereoButton)
        stereoGroup.setLayout(stereoLayout)
        stereoGroup.setFixedHeight(300)

        # 右下角 SIFT 區域
        siftGroup = QGroupBox("4. SIFT")
        siftLayout = QVBoxLayout()
        siftLayout.setSpacing(10)
        
        loadImage1Button = QPushButton("Load Image1")
        loadImage1Button.setFixedWidth(120)
        
        loadImage2Button = QPushButton("Load Image2")
        loadImage2Button.setFixedWidth(120)
        
        # 4.1
        keypointsButton = QPushButton("4.1 Keypoints")
        keypointsButton.setFixedWidth(120)
        keypointsButton.clicked.connect(lambda: q4.createKeyPoint(self))
        
        # 4.2
        matchedKeypointsButton = QPushButton("4.2 Matched Keypoints")
        matchedKeypointsButton.setFixedWidth(120)
        matchedKeypointsButton.clicked.connect(lambda: q4.matchedKeyPoint(self))

        siftLayout.addWidget(loadImage1Button)
        siftLayout.addWidget(loadImage2Button)
        siftLayout.addWidget(keypointsButton)
        siftLayout.addWidget(matchedKeypointsButton)
        siftGroup.setLayout(siftLayout)
        siftGroup.setFixedHeight(300)

        # 將所有組件加到主佈局中
        mainLayout.addWidget(loadImageGroup)
        mainLayout.addWidget(calibrationGroup)
        mainLayout.addWidget(arGroup)
        mainLayout.addWidget(stereoGroup)
        mainLayout.addWidget(siftGroup)

        # 設定所有按鈕的高度
        all_buttons = [
            loadFolderButton, loadImageLButton, loadImageRButton,
            findCornersButton, findIntrinsicButton, findExtrinsicButton,
            findDistortionButton, showResultButton,
            showWordsOnBoardButton, showWordsVerticalButton,
            stereoButton, loadImage1Button, loadImage2Button,
            keypointsButton, matchedKeypointsButton
        ]
        self.set_buttons_height(all_buttons, height=40)

        self.setCentralWidget(mainWidget)

    # 設定所有按鈕高度的函式
    def set_buttons_height(self, buttons, height=40):
        for button in buttons:
            button.setFixedHeight(height)





if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
