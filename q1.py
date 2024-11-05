import cv2
import numpy as np
import os
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from natsort import natsorted


def load_folder(self):
    """讓使用者選擇資料夾並加載圖片的相對路徑"""
    folder_path = QFileDialog.getExistingDirectory(None, "選擇資料夾")
    if folder_path:
        # 將每張圖片的路徑轉換為相對路徑
        self.files = natsorted(
            [os.path.relpath(os.path.join(folder_path, f), start=os.getcwd())
             for f in os.listdir(folder_path) if f.endswith('.bmp')]
        )
        print(f"已載入資料夾：{folder_path}")
        print(f"找到圖片（相對路徑）：{self.files}")

def findCorners(self):
    # termination criterias
    criterias = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    point_3D = []
    point_2D = []
    for file in self.files:
        img = cv2.imread(os.path.join(self.loadAllFile, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (self.width, self.height), None)
        if ret == True:
            new_corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criterias)
            img_out = cv2.drawChessboardCorners(
                img, (self.width, self.height), new_corners, ret
            )
            point_3D.append(self.obj_point)
            point_2D.append(new_corners)
            cv2.namedWindow("Corners", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Corners", 1024, 1024)
            cv2.imshow("Corners", img_out)
            cv2.waitKey(400)
    self.matrix = cv2.calibrateCamera(
        point_3D, point_2D, gray.shape[::-1], None, None
    )
    cv2.destroyAllWindows()


def findInstrinsic(self):
    criterias = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    point_3D = []
    point_2D = []
    for file in self.files:
        img = cv2.imread(os.path.join(self.loadAllFile, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.width, self.height), None)
        if ret == True:
            new_corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criterias)
            point_3D.append(self.obj_point)
            point_2D.append(new_corners)
    self.matrix = cv2.calibrateCamera(
        point_3D, point_2D, gray.shape[::-1], None, None
    )
    print(self.matrix)
    print("Intrinsic:", self.matrix[1], sep="\n")

    msg_box = QMessageBox()
    msg_box.setWindowTitle("Intrinsic Matrix")
    msg_box.setText(f"Intrinsic matrix (camera matrix):\n{self.matrix[1]}")
    msg_box.exec_()


def findExtrinsic(self, imgIdx):
    criterias = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    point_3D = []
    point_2D = []
    for file in self.files:
        img = cv2.imread(os.path.join(self.loadAllFile, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.width, self.height), None)
        if ret == True:
            new_corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criterias)
            point_3D.append(self.obj_point)
            point_2D.append(new_corners)
    self.matrix = cv2.calibrateCamera(
        point_3D, point_2D, gray.shape[::-1], None, None
    )
    i = int(imgIdx) - 1
    rotation_matrix = cv2.Rodrigues(self.matrix[3][i])[0]
    extrinsic_matrix = np.hstack([rotation_matrix, self.matrix[4][i]])
    print("Extrinsic", extrinsic_matrix, sep="\n")

    msg_box = QMessageBox()
    msg_box.setWindowTitle("Extrinsic Matrix")
    msg_box.setText(f"Extrinsic Matrix for Image {imgIdx + 1}:\n{extrinsic_matrix}")
    msg_box.exec_()


def findDistorsion(self):
    criterias = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    point_3D = []
    point_2D = []
    for file in self.files:
        img = cv2.imread(os.path.join(self.loadAllFile, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.width, self.height), None)
        if ret == True:
            new_corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criterias)
            point_3D.append(self.obj_point)
            point_2D.append(new_corners)
    self.matrix = cv2.calibrateCamera(
        point_3D, point_2D, gray.shape[::-1], None, None
    )
    print("Distorsion:", self.matrix[2], sep="\n")

    msg_box = QMessageBox()
    msg_box.setWindowTitle("Distortion Matrix")
    msg_box.setText(f"Distortion Coefficients:\n{self.matrix[2]}")
    msg_box.exec_()


def showResultClick(self):
    criterias = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    point_3D = []
    point_2D = []
    for file in self.files:
        img = cv2.imread(os.path.join(self.loadAllFile, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.width, self.height), None)
        if ret == True:
            new_corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criterias)
            point_3D.append(self.obj_point)
            point_2D.append(new_corners)
    self.matrix = cv2.calibrateCamera(
        point_3D, point_2D, gray.shape[::-1], None, None
    )
    for file in self.files:
        img = cv2.imread(os.path.join(self.loadAllFile, file))
        h, w = img.shape[:2]
        newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(
            self.matrix[1], self.matrix[2], (w, h), 0, (w, h)
        )
        dst = cv2.undistort(
            img, self.matrix[1], self.matrix[2], None, newcameramatrix
        )

        x, y, w, h = roi
        dst = dst[y : y + h, x : x + w]
        img = cv2.resize(img, (480, 480))
        dst = cv2.resize(dst, (480, 480))
        imgs = np.hstack([dst, img])
        cv2.imshow("undistorted result", imgs)
        cv2.waitKey(400)
    cv2.destroyAllWindows()