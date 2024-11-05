import cv2
import numpy as np
from PyQt5.QtWidgets import QMessageBox,QApplication, QMainWindow, QPushButton, QLabel, QGroupBox, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QSpinBox, QTextEdit
import os
from natsort import natsorted

# 全域變數，用於儲存圖片路徑
image_paths = []
object_points = []  # 3D 點在真實世界的座標
image_points = []   # 2D 點在影像中的座標
rvec = []          # 儲存旋轉向量
tvec = []          # 儲存平移向量
dist = None         # 儲存失真係數
K = None            # 儲存內參矩陣

# 棋盤格的尺寸和每格的大小
chessboard_size = (11, 8, 1)
square_size = 0.02  # 每格的邊長，單位為米

def load_folder():
    """讓使用者選擇資料夾並加載圖片的相對路徑"""
    folder_path = QFileDialog.getExistingDirectory(None, "選擇資料夾")
    global image_paths
    if folder_path:
        # 將每張圖片的路徑轉換為相對路徑
        image_paths = natsorted(
            [os.path.relpath(os.path.join(folder_path, f), start=os.getcwd())
             for f in os.listdir(folder_path) if f.endswith('.bmp')]
        )
        print(f"已載入資料夾：{folder_path}")
        print(f"找到圖片（相對路徑）：{image_paths}")

def detect_corners_from_folder():
    """偵測每張圖片的角點並準備校正數據"""
    global object_points, image_points, image_paths, corners
    object_points = []  # 清空先前的資料
    image_points = []
    winSize = (5,5)
    zeroZone = (-1,-1)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

    #3D 物體點座標（假設 Z = 0）
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
    # 設定圖片顯示的大小
    display_width = 800
    display_height = 600

    ######################
    # 單獨測試 4.bmp
    # img_path = 'Q1_Image/4.bmp'  # 修改為 `4.bmp` 的相對路徑
    # img = cv2.imread(img_path)
    # if img is not None:
    #     img = cv2.resize(img, (display_width, display_height))
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     ret, corners = cv2.findChessboardCorners(gray, chessboard_size[0:2])
    #     if ret:
    #         print("找到角點")
    #         corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    #         img = cv2.drawChessboardCorners(img, chessboard_size[0:2], corners, ret)
    #         cv2.imshow("Detected Corners - 4.bmp", img)
    #         cv2.waitKey(0)
    #     else:
    #         print("未找到角點")
    # cv2.destroyAllWindows()
    ##########################

    for img_path in image_paths:
        print(img_path)
        img = cv2.imread(img_path)
        if img_path is None:
            print(f"警告：無法讀取圖片 '{img_path}'。請檢查路徑或檔案格式。")
            continue
        
        img = cv2.resize(img, (display_width, display_height))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size[0:2])
        if ret:
            print('ret:', img_path)
            object_points.append(objp)
            corners = cv2.cornerSubPix(gray, corners, winSize, zeroZone,criteria)
            image_points.append(corners)
            img = cv2.drawChessboardCorners(img, chessboard_size[0:2], corners, ret)
            
            window_name = f'Detected Corners - {os.path.basename(img_path)}'
            cv2.imshow(window_name, img)
            cv2.waitKey(500)
            cv2.destroyWindow(window_name)
    cv2.destroyAllWindows()



def find_intrinsic_matrix():
    
    global K, dist, rvec, tvec
    """計算並顯示內參矩陣"""
    if len(object_points) == 0 or len(image_points) == 0:
        print("沒有足夠的資料來計算內參矩陣。請先執行角點偵測。")
        return

    # 假設所有圖片的尺寸相同，這裡可以取第一張圖片的尺寸
    img = cv2.imread(image_paths[0])
    img_shape = (2048, 2048)  # (寬, 高)

    # 使用 cv2.calibrateCamera 計算相機的內參矩陣
    ret, K, dist, rvec, tvec = cv2.calibrateCamera(object_points, image_points, img_shape, None, None)

    # 顯示內參矩陣
    print("Intrinsic Matrix (Camera Matrix):")
    print(K)

    # 顯示結果在訊息框中
    msg_box = QMessageBox()
    msg_box.setWindowTitle("Intrinsic Matrix")
    msg_box.setText(f"Intrinsic matrix (camera matrix):\n{K}")
    msg_box.exec_()

    return K, dist, rvec, tvec


def find_extrinsic_matrix(image_index):
    """
    計算並顯示指定圖像的外參矩陣。
    image_index: 要計算的影像索引（從0開始）
    rvec: 從 cv2.calibrateCamera 計算出的旋轉向量
    tvec: 從 cv2.calibrateCamera 計算出的平移向量
    """
    if image_index < 0 or image_index >= len(rvec):
        print("無效的圖像索引")
        return

    # 使用 Rodrigues 函式將旋轉向量轉換為旋轉矩陣
    rotation_matrix, _ = cv2.Rodrigues(rvec[image_index])

    # 將旋轉矩陣和平移向量合併成外參矩陣
    extrinsic_matrix = np.hstack((rotation_matrix, tvec[image_index]))

    # 顯示外參矩陣
    print(f"Extrinsic Matrix for Image {image_index + 1}:")
    print(extrinsic_matrix)

    # 使用 QMessageBox 顯示外參矩陣
    msg_box = QMessageBox()
    msg_box.setWindowTitle("Extrinsic Matrix")
    msg_box.setText(f"Extrinsic Matrix for Image {image_index + 1}:\n{extrinsic_matrix}")
    msg_box.exec_()

    return extrinsic_matrix

def find_distortion_matrix():
    """顯示失真矩陣"""
    global dist
    if dist is None:
        print("請先執行內參矩陣計算以獲取失真係數。")
        return

    # 顯示失真係數
    print("Distortion Coefficients:")
    print(dist)

    # 使用 QMessageBox 顯示失真係數
    msg_box = QMessageBox()
    msg_box.setWindowTitle("Distortion Matrix")
    msg_box.setText(f"Distortion Coefficients:\n{dist}")
    msg_box.exec_()

    return dist

def show_undistorted_result():
    """顯示失真和未失真結果"""
    global K, dist, image_paths
    if K is None or dist is None:
        print("請先執行內參和失真係數計算。")
        return

    # 設定顯示影像的寬度和高度
    display_width = 800
    display_height = 600

    # 遍歷每張圖片，顯示失真和未失真結果
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：無法讀取圖片 '{img_path}'。請檢查路徑或檔案格式。")
            continue

        # 將影像轉為灰度影像
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 使用內參矩陣 K 和失真係數 dist 矯正影像
        result_img = cv2.undistort(gray_img, K, dist)

        # 調整影像大小
        img_resized = cv2.resize(gray_img, (display_width, display_height))
        undistorted_resized = cv2.resize(result_img, (display_width, display_height))

        # 將原始（失真）影像和未失真影像並排顯示
        combined = np.hstack((img_resized, undistorted_resized))
        window_name = f'Undistorted Result - {os.path.basename(img_path)}'
        cv2.imshow(window_name, combined)
        
        # 等待使用者按下任意鍵以切換到下一張圖片
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    cv2.destroyAllWindows()



#################-------------Q2--------------####################
import cv2
import numpy as np
from PyQt5.QtWidgets import QMessageBox, QFileDialog
import os

# 假設之前已經獲得 K, dist, rvecs, 和 tvecs 的值

def load_alphabet_coordinates(letter, filename="Q2_Image/Q2_db/alphabet_db_onboard.txt"):
    """從文件中加載指定字母的 3D 座標"""
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    charPoints = fs.getNode(letter).mat()  # 加載字母的 3D 座標
    fs.release()
    return charPoints

def show_words_on_board(letter="K", display_width=800, display_height=600):
    """將字母顯示在棋盤上，並調整圖像顯示大小"""
    global K, dist, rvec, tvec, image_paths
    if K is None or dist is None or not rvec or not tvec:
        print("請先完成相機校正")
        return

    # 加載字母的 3D 坐標
    charPoints = load_alphabet_coordinates(letter)
    if charPoints is None:
        print(f"無法找到字母 {letter} 的座標")
        return
    print("charpoints:", charPoints)
    
    # 確保 charPoints 是正確的格式
    charPoints = charPoints.astype(np.float32).reshape(-1, 1, 3)

    # 遍歷每張圖像
    for i, img_path in enumerate(image_paths[:5]):  # 只處理前 5 張圖片
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：無法讀取圖片 '{img_path}'。請檢查路徑或檔案格式。")
            continue

        # 調整顯示圖像的大小
        img = cv2.resize(img, (display_width, display_height))

        # 投影 3D 字母點到 2D 影像
        newCharPoints, _ = cv2.projectPoints(charPoints, rvec[i], tvec[i], K, dist)
        
        # 列出投影後的 2D 坐標
        print(f"Projected 2D points for image {i + 1}:")
        for point in newCharPoints:
            print(f"({point[0][0]:.2f}, {point[0][1]:.2f})")

        # 畫出字母的線條
        for j in range(0, len(newCharPoints) - 1, 2):
            try:
                pointA = (int(newCharPoints[j][0][0]), int(newCharPoints[j][0][1]))
                pointB = (int(newCharPoints[j + 1][0][0]), int(newCharPoints[j + 1][0][1]))
            except (TypeError, ValueError) as e:
                print(f"跳過無效點對，錯誤：{e}")
                continue

            # 檢查座標是否在圖像範圍內
            if 0 <= pointA[0] < display_width and 0 <= pointA[1] < display_height and \
               0 <= pointB[0] < display_width and 0 <= pointB[1] < display_height:
                cv2.line(img, pointA, pointB, (0, 255, 0), 2)
            else:
                print("跳過超出範圍的點對")

        # 顯示結果
        cv2.imshow(f"Word on Board - {letter}", img)
        cv2.waitKey(1000)  # 每張圖片顯示 1 秒
        cv2.destroyAllWindows()

