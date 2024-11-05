import cv2
import numpy as np
import os

def calibration2(files, load_folder, obj_point, width, height):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    point_3D = []
    point_2D = []

    for file in files:
        img = cv2.imread(os.path.join(load_folder, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
        if ret:
            new_corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criteria)
            point_3D.append(obj_point)
            point_2D.append(new_corners)

    matrix01 = cv2.calibrateCamera(point_3D, point_2D, gray.shape[::-1], None, None)
    return matrix01

def draw(img, corners, img_point, length):
    img_point = np.int32(img_point).reshape(-1, 2)
    for i in range(length):
        img = cv2.line(img, tuple(img_point[2 * i]), tuple(img_point[2 * i + 1]), (0, 0, 255), 15)
    return img

def horizontallyClick(files, load_folder, in_word, matrix01, obj_point, width, height):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    word = []
    text = in_word.upper()
    lib = os.path.join(load_folder, "Q2_lib/alphabet_lib_onboard.txt")
    fs = cv2.FileStorage(lib, cv2.FILE_STORAGE_READ)

    length = 0
    for char in text:
        if char.isalpha():
            word.append(fs.getNode(char).mat())
            length += 1

    pos_adjust = [[7, 5, 0], [4, 5, 0], [1, 5, 0], [7, 2, 0], [4, 2, 0], [1, 2, 0]]
    for i in range(length):
        for j in range(len(word[i])):
            word[i][j][0] = [a + b for a, b in zip(word[i][j][0], pos_adjust[i])]
            word[i][j][1] = [a + b for a, b in zip(word[i][j][1], pos_adjust[i])]

    for i, file in enumerate(files):
        img = cv2.imread(os.path.join(load_folder, file))
        rotation_vector = cv2.Rodrigues(matrix01[3][i])[0]
        transform_vector = matrix01[4][i]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, corners = cv2.findChessboardCorners(gray, (width, height), None)
        new_corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criteria)

        for j in range(len(word)):
            axis = np.array(word[j], dtype=np.float32).reshape(-1, 3)
            img_points, _ = cv2.projectPoints(axis, rotation_vector, transform_vector, matrix01[1], matrix01[2])
            img = draw(img, new_corners, img_points, len(word[j]))

        cv2.imshow("Augmented Reality", img)
        cv2.waitKey(1000)
    cv2.destroyAllWindows()

# `verticallyClick` 與 `horizontallyClick` 類似，只需改變座標加載的文件路徑
