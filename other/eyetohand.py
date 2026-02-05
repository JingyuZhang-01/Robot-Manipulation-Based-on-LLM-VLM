import yaml
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import re

#============获取棋盘格在相机下位姿的相关函数===============
def load_image_paths_from_folder(folder_path, suffix=".jpg"):
    """
    从文件夹中读取所有图片路径，并按数字顺序排序
    例如: 1.jpg, 2.jpg, 10.jpg
    """
    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(suffix)
    ]

    # 按文件名中的数字排序（核心）
    files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))

    image_paths = [
        os.path.join(folder_path, f) for f in files
    ]

    return image_paths


def compute_chessboard_poses_from_images(image_paths):
    """
    输入:
        image_paths: list[str], 长度为24的图片路径（按顺序）
    输出:
        poses: list, 每个元素是 4x4 齐次变换矩阵 (cam -> board)，
               若该图像未成功检测棋盘，则为 None
    """

    # ================== 参数区（与你原代码完全一致） ==================
    XX = 7
    YY = 4
    L  = 0.02

    #========================right=========================
    # K = np.array([
    #     [573.92496691,   0.  ,       321.00277595],
    #     [  0.   ,      575.10268636, 258.62855548],
    #     [  0.   ,        0.    ,       1.        ]
    # ])

    #right color
    # K = np.array([
    # [606.1142578125, 0.0, 329.6438903808594],
    # [0.0, 605.7283935546875, 256.7432556152344],
    # [0.0, 0.0, 1.0]
    # ])

    dist = np.array([[-0.06348319,  1.46949519, -0.00626365, -0.0066574,  -4.4033578]])

    #============================left=============================
    K = np.array([
    [512.07600085,   0.  ,       308.42023199],
    [  0.     ,    508.92735394, 267.06333321],
    [  0.     ,      0.      ,     1.        ]])


    # #left color
    # K = np.array([
    #     [607.93701171875, 0.0, 310.13470458984375],
    #     [0.0, 607.7496948242188, 250.61622619628906],
    #     [0.0, 0.0, 1.0]
    # ])
    dist = np.array([[-1.98054329e-01,  1.32939597e+00,  1.37618009e-02, -1.03789197e-03, -3.00825429e+00]])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    # 世界坐标系下棋盘角点（Z=0）
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)
    objp *= L
    # ===============================================================

    poses = []

    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] 第 {idx} 张图像读取失败: {img_path}")
            poses.append(None)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, (XX, YY), None)

        if not found:
            print(f"[WARN] 第 {idx} 张图像未检测到棋盘")
            poses.append(None)
            continue

        corners_sub = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        ok, rvec, tvec = cv2.solvePnP(
            objp, corners_sub, K, dist
        )

        if not ok:
            print(f"[WARN] 第 {idx} 张图像 solvePnP 失败")
            poses.append(None)
            continue


        R, _ = cv2.Rodrigues(rvec)
        tvec_value = tvec.ravel()
        print(f"第 {idx} 张图像 tvec.ravel() = [{tvec_value[0]:.6f}, {tvec_value[1]:.6f}, {tvec_value[2]:.6f}]")
        # 构造齐次变换矩阵 T_cam2board
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = tvec.ravel()

        poses.append(T)

    return poses




#============获取机械臂末端在基座标的位姿相关函数===============
def CvtEulerAngleToRotationMatrix(pose):
    x, y, z, rx, ry, rz = pose
    ptrEulerAngle = np.radians([rx, ry, rz])
    ptrSinAngle = np.sin(ptrEulerAngle)
    ptrCosAngle = np.cos(ptrEulerAngle)
    ptrRotationMatrix = np.zeros((3, 3))

    ptrRotationMatrix[0, 0] = ptrCosAngle[2] * ptrCosAngle[1]
    ptrRotationMatrix[0, 1] = ptrCosAngle[2] * ptrSinAngle[1] * ptrSinAngle[0] - ptrSinAngle[2] * ptrCosAngle[0]
    ptrRotationMatrix[0, 2] = ptrCosAngle[2] * ptrSinAngle[1] * ptrCosAngle[0] + ptrSinAngle[2] * ptrSinAngle[0]

    ptrRotationMatrix[1, 0] = ptrSinAngle[2] * ptrCosAngle[1]
    ptrRotationMatrix[1, 1] = ptrSinAngle[2] * ptrSinAngle[1] * ptrSinAngle[0] + ptrCosAngle[2] * ptrCosAngle[0]
    ptrRotationMatrix[1, 2] = ptrSinAngle[2] * ptrSinAngle[1] * ptrCosAngle[0] - ptrCosAngle[2] * ptrSinAngle[0]

    ptrRotationMatrix[2, 0] = -ptrSinAngle[1]
    ptrRotationMatrix[2, 1] = ptrCosAngle[1] * ptrSinAngle[0]
    ptrRotationMatrix[2, 2] = ptrCosAngle[1] * ptrCosAngle[0]

    T = np.eye(4)
    T[0:3, 0:3] = ptrRotationMatrix
    T[0:3, 3] = [x, y, z]
    return T



def load_ee_base_inverse_transforms(txt_path):
    """
    输入: txt_path: 末端位姿txt文件路径
    输出: T_ee2base_list: list[np.ndarray], 每个元素是 4x4 的齐次矩阵     顺序与 txt 中行顺序完全一致
    """

    T_ee2base_list = []

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # 解析一行 pose
        pose = list(map(float, line.split(',')))
        if len(pose) != 6:
            raise ValueError(f"第 {idx} 行格式错误: {line}")

        # 基座标下的末端
        T = CvtEulerAngleToRotationMatrix(pose)

        T_ee2base_list.append(T)

    return T_ee2base_list



#=================求解相机在基座标系下的齐次矩阵===============
def handeye_calibration_from_T(T_ee2base_all, T_target2cam):
    """
    完全仿照 handeye_calibration() 的计算逻辑，
    只是输入从 xyzrxryrz 换成齐次矩阵 T

    Args:
        T_ee2base_all: list / array of (4,4)
                       末端 -> 基座
        T_target2cam: list / array of (4,4)
                       标定板 -> 相机

    Returns:
        T_cam2base: (4,4) 相机 -> 基座 齐次矩阵
    """

    end2base_R, end2base_t = [], []
    board2cam_R, board2cam_t = [], []

    # ===== 1. ee2base -> end2base_R, end2base_t =====
    for T in T_ee2base_all:
        if T is None:
            continue
        R = T[:3, :3]
        t = T[:3, 3].reshape(3, 1)
        end2base_R.append(R)
        end2base_t.append(t)

    # ===== 2. target2cam -> board2cam_R, board2cam_t =====
    for T in T_target2cam:
        if T is None:
            continue
        R = T[:3, :3]
        t = T[:3, 3].reshape(3, 1)
        board2cam_R.append(R)
        board2cam_t.append(t)

    # ===== 3. end2base -> base2end（完全按你给的方式再求逆）=====
    base2end_R, base2end_t = [], []
    for R, t in zip(end2base_R, end2base_t):
        R_b2e = R.T
        t_b2e = -R_b2e @ t
        base2end_R.append(R_b2e)
        base2end_t.append(t_b2e)

    # ===== 4. 使用 Tsai 方法 =====
    cam2base_R, cam2base_t = cv2.calibrateHandEye(
        base2end_R,
        base2end_t,
        board2cam_R,
        board2cam_t,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    print(f'@@TSAI@@\ncam2base_R:\n{cam2base_R}\ncam2base_t:\n{cam2base_t}')

    # ===== 5. 使用 Park 方法（与你原函数一致）=====
    cam2base_R, cam2base_t = cv2.calibrateHandEye(
        base2end_R,
        base2end_t,
        board2cam_R,
        board2cam_t,
        method=cv2.CALIB_HAND_EYE_PARK
    )

    print(f'@@PARK@@\ncam2base_R:\n{cam2base_R}\ncam2base_t:\n{cam2base_t}')

    # ===== 6. 组装齐次矩阵 cam2base =====
    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = cam2base_R
    T_cam2base[:3, 3] = cam2base_t.reshape(3)

    return T_cam2base


#===============================主函数======================================
#0106左 010602右
folder_path = "/home/elephant/Grasp/Vox/hand_eye_calibration-main/eye_hand_data/data20260106" #保存的棋盘格数据
txt_path = "/home/elephant/Grasp/Vox/hand_eye_calibration-main/eye_hand_data/data20260106/poses.txt" #机械臂末端数据
image_paths = load_image_paths_from_folder(folder_path)

T_target2cam = compute_chessboard_poses_from_images(image_paths)

T_ee2base_all = load_ee_base_inverse_transforms(txt_path)

print(f"[INFO] target2cam 数量: {len(T_target2cam)}")
print(f"[INFO] ee2base 数量  : {len(T_ee2base_all)}")

T_cam2base = handeye_calibration_from_T(
    T_ee2base_all,
    T_target2cam
)





