import os
import sys
import time
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from graspnetAPI import GraspGroup
import pyrealsense2 as rs

# ============================================================
#  路径 & import（与你参考代码一致）
# ============================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
# =========================
#  相机 → 基座标 变换矩阵
# =========================

T_base_left = np.array([
    [ 0.96974678, 0.05240736, -0.23842117, 0.29],
    [ 0.16478827, -0.86110891, 0.4809743, -0.30],
    [-0.1801, -0.50571229, -0.84369371, 0.42],
    [ 0.0, 0.0, 0.0, 1.0]
])

T_base_right = np.array([
    [-0.97616224, -0.03089485, -0.21483199, 0.37],
    [ 0.09935684, 0.81640497, -0.5688683, 0.35],
    [ 0.19296501, -0.57665278, -0.79387409, 0.38],
    [ 0.0, 0.0, 0.0, 1.0]
])

# =========================
#  相机初始化以获得内参
# =========================

K_left = {'fx': 607.93701171875,'fy': 607.7496948242188,'cx': 310.13470458984375,'cy': 250.61622619628906}
K_right = {'fx': 606.1142578125,'fy': 605.7283935546875,'cx': 329.6438903808594,'cy': 256.7432556152344}




# =========================
#  路径
# =========================
CHECKPOINT_PATH = '/home/elephant/Grasp/graspnet-baseline/logs/log_rs/checkpoint-rs.tar'
NUM_VIEW = 300
NUM_POINT = 4096
VOXEL_SIZE = 0.005
COLLISION_THRESH = 0.01

DEPTH_FACTOR = 1000.0

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "/home/elephant/Grasp/github/Robot-Manipulation-Based-on-LLM-VLM-main/data"
LEFT_DIR = os.path.join(DATA_ROOT, "left")
RIGHT_DIR = os.path.join(DATA_ROOT, "right")
SAVE_DIR = os.path.join(DATA_ROOT, "grasp")
os.makedirs(SAVE_DIR, exist_ok=True)

#============================================================
#  GraspNet 网络加载（一次）
# ============================================================

def load_graspnet():
    net = GraspNet(
        input_feature_dim=0,
        num_view=NUM_VIEW,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False
    ).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net


# ============================================================
#  图像处理（与你给的 get_and_process_data 完全一致）
# ============================================================

def get_and_process_data(color_path, depth_path, mask_path, K_depth):
    color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    depth = np.array(Image.open(depth_path))
    workspace_mask = np.array(Image.open(mask_path))

    TARGET_W, TARGET_H = 640, 480

    if depth.shape != (TARGET_H, TARGET_W):
        depth = np.array(Image.fromarray(depth).resize((TARGET_W, TARGET_H), Image.NEAREST))
    if color.shape[:2] != (TARGET_H, TARGET_W):
        color = np.array(
            Image.fromarray((color * 255).astype(np.uint8)).resize((TARGET_W, TARGET_H))
        ) / 255.0
    if workspace_mask.shape != (TARGET_H, TARGET_W):
        workspace_mask = np.array(
            Image.fromarray(workspace_mask).resize((TARGET_W, TARGET_H), Image.NEAREST)
        )

    camera = CameraInfo(
        width=640,
        height=480,
        fx=K_depth['fx'],
        fy=K_depth['fy'],
        cx=K_depth['cx'],
        cy=K_depth['cy'],
        scale=DEPTH_FACTOR
    )
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    # 全点云（不 mask）
    cloud_all = cloud.reshape(-1, 3)
    color_all = color.reshape(-1, 3)
    valid = depth.reshape(-1) > 0

    cloud_all = cloud_all[valid]
    color_all = color_all[valid]

    # 目标点云
    obj_mask = (workspace_mask > 0) & (depth > 0)
    cloud_obj = cloud[obj_mask]
    color_obj = color[obj_mask]
    # cloud_obj = filter_by_local_density(cloud_obj, k=5)
    # cloud_obj = filter_largest_cluster(cloud_obj, eps=0.01, min_points=100)
    

    # 环境点云 = 全部 - 目标
    env_mask = (~workspace_mask.astype(bool)) & (depth > 0)
    cloud_env = cloud[env_mask]
    color_env = color[env_mask]

    cloud_o3d_obj = o3d.geometry.PointCloud()
    cloud_o3d_obj.points = o3d.utility.Vector3dVector(cloud_obj)
    cloud_o3d_obj.colors = o3d.utility.Vector3dVector(color_obj)

    cloud_o3d_env = o3d.geometry.PointCloud()
    cloud_o3d_env.points = o3d.utility.Vector3dVector(cloud_env)
    cloud_o3d_env.colors = o3d.utility.Vector3dVector(color_env)


    if len(cloud_obj) >= NUM_POINT:
        idxs = np.random.choice(len(cloud_obj), NUM_POINT, replace=False)
    else:
        idxs = np.concatenate([
            np.arange(len(cloud_obj)),
            np.random.choice(len(cloud_obj), NUM_POINT - len(cloud_obj), replace=True)
        ])

    cloud_sampled = cloud_obj[idxs]
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(DEVICE)

    end_points = {'point_clouds': cloud_sampled}

    object_center = cloud_obj.mean(axis=0)
    return end_points, cloud_o3d_obj, cloud_o3d_env, object_center



#======================离群点处理=============================
def filter_largest_cluster(cloud_np, eps=0.01, min_points=50):
    """
    cloud_np: (N,3)
    return: filtered cloud (M,3)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_np)

    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )

    if labels.max() < 0:
        print("[WARN] No clusters found, skip clustering")
        return cloud_np

    largest_label = np.argmax(np.bincount(labels[labels >= 0]))
    return cloud_np[labels == largest_label]

def filter_by_local_density(cloud_np, k=20, dist_thresh=None):
    """
    cloud_np: (N,3)
    k: KNN 邻居数
    dist_thresh: 距离阈值（None 表示自适应）
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_np)

    kdtree = o3d.geometry.KDTreeFlann(pcd)

    mean_dists = []
    for i, p in enumerate(cloud_np):
        _, idx, dists = kdtree.search_knn_vector_3d(p, k)
        mean_dists.append(np.mean(np.sqrt(dists[1:])))  # 排除自己

    mean_dists = np.array(mean_dists)

    # ✅ 自适应阈值（关键）
    if dist_thresh is None:
        dist_thresh = np.percentile(mean_dists, 70)

    keep = mean_dists < dist_thresh
    return cloud_np[keep]


# ============================================================
#  单相机抓取推理（完全等价于 run_grasp_inference）
# ============================================================
# ==================== 碰撞检测 ====================
def collision_detection(gg, cloud_points):
    mfcdetector = ModelFreeCollisionDetector(cloud_points, voxel_size=VOXEL_SIZE)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=COLLISION_THRESH)
    return gg[~collision_mask]

def infer_best_grasp(color_path, depth_path, mask_path, K, visualize=False):
    """
    输入：
        color_path, depth_path, mask_path : str
        visualize : bool, 是否可视化 top 抓取

    输出：
        best_grasp : dict
            {
                'translation': np.array(3,),
                'rotation': np.array(3,3),
                'width': float
            }
    """

    # 1️⃣ 加载网络
    
    net = load_graspnet()

    # 2️⃣ 数据处理
    end_points, cloud_o3d_obs,cloud_o3d_env, object_center = get_and_process_data(color_path, depth_path, mask_path, K )

    # 3️⃣ 前向推理
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)

    # 4️⃣ 封装成 GraspGroup
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())

    # 5️⃣ 碰撞检测
    if COLLISION_THRESH > 0:
        gg = collision_detection(gg, np.asarray(cloud_o3d_env.points))

    # 6️⃣ NMS 去重 + 按得分排序
    gg.nms().sort_by_score()


    # ===============================
    # 7️⃣ 垂直角度筛选
    # ===============================
    all_grasps = list(gg)
    filtered = all_grasps

    # ===============================
    # 8️⃣ Top-K → 在 K 个里选最优
    # ===============================
    TOP_K = 30
    topk = min(TOP_K, len(filtered))

    top_grasps = filtered[:topk]  # 已经按 score 排好序
    # best_grasp = max(top_grasps, key=lambda g: g.score)
    best_grasp = max(
        top_grasps,
        key=lambda g: grasp_center_score(g, object_center, lambda_center=3.0)
    )


    print(f"[BEST GRASP] score = {best_grasp.score:.6f}")

    if visualize:
        gripper_geom = [best_grasp.to_open3d_geometry()]
        o3d.visualization.draw_geometries([cloud_o3d_obs, *gripper_geom])

    return best_grasp



def grasp_center_score(grasp, center, lambda_center=2.0):
    """
    grasp: 单个 Grasp
    center: 物体点云中心 (3,)
    lambda_center: 中心惩罚权重（建议 1~5）
    """
    dist = np.linalg.norm(grasp.translation - center)
    return grasp.score - lambda_center * dist


# ============================================================
#  保存（基座标系）
# ============================================================

def save_grasp(name, grasp_base):
    """
    保存基座标系下的抓取：
    - npz：translation / rotation / width
    - ply：官方 GraspNet 夹爪几何
    """
    npz_path = os.path.join(SAVE_DIR, f"{name}.npz")
    ply_path = os.path.join(SAVE_DIR, f"{name}.ply")

    # ---------- 保存数值 ----------
    np.savez(
        npz_path,
        translation=grasp_base.translation,
        rotation=grasp_base.rotation_matrix,
        width=grasp_base.width,
        timestamp=time.time()
    )

    # ---------- 保存夹爪几何（关键） ----------
    gripper_mesh = grasp_base.to_open3d_geometry()
    gripper_mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(ply_path, gripper_mesh)

    print("[GraspNet] grasp saved:")
    print("  npz ->", npz_path)
    print("  ply ->", ply_path)

# ============================================================
#  点云转化
# ============================================================
def transform_grasp_to_base(grasp_cam, T_base_camera):
    """
    将 GraspNet 输出的 grasp 从相机坐标系转换到基坐标系
    """
    T = np.eye(4)
    T[:3, :3] = grasp_cam.rotation_matrix
    T[:3, 3] = grasp_cam.translation

    T_base = T_base_camera @ T

    grasp_cam.translation = T_base[:3, 3]
    grasp_cam.rotation_matrix = T_base[:3, :3]
    return grasp_cam


# ============================================================
#  主循环
# ============================================================


while True:
    try:
        # ================= LEFT =================
        best_grasp_cam = infer_best_grasp(
            f"{LEFT_DIR}/color_latest.png",
            f"{LEFT_DIR}/depth_latest.png",
            f"{LEFT_DIR}/mask_latest.png",
            K_left,
            visualize=False
        )
        

        # 相机系 → 基座标系
        best_grasp_base = transform_grasp_to_base(
            best_grasp_cam,
            T_base_left
        )

        save_grasp("left_grasp", best_grasp_base)
        print("[LEFT] updated")

        # ================= RIGHT =================
        best_grasp_cam = infer_best_grasp(
            f"{RIGHT_DIR}/color_latest.png",
            f"{RIGHT_DIR}/depth_latest.png",
            f"{RIGHT_DIR}/mask_latest.png",
            K_right,
            visualize=False
        )

        best_grasp_base = transform_grasp_to_base(
            best_grasp_cam,
            T_base_right
        )

        save_grasp("right_grasp", best_grasp_base)
        print("[RIGHT] updated")

        time.sleep(0.5)

    except KeyboardInterrupt:
        break
    except Exception as e:
        print("⚠️", e)
        time.sleep(0.5)
