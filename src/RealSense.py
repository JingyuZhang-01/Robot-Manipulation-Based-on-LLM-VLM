import numpy as np
import cv2
import open3d as o3d
import pyrealsense2 as rs
import torch
from PIL import Image

# 引入 GroundingDINO + SAM2（路径不得修改，保持你原来的）
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor



# =============================
# 接口要求的容器(与物体交互使用未写)
# =============================
class LatestObs:
    pass



# ============================================================
# =================== RealSense 环境类 ===============
# ============================================================

class RealSenseEnv:

    # ==========================
    # 初始化
    # ==========================
    def __init__(self,
                 left_serial,
                 right_serial,            
                 T_base_left,T_base_right,
                 text_list):

        print("Initializing RealSenseEnv ...")

        # 保存参数
        self.left_serial = left_serial
        self.right_serial = right_serial

        self.T_base_left = T_base_left  # 左——>基的齐次矩阵
        self.T_base_right = T_base_right # 右——>基的齐次矩阵
        self.text_list = text_list

        self.camera_names = ["left", "right"]
        # ===== Workspace bounds=====
        # 实机桌面空间的 min/max（单位：米）

        self.workspace_bounds_min = np.array([0.0, -0.3, -0.03])
        self.workspace_bounds_max = np.array([0.4,  0.3, 0.45])

        # ===== 用 BASE 坐标系计算 lookat vectors =====

        # 光轴方向（相机坐标系）
        optical_axis = np.array([0, 0, 1.0])

        # 左相机 R_base_left
        R_base_left = self.T_base_left[:3, :3]
        left_lookat = R_base_left @ optical_axis
        left_lookat = left_lookat / np.linalg.norm(left_lookat)

        # 右相机 R_base_right
        R_base_right = self.T_base_right[:3, :3]
        right_lookat = R_base_right @ optical_axis
        right_lookat = right_lookat / np.linalg.norm(right_lookat)

        self.lookat_vectors = {
            "left": left_lookat,
            "right": right_lookat,
        }

        # ===== Robot & grasped object masks（必须有，否则忽略逻辑报错）=====
        self.robot_mask_ids = []
        self.arm_mask_ids = []
        self.gripper_mask_ids = []
        self.grasped_obj_ids = []

        # ===== Object id lookup（realenv 有，这里必须补齐）=====
        self.id2name = {}    # mask → 物体名称


        # 给每个目标分配固定 id（多相机一致）
        self.name2ids = {name: [(i + 1) * 10] for i, name in enumerate(text_list)}

        # RealSense 初始化
        self._init_realsense()

        # GroundingDINO + SAM2
        self._init_models()

        # 存储最新观测
        self.latest_obs = LatestObs()


        self.ee_pos_base = None   # 末端在 base 坐标系下的位置



    # =====================================================
    # 初始化两个 RealSense
    # =====================================================
    def _init_realsense(self):

        print("Initializing RealSense devices ...")

        self.pipeline_left = rs.pipeline()
        cfg_l = rs.config()
        cfg_l.enable_device(self.left_serial)
        cfg_l.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        cfg_l.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        prof_l = self.pipeline_left.start(cfg_l)

        self.pipeline_right = rs.pipeline()
        cfg_r = rs.config()
        cfg_r.enable_device(self.right_serial)
        cfg_r.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        cfg_r.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        prof_r = self.pipeline_right.start(cfg_r)

        # depth scale
        self.depth_scale_left = prof_l.get_device().first_depth_sensor().get_depth_scale()
        self.depth_scale_right = prof_r.get_device().first_depth_sensor().get_depth_scale()
        # 对齐对象 - 左
        self.align_left = rs.align(rs.stream.color)
        # 对齐对象 - 右
        self.align_right = rs.align(rs.stream.color)


        print("Left depth scale:", self.depth_scale_left)
        print("Right depth scale:", self.depth_scale_right)


    # =====================================================
    # 初始化 GDINO + SAM2
    # =====================================================
    def _init_models(self):
        print("Initializing GroundingDINO & SAM2...")

        # GDINO
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MODEL = "IDEA-Research/grounding-dino-base"
        self.processor_gd = AutoProcessor.from_pretrained(MODEL)
        self.model_gd = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL).to(self.device)

        # SAM2
        checkpoint = "/home/elephant/Grasp/Vox/sam2/checkpoints/sam2.1_hiera_tiny.pt"
        config_path = "configs/sam2.1/sam2.1_hiera_t.yaml"


        self.sam_model = build_sam2(config_path, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam_model)

        self.predictor.set_image(np.zeros((480, 640, 3), dtype=np.uint8))


    # =====================================================================
    # 将 depth 转为全图点云
    # =====================================================================
    def depth_to_full_cloud(self, depth, K, depth_scale):
        H, W = depth.shape
        fx, fy, cx, cy = K
        Z = depth.astype(np.float32) * depth_scale

        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        X = (xs - cx) * Z / fx
        Y = (ys - cy) * Z / fy
        cloud = np.stack([X, Y, Z], axis=-1)
        return cloud  # (H, W, 3)


    # =====================================================================
    # 相机点云转换到基相机系
    # =====================================================================
    def transform_cam_to_base(self, cloud,T_cam_base):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud.reshape(-1, 3))
        pcd.transform(T_cam_base)
        return np.asarray(pcd.points).reshape(cloud.shape)


    # =====================================================================
    # SAM mask 转 full-resolution mask id
    # =====================================================================
    def apply_mask_to_full(self, full_mask, mask_small, obj_id):
        # mask_small shape = (H,W)
        full_mask[mask_small.astype(bool)] = obj_id
        return full_mask


    # =====================================================================
    # ======= 主接口：获取当前双目相机的完整观测 =============================
    # =====================================================================
    def update(self):

        # ====== 左相机 ======
        frames_l = self.pipeline_left.wait_for_frames()
        aligned_l = self.align_left.process(frames_l)

        depth_l_frame = aligned_l.get_depth_frame()
        color_l_frame = aligned_l.get_color_frame()

        depth_l_raw = np.asanyarray(depth_l_frame.get_data())    # 不要乘 depth_scale
        color_l = np.asanyarray(color_l_frame.get_data())        # 不要 undistort

        # color内参
        intr_l = color_l_frame.profile.as_video_stream_profile().get_intrinsics()
        Kl = (intr_l.fx, intr_l.fy, intr_l.ppx, intr_l.ppy)

        # 左点云（相机坐标）
        cloud_l = self.depth_to_full_cloud(depth_l_raw, Kl, self.depth_scale_left)

        # 转 BASE
        cloud_l = self.transform_cam_to_base(cloud_l, self.T_base_left)

        # ====== 右相机 ======
        frames_r = self.pipeline_right.wait_for_frames()
        aligned_r = self.align_right.process(frames_r)

        depth_r_frame = aligned_r.get_depth_frame()
        color_r_frame = aligned_r.get_color_frame()

        depth_r_raw = np.asanyarray(depth_r_frame.get_data())
        color_r = np.asanyarray(color_r_frame.get_data())

        # color内参
        intr_r = color_r_frame.profile.as_video_stream_profile().get_intrinsics()
        Kr = (intr_r.fx, intr_r.fy, intr_r.ppx, intr_r.ppy)

        # 点云
        cloud_r = self.depth_to_full_cloud(depth_r_raw, Kr, self.depth_scale_right)
        cloud_r = self.transform_cam_to_base(cloud_r, self.T_base_right)

        # ===== 检测+分割 =====
        mask_l = self._detect_and_segment(color_l, depth_l_raw, Kl, np.zeros((480,640),dtype=np.int32))
        mask_r = self._detect_and_segment(color_r, depth_r_raw, Kr, np.zeros((480,640),dtype=np.int32))

        # 存储
        lo = self.latest_obs
        lo.left_point_cloud  = cloud_l.reshape(-1,3)
        lo.left_rgb          = color_l.reshape(-1,3)
        lo.left_mask         = mask_l.reshape(-1)

        lo.right_point_cloud = cloud_r.reshape(-1,3)
        lo.right_rgb         = color_r.reshape(-1,3)
        lo.right_mask        = mask_r.reshape(-1)

        lo.left_depth = depth_l_raw
        lo.right_depth = depth_r_raw



    # =====================================================================
    # 检测 + SAM2
    # =====================================================================
    def _detect_and_segment(self, color, depth, K, full_mask, is_left=True):

        pil = Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))

        inputs = self.processor_gd(images=pil, text=self.text_list, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model_gd(**inputs)

        results_list = self.processor_gd.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            threshold=0.4,
            text_threshold=0.4,
            target_sizes=[pil.size[::-1]],
        )

        boxes = results_list[0]["boxes"]
        labels = results_list[0]["labels"]
        scores = results_list[0]["scores"]

        self.predictor.set_image(color)

        # 遍历每个物体框
        for i in range(len(boxes)):
            if float(scores[i]) < 0.3:   # 同你的逻辑
                continue

            #label_text = labels[i]

            raw_label = labels[i]    # GDINO 返回的标签

            # 找出与 raw_label 最相似的 prompt
            label_text = self._match_label(raw_label)


            obj_id = self.name2ids[label_text][0]

            box = boxes[i].cpu().numpy().astype(int)
            masks, _, _ = self.predictor.predict(box=box[None, :], multimask_output=False)

            if masks.ndim == 4:
                mask = masks[0][0]
            elif masks.ndim == 3:
                mask = masks[0]
            else:
                mask = masks

            if mask.sum() == 0:
                continue

            # 将 mask 写入 full-size mask
            full_mask = self.apply_mask_to_full(full_mask, mask.astype(np.uint8), obj_id)

        return full_mask

     
    #=====================需要 label 映射与匹配。==============
    def _match_label(self, raw_label):
        """
        将 GDINO 返回的 label_text 映射到 text_list 中最接近的字符串。
        例如：
        raw_label="a bottle" → "a bottle of water"
        """

        raw_label = raw_label.lower().strip()

        best_key = None
        best_score = -1

        for key in self.name2ids.keys():
            k = key.lower()

            # 分数规则：交集词越多得分越高
            score = len(set(k.split()) & set(raw_label.split()))

            if score > best_score:
                best_score = score
                best_key = key

        return best_key


    # =====================================================================
    # 接口：获取整个场景点云
    # =====================================================================
    def get_scene_3d_obs(self, ignore_robot=True, ignore_grasped_obj=True):
        """
        获取整个场景点云和彩色信息（实机版）
        """
        points_all = []
        colors_all = []
        masks_all  = []

        for cam in self.camera_names:
            points_all.append(getattr(self.latest_obs, f"{cam}_point_cloud").reshape(-1, 3))
            colors_all.append(getattr(self.latest_obs, f"{cam}_rgb").reshape(-1, 3))
            masks_all.append(getattr(self.latest_obs, f"{cam}_mask").reshape(-1))

        # === 拼接 ===
        points_all = np.concatenate(points_all, axis=0)
        colors_all = np.concatenate(colors_all, axis=0)
        masks_all  = np.concatenate(masks_all, axis=0)

        # === workspace filtering ===
        cond_x = (points_all[:, 0] > self.workspace_bounds_min[0]) & (points_all[:, 0] < self.workspace_bounds_max[0])
        cond_y = (points_all[:, 1] > self.workspace_bounds_min[1]) & (points_all[:, 1] < self.workspace_bounds_max[1])
        cond_z = (points_all[:, 2] > self.workspace_bounds_min[2]) & (points_all[:, 2] < self.workspace_bounds_max[2])
        cond = cond_x & cond_y & cond_z

        points_all = points_all[cond]
        colors_all = colors_all[cond]
        masks_all  = masks_all[cond]

        # === 忽略机器人 ===
        if ignore_robot:
            robot_mask = np.isin(masks_all, self.robot_mask_ids)
            points_all = points_all[~robot_mask]
            colors_all = colors_all[~robot_mask]
            masks_all  = masks_all[~robot_mask]

        # === 忽略抓取中的物体 ===
        if ignore_grasped_obj and len(self.grasped_obj_ids) > 0:
            grasp_mask = np.isin(masks_all, self.grasped_obj_ids)
            points_all = points_all[~grasp_mask]
            colors_all = colors_all[~grasp_mask]
            masks_all  = masks_all[~grasp_mask]

        #=== 忽略夹爪附近8cm的物体 ===
        if ignore_robot or ignore_grasped_obj:
            keep = self.remove_ee_attached_geometry(points_all, radius=0.015)
            points_all = points_all[keep]
            colors_all = colors_all[keep]
            masks_all  = masks_all[keep]



        # === 下采样 ===
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_all)
        pcd.colors = o3d.utility.Vector3dVector(colors_all / 255.0)

        pcd = pcd.voxel_down_sample(voxel_size=0.001)



        return np.asarray(pcd.points), (np.asarray(pcd.colors) * 255).astype(np.uint8)

    def remove_ee_attached_geometry(self, points, radius=0.08): #删除夹爪附近区域的点云
        """
        删除 EE / 抓取物体附近的点云
        """
        if self.ee_pos_base is None:
            return np.ones(len(points), dtype=bool)

        dist = np.linalg.norm(points - self.ee_pos_base[None, :], axis=1)
        keep = dist > radius
        return keep
    
    def update_ee_pos(self, ee_pos):
        """
        ee_pos: np.ndarray, shape (3,), base frame
        """
        self.ee_pos_base = np.array(ee_pos, dtype=float)



    # =====================================================================
    # 接口：根据物体名获取 3D 点云 + 法向
    # =====================================================================
    def get_3d_obs_by_name(self, query_name):
        """
        从多视角点云与mask中提取特定物体的3D点云和法向量
        """

        assert query_name in self.name2ids, f"Unknown object name: {query_name}"
        obj_ids = self.name2ids[query_name]   # 该物体对应的一组 object_id

        points_all = []
        masks_all = []
        normals_all = []

        for cam in self.camera_names:

            # === 读取每个相机的点云、mask ===
            cam_points = getattr(self.latest_obs, f"{cam}_point_cloud").reshape(-1, 3)
            cam_mask   = getattr(self.latest_obs, f"{cam}_mask").reshape(-1)

            # === 计算法向量 ===
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cam_points)
            pcd.estimate_normals()

            cam_normals = np.asarray(pcd.normals)

            # === 法向量方向校准 ===
            lookat = self.lookat_vectors[cam]  # shape (3,)
            flip = np.dot(cam_normals, lookat) > 0
            cam_normals[flip] *= -1

            # === 保存 ===
            points_all.append(cam_points)
            masks_all.append(cam_mask)
            normals_all.append(cam_normals)

        # === 拼接所有相机 ===
        points_all  = np.concatenate(points_all, axis=0)
        masks_all   = np.concatenate(masks_all, axis=0)
        normals_all = np.concatenate(normals_all, axis=0)

        # === 过滤目标物体 ===
        chosen = np.isin(masks_all, obj_ids)
        obj_points = points_all[chosen]
        obj_normals = normals_all[chosen]

        if len(obj_points) == 0:
            raise ValueError(f"Object {query_name} not found.")

        # === voxel 下采样 ===
        pcd = o3d.geometry.PointCloud()
        pcd.points  = o3d.utility.Vector3dVector(obj_points)
        pcd.normals = o3d.utility.Vector3dVector(obj_normals)

        pcd = pcd.voxel_down_sample(voxel_size=0.001)
        # print("min", np.min(pcd.points, axis=0))
        # print("max", np.max(pcd.points, axis=0))

        return np.asarray(pcd.points), np.asarray(pcd.normals)
    

    def get_camera_data(self, side, obj_name):
        """
        返回指定相机 side ('left' 或 'right') 对应目标物体的 color, depth, mask
        - color: (H,W,3) uint8
        - depth: (H,W) uint16（单位 mm）
        - mask:  (H,W) 0/1 uint8
        """
        assert side in ['left','right']
        assert obj_name in self.name2ids

        obj_id = self.name2ids[obj_name][0]

        # 获取最新帧
        self.update()  # 保证最新数据

        # ===== depth =====
        depth_raw = getattr(self.latest_obs, f"{side}_depth")  # 原始 RealSense uint16
        if depth_raw.dtype != np.uint16:
            # 乘以 depth_scale 并转 mm
            scale = self.depth_scale_left if side=='left' else self.depth_scale_right
            depth_raw = (depth_raw.astype(np.float32) * scale * 1000).astype(np.uint16)

        # ===== color =====
        # 获取原始 color (H,W,3)
        if side=='left':
            frames = self.pipeline_left.wait_for_frames()
            aligned = self.align_left.process(frames)
            color_frame = aligned.get_color_frame()
        else:
            frames = self.pipeline_right.wait_for_frames()
            aligned = self.align_right.process(frames)
            color_frame = aligned.get_color_frame()
        color = np.asanyarray(color_frame.get_data())  # (H,W,3) uint8

        # ===== mask =====
        mask_full = getattr(self.latest_obs, f"{side}_mask").reshape(480,640)  # 0/obj_id
        mask = (mask_full == obj_id).astype(np.uint8)  # 0/1

        return color, depth_raw, mask



