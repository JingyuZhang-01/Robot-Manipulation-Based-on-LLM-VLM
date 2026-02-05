import numpy as np
import time
from pymycobot import MercurySocket
import time




class MyCobotRobotInterface:
    """将 pymycobot.MercurySocket 封装成可用的机器人接口（姿态均基于旋转矩阵转换）"""

    def __init__(self, mc: MercurySocket):
        self.mc = mc
        self.default_speed = 80
        self.home_angles = [-15, 4, 0, -60, 10, 100, 0] #init angles
        if not mc.is_power_on():
            mc.power_on()
            time.sleep(1)

    # --------------------------------------------------------------
    # Robot movement
    # --------------------------------------------------------------
    def move_to_home(self):
        self.mc.send_angles(self.home_angles, self.default_speed,_async = True)
        self._wait_until_reach(self.home_angles)
        print("[Robot] Reached HOME.")

    def move_to_pose(self, pose, speed=None):
        """
        pose = [x, y, z, qw, qx, qy, qz]  (meters)
        MyCobot expects [x, y, z, rx, ry, rz]  (mm + degrees)
        """
        # ===== 单位换算 m → mm =====
        x_mm = pose[0] * 1000
        y_mm = pose[1] * 1000
        z_mm = pose[2] * 1000

        quat = pose[3:]

        # Step 1: quat → Rotation Matrix
        R = self.quat_to_matrix(quat)

        # Step 2: Rotation Matrix → MyCobot Euler (rx,ry,rz in deg)
        # 旋转矩阵 → 欧拉角（rad）
        euler_rad = self.CvtRotationMatrixToEulerAngle(R)

        # rad → degree
        euler_deg = np.degrees(euler_rad)

        #euler = self.CvtRotationMatrixToEulerAngle(R) * 180 / np.pi
        rx, ry, rz = euler_deg.tolist()

        coords = [x_mm, y_mm, z_mm, rx, ry, rz]
        speed = speed if speed is not None else self.default_speed

        self.mc.send_coords(coords, speed,_async = True)





    def move_to_home_nonblocking(self):
        """
        仅发送回 home 指令，不等待机器人报告到位（避免阻塞）
        """
        print("[Robot] Moving to HOME (non-blocking)...")
        self.mc.send_angles(self.home_angles, self.default_speed,_async = True)
        # 不调用等待函数
        time.sleep(0.2)  # 给指令传输一些最小延时

    # --------------------------------------------------------------
    # 获取当前末端姿态
    # --------------------------------------------------------------
    def get_ee_pose(self, retry=10):
        """
        Return pose :
        [x, y, z, qw, qx, qy, qz]  (meters + quaternion)
        """
        
        for _ in range(retry):
            xyz_rxryrz = self.mc.get_coords()
            if xyz_rxryrz and len(xyz_rxryrz) == 6:
                break
            # time.sleep(0.05)
        else:
            raise RuntimeError("get_coords() 连续返回 None，机械臂可能未准备好")

        x_mm, y_mm, z_mm, rx, ry, rz = xyz_rxryrz

        # ===== 单位换算 mm → m =====
        x = x_mm / 1000
        y = y_mm / 1000
        z = z_mm / 1000

        # Step 1: MyCobot Euler → Rotation Matrix
        pose_tmp = [x_mm, y_mm, z_mm, rx, ry, rz]
        T = self.CvtEulerAngleToRotationMatrix(pose_tmp)
        R = T[:3, :3]

        # Step 2: Rotation Matrix → quaternion
        quat = self.matrix_to_quat(R)
        

        return np.array([x, y, z, *quat])

    # --------------------------------------------------------------
    # 工具函数：m转mm函数
    # --------------------------------------------------------------
    def pose_to_mycobot_coords(self, pose):
        """
        pose (m, quat) → MyCobot [mm, deg]
        """
        x_mm = pose[0] * 1000
        y_mm = pose[1] * 1000
        z_mm = pose[2] * 1000
        quat = pose[3:]

        # quat → Rotation matrix
        R = self.quat_to_matrix(quat)

        # Rot → factory Euler
        euler_deg = self.CvtRotationMatrixToEulerAngle(R) * 180 / np.pi

        return [x_mm, y_mm, z_mm, *euler_deg]

    # --------------------------------------------------------------
    # 工具函数：等待运动结束
    # --------------------------------------------------------------
    def _wait_until_reach(self, target, timeout=10.0):
        """
        自动判断输入是角度还是末端 pose
        target:
            若为长度 6 → [x_mm, y_mm, z_mm, rx, ry, rz] (末端)
            若为长度 7 → [j1..j7]（关节角）
            若为长度 7 的 pose (m + quat) → 自动转换成厂家坐标
        """
        # -------------------------
        # 判断输入格式
        # -------------------------
        if len(target) == 7:
            # 可能是关节角，也可能是 pose
            if abs(target[3]) <= 1 and abs(target[4]) <= 1:  
                # 说明可能是 quat → 使用末端 pose 模式
                target = self.pose_to_mycobot_coords(target)
                flag = 1
            else:
                # 是关节角
                flag = 0

        elif len(target) == 6:
            flag = 1

        else:
            raise ValueError("target 必须是 6（协调）或 7（角度/pose）长度")

        # -------------------------
        # 等待到位
        # -------------------------
        start = time.time()

        while True:
            result = self.mc.is_in_position(target, flag)

            if result == 1:
                return True

            if result == -1:
                print("[ERROR] is_in_position() 返回 -1，可能通信错误")
                return False

            if time.time() - start > timeout:
                print("[WARN] wait timeout, may not reach exact target")
                return False

            time.sleep(0.05)




    # --------------------------------------------------------------
    # 姿态转换：quat ↔ matrix
    # --------------------------------------------------------------
    def quat_to_matrix(self, quat):
        """
        四元数 → 旋转矩阵
        (保持右手系)
        """
        qw, qx, qy, qz = quat
        R = np.zeros((3, 3))
        R[0, 0] = 1 - 2*(qy*qy + qz*qz)
        R[0, 1] = 2*(qx*qy - qz*qw)
        R[0, 2] = 2*(qx*qz + qy*qw)
        R[1, 0] = 2*(qx*qy + qz*qw)
        R[1, 1] = 1 - 2*(qx*qx + qz*qz)
        R[1, 2] = 2*(qy*qz - qx*qw)
        R[2, 0] = 2*(qx*qz - qy*qw)
        R[2, 1] = 2*(qy*qz + qx*qw)
        R[2, 2] = 1 - 2*(qx*qx + qy*qy)
        return R

    def matrix_to_quat(self, R):
        """
        旋转矩阵 → 四元数
        (保持格式 [qw, qx, qy, qz])
        """
        qw = np.sqrt(1 + np.trace(R)) / 2
        qx = (R[2, 1] - R[1, 2]) / (4 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4 * qw)
        return [qw, qx, qy, qz]

    # --------------------------------------------------------------
    # 旋转公式（核心）
    # --------------------------------------------------------------
    def CvtRotationMatrixToEulerAngle(self, R):
        pdtEulerAngle = np.zeros(3)
        pdtEulerAngle[2] = np.arctan2(R[1, 0], R[0, 0])
        fCosRoll = np.cos(pdtEulerAngle[2])
        fSinRoll = np.sin(pdtEulerAngle[2])
        pdtEulerAngle[1] = np.arctan2(-R[2, 0],
                                     (fCosRoll * R[0, 0]) + (fSinRoll * R[1, 0]))
        pdtEulerAngle[0] = np.arctan2((fSinRoll * R[0, 2]) - (fCosRoll * R[1, 2]),
                                     (-fSinRoll * R[0, 1]) + (fCosRoll * R[1, 1]))
        return pdtEulerAngle  # rad

    def CvtEulerAngleToRotationMatrix(self, pose):
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

import time

class MyCobotGripperInterface:
    """
    适配的夹爪接口
    底层使用 pymycobot.MercurySocket
    """

    def __init__(self, mc, speed=80):
        self.mc = mc
        self.speed = speed
        self._state_value = 100     # 0~100 记录位置 (0闭/100开)
        self._state_binary = 0      # 0=open, 1=close

        # 初始化485模式
        self.mc.set_gripper_mode(0)
        time.sleep(0.2)

    # ----------------------------------------------------
    # 接口：打开夹爪
    # ----------------------------------------------------
    def open(self):
        """最大张开"""
        self.mc.set_gripper_state(0, self.speed)     # 0=open
        self._state_binary = 0
        self._state_value = 100
        time.sleep(0.05)

    # ----------------------------------------------------
    # 接口：闭合夹爪
    # ----------------------------------------------------
    def close(self):
        """完全闭合"""
        self.mc.set_gripper_state(1, self.speed)     # 1=close
        self._state_binary = 1
        self._state_value = 0
        time.sleep(0.05)

    # ----------------------------------------------------
    # 接口：设置夹爪开合程度 (0~1)
    # ----------------------------------------------------
    def set_state(self, v):
        """
        v ∈ [0, 1]
        1 = fully open ；0 = fully close
        """
        v = float(v)
        value = int(v * 100)    # 转成0~100
        self.mc.set_gripper_value(value, self.speed)
        self._state_value = value
        self._state_binary = 0 if v >= 0.5 else 1
        time.sleep(0.05)

    # ----------------------------------------------------
    # 接口：用于obs生成
    # return float ∈ [0,1]
    # ----------------------------------------------------
    def get_state(self):
        return self._state_value / 100.0


import numpy as np
import time

class RealEnv:
    """
    Sim2Real 版本的环境
    ⬆ 完全替代 realenv 接口，兼容推理流程
    """

    # -------------------------------
    # 内置 RealObs（与 realenv Observation 对齐）
    # -------------------------------
    class RealObs:
        def __init__(self, ee_pose, gripper_open):
            # xyz + quat (7,)
            self.gripper_pose = np.array(ee_pose, dtype=float)
            # ∈ [0,1]
            self.gripper_open = float(gripper_open)

    # -------------------------------
    # 构造函数
    # -------------------------------
    def __init__(self, robot_interface, gripper_interface, realsense_env, visualizer=None):
        print("Initializing RealEnv ...")

        # 外部接口
        self.robot = robot_interface
        self.gripper = gripper_interface
        self.rs = realsense_env
        self.visualizer = visualizer  # 未使用但为了兼容 realenv 顶层调用必留

        # realenv 状态变量（必须保留）
        self.latest_obs = None
        self.init_obs = None
        self.latest_action = None
        self.latest_reward = 0
        self.latest_terminate = False

        self.task_name = "real_world_task"

        # 夹持的物体 ID 用于 3D 过滤 （如果后续加入可抓取检测）
        self.grasped_obj_ids = []

        # 与 RealSenseEnv 同步 workspace
        self.workspace_bounds_min = self.rs.workspace_bounds_min
        self.workspace_bounds_max = self.rs.workspace_bounds_max


    # ----------------------------------------------------
    # realenv 接口：加载任务
    # ----------------------------------------------------
    def load_task(self, task_name_or_objects):
        """
        Real world 版本 load_task():
        - 不切换真实场景
        - 只是设置当前实验中有哪些可交互物体名称
        输入:
            task_name_or_objects: 传入语义物体名称列表即可，例如 ["bottle", "trash_bin"]
        返回:
            任务描述文本列表（供自然语言接口使用）
        """

        print(f"[RealEnv] Loading real-world task with objects: {task_name_or_objects}")

        # 设置任务名称
        if isinstance(task_name_or_objects, list):
            self.task_name = "real_world_custom"
            object_names = task_name_or_objects
        else:
            self.task_name = str(task_name_or_objects)
            if hasattr(task_name_or_objects, "OBJECTS"):
                object_names = task_name_or_objects.OBJECTS
            else:
                raise ValueError("实物实验必须提供可识别物体名称列表")

        # 更新 RealSenseEnv 识别词列表
        self.rs.text_list = object_names
        self.rs.name2ids = {name: [(i + 1) * 10] for i, name in enumerate(object_names)}

        # 记录用于 get_object_names()
        self.object_names = object_names

        # 构造自然语言描述
        descriptions = [f"Find the {name} and manipulate it." for name in object_names]

        return descriptions




    def reset(self):
        print("[RealEnv] Resetting...")

        # 1️⃣ 机械臂回 home（不等待到位，只等待少量时间）
        self.robot.move_to_home()

        time.sleep(0.3)  # 机械臂需要启动动作的时间

        # 2️⃣ 夹爪打开
        self.gripper.open()
        time.sleep(0.2)

        # 3️⃣ 视觉更新
        self.rs.update()

        # 4️⃣ 读取一次姿态（多次重试更稳）
        ee_pose = self.robot.get_ee_pose(retry=10)
        self.rs.update_ee_pos(ee_pose[:3])


        # 5️⃣ 构建 obs
        grip_state = self.gripper.get_state()
        self.latest_obs = self.RealObs(ee_pose, grip_state)
        self.init_obs = self.latest_obs
        self.latest_action = None
        self.init_obs   = self.latest_obs

        scene_descriptions = [
        f"There is a {name} in the scene."
        for name in self.object_names
     ]

        return scene_descriptions, self.latest_obs




    def apply_action(self, action, speed=None):
        pose = action[:7]
        g = action[7]
        # print("g:",g)

        # move robot
        self.robot.move_to_pose(pose, speed=speed)

        # gripper        
        if g > 0.5:
            self.gripper.open()
        else:
            self.gripper.close()

        # update obs
        self.rs.update()
        ee_pose = self.robot.get_ee_pose()
        self.rs.update_ee_pos(ee_pose[:3])

        grip_state = self.gripper.get_state()
        self.latest_obs = self.RealObs(ee_pose, grip_state)
        self.latest_action = action

        reward = 0
        terminate = False
        return self.latest_obs, reward, terminate

    def apply_action_wait(self, action, speed=None):
        pose = action[:7]
        g = action[7]

        # move robot
        self.robot.move_to_pose(pose, speed=speed)

        # gripper
        if g > 0.5:
            self.gripper.open()
        else:
            self.gripper.close()

        # update obs
        self.rs.update()
        time.sleep(5)
        ee_pose = self.robot.get_ee_pose()
        self.rs.update_ee_pos(ee_pose[:3])

        grip_state = self.gripper.get_state()
        self.latest_obs = self.RealObs(ee_pose, grip_state)
        self.latest_action = action

        reward = 0
        terminate = False
        return self.latest_obs, reward, terminate


    def move_to_pose(self, pose, speed=None):
        """
        pose: [x, y, z, qw, qx, qy, qz]
        speed: optional velocity parameter from controller
        """
        last_g = self.get_last_gripper_action()

        action = np.concatenate([pose, [last_g]])
    
        return self.apply_action(action, speed=speed)
    
    def move_to_pose_wait(self, pose, speed=None): 
        """
        pose: [x, y, z, qw, qx, qy, qz]
        speed: optional velocity parameter from controller
        #由于移动时就读取导致读到的不是最新的末端位置所以添加等待时间等到位置后再读取
        """
        last_g = self.get_last_gripper_action()

        action = np.concatenate([pose, [last_g]])
        self.apply_action_wait(action, speed=speed)
        time.sleep(15)

        return


    def open_gripper(self):
        return self.apply_action(np.concatenate([self.latest_obs.gripper_pose, [1.0]]))

    def close_gripper(self):
        return self.apply_action(np.concatenate([self.latest_obs.gripper_pose, [0.0]]))

    def set_gripper_state(self, g):
        return self.apply_action(np.concatenate([self.latest_obs.gripper_pose, [g]]))

    def reset_to_default_pose(self):
        self.move_to_pose_wait(self.init_obs.gripper_pose)
        ee_pose = self.robot.get_ee_pose()
        self.latest_obs = self.RealObs(ee_pose, self.gripper.get_state())
        return
    


    # ----------------------------------------------------
    # 3D 视觉接口（核心）——直接调用 RealSenseEnv
    # ----------------------------------------------------
    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        return self.rs.get_scene_3d_obs(ignore_robot, ignore_grasped_obj)

    def get_3d_obs_by_name(self, name):
        return self.rs.get_3d_obs_by_name(name)

    def get_object_names(self):
        """用于 set_lmp_objects(lmps, env.get_object_names())"""
        return list(self.rs.name2ids.keys()) #返回环境中所有对象名称的列表
    
    def update(self):
        return self.rs.update()
    
    def get_camera_data(self, side, obj_name):
        return self.rs.get_camera_data(side, obj_name)

    # ----------------------------------------------------
    # 访问上一次执行状态
    # ----------------------------------------------------
    def get_ee_pose(self):
        return self.latest_obs.gripper_pose

    def get_ee_pos(self):
        return self.latest_obs.gripper_pose[:3]

    def get_ee_quat(self):
        return self.latest_obs.gripper_pose[3:]

    def get_last_gripper_action(self):
        return (
            self.latest_action[7]
            if self.latest_action is not None
            else self.latest_obs.gripper_open
        )
