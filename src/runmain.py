import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

import openai
import numpy as np
from arguments import get_config
from interfaces import setup_LMP

from utils import set_lmp_objects

# 你自己的接口
from RealSense import RealSenseEnv
from visualizers import ValueMapVisualizer
from envs.RealEnv import RealEnv, MyCobotRobotInterface, MyCobotGripperInterface

from pymycobot import MercurySocket
# 设置 OpenAI API 密钥
openai.api_key = "YOUR API KEY"

mc = MercurySocket("A1 IP", 9000, debug=0) #socket通信

def main(mc):

    print("========== 初始化硬件 ==========")

    robot = MyCobotRobotInterface(mc)
    gripper = MyCobotGripperInterface(mc)

    rs = RealSenseEnv(
        left_serial="238322072021", #相机编号
        right_serial="244222070576", 
        T_base_left = np.array([
            [ 0.96974678, 0.05240736, -0.23842117, 0.29],
            [ 0.16478827, -0.86110891, 0.4809743, -0.30],
            [-0.1801, -0.50571229, -0.84369371, 0.42],
            [ 0.0, 0.0, 0.0, 1.0]
        ]),
        T_base_right = np.array([
            [-0.97616224, -0.03089485, -0.21483199, 0.37],
            [ 0.09935684, 0.81640497, -0.5688683, 0.35],
            [ 0.19296501, -0.57665278, -0.79387409, 0.38],
            [ 0.0, 0.0, 0.0, 1.0]
        ]), #手眼矩阵


        text_list=[]
    )

    print("========== 初始化环境与 LMP ==========")
    config = get_config("realenv")

    env = RealEnv(robot, gripper, rs)
    env.visualizer = None  # 确保visualizer为None

    lmps, lmp_env = setup_LMP(env, config, debug=False)
    ui = lmps["plan_ui"]

    #  tasks → load_task(["bottle", "trash_bin"])
    env.load_task(["bread","banana", "pen","pencil box","bottle","plate","robot arm", "gripper"])

    print("========== reset ==========")    
    scene_descriptions, obs = env.reset()
    set_lmp_objects(lmps, env.get_object_names()) 

    instruction = "Grasp bread, then place it in the plate"
    # instruction = "Put the pen in the pencil box."
    # instruction = "Grasp the bread while staying at least 10cm from the bottle, then place it in the plate"
    # instruction = "Place every object on the table into the plate, one by one."

    print(f"执行指令: {instruction}")

    print("========== 开始推理 ==========")
    ui(instruction)

    print("========== 任务完成 ==========")

if __name__ == "__main__":
    main(mc)
