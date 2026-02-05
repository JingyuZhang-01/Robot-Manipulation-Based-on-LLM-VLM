Robot Manipulation
Building on the Voxposer framework, this project introduces GraspNet-based grasp pose generation and integrates visual feedback with rotation planning. The implemented system was ultimately tested and validated on the Mercury A1 robotic arm from Elephant Company.
![video](https://github.com/JingyuZhang-01/Robot-Manipulation-Based-on-LLM-VLM/media/2.mp4)


https://github.com/user-attachments/assets/f60252c8-516a-45fc-b8b3-9da5d9a0b0af



# Contents
- [Environment Setup](#Environment_setup)
- [Project Overview](#Project_Overview)
- [Software Environment Setup](#Software_Environment_Setup)
- [Running the Experiment](#Running_Experiment)
- [Code Structure](#Code_Structure)
- [Acknowledgments](#Acknowledgments)

# Environment Setup
This project involves a physical experiment setup with no simulation component.
- Cameras: Two Intel RealSense D435 cameras, positioned in front of the robotic arm.
- Robotic Arm: One Mercury A1 (7-DOF) robotic arm.
- Gripper: One robotic gripper.
The hardware installation layout is as shown in the figure below.
![harnware](https://github.com/JingyuZhang-01/Robot-Manipulation-Based-on-LLM-VLM/media/1.jpeg)
![1](https://github.com/user-attachments/assets/43dac6ba-e191-4a6b-8d07-135e1e7ded51)

---


# Project Overview
The project structure is inspired by Voxposer, but implements significant modifications:
- Different AI Models: Replaces the original LLM (Large Language Model) and VLM (Vision Language Model) components.
- Core Technologies Used:
  - Qwen API for language understanding and code .
  - Grounding DINO for open-vocabulary object detection.
  - SAM 2 (Segment Anything Model 2) for high-performance image segmentation.
  - GraspNet for robotic grasp pose generation.


# Software Environment Setup
-  Create the first conda environment:
```bash
conda create -n Realenv python=3.9
conda activate Realenv
```
- See Instructions to install [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and [SAM2](https://github.com/facebookresearch/sam2) (Note: install these inside the created conda environment).
- Install other dependencies:
```bash
pip install -r requirements.txt
```
- Obtain an [OpenAI API](https://www.aliyun.com/product/tongyi) key, and put it inside the first cell of ```runmain.py```.


Because  GraspNet relies on older libraries that are incompatible with the  PyTorch version required by SAM 2, it needs to be installed in another Conda environment
- Create a second Conda environment:
```bash
conda create -n graspnet python=3.9
conda activate graspnet
```
Then, install Graspnet according to its requirements within this environment.


# Running Experiment
1 Robotic Arm Connection via Socket

Establish a remote socket connection between the host machine and the Mercury A1 robotic arm.


On the Mercury A1 side (Robotic Arm Controller):
```bash
python server_A1_close_loop.py
```
2 Running on the Host Machine


Two separate Conda environments need to be active, each running a specific process.
- In the RealEnv environment:
```bash
conda activate RealEnv
python runmain.py
```
- In the graspnet environment:

```bash
conda activate graspnet
python grasp_github.py
```
3 Data Flow & Paths
- runmain.py automatically saves the latest detected target object information in:
  - /data/left/
  - /data/right/


Note:  The system selects the camera data (left or right) based on point cloud  density to avoid occlusion and ensure accurate grasp pose generation.
- grasp_github.py generates and saves the latest grasp poses in:
  - /data/grasp/
 


# Code Structure 
Compared  to the original Voxposer implementation, the following files have been  significantly modified or newly added to adapt the system to real-world  experiments:


1. Hardware Interaction Modules
- ```Realsense.py```:  Handles initialization of the RealSense D435 cameras, object  recognition, and fusion of point cloud data from multiple sources.
- ```Realenv.py```: Implements the real-world experiment environment, replacing the ```rlbench_env``` from Voxposer. This module provides the primary interface for controlling the Mercury A1 robotic arm's motion.
2. LLM Prompt Files

  
The  prompts for the LLM (Large Language Model) have been updated to integrate new functionalities for grasp pose generation, rotation planning, and visual feedback. The modified files include:
- ```composer_prompt.txt```: Major modifications to incorporate visual feedback mechanisms.
- ```affordance_map_prompt.txt```: Updated to refine the definition and selection of grasping targets.
- ```rotation_map_prompt.txt```: Modified to adjust the criteria for generating grasping orientations.
- ```parse_query_obj_prompt.txt```: Updated for loading and interpreting the grasp poses of target objects.
- ```get_gripper_map_prompt.txt```: Adjusted to specify the gripper closure goals and actions.


# Acknowledgments
- The system architecture is based on [Voxposer](https://github.com/huangwl18/VoxPoser).
- Implementation of Language Model Programs (LMPs) is based on [Code as Policies](https://code-as-policies.github.io/).
