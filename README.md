# Robot-Manipulation-Based-on-LLM-VLM
Building on the Voxposer framework, this project introduces GraspNet-based grasp pose generation and integrates visual feedback with rotation planning. The implemented system was ultimately tested and validated on the Mercury A1 robotic arm from Elephant Company.

# 目录
- [Environment Setup](#Environment_setup)
- [Project Overview](#Project_Overview)
- [sim2sim](#sim2sim)

# Environment_Setup
This project involves a physical experiment setup with no simulation component.
1. Hardware Setup
- Cameras: Two Intel RealSense D435 cameras, positioned in front of the robotic arm.
- Robotic Arm: One Mercury A1 (7-DOF) robotic arm.
- Gripper: One robotic gripper.
The hardware installation layout is as shown in the figure below.
![harnware](https://github.com/JingyuZhang-01/Robot-Manipulation-Based-on-LLM-VLM/image/1.png)
---


# Project_Overview
The project structure is inspired by Voxposer, but implements significant modifications:
- Different AI Models: Replaces the original LLM (Large Language Model) and VLM (Vision Language Model) components.
- Core Technologies Used:
  - Qwen API for language understanding and code .
  - Grounding DINO for open-vocabulary object detection.
  - SAM 2 (Segment Anything Model 2) for high-performance image segmentation.
  - GraspNet for robotic grasp pose generation.
3. Software Environment Setup
```bash
-  Create the first conda environment:
conda create -n Realenv python=3.9
conda activate Realenv
```
- See Instructions to install GroundingDINO and SAM2 (Note: install these inside the created conda environment).
- Install other dependencies:
```bash
pip install -r requirements.txt
```
- Obtain an OpenAI API key, and put it inside the first cell of runmain.py.
Because  GraspNet relies on older libraries that are incompatible with the  PyTorch version required by SAM 2, it needs to be installed in another Conda environment
- Create a second Conda environment:
```bash
conda create -n graspnet python=3.9
conda activate graspnet
```
Then, install Graspnet according to its requirements within this environment.
