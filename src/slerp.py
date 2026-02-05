import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def slerp(q1, q2, t):
    """
    SLERP球面线性插值
    
    参数:
        q1: 起始四元数 [w, x, y, z]
        q2: 目标四元数 [w, x, y, z] 
        t: 插值参数 [0, 1]
    
    返回:
        插值后的四元数
    """
    # 确保四元数归一化
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # 计算点积
    dot = np.dot(q1, q2)
    
    # 选择最短路径（处理四元数双重覆盖性）
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    # 处理接近平行的情况（数值稳定性）
    if dot > 0.9995:
        # 退化为线性插值
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    # 标准SLERP计算
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    return w1 * q1 + w2 * q2

