from pymycobot import MercurySocket
import time
# import numpy
mc = MercurySocket("192.168.123.234", 9000,debug = 1)

if mc.is_power_on() != 1:
    mc.power_on()

mc.send_angles([0, 0, 0, -80, 0, 90,0],90) #机械臂初始位姿
exit()
mc.set_limit_switch(2,0) #关闭闭环
# mc.set_vr_mode(1)
mc.set_movement_type(4) #切换连续运动模式



mc.set_gripper_mode(0)
#close
mc.set_gripper_value(1, 60)
#open
mc.set_gripper_value(100, 60)


#========机械臂关节位置发送接口==========
mc.send_angles([0, 0, 0, -80, 0, 90,20],90) #机械臂初始位姿
print(mc.get_angles())


#=====================末端工具坐标系==========
print(mc.get_tool_reference())
coords = [0.0, 0.0, 130, 0.0, 0.0, 0.0]
mc.set_tool_reference(coords)
print(mc.get_tool_reference())
mc.set_end_type(1) #设置为工具坐标系



