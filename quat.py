import math


pitch = -171/180*math.pi
yaw = -7/180*math.pi
roll = -11.6/180*math.pi
x = math.sin(pitch/2)*math.cos(yaw/2)*math.cos(roll/2) - \
                 math.cos(pitch/2)*math.sin(yaw/2)*math.sin(roll/2)
y = math.cos(pitch/2)*math.sin(yaw/2)*math.cos(roll/2) + \
        math.sin(pitch/2)*math.cos(yaw/2)*math.sin(roll/2)
z = math.cos(pitch/2)*math.cos(yaw/2)*math.sin(roll/2) - \
        math.sin(pitch/2)*math.sin(yaw/2)*math.cos(roll/2)
w = math.cos(pitch/2)*math.cos(yaw/2)*math.cos(roll/2) + \
        math.sin(pitch/2)*math.sin(yaw/2)*math.sin(roll/2)


a=1
