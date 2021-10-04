import math
import numpy as np


def quaternions(angle_a, angle_b, angle_r):
    w = math.cos(angle_a/2)*math.cos(angle_b/2)*math.cos(angle_r/2) + \
        math.sin(angle_a/2)*math.sin(angle_b/2)*math.sin(angle_r/2)
    x = math.sin(angle_a/2)*math.cos(angle_b/2)*math.cos(angle_r/2) - \
        math.cos(angle_a/2)*math.sin(angle_b/2)*math.sin(angle_r/2)
    y = math.cos(angle_a/2)*math.sin(angle_b/2)*math.cos(angle_r/2) + \
        math.sin(angle_a/2)*math.cos(angle_b/2)*math.sin(angle_r/2)
    z = math.cos(angle_a/2)*math.cos(angle_b/2)*math.sin(angle_r/2) - \
        math.sin(angle_a/2)*math.sin(angle_b/2)*math.cos(angle_r/2)
    q = [w, x, y, z]
    return q

n = (0, 0, -1)

angleY = math.atan2(n[0], n[2]) * 180/math.pi
if angleY < 0:
    angleY = 180+angleY
else:
    angleY = angleY-180

angleX = math.atan2(n[1], n[2]) * 180/math.pi
if angleX < 0:
    angleX = 180+angleX
else:
    angleX = angleX-180
print("Angle-α: %.3f" % angleX)
print("Angle-β: %.3f" % angleY)

Rm = [0.012603671, 0.998969251, 0.043607133,
      -0.999920493, 0.012608848, 0.000156353,
      0.000393644, 0.043605637, -0.999048744
      ]
Tm = [-0.559843133,
      -0.005578311,
      0.961478158
      ]
angle_robot = [-(n[0]*Rm[0]+n[1]*Rm[1]+n[2]*Rm[2]),
               -(n[0]*Rm[3]+n[1]*Rm[4]+n[2]*Rm[5]),
               -(n[0]*Rm[6]+n[1]*Rm[7]+n[2]*Rm[8])]
# angle_robot = [-0.5,0,1]
# RPY角度变换
angle_a = math.atan2(angle_robot[2], angle_robot[1])-math.pi/2
# if angle_a <= -3.4:
#     angle_a = -3.4
# if angle_a >= -2.88:
#     angle_a = -2.88

angle_b = -math.atan2(angle_robot[2], angle_robot[0])-math.pi/2
# if angle_b >= 0.26:
#     angle_b = 0.26
# if angle_b <= -0.26:
#     angle_b = -0.26
# angle_a = math.pi - 0.7853982
# angle_b = 0
print("Angle-α: %.3f" % (angle_a/(3.14159)*180))
print("Angle-β: %.3f" % (angle_b/(3.14159)*180))

# angle_a = math.pi
# angle_b = math.pi/4
angle_r = 0

q1 = quaternions(angle_a, angle_b, angle_r)
q2 = quaternions(0, 0, 0)

q_f = [q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3],
       q1[1]*q2[0]+q1[0]*q2[1]-q1[3]*q2[2]+q1[2]*q2[3],
       q1[2]*q2[0]+q1[3]*q2[1]+q1[0]*q2[2]-q1[1]*q2[3],
       q1[3]*q2[0]-q1[2]*q2[1]+q1[1]*q2[2]+q1[0]*q2[3]]

path_point = []
waypoint = []
pose = [0, 0, 0, 0, 0, 0, 0]
pose[0] = -0.5
pose[1] = 0
pose[2] = 0.2
pose[3] = q_f[0]
pose[4] = q_f[1]
pose[5] = q_f[2]
pose[6] = q_f[3]
waypoint.append(pose)

waypoint.append(pose)
path_point.append(waypoint)
np.save('/home/ye/shot/pose.npy', path_point)
end = 1
