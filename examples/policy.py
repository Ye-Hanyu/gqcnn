# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents).
All Rights Reserved. Permission to use, copy, modify, and distribute this
software and its documentation for educational, research, and not-for-profit
purposes, without fee and without a signed licensing agreement, is hereby
granted, provided that the above copyright notice, this paragraph and the
following two paragraphs appear in all copies, modifications, and
distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150
Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
otl@berkeley.edu,
http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Displays robust grasps planned using a GQ-CNN grapsing policy on a set of saved
RGB-D images. The default configuration for the standard GQ-CNN policy is
`cfg/examples/cfg/examples/gqcnn_pj.yaml`. The default configuration for the
Fully-Convolutional GQ-CNN policy is `cfg/examples/fc_gqcnn_pj.yaml`.

Author
------
Jeff Mahler & Vishal Satish
"""
import argparse
import json
import os
import time
import shutil
import numpy as np
import cv2
import math

from autolab_core import YamlConfig, Logger
from perception import (BinaryImage, CameraIntrinsics, ColorImage, DepthImage,
                        RgbdImage)
from visualization import Visualizer2D as vis

from gqcnn.grasping import (RobustGraspingPolicy,
                            CrossEntropyRobustGraspingPolicy, RgbdImageState,
                            FullyConvolutionalGraspingPolicyParallelJaw,
                            FullyConvolutionalGraspingPolicySuction)
from gqcnn.utils import GripperMode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up logger.
logger = Logger.get_logger("examples/policy.py")


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


if __name__ == "__main__":
    # Parse args.
    # parser = argparse.ArgumentParser(
    #     description="Run a grasping policy on an example image")
    # parser.add_argument("model_name",
    #                     type=str,
    #                     default="GQCNN-4.0-PJ",
    #                     help="name of a trained model to run")
    # parser.add_argument(
    #     "--depth_image",
    #     type=str,
    #     default="/home/ye/img/depth/depth-d.npy",
    #     help="path to a test depth image stored as a .npy file")
    # parser.add_argument("--segmask",
    #                     type=str,
    #                     default="/home/ye/img/mask/gear-shaft-mask.jpg",
    #                     help="path to an optional segmask to use")
    # parser.add_argument("--camera_intr",
    #                     type=str,
    #                     default="/home/ye/shot/kinect.intr",
    #                     help="path to the camera intrinsics")
    # parser.add_argument("--model_dir",
    #                     type=str,
    #                     default=None,
    #                     help="path to the folder in which the model is stored")
    # parser.add_argument("--config_filename",
    #                     type=str,
    #                     default=None,
    #                     help="path to configuration file to use")
    # parser.add_argument(
    #     "--fully_conv",
    #     action="store_true",
    #     help=("run Fully-Convolutional GQ-CNN policy instead of standard"
    #           " GQ-CNN policy"))
    # args = parser.parse_args()

    # model_name = args.model_name
    # depth_im_filename = args.depth_image
    # segmask_filename = args.segmask
    # camera_intr_filename = args.camera_intr
    # model_dir = args.model_dir
    # config_filename = args.config_filename
    # fully_conv = args.fully_conv
    shutil.rmtree('/home/ye/shot/grasp')
    os.mkdir('/home/ye/shot/grasp')
    part_list = np.load('/home/ye/shot/mask/class.npy')
    model_name = "GQCNN-4.0-PJ"
    depth_im_filename = "/home/ye/shot/depth/depth-d.npy"
    camera_intr_filename = "/home/ye/shot/kinect.intr"
    model_dir = "/home/ye/gqcnn/models"
    config_filename = "/home/ye/gqcnn/cfg/examples/gqcnn_pj.yaml"
    fully_conv = False
    path_point = []

    for part_num in range(part_list.shape[0]):
        segmask_filename = "/home/ye/shot/mask/" + \
            part_list[part_num][0]+"-mask.jpg"
        full_path = '/home/ye/shot/grasp/' + part_list[part_num][0] + '.txt'
        txt_file = open(full_path, 'a')

        assert not (fully_conv and depth_im_filename is not None
                    and segmask_filename is None
                    ), "Fully-Convolutional policy expects a segmask."

        if depth_im_filename is None:
            if fully_conv:
                depth_im_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "..",
                    "data/examples/clutter/primesense/depth_0.npy")
            else:
                depth_im_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "..",
                    "data/examples/single_object/primesense/depth_0.npy")
        if fully_conv and segmask_filename is None:
            segmask_filename = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "..",
                "data/examples/clutter/primesense/segmask_0.png")
        if camera_intr_filename is None:
            camera_intr_filename = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "..",
                "data/calib/primesense/primesense.intr")

        # Set model if provided.
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    "../models")
        model_path = os.path.join(model_dir, model_name)

        # Get configs.
        model_config = json.load(open(os.path.join(model_path, "config.json"),
                                    "r"))
        try:
            gqcnn_config = model_config["gqcnn"]
            gripper_mode = gqcnn_config["gripper_mode"]
        except KeyError:
            gqcnn_config = model_config["gqcnn_config"]
            input_data_mode = gqcnn_config["input_data_mode"]
            if input_data_mode == "tf_image":
                gripper_mode = GripperMode.LEGACY_PARALLEL_JAW
            elif input_data_mode == "tf_image_suction":
                gripper_mode = GripperMode.LEGACY_SUCTION
            elif input_data_mode == "suction":
                gripper_mode = GripperMode.SUCTION
            elif input_data_mode == "multi_suction":
                gripper_mode = GripperMode.MULTI_SUCTION
            elif input_data_mode == "parallel_jaw":
                gripper_mode = GripperMode.PARALLEL_JAW
            else:
                raise ValueError(
                    "Input data mode {} not supported!".format(input_data_mode))

        # Set config.
        if config_filename is None:
            if (gripper_mode == GripperMode.LEGACY_PARALLEL_JAW
                    or gripper_mode == GripperMode.PARALLEL_JAW):
                if fully_conv:
                    config_filename = os.path.join(
                        os.path.dirname(os.path.realpath(__file__)), "..",
                        "cfg/examples/fc_gqcnn_pj.yaml")
                else:
                    config_filename = os.path.join(
                        os.path.dirname(os.path.realpath(__file__)), "..",
                        "cfg/examples/gqcnn_pj.yaml")
            elif (gripper_mode == GripperMode.LEGACY_SUCTION
                or gripper_mode == GripperMode.SUCTION):
                if fully_conv:
                    config_filename = os.path.join(
                        os.path.dirname(os.path.realpath(__file__)), "..",
                        "cfg/examples/fc_gqcnn_suction.yaml")
                else:
                    config_filename = os.path.join(
                        os.path.dirname(os.path.realpath(__file__)), "..",
                        "cfg/examples/gqcnn_suction.yaml")

        # Read config.
        config = YamlConfig(config_filename)
        inpaint_rescale_factor = config["inpaint_rescale_factor"]
        policy_config = config["policy"]

        # Make relative paths absolute.
        if "gqcnn_model" in policy_config["metric"]:
            policy_config["metric"]["gqcnn_model"] = model_path
            if not os.path.isabs(policy_config["metric"]["gqcnn_model"]):
                policy_config["metric"]["gqcnn_model"] = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "..",
                    policy_config["metric"]["gqcnn_model"])

        # Setup sensor.
        camera_intr = CameraIntrinsics.load(camera_intr_filename)

        # Read images.
        depth_data = np.load(depth_im_filename)
        depth_im = DepthImage(depth_data, frame=camera_intr.frame)
        color_im = ColorImage(np.zeros([depth_im.height, depth_im.width,
                                        3]).astype(np.uint8),
                            frame=camera_intr.frame)

        # Optionally read a segmask.
        segmask = None
        if segmask_filename is not None:
            segmask = BinaryImage.open(segmask_filename)
        valid_px_mask = depth_im.invalid_pixel_mask().inverse()
        if segmask is None:
            segmask = valid_px_mask
        else:
            segmask = segmask.mask_binary(valid_px_mask)

        # Inpaint.
        depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)

        if "input_images" in policy_config["vis"] and policy_config["vis"][
                "input_images"]:
            vis.figure(size=(10, 10))
            num_plot = 1
            if segmask is not None:
                num_plot = 2
            vis.subplot(1, num_plot, 1)
            vis.imshow(depth_im)
            if segmask is not None:
                vis.subplot(1, num_plot, 2)
                vis.imshow(segmask)
            vis.show()

        # Create state.
        rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
        state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)

        # Set input sizes for fully-convolutional policy.
        if fully_conv:
            policy_config["metric"]["fully_conv_gqcnn_config"][
                "im_height"] = depth_im.shape[0]
            policy_config["metric"]["fully_conv_gqcnn_config"][
                "im_width"] = depth_im.shape[1]

        # Init policy.
        if fully_conv:
            # TODO(vsatish): We should really be doing this in some factory policy.
            if policy_config["type"] == "fully_conv_suction":
                policy = FullyConvolutionalGraspingPolicySuction(policy_config)
            elif policy_config["type"] == "fully_conv_pj":
                policy = FullyConvolutionalGraspingPolicyParallelJaw(policy_config)
            else:
                raise ValueError(
                    "Invalid fully-convolutional policy type: {}".format(
                        policy_config["type"]))
        else:
            policy_type = "cem"
            if "type" in policy_config:
                policy_type = policy_config["type"]
            if policy_type == "ranking":
                policy = RobustGraspingPolicy(policy_config)
            elif policy_type == "cem":
                policy = CrossEntropyRobustGraspingPolicy(policy_config)
            else:
                raise ValueError("Invalid policy type: {}".format(policy_type))

        # Query policy.
        policy_start = time.time()
        action = policy(state)
        logger.info("Planning took %.3f sec" % (time.time() - policy_start))

        # Vis final grasp.
        if policy_config["vis"]["final_grasp"]:

            colorimg = plt.imread('/home/ye/shot/mask/cut.png')
            vis.figure(size=(10, 10))
            plt.imshow(colorimg)
            # vis.imshow(rgbd_im.depth,
            #            vmin=policy_config["vis"]["vmin"],
            #            vmax=policy_config["vis"]["vmax"])
            vis.title("Planned grasp at depth {0:.3f}m with Q={1:.3f}".format(
                action.grasp.depth, action.q_value))
            vis.grasp(action.grasp, scale=2.5, show_center=True, show_axis=True)
            plt.savefig("/home/ye/shot/grasp/" + part_list[part_num][0] + "-grasp.png")
            logger.info("Pixel Center: (%d, %d)" %
                        (action.grasp.center.data[0], action.grasp.center.data[1]))
            txt_file.write("Pixel Center: (%d, %d)" %
                           (action.grasp.center.data[0], action.grasp.center.data[1]))
            txt_file.write('\n')
            logger.info("Width: %d" % (action.grasp.width*1000))
            # 以图像中心为旋转中心
            center = (action.grasp.center.data[0].astype(int),
                    action.grasp.center.data[1].astype(int))
            angle = (action.grasp.angle / (3.14159 * 2) * 360)                 # 顺时针旋转90°
            scale = 1                  # 等比例旋转，即旋转后尺度不变
            img = np.load('/home/ye/shot/depth/depth-A.npy')
            logger.info("World Center: (%.2f, %.2f, %.2f)" %
                        (img[center[1], center[0], 0]*1000, img[center[1], center[0], 1]*1000, img[center[1], center[0], 2]*1000))
            txt_file.write("World Center: (%.2f, %.2f, %.2f)" %
                           (img[center[1], center[0], 0]*1000, img[center[1], center[0], 1]*1000, img[center[1], center[0], 2]*1000))
            txt_file.write('\n')
            M = cv2.getRotationMatrix2D(center, angle, scale)
            image_rotation = cv2.warpAffine(src=img, M=M, dsize=(
                640, 480))
            # cv2.imshow('1', image_rotation)
            # cv2.waitKey(1000)
            # cv2.imwrite('/home/ye/img/test/rotation.png', image_rotation*255)
            cutwidth = int(action.grasp.width_px / 2)
            dst = image_rotation[int(center[1] - cutwidth):int(center[1] + cutwidth),
                                int(center[0] - cutwidth):int(center[0] + cutwidth)]   # 裁剪坐标为[y0:y1, x0:x1]
            res = cv2.resize(dst, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
            # np.save('cut.npy', dst)
            # cv2.imshow('2', dst)
            # cv2.waitKey(1000)
            # cv2.imwrite('/home/ye/img/test/cut.png', dst*255)

            # 抓取点真实坐标
            g_p = [img[center[1], center[0], 0], img[center[1],
                                                     center[0], 1], img[center[1], center[0], 2]]
            s = dst.shape[0]

            xyzs = np.zeros((480*640, 3))

            xyzs[:, 0] = img[:, :, 0].flatten()*1000
            xyzs[:, 1] = img[:, :, 1].flatten()*1000
            xyzs[:, 2] = img[:, :, 2].flatten()*1000

            # test data
            xyz = []
            th = 20  # 筛除阈值

            for i in range(480*640):  # 筛除外点
                if xyzs[i, 2] <= g_p[2]*1000 + th and xyzs[i, 2] >= g_p[2]*1000 - th:
                    if xyzs[i, 0] >= g_p[0]*1000 - 40 and xyzs[i, 0] <= g_p[0]*1000 + 40:
                        if xyzs[i, 1] >= g_p[1]*1000 - 40 and xyzs[i, 1] <= g_p[1]*1000 + 40:
                            xyz.append(xyzs[i])
            xyzm = np.array(xyz)

            x2 = xyzm[:, 0]
            y2 = xyzm[:, 1]
            z2 = xyzm[:, 2]

            # 创建系数矩阵A
            A = np.zeros((3, 3))
            for i in range(0, xyzm.shape[0]):
                A[0, 0] = A[0, 0]+x2[i]**2
                A[0, 1] = A[0, 1]+x2[i]*y2[i]
                A[0, 2] = A[0, 2]+x2[i]
                A[1, 0] = A[0, 1]
                A[1, 1] = A[1, 1]+y2[i]**2
                A[1, 2] = A[1, 2]+y2[i]
                A[2, 0] = A[0, 2]
                A[2, 1] = A[1, 2]
                A[2, 2] = xyzm.shape[0]
            # print(A)

            # 创建b
            b = np.zeros((3, 1))
            for i in range(0, xyzm.shape[0]):
                b[0, 0] = b[0, 0]+x2[i]*z2[i]
                b[1, 0] = b[1, 0]+y2[i]*z2[i]
                b[2, 0] = b[2, 0]+z2[i]
            # print(b)

            # 求解X
            A_inv = np.linalg.inv(A)
            X = np.dot(A_inv, b)
            print('平面拟合结果为：z = %.3f * x + %.3f * y + %.3f' % (X[0, 0], X[1, 0], X[2, 0]))
            txt_file.write('平面拟合结果为：z = %.3f * x + %.3f * y + %.3f' %
                           (X[0, 0], X[1, 0], X[2, 0]))
            txt_file.write('\n')

            # 计算方差
            R = 0
            for i in range(0, xyzm.shape[0]):
                R = R+(X[0, 0] * x2[i] + X[1, 0] * y2[i] + X[2, 0] - z2[i])**2
            print('方差为：%.*f' % (3, R))

            # 展示图像
            fig1 = plt.figure()
            
            ax1 = fig1.add_subplot(111, projection='3d')
            ax1.set_aspect('auto')
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.set_zlabel("z")
            ax1.scatter(x2, y2, z2, c='b', marker='.')
            x_p = np.linspace(g_p[0]*1000-50, g_p[0]*1000+50, 100)
            y_p = np.linspace(g_p[1]*1000-50, g_p[1]*1000+50, 100)
            x_p, y_p = np.meshgrid(x_p, y_p)
            z_p = X[0, 0] * x_p + X[1, 0] * y_p + X[2, 0]
            
            ax1.plot_wireframe(x_p, y_p, z_p, color='green', rstride=10, cstride=10)

            px = x_p[50, 50]
            py = y_p[50, 50]
            pz = X[0, 0] * px + X[1, 0] * py + X[2, 0]
            fz = np.linspace(850, 950, 100)
            fx = (X[0, 0]/(-1))*(fz-pz)+px
            fy = (X[1, 0]/(-1))*(fz-pz)+py
            plt.savefig("/home/ye/shot/grasp/" +
                        part_list[part_num][0] + "-point.png")

            # 得到法向量
            n = (X[0, 0], X[1, 0], -1)
            print('Vector = ', n)
            txt_file.write('Vector = ')
            txt_file.write('\n')
            txt_file.write(str(n))
            txt_file.write('\n')

            # 法向量旋转回原坐标
            # m = (n[0]*math.cos(action.grasp.angle)+n[1]*math.sin(action.grasp.angle),
            #      -n[0]*math.sin(action.grasp.angle)+n[1]*math.cos(action.grasp.angle), -1)
            # print('Vector = ', m)
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
            logger.info("Angle-x: %.3f" % angleX)
            logger.info("Angle-y: %.3f" % angleY)
            logger.info("Angle-z: %.3f" %
                        (action.grasp.angle / (3.14159 * 2) * 360))
            txt_file.write("Angle-x: %.3f" % angleX)
            txt_file.write('\n')
            txt_file.write("Angle-y: %.3f" % angleY)
            txt_file.write('\n')
            txt_file.write("Angle-z: %.3f" %
                           (action.grasp.angle / (3.14159 * 2) * 360))
            txt_file.write('\n')

            plt.plot(fx, fy, fz, 'r', linewidth=4)

            # 计算悬停时ur5末端位姿
            x_tool = img[center[1], center[0], 0]
            y_tool = img[center[1], center[0], 1]           
            z_tool = img[center[1], center[0], 2]

            # 手眼标定结果
            Rm = [0.012603671, 0.998969251, 0.043607133,
                  -0.999920493, 0.012608848, 0.000156353,
                  0.000393644, 0.043605637, -0.999048744
                  ]
            Tm = [-0.559843133,
                  -0.005578311,
                  0.961478158
                  ]

            # 机械爪抓取向量计算
            angle_robot = [-(n[0]*Rm[0]+n[1]*Rm[1]+n[2]*Rm[2]),
                           -(n[0]*Rm[3]+n[1]*Rm[4]+n[2]*Rm[5]),
                           -(n[0]*Rm[6]+n[1]*Rm[7]+n[2]*Rm[8])]
            print(angle_robot)
            txt_file.write("Vector-robot:")
            txt_file.write('\n')
            txt_file.write(str(angle_robot))
            txt_file.write('\n')

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
            logger.info("Angle-X: %.3f" % (angle_a/math.pi*180))
            logger.info("Angle-Y: %.3f" % (angle_b/math.pi*180))
            txt_file.write("Angle-X: %.3f" % (angle_a/math.pi*180))
            txt_file.write('\n')
            txt_file.write("Angle-Y: %.3f" % (angle_b/math.pi*180))
            txt_file.write('\n')
            angle_r = 0

            g_e_p1 = img[int(action.grasp.endpoints[0][1]),
                         int(action.grasp.endpoints[0][0])]
            g_e_p2 = img[int(action.grasp.endpoints[1][1]),
                         int(action.grasp.endpoints[1][0])]
            angle_g = math.atan2(g_e_p2[1]-g_e_p1[1], g_e_p2[0]-g_e_p1[0])
            n_g = [1, math.tan(angle_g), 0]

            # 机械爪旋转向量计算
            angle_grip = [(n_g[0]*Rm[0]+n_g[1]*Rm[1]+n_g[2]*Rm[2]),
                          (n_g[0]*Rm[3]+n_g[1]*Rm[4]+n_g[2]*Rm[5]),
                          (n_g[0]*Rm[6]+n_g[1]*Rm[7]+n_g[2]*Rm[8])]
            print(angle_grip)
            txt_file.write("Vector-robot:")
            txt_file.write('\n')
            txt_file.write(str(angle_grip))
            txt_file.write('\n')

            angle_rg = -math.atan2(angle_grip[1], angle_grip[0])
            logger.info("Angle-Z: %.3f" % (angle_rg/math.pi*180))
            txt_file.write("Angle-Z: %.3f" % (angle_rg/math.pi*180))
            txt_file.write('\n')
            
            # q1为X，Y轴旋转所得四元数，q2为抓取检测所得旋转角度四元数
            q1 = quaternions(angle_a, angle_b, angle_r)
            q2 = quaternions(0, 0, angle_rg)

            # 两四元数叠加
            q_f = [q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3],
                   q1[1]*q2[0]+q1[0]*q2[1]-q1[3]*q2[2]+q1[2]*q2[3],
                   q1[2]*q2[0]+q1[3]*q2[1]+q1[0]*q2[2]-q1[1]*q2[3],
                   q1[3]*q2[0]-q1[2]*q2[1]+q1[1]*q2[2]+q1[0]*q2[3]]

            # 写入悬停路径点
            waypoint = []
            pose = [0, 0, 0, 0, 0, 0, 0]
            pose[0] = x_tool*Rm[0]+y_tool*Rm[1] + \
                z_tool*Rm[2]+Tm[0]+angle_robot[0]/angle_robot[2]*0.15
            pose[1] = x_tool*Rm[3]+y_tool*Rm[4]+z_tool * \
                Rm[5]+Tm[1]+angle_robot[1]/angle_robot[2]*0.15
            pose[2] = x_tool*Rm[6]+y_tool*Rm[7]+z_tool*Rm[8]+Tm[2]+0.15
            pose[3] = q_f[0]
            pose[4] = q_f[1]
            pose[5] = q_f[2]
            pose[6] = q_f[3]
            waypoint.append(pose)
            logger.info("悬停点：Pose: (%.4f, %.4f, %.4f)" % (pose[0], pose[1], pose[2]))
            logger.info("Orient: (%.4f, %.4f, %.4f, %.4f)" %
                        (pose[3], pose[4], pose[5], pose[6]))
            txt_file.write("悬停点")
            txt_file.write('\n')
            txt_file.write(str(pose))
            txt_file.write('\n')

            # 抓取点
            pose = [0, 0, 0, 0, 0, 0, 0]
            pose[0] = x_tool*Rm[0]+y_tool*Rm[1]+z_tool*Rm[2]+Tm[0]
            pose[1] = x_tool*Rm[3]+y_tool*Rm[4]+z_tool*Rm[5]+Tm[1]
            pose[2] = x_tool*Rm[6]+y_tool*Rm[7]+z_tool*Rm[8]+Tm[2]-0.015
            pose[3] = q_f[0]
            pose[4] = q_f[1]
            pose[5] = q_f[2]
            pose[6] = q_f[3]
            waypoint.append(pose)
            logger.info("抓取点：Pose: (%.4f, %.4f, %.4f)" %
                        (pose[0], pose[1], pose[2]))
            logger.info("Orient: (%.4f, %.4f, %.4f, %.4f)" %
                        (pose[3], pose[4], pose[5], pose[6]))
            txt_file.write("抓取点")
            txt_file.write('\n')
            txt_file.write(str(pose))
            txt_file.write('\n')
            txt_file.close()

            if part_list[part_num][0] == 'gear-shaft':
                box_point1 = [-0.55, 0.3, 0.2, 0, 1, 0, 0]
                box_point2 = [-0.55, 0.3, 0.15, 0, 1, 0, 0]
                waypoint.append(box_point1)
                waypoint.append(box_point2)

            if part_list[part_num][0] == 'gear1':
                waypoint[0][2] += 0.00
                box_point1 = [-0.55, 0.3, 0.2, 0, 1, 0, 0]
                box_point2 = [-0.55, 0.3, 0.15, 0, 1, 0, 0]
                waypoint.append(box_point1)
                waypoint.append(box_point2)

            if part_list[part_num][0] == 'gear2':
                waypoint[0][2] += 0.00
                box_point1 = [-0.55, 0.3, 0.2, 0, 1, 0, 0]
                box_point2 = [-0.55, 0.3, 0.15, 0, 1, 0, 0]
                waypoint.append(box_point1)
                waypoint.append(box_point2)

            if part_list[part_num][0] == 'planet-carrier':
                waypoint[0][2] += 0.00
                box_point1 = [-0.55, 0.3, 0.2, 0, 1, 0, 0]
                box_point2 = [-0.55, 0.3, 0.15, 0, 1, 0, 0]
                waypoint.append(box_point1)
                waypoint.append(box_point2)

            if part_list[part_num][0] == 'Screw':
                box_point1 = [-0.8, 0.3, 0.3, 0, 1, 0, 0]
                box_point2 = [-0.8, 0.3, 0.15, 0, 1, 0, 0]
                waypoint.append(box_point1)
                waypoint.append(box_point2)

            vis.show()
            path_point.append(waypoint)
    np.save('/home/ye/shot/pose.npy', path_point)
