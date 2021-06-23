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

import numpy as np
import cv2

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
import math

# Set up logger.
logger = Logger.get_logger("examples/policy.py")

if __name__ == "__main__":
    # Parse args.
    parser = argparse.ArgumentParser(
        description="Run a grasping policy on an example image")
    parser.add_argument("model_name",
                        type=str,
                        default=None,
                        help="name of a trained model to run")
    parser.add_argument(
        "--depth_image",
        type=str,
        default=None,
        help="path to a test depth image stored as a .npy file")
    parser.add_argument("--segmask",
                        type=str,
                        default=None,
                        help="path to an optional segmask to use")
    parser.add_argument("--camera_intr",
                        type=str,
                        default=None,
                        help="path to the camera intrinsics")
    parser.add_argument("--model_dir",
                        type=str,
                        default=None,
                        help="path to the folder in which the model is stored")
    parser.add_argument("--config_filename",
                        type=str,
                        default=None,
                        help="path to configuration file to use")
    parser.add_argument(
        "--fully_conv",
        action="store_true",
        help=("run Fully-Convolutional GQ-CNN policy instead of standard"
              " GQ-CNN policy"))
    args = parser.parse_args()
    model_name = args.model_name
    depth_im_filename = args.depth_image
    segmask_filename = args.segmask
    camera_intr_filename = args.camera_intr
    model_dir = args.model_dir
    config_filename = args.config_filename
    fully_conv = args.fully_conv

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
        colorimg = plt.imread('/home/ye/img/mask/rgb-22.png-cut.png')
        vis.figure(size=(10, 10))
        plt.imshow(colorimg)
        # vis.imshow(rgbd_im.depth,
        #            vmin=policy_config["vis"]["vmin"],
        #            vmax=policy_config["vis"]["vmax"])
        vis.title("Planned grasp at depth {0:.3f}m with Q={1:.3f}".format(
            action.grasp.depth, action.q_value))
        vis.grasp(action.grasp, scale=2.5, show_center=True, show_axis=True)
        logger.info("Center Point: (%d, %d)" %
                    (action.grasp.center.data[0], action.grasp.center.data[1]))
        
        logger.info("Width: %d" % (action.grasp.width*1000))
        # 以图像中心为旋转中心
        center = (action.grasp.center.data[0].astype(int),
                  action.grasp.center.data[1].astype(int))
        angle = (action.grasp.angle / (3.14159 * 2) * 360)                 # 顺时针旋转90°
        scale = 1                  # 等比例旋转，即旋转后尺度不变
        img = np.load('/home/ye/img/depth/depth-A.npy')
        logger.info("Center: (%.2f, %.2f, %.2f)" %
                    (img[center[0], center[1], 0]*1000, img[center[0], center[1], 1]*1000, img[center[0], center[1], 2]*1000))
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
        

        s = dst.shape[0]

        xyzs = np.zeros((s*s, 3))

        xyzs[:, 0] = dst[:, :, 0].flatten()*1000
        xyzs[:, 1] = dst[:, :, 1].flatten()*1000
        xyzs[:, 2] = dst[:, :, 2].flatten()*1000

        # test data
        xyz = []

        for i in range(s*s):  # 筛除外点
            if xyzs[i, 2] <= 830:
                if xyzs[i, 2] >= 600:
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

        #计算方差
        R = 0
        for i in range(0, xyzm.shape[0]):
            R = R+(X[0, 0] * x2[i] + X[1, 0] * y2[i] + X[2, 0] - z2[i])**2
        print('方差为：%.*f' % (3, R))

        # 展示图像
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        ax1.scatter(x2, y2, z2, c='r', marker='o')
        x_p = np.linspace(40, 120, 100)
        y_p = np.linspace(0, 80, 100)
        x_p, y_p = np.meshgrid(x_p, y_p)
        z_p = X[0, 0] * x_p + X[1, 0] * y_p + X[2, 0]
        ax1.plot_wireframe(x_p, y_p, z_p, rstride=10, cstride=10)

        px = x2[900]
        py = y2[900]
        pz = X[0, 0] * px + X[1, 0] * py + X[2, 0]
        fz = np.linspace(740, 880, 100)
        fx = (X[0, 0]/(-1))*(fz-pz)+px
        fy = (X[1, 0]/(-1))*(fz-pz)+py
        # 得到法向量
        n = (X[0, 0], X[1, 0], -1)
        print('Vector = ', n)
        # 法向量旋转回原坐标
        m = (n[0]*math.cos(action.grasp.angle)+n[1]*math.sin(action.grasp.angle),
             -n[0]*math.sin(action.grasp.angle)+n[1]*math.cos(action.grasp.angle), -1)
        print('Vector = ', m)
        angleY = math.atan2(m[0], -1) * 180/math.pi
        if angleY < 0:
            angleY = -180-angleY
        else:
            angleY = -angleY+180
        
        
        angleX = math.atan2(m[1], -1) * 180/math.pi
        if angleX < 0:
            angleX = 180+angleX
        else:
            angleX = angleX-180
        logger.info("Angle-α: %.3f" % angleX)
        logger.info("Angle-β: %.3f" % angleY)
        logger.info("Angle-θ: %.3f" %
                    (action.grasp.angle / (3.14159 * 2) * 360))
        plt.plot(fx, fy, fz, 'y', linewidth=5)

        vis.show()
