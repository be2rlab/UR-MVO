import os
import time
import subprocess as sp

import cv2 as cv
import cv_bridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image as ImageROS
from tqdm import tqdm

from ur_mvo.components import Frame, Image, Setup
from ur_mvo.engine import UR_MVO


class vo(Node):
    def __init__(self, images_path=None, gt_path=None, results_path=None):

        super().__init__('ur_mvo_node', parameter_overrides=[])
        self.bridge = cv_bridge.CvBridge()
        self.path_pub = self.create_publisher(Path, '/UR_MVO/path', 10)
        self.img_pub = self.create_publisher(ImageROS, '/UR_MVO/image_orig', 10)

        config = {"config_path": "configs_aqua.yaml"}

        VO = UR_MVO(config, Setup.MONO)

        if images_path is None:
            images_path = '/data/aqua/harbor/raw_data2/harbor_images_sequence_02'
        if gt_path is None:
            gt_path = '/data/aqua/harbor/raw_data2/new_harbor_colmap_traj_sequence_02.txt'
        if results_path is None:
            results_path = '/data/aqua/harbor/raw_data2/results_urmvo'

        image_stamps = sorted(os.listdir(images_path))
        print(f"Reading {len(image_stamps)} Images")
        self.poses = []
        self.count = 0
        self.path_msg = Path()
        self.init = False

        for image_stamp in tqdm(image_stamps):
            if not self.init:
                self.count += 1

            image = cv.imread(os.path.join(images_path, image_stamp), cv.IMREAD_GRAYSCALE)
            image_msg = self.bridge.cv2_to_imgmsg(image)
            self.img_pub.publish(image_msg)
            frame = Frame(image=Image(image))
            pose = VO.process(frame)
            if (pose is not None):
                self.init = True
                self.poses += pose
                P = PoseStamped()
                for i in range(len(pose)):
                    P.pose.position.x, P.pose.position.y, P.pose.position.z = pose[i].translation.numpy()
                    R_mat = R.from_matrix(pose[i].rotation_matrix.numpy())
                    P.pose.orientation.x, P.pose.orientation.y, P.pose.orientation.z, P.pose.orientation.w = R_mat.as_quat()
                    self.path_msg.poses.append(P)
            # time.sleep(1/30)
        poses_path = os.path.join(results_path, 'poses.txt')
        with open(poses_path, 'w+') as f:
            for i in range(len(self.path_msg.poses)):
                if (i % 5 == 0):  # ground truth for aqua is every 5 frames
                    path_msg = self.path_msg
                    x, y, z = path_msg.poses[i].pose.position.x, path_msg.poses[i].pose.position.y, path_msg.poses[i].pose.position.z
                    x_r, y_r, z_r, w_r = path_msg.poses[i].pose.orientation.x, path_msg.poses[i].pose.orientation.y, path_msg.poses[i].pose.orientation.z, path_msg.poses[i].pose.orientation.w
                    f.write(f"{i} {x} {y} {z} {x_r} {y_r} {z_r} {w_r}\n")
        cmd = f"evo_ape tum {gt_path.strip()} {poses_path.strip()} --save_results {os.path.join(results_path, 'res.zip').strip()} --align --correct_scale --pose_relation trans_part --plot --plot_mode xyz --save_plot {os.path.join(results_path, 'plot.pdf').strip()} --t_start {self.count}"
        print(f"Executing: {cmd}")
        sp.call(cmd.split(" "))


def main(args=None):
    rclpy.init(args=args)
    ur_mvo_node = vo()
    ur_mvo_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
