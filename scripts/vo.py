from ur_mvo.engine import UR_MVO

import os
import cv2 as cv

from nav_msgs.msg import Path
from sensor_msgs.msg import Image as ImageROS
from geometry_msgs.msg import Pose as PoseMsg
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import cv_bridge



import time
import rclpy
from rclpy.node import Node
import time

from ur_mvo.components import Setup, Frame, Mask, DepthMap, Image
class vo(Node):
    def __init__(self):

        super().__init__('ur_mvo_node')
        self.bridge = cv_bridge.CvBridge()
        config = {}

        config["config_path"] = "configs_aqua.yaml"
        # config["config_path"] = "configs_cave.yaml"

        vo = UR_MVO(config, Setup.MONO)

        # poses = vo.process_directory('/data/harbor/raw_data1/harbor_images_sequence_01')
        directory = '/data/aqua/harbor/raw_data6/harbor_images_sequence_06'
        # directory = '/data/cave_cam0'
        image_stamps = sorted(os.listdir(directory))
        self.pub = self.create_publisher(Path, 'unscaled_pose_topic', 10)
        self.im_pub = self.create_publisher(ImageROS, 'raw_image', 10)
        # self.timer=self.create_timer(1.0,self.timer_callback)
        print(f"Reading {len(image_stamps)} Images")
        self.poses=[]
        self.count=0
        self.path_msg=Path()
        i = 0
        for image_stamp in image_stamps:
            # if i < 1000:
            #     i+= 1  
            #     continue
            image = cv.imread(os.path.join(directory, image_stamp), cv.IMREAD_GRAYSCALE)
            # image = cv.resize(image, (0,0), fx= 0.5, fy= 0.5)
            image_msg = self.bridge.cv2_to_imgmsg(image)
            self.im_pub.publish(image_msg)
            self.count+=1
            cv.imshow('Image', image)
            cv.waitKey(1)
            # if self.count==2000:
            #     break
            frame = Frame(image=Image(image))
            pose = vo.process(frame)
            if(pose!=None):
                    self.poses += pose
                    P=PoseStamped()
                    for i in range(len(pose)):
                        P.pose.position.x,P.pose.position.y,P.pose.position.z = pose[i].translation.numpy()
                        R_mat=R.from_matrix(pose[i].rotation_matrix.numpy())
                        P.pose.orientation.x,P.pose.orientation.y,P.pose.orientation.z,P.pose.orientation.w = R_mat.as_quat()
                        self.path_msg.poses.append(P)
            time.sleep(30 / 1000)
                
        while(True):
          self.pub.publish(self.path_msg)  
          time.sleep(1)
def main(args=None):
    rclpy.init(args=args)
    ur_mvo_node = vo()
    # rclpy.spin(ur_mvo_node)
    ur_mvo_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
