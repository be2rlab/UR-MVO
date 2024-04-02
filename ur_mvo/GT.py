import numpy as np
import rclpy 
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation as R
from scipy.linalg import orthogonal_procrustes

global_path = Path()

class GTNode(Node):
    def __init__(self):
        super().__init__('gt_publisher')
        self.publisher_ = self.create_publisher(Path, '/gt_path', 10)
        data_path = '/workspace/ros2_ws/src/visual_odometry/data/harbor_colmap_traj_sequence_01.txt'
        data = np.loadtxt(data_path, delimiter=' ') 
        path = Path()
        path.header.frame_id = "map"
        path.poses = []
        print("data.shape[0]: ", data.shape[0])
        for i in range(data.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = data[i, 1]
            pose.pose.position.y = data[i, 2]
            pose.pose.position.z = data[i, 3]
            pose.pose.orientation.x = data[i, 4]
            pose.pose.orientation.y = data[i, 5]
            pose.pose.orientation.z = data[i, 6]
            pose.pose.orientation.w = data[i, 7]
            path.poses.append(pose)
        print(path.poses[0])
        global_path = path
        self.path = path
        self.timer = self.create_timer(0.1, self.timer_callback)
    def timer_callback(self):
        self.publisher_.publish(self.path)

def allign_paths(traj1:Path, traj2:Path):
    traj1 = [[pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w] for pose in traj1.poses]
    traj2 = [[pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w] for pose in traj2.poses]
    # Assuming traj1 and traj2 are your two trajectories, each a list of [x, y, z, qx, qy, qz, qw]
    traj1_points = np.array([point[:3] for point in traj1])
    traj2_points = np.array([point[:3] for point in traj2])

    # Use Umeyama's method to find the best alignment
    R, scale = orthogonal_procrustes(traj1_points, traj2_points)

    # Apply the transformation to the first trajectory
    traj1_points_transformed = scale * np.dot(traj1_points, R)

    # For the quaternion part, you might need to convert them to rotation matrices first, 
    # perform the rotation, and then convert back to quaternions
    traj1_quaternions = [R.from_quat(point[3:]) for point in traj1]
    traj1_rotations = [quat.as_matrix() for quat in traj1_quaternions]

    # Apply the rotation to the first trajectory's rotations
    traj1_rotations_transformed = [np.dot(R, rotation) for rotation in traj1_rotations]

    # Convert back to quaternions
    traj1_quaternions_transformed = [R.from_matrix(rotation).as_quat() for rotation in traj1_rotations_transformed]

    returned_traj1 = [np.concatenate([point, quat]) for point, quat in zip(traj1_points_transformed, traj1_quaternions_transformed)]
    returned_path = Path()
    returned_path.header.frame_id = "map"
    returned_path.poses = []
    for pose in returned_traj1:
        pose_msg = PoseStamped()
        pose_msg.pose.position.x = pose[0]
        pose_msg.pose.position.y = pose[1]
        pose_msg.pose.position.z = pose[2]
        pose_msg.pose.orientation.x = pose[3]
        pose_msg.pose.orientation.y = pose[4]
        pose_msg.pose.orientation.z = pose[5]
        pose_msg.pose.orientation.w = pose[6]
        returned_path.poses.append(pose_msg)
    return returned_path


class PoseAlignmentNode(Node):
    def __init__(self):
        super().__init__('pose_alignment')
        self.subscription = self.create_subscription(Path, '/visual_odometry/odometry', self.path_callback, 10)
        self.save_path = '/workspace/ros2_ws/src/visual_odometry/data/est.txt'
    def path_callback(self, msg):
        # overwrite the previous path
        i=1
        with open(self.save_path, 'w') as f:
            for pose in msg.poses:
                i=i+1
                f.write(f"{i} {pose.pose.position.x} {pose.pose.position.y} {pose.pose.position.z} {pose.pose.orientation.x} {pose.pose.orientation.y} {pose.pose.orientation.z} {pose.pose.orientation.w}\n")

def main(args=None):
    rclpy.init(args=args)
    gt_node = GTNode()
    pose_alignment_node = PoseAlignmentNode()
    rclpy.spin(pose_alignment_node)
    gt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':  
    main()
