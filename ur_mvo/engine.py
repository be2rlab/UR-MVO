from py_ur_mvo import UR_MVO as VO
from ur_mvo.components import Setup, Frame, Mask, DepthMap, Image, Pose

import cv2 as cv
import numpy as np
import torch
from typing import Any, Dict, List, Tuple
import os
from pathlib import Path


from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion

def list_immediate_folders(directory):
    if os.path.exists(directory):
        folders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
        return folders
    else:
        return []

def interpolate(p1: Pose, p2: Pose, samples:int = 1) -> List[Pose]:
    pose1_translation = p1.translation.numpy()
    rot1 = Quaternion([p1.orientation.w, p1.orientation.x, p1.orientation.y, p1.orientation.z])

    pose2_translation = p2.translation.numpy()
    rot2 = Quaternion([p2.orientation.w, p2.orientation.x, p2.orientation.y, p2.orientation.z])


    poses = []
    for i in range(samples+1):
        timestamp = (1.0 / (float(samples) + 1.0)) * float(i)
        interpolated_rotation = Quaternion.slerp(rot1, rot2, timestamp)
        
        interpolated_translation = pose1_translation + (pose2_translation - pose1_translation) * timestamp
        interpolated_quaternion = interpolated_rotation

        p = Pose(torch.from_numpy(interpolated_quaternion.rotation_matrix), torch.from_numpy(interpolated_translation))
        poses.append(p)
    return poses

class UR_MVO:
    def __init__(self, config: dict, setup: Setup):
        self.config = config
        self.setup = setup
        self.vo = VO(config, setup.value)
        self.last_pose = None
        self.accumulated_samples = 0

    def process(self, data: Frame) -> List[Pose]:
        if self.setup == Setup.MONO:
            if data.mask is None:
                pose = self.vo.processMono(data.image.get_image())
                if pose[3,3] < 0.5:
                    self.accumulated_samples += 1
                    return None
                
                rot = torch.from_numpy(pose[0:3, 0:3])
                trans = torch.from_numpy(pose[0:3, 3])
                current_pose = Pose(rot, trans, torch.eye(6))
                
                if self.last_pose is None:
                    self.last_pose = current_pose
                    return [self.last_pose]
                res = interpolate(self.last_pose, current_pose, self.accumulated_samples)
                self.accumulated_samples = 0
                self.last_pose = current_pose
                return res
            else:
                pose = self.vo.processMonoWithMask(data.image.get_image(), data.mask.get_mask())
                if pose[3,3] < 0.5:
                    self.accumulated_samples += 1
                    return None
                
                rot = torch.from_numpy(pose[0:3, 0:3])
                trans = torch.from_numpy(pose[0:3, 3])
                current_pose = Pose(rot, trans, torch.eye(6))
                
                if self.last_pose is None:
                    self.last_pose = current_pose
                    return [self.last_pose]
                res = interpolate(self.last_pose, current_pose, self.accumulated_samples)
                self.accumulated_samples = 0
                self.last_pose = current_pose
                return res
        elif self.setup == Setup.STEREO:
            if data.mask is None:
                pose = self.vo.processStereo(data.image.get_image(), data.right_image.get_image())
                if pose[3,3] < 0.5:
                    self.accumulated_samples += 1
                    return None
                
                rot = torch.from_numpy(pose[0:3, 0:3])
                trans = torch.from_numpy(pose[0:3, 3])
                current_pose = Pose(rot, trans, torch.eye(6))
                
                if self.last_pose is None:
                    self.last_pose = current_pose
                    return [self.last_pose]
                res = interpolate(self.last_pose, current_pose, self.accumulated_samples)
                self.accumulated_samples = 0
                self.last_pose = current_pose
                return res
            else:
                pose = self.vo.processStereoWithMask(data.image.get_image(), data.right_image.get_image(), data.mask.get_mask())
                if pose[3,3] < 0.5:
                    self.accumulated_samples += 1
                    return None
                
                rot = torch.from_numpy(pose[0:3, 0:3])
                trans = torch.from_numpy(pose[0:3, 3])
                current_pose = Pose(rot, trans, torch.eye(6))
                
                if self.last_pose is None:
                    self.last_pose = current_pose
                    return [self.last_pose]
                res = interpolate(self.last_pose, current_pose, self.accumulated_samples)
                self.accumulated_samples = 0
                self.last_pose = current_pose
                return res
        elif self.setup == Setup.RGBD:
            if data.mask is None:
                pose = self.vo.processRGBD(data.image.get_image(), data.depth_map.get_depth_map())
                if pose[3,3] < 0.5:
                    self.accumulated_samples += 1
                    return None
                
                rot = torch.from_numpy(pose[0:3, 0:3])
                trans = torch.from_numpy(pose[0:3, 3])
                current_pose = Pose(rot, trans, torch.eye(6))
                
                if self.last_pose is None:
                    self.last_pose = current_pose
                    return [self.last_pose]
                res = interpolate(self.last_pose, current_pose, self.accumulated_samples)
                self.accumulated_samples = 0
                self.last_pose = current_pose
                return res
            else:
                pose = self.vo.processRGBDWithMask(data.image.get_image(), data.depth_map.get_depth_map(), data.mask.get_mask())
                if pose[3,3] < 0.5:
                    self.accumulated_samples += 1
                    return None
                
                rot = torch.from_numpy(pose[0:3, 0:3])
                trans = torch.from_numpy(pose[0:3, 3])
                current_pose = Pose(rot, trans, torch.eye(6))
                
                if self.last_pose is None:
                    self.last_pose = current_pose
                    return [self.last_pose]
                res = interpolate(self.last_pose, current_pose, self.accumulated_samples)
                self.accumulated_samples = 0
                self.last_pose = current_pose
                return res
    
    def process_directory(self, directory:str) -> List[Pose]:
        poses = []
        assert directory is not None, "Directory is None"
        directory = Path(directory)
        cams = list_immediate_folders(directory)
        num_cams = len(cams)
        for i in range(num_cams):
            assert os.path.isdir(os.path.join(directory, f"cam{i}")), f"Folder {i} does not exist"
            assert os.path.isdir(os.path.join(directory, f"cam{i}", "data")), f"Folder {i}/data does not exist"
        
        image_stamps = sorted(os.listdir(os.path.join(directory, "cam0", "data")))
        for image_stamp in image_stamps:
            image = cv.imread(os.path.join(directory, "cam0", "data", image_stamp), cv.IMREAD_GRAYSCALE)
            if self.setup == Setup.STEREO:
                image_right = cv.imread(os.path.join(directory, "cam1", "data", image_stamp), cv.IMREAD_GRAYSCALE)
                frame = Frame(image=Image(image), right_image=Image(image_right))
            else:
                frame = Frame(image=Image(image))
            pose = self.process(frame)
            if pose is not None:
                poses += pose
            # poses.append(pose)
        
        return poses

    def reset(self, config: dict = None, setup: Setup = None) -> None:
        if config is not None:
            self.config = config
        if setup is not None:
            self.setup = setup
        self.vo.reset(self.config, self.setup.value)

    def shutdown(self) -> None:
        pass

