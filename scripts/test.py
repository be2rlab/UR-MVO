from ur_mvo.engine import UR_MVO
from ur_mvo.components import Setup, Frame, Mask, DepthMap, Image

import os
import cv2 as cv
import time
config = {}

config["config_path"] = "configs_aqua.yaml"
# config["config_path"] = "configs_cave.yaml"

vo = UR_MVO(config, Setup.MONO)

# poses = vo.process_directory('/data/harbor/raw_data1/harbor_images_sequence_01')
directory = '/data/aqua/harbor/raw_data1/harbor_images_sequence_01'
# directory = '/data/cave_cam0'
image_stamps = sorted(os.listdir(directory))
print(f"Reading {len(image_stamps)} Images")
poses = []
i = 0
for image_stamp in image_stamps:
    image = cv.imread(os.path.join(directory, image_stamp), cv.IMREAD_GRAYSCALE)
    
    cv.imshow('Image', image)
    cv.waitKey(1)
    frame = Frame(image=Image(image))
    pose = vo.process(frame)
    if pose is not None:
        poses += pose

    # time.sleep(0.1)
    # poses.append(pose)

