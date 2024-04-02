from __future__ import annotations
from time import time
from typing import List
import numpy as np
from pyquaternion import Quaternion

import cv2 as cv
from enum import Enum
import torch

class Setup(Enum):
    MONO = "mono"
    STEREO = "stereo"


class Image:
    """
    Class representing an image object.
    """

    def __init__(self, image: np.ndarray, timestamp: float = None):
        """
        Initialize the Image object.

        Args:
            image (np.ndarray): The image data as a numpy array.
            timestamp (float, optional): The timestamp associated with the image. Defaults to None.
        """
        self._image = image
        self._timestamp = time() if timestamp is None else timestamp

    def get_image(self):
        """
        Get the image data.

        Returns:
            np.ndarray: The image data.
        """
        return self._image

    def get_timestamp(self):
        """
        Get the timestamp associated with the image.

        Returns:
            float: The timestamp.
        """
        return self._timestamp

    def set_image(self, image):
        """
        Set the image data.

        Args:
            image (np.ndarray): The image data as a numpy array.
        """
        self._image = image

    def set_timestamp(self, timestamp):
        """
        Set the timestamp associated with the image.

        Args:
            timestamp (float): The timestamp.
        """
        self._timestamp = timestamp

    def copy(self):
        """
        Create a copy of the Image object.

        Returns:
            Image: The copied Image object.
        """
        return Image(self._image.copy(), self._timestamp)


class Mask:
    """
    Class representing a mask object.
    """

    def __init__(self, mask: np.ndarray, timestamp: float = None):
        """
        Initialize the Mask object.

        Args:
            mask (np.ndarray): The mask data as a numpy array.
            timestamp (float, optional): The timestamp associated with the mask. Defaults to None.
        """
        self._mask = mask
        self._timestamp = time() if timestamp is None else timestamp

    def get_mask(self):
        """
        Get the mask data.

        Returns:
            np.ndarray: The mask data.
        """
        return self._mask

    def get_timestamp(self):
        """
        Get the timestamp associated with the mask.

        Returns:
            float: The timestamp.
        """
        return self._timestamp

    def set_mask(self, mask):
        """
        Set the mask data.

        Args:
            mask (np.ndarray): The mask data as a numpy array.
        """
        self._mask = mask

    def set_timestamp(self, timestamp):
        """
        Set the timestamp associated with the mask.

        Args:
            timestamp (float): The timestamp.
        """
        self._timestamp = timestamp

    def copy(self):
        """
        Create a copy of the Mask object.

        Returns:
            Mask: The copied Mask object.
        """
        return Mask(self._mask.copy(), self._timestamp)


class DepthMap:
    """
    Class representing a depth map object.
    """

    def __init__(self, depth_map: np.ndarray, timestamp: float = None):
        """
        Initialize the DepthMap object.

        Args:
            depth_map (np.ndarray): The depth map data as a numpy array.
            timestamp (float, optional): The timestamp associated with the depth map. Defaults to None.
        """
        self._depth_map = depth_map
        self._timestamp = time() if timestamp is None else timestamp

    def get_depth_map(self):
        """
        Get the depth map data.

        Returns:
            np.ndarray: The depth map data.
        """
        return self._depth_map

    def get_timestamp(self):
        """
        Get the timestamp associated with the depth map.

        Returns:
            float: The timestamp.
        """
        return self._timestamp

    def set_depth_map(self, depth_map):
        """
        Set the depth map data.

        Args:
            depth_map (np.ndarray): The depth map data as a numpy array.
        """
        self._depth_map = depth_map

    def set_timestamp(self, timestamp):
        """
        Set the timestamp associated with the depth map.

        Args:
            timestamp (float): The timestamp.
        """
        self._timestamp = timestamp

    def copy(self):
        """
        Create a copy of the DepthMap object.

        Returns:
            DepthMap: The copied DepthMap object.
        """
        return DepthMap(self._depth_map.copy(), self._timestamp)


class Frame:
    """
    Frame that consists of image, pose, depth map and mask
    """

    __id: int = 0

    def __init__(
        self,
        frame_id: int = None,
        timestamp: float = None,
        pose: Pose = None,
        image: Image = None,
        right_image: Image = None,
        depth_map: DepthMap = None,
        mask: Mask = None,
    ) -> None:
        if frame_id is None:
            self._id = Frame.__id
            Frame.__id += 1
        else:
            self._id = frame_id
        self._timestamp = time() if timestamp is None else timestamp
        self._pose = pose
        self._image = image
        self._right_image = right_image
        self._depth_map = depth_map
        self._mask = mask

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, id_index: int):
        self._id = id_index

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @property
    def pose(self) -> Pose:
        return self._pose

    @pose.setter
    def pose(self, pose: Pose):
        self._pose = pose

    @property
    def image(self) -> Image:
        return self._image

    @property
    def right_image(self) -> Image:
        return self._right_image

    @property
    def depth_map(self) -> DepthMap:
        return self._depth_map

    @property
    def mask(self) -> Mask:
        return self._mask

    def copy(self):
        frame_copy = Frame(
            frame_id=self._id,
            timestamp=self._timestamp,
            pose=self._pose.copy() if self._pose is not None else None,
            image=self.image.copy() if self.image is not None else None,
            right_image=(
                self.right_image.copy() if self.right_image is not None else None
            ),
            depth_map=self.depth_map.copy() if self.depth_map is not None else None,
            mask=self.mask.copy() if self.mask is not None else None,
        )
        return frame_copy


DTYPE = torch.float64


class Pose:
    def __init__(
        self,
        rotation_matrix: torch.Tensor = torch.eye(3),
        translation: torch.Tensor = torch.zeros((3,)),
        covariance: torch.Tensor = torch.eye(6),
    ):
        self._rotation_matrix: torch.Tensor = rotation_matrix.to(dtype=DTYPE)
        self._translation: torch.Tensor = translation.to(dtype=DTYPE)
        self._covariance: torch.Tensor = covariance.to(dtype=DTYPE)

    @property
    def translation(self) -> torch.Tensor:
        return self._translation.to(dtype=DTYPE)

    @translation.setter
    def translation(self, translation: torch.Tensor):
        assert translation.shape == (3, 1) or translation.shape == (
            3,
        ), f"translation.shape == {translation.shape}"
        self._translation = translation

    @property
    def rotation_matrix(self) -> torch.Tensor:
        return self._rotation_matrix.to(dtype=DTYPE)

    @rotation_matrix.setter
    def rotation_matrix(self, rotation_matrix: torch.Tensor):
        assert rotation_matrix.shape == (
            3,
            3,
        ), f"rotation_matrix.shape == {rotation_matrix.shape}"
        self._rotation_matrix = rotation_matrix

    @property
    def orientation(self) -> Quaternion:
        return Quaternion(matrix=self._rotation_matrix.numpy(), rtol=1e-02, atol=1e-02)

    @orientation.setter
    def orientation(self, quaternion: Quaternion):
        self._rotation_matrix = torch.from_numpy(quaternion.rotation_matrix)

    @property
    def transformation_matrix(self) -> torch.Tensor:
        """
        Column-major transformation matrix
        """
        transformation = torch.eye(4, dtype=DTYPE)
        transformation[:3, :3] = self._rotation_matrix
        transformation[:3, 3] = self._translation
        return transformation

    @transformation_matrix.setter
    def transformation_matrix(self, transformation: torch.Tensor):
        assert transformation.shape == (
            4,
            4,
        ), f"rotation_matrix.shape == {transformation.shape}"
        self._rotation_matrix = transformation[:3, :3]
        self._translation = transformation[:3, 3]

    def copy(self) -> Pose:
        return Pose(
            rotation_matrix=self._rotation_matrix.clone(),
            translation=self._translation.clone(),
            covariance=self._covariance.clone(),
        )
