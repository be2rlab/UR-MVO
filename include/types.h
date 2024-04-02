
#ifndef OPTIMIZATION_3D_TYPES_H_
#define OPTIMIZATION_3D_TYPES_H_

#include <istream>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam3d_addons/types_slam3d_addons.h>

#include "utils.h"

struct Pose3d {
  bool fixed;
  Eigen::Vector3d p;
  Eigen::Quaterniond q;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Pose3d() {}
  Pose3d &operator=(Pose3d &other) {
    fixed = other.fixed;
    p = other.p;
    q = other.q;
    return *this;
  }
};
typedef std::map<int, Pose3d, std::less<int>,
                 Eigen::aligned_allocator<std::pair<const int, Pose3d>>>
    MapOfPoses;

struct Position3d {
  bool fixed;
  Eigen::Vector3d p;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Position3d() {}
  Position3d &operator=(Position3d &other) {
    fixed = other.fixed;
    p = other.p;
    return *this;
  }
};
typedef std::map<int, Position3d, std::less<int>,
                 Eigen::aligned_allocator<std::pair<const int, Position3d>>>
    MapOfPoints3d;

struct MonoPointConstraint {
  int id_pose;
  int id_point;
  int id_camera;
  bool inlier;

  // x_left, y_left
  Eigen::Vector2d keypoint;
  double pixel_sigma;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MonoPointConstraint() {}
  MonoPointConstraint &operator=(MonoPointConstraint &other) {
    id_pose = other.id_pose;
    id_point = other.id_point;
    id_camera = other.id_camera;
    inlier = other.inlier;
    keypoint = other.keypoint;
    pixel_sigma = other.pixel_sigma;
    return *this;
  }
};
typedef std::shared_ptr<MonoPointConstraint> MonoPointConstraintPtr;
typedef std::vector<MonoPointConstraintPtr> VectorOfMonoPointConstraints;

struct StereoPointConstraint {
  int id_pose;
  int id_point;
  int id_camera;
  bool inlier;

  // x_left, y_left, x_right
  Eigen::Vector3d keypoint;
  double pixel_sigma;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  StereoPointConstraint() {}
  StereoPointConstraint &operator=(StereoPointConstraint &other) {
    id_pose = other.id_pose;
    id_point = other.id_point;
    id_camera = other.id_camera;
    inlier = other.inlier;
    keypoint = other.keypoint;
    pixel_sigma = other.pixel_sigma;
    return *this;
  }
};
typedef std::shared_ptr<StereoPointConstraint> StereoPointConstraintPtr;
typedef std::vector<StereoPointConstraintPtr> VectorOfStereoPointConstraints;

#endif // OPTIMIZATION_3D_TYPES_H_
