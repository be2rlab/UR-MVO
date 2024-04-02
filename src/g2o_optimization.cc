#include "g2o_optimization.h"

#include "camera.h"
#include <fstream>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include "read_configs.h"

void LocalmapOptimization(
    MapOfPoses &poses, MapOfPoints3d &points,
    std::vector<CameraPtr> &camera_list,
    VectorOfMonoPointConstraints &mono_point_constraints,
    VectorOfStereoPointConstraints &stereo_point_constraints,
    const OptimizationConfig &cfg) {
  // Setup optimizer
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1>> SlamBlockSolver;
  typedef g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType>
      SlamLinearSolver;

  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);

  auto linear_solver = std::make_unique<SlamLinearSolver>();
  linear_solver->setBlockOrdering(false);
  g2o::OptimizationAlgorithmLevenberg *solver =
      new g2o::OptimizationAlgorithmLevenberg(
          std::make_unique<SlamBlockSolver>(std::move(linear_solver)));
  optimizer.setAlgorithm(solver);

  // frame vertex
  int max_frame_id = 0;
  for (auto &kv : poses) {
    g2o::VertexSE3Expmap *frame_vertex = new g2o::VertexSE3Expmap();
    frame_vertex->setEstimate(g2o::SE3Quat(kv.second.q, kv.second.p).inverse());
    frame_vertex->setId(kv.first);
    frame_vertex->setFixed(kv.second.fixed);
    max_frame_id = std::max(max_frame_id, kv.first);
    optimizer.addVertex(frame_vertex);
  }
  max_frame_id++;

  // point vertex
  int max_point_id = 0;
  for (auto &kv : points) {
    g2o::VertexPointXYZ *point_vertex = new g2o::VertexPointXYZ();
    point_vertex->setEstimate(kv.second.p);
    int point_id = kv.first + max_frame_id;
    point_vertex->setId((point_id));
    max_point_id = std::max(max_point_id, point_id);
    point_vertex->setMarginalized(true);
    optimizer.addVertex(point_vertex);
  }
  max_point_id++;

  // point edges
  std::vector<g2o::EdgeSE3ProjectXYZ *> mono_edges;
  mono_edges.reserve(mono_point_constraints.size());
  std::vector<g2o::EdgeStereoSE3ProjectXYZ *> stereo_edges;
  stereo_edges.reserve(stereo_point_constraints.size());
  const float thHuberMonoPoint = sqrt(cfg.mono_point);
  const float thHuberStereoPoint = sqrt(cfg.stereo_point);
  CameraType camType = CameraType::MONO;
  // mono point edges
  for (MonoPointConstraintPtr &mpc : mono_point_constraints) {
    g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                        optimizer.vertex((mpc->id_point + max_frame_id))));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                        optimizer.vertex(mpc->id_pose)));
    e->setMeasurement(mpc->keypoint);
    e->setInformation(Eigen::Matrix2d::Identity());
    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(thHuberMonoPoint);
    e->fx = camera_list[mpc->id_camera]->Fx();
    e->fy = camera_list[mpc->id_camera]->Fy();
    e->cx = camera_list[mpc->id_camera]->Cx();
    e->cy = camera_list[mpc->id_camera]->Cy();
    camType = camera_list[mpc->id_camera]->GetCameraType();
    optimizer.addEdge(e);
    mono_edges.push_back(e);
  }

  // stereo point edges
  if (camType == CameraType::STEREO) {
    for (StereoPointConstraintPtr &spc : stereo_point_constraints) {
      g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

      e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                          optimizer.vertex((spc->id_point + max_frame_id))));
      e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                          optimizer.vertex(spc->id_pose)));
      e->setMeasurement(spc->keypoint);
      e->setInformation(Eigen::Matrix3d::Identity());
      g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
      e->setRobustKernel(rk);
      rk->setDelta(thHuberStereoPoint);
      e->fx = camera_list[spc->id_camera]->Fx();
      e->fy = camera_list[spc->id_camera]->Fy();
      e->cx = camera_list[spc->id_camera]->Cx();
      e->cy = camera_list[spc->id_camera]->Cy();
      e->bf = camera_list[spc->id_camera]->BF();

      optimizer.addEdge(e);
      stereo_edges.push_back(e);
    }
  }
  /**
   * @todo(Jaafar): Here I commented Stereo edge for mono, is there any need for
   * replacment?
   */

  // solve
  optimizer.initializeOptimization();
  optimizer.optimize(10);

  // check inlier observations
  for (size_t i = 0; i < mono_edges.size(); i++) {
    g2o::EdgeSE3ProjectXYZ *e = mono_edges[i];
    if (e->chi2() > cfg.mono_point || !e->isDepthPositive()) {
      e->setLevel(1);
    }
    e->setRobustKernel(0);
  }

  for (size_t i = 0; i < stereo_edges.size(); i++) {
    g2o::EdgeStereoSE3ProjectXYZ *e = stereo_edges[i];
    if (e->chi2() > cfg.stereo_point || !e->isDepthPositive()) {
      e->setLevel(1);
    }
    e->setRobustKernel(0);
  }

  // optimize again without the outliers
  optimizer.initializeOptimization(0);
  optimizer.optimize(5);

  // check inlier observations
  for (size_t i = 0; i < mono_edges.size(); i++) {
    g2o::EdgeSE3ProjectXYZ *e = mono_edges[i];
    mono_point_constraints[i]->inlier =
        (e->chi2() <= cfg.mono_point && e->isDepthPositive());
  }

  for (size_t i = 0; i < stereo_edges.size(); i++) {
    g2o::EdgeStereoSE3ProjectXYZ *e = stereo_edges[i];
    stereo_point_constraints[i]->inlier =
        (e->chi2() <= cfg.stereo_point && e->isDepthPositive());
  }

  // Recover optimized data
  // Keyframes
  for (MapOfPoses::iterator it = poses.begin(); it != poses.end(); ++it) {
    g2o::VertexSE3Expmap *frame_vertex =
        static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(it->first));
    g2o::SE3Quat SE3quat = frame_vertex->estimate().inverse();
    it->second.p = SE3quat.translation();
    it->second.q = SE3quat.rotation();
  }
  // Points
  for (MapOfPoints3d::iterator it = points.begin(); it != points.end(); ++it) {
    g2o::VertexPointXYZ *point_vertex = static_cast<g2o::VertexPointXYZ *>(
        optimizer.vertex(it->first + max_frame_id));
    it->second.p = point_vertex->estimate();
  }
}

int FrameOptimization(MapOfPoses &poses, MapOfPoints3d &points,
                      std::vector<CameraPtr> &camera_list,
                      VectorOfMonoPointConstraints &mono_point_constraints,
                      VectorOfStereoPointConstraints &stereo_point_constraints,
                      const OptimizationConfig &cfg) {
  assert(poses.size() == 1);
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);
  std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linear_solver;
  linear_solver = std::make_unique<
      g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
  g2o::OptimizationAlgorithmLevenberg *solver =
      new g2o::OptimizationAlgorithmLevenberg(
          std::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver)));
  optimizer.setAlgorithm(solver);

  // frame vertex
  MapOfPoses::iterator pose_it = poses.begin();
  g2o::VertexSE3Expmap *frame_vertex = new g2o::VertexSE3Expmap();
  frame_vertex->setEstimate(
      g2o::SE3Quat(pose_it->second.q, pose_it->second.p).inverse());
  frame_vertex->setId(0);
  frame_vertex->setFixed(false);
  optimizer.addVertex(frame_vertex);

  // point edges
  std::vector<g2o::EdgeSE3ProjectXYZOnlyPose *> mono_edges;
  mono_edges.reserve(mono_point_constraints.size());
  std::vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> stereo_edges;
  stereo_edges.reserve(stereo_point_constraints.size());
  const float deltaMonoPoint = sqrt(cfg.mono_point);
  const float deltaStereoPoint = sqrt(cfg.stereo_point);
  CameraType camType = CameraType::MONO;
  // mono edges
  for (MonoPointConstraintPtr &mpc : mono_point_constraints) {
    Position3d point = points[mpc->id_point];

    g2o::EdgeSE3ProjectXYZOnlyPose *e = new g2o::EdgeSE3ProjectXYZOnlyPose();
    e->setVertex(
        0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
    e->setMeasurement(mpc->keypoint);
    e->setInformation(Eigen::Matrix2d::Identity());
    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(deltaMonoPoint);
    e->fx = camera_list[mpc->id_camera]->Fx();
    e->fy = camera_list[mpc->id_camera]->Fy();
    e->cx = camera_list[mpc->id_camera]->Cx();
    e->cy = camera_list[mpc->id_camera]->Cy();
    camType = camera_list[mpc->id_camera]->GetCameraType();
    e->Xw = point.p;

    optimizer.addEdge(e);
    mono_edges.push_back(e);
  }
  // stereo edges
  if (camType == CameraType::STEREO) {
    for (StereoPointConstraintPtr &spc : stereo_point_constraints) {
      Position3d point = points[spc->id_point];

      g2o::EdgeStereoSE3ProjectXYZOnlyPose *e =
          new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
      e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                          optimizer.vertex(0)));
      e->setMeasurement(spc->keypoint);
      e->setInformation(Eigen::Matrix3d::Identity());
      g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
      e->setRobustKernel(rk);
      rk->setDelta(deltaStereoPoint);
      e->fx = camera_list[spc->id_camera]->Fx();
      e->fy = camera_list[spc->id_camera]->Fy();
      e->cx = camera_list[spc->id_camera]->Cx();
      e->cy = camera_list[spc->id_camera]->Cy();
      e->bf = camera_list[spc->id_camera]->BF();
      e->Xw = point.p;

      optimizer.addEdge(e);
      stereo_edges.push_back(e);
    }
  }

  // solve
  const int its[4] = {10, 10, 10, 10};

  int num_outlier = 0;
  for (size_t iter = 0; iter < 4; iter++) {
    frame_vertex->setEstimate(
        g2o::SE3Quat(pose_it->second.q, pose_it->second.p).inverse());
    optimizer.initializeOptimization(0);
    optimizer.optimize(its[iter]);

    num_outlier = 0;
    for (size_t i = 0; i < mono_edges.size(); i++) {
      g2o::EdgeSE3ProjectXYZOnlyPose *e = mono_edges[i];
      if (!mono_point_constraints[i]->inlier) {
        e->computeError();
      }

      const float chi2 = e->chi2();
      if (chi2 > cfg.mono_point) {
        mono_point_constraints[i]->inlier = false;
        e->setLevel(1);
        num_outlier++;
      } else {
        mono_point_constraints[i]->inlier = true;
        e->setLevel(0);
      }

      if (iter == 2)
        e->setRobustKernel(0);
    }

    for (size_t i = 0; i < stereo_edges.size(); i++) {
      g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = stereo_edges[i];
      if (!stereo_point_constraints[i]->inlier) {
        e->computeError();
      }

      const float chi2 = e->chi2();
      if (chi2 > cfg.stereo_point) {
        stereo_point_constraints[i]->inlier = false;
        e->setLevel(1);
        num_outlier++;
      } else {
        stereo_point_constraints[i]->inlier = true;
        e->setLevel(0);
      }
      if (iter == 2)
        e->setRobustKernel(0);
    }

    if (optimizer.edges().size() < 10)
      break;
  }

  // recover optimized data
  g2o::SE3Quat SE3quat = frame_vertex->estimate().inverse();
  pose_it->second.p = SE3quat.translation();
  pose_it->second.q = SE3quat.rotation();

  return (mono_point_constraints.size() + stereo_point_constraints.size() -
          num_outlier);
}

int SolvePnPWithCV(FramePtr frame, std::vector<MappointPtr> &mappoints,
                   Eigen::Matrix4d &pose, std::vector<int> &inliers) {
  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
  std::vector<int> point_indexes;
  cv::Mat camera_matrix, dist_coeffs;
  CameraPtr camera = frame->GetCamera();
  camera->GetCamerMatrix(camera_matrix);
  camera->GetDistCoeffs(dist_coeffs);
  cv::Mat rotation_vector;
  cv::Mat translation_vector;
  cv::Mat cv_inliers;

  for (size_t i = 0; i < mappoints.size(); i++) {
    MappointPtr mpt = mappoints[i];
    if (mpt == nullptr || !mpt->IsValid())
      continue;
    Eigen::Vector2d keypoint;
    if (!frame->GetKeypointPosition(i, keypoint))
      continue;
    const Eigen::Vector3d &point_position = mpt->GetPosition();
    object_points.emplace_back(point_position(0), point_position(1),
                               point_position(2));
    image_points.emplace_back(keypoint(0), keypoint(1));
    point_indexes.emplace_back(i);
  }
  if (object_points.size() < 8)
    return 0;

  try {
    cv::solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs,
                       rotation_vector, translation_vector, false, 100, 20.0,
                       0.99, cv_inliers);
  } catch (...) {
    return 0;
  }

  cv::Mat cv_Rcw;
  cv::Rodrigues(rotation_vector, cv_Rcw);
  Eigen::Matrix3d eigen_Rcw;
  Eigen::Vector3d eigen_tcw;
  cv::cv2eigen(cv_Rcw, eigen_Rcw);
  cv::cv2eigen(translation_vector, eigen_tcw);
  Eigen::Matrix3d eigen_Rwc = eigen_Rcw.transpose();
  pose.block<3, 3>(0, 0) = eigen_Rwc;
  pose.block<3, 1>(0, 3) = eigen_Rwc * (-eigen_tcw);

  inliers = std::vector<int>(mappoints.size(), -1);
  for (int i = 0; i < cv_inliers.rows; i++) {
    int inlier_idx = cv_inliers.at<int>(i, 0);
    int point_idx = point_indexes[inlier_idx];
    inliers[point_idx] = mappoints[point_idx]->GetId();
  }
  return cv_inliers.rows;
}
