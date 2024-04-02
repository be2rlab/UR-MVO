#ifndef FRAME_H_
#define FRAME_H_

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <opencv2/opencv.hpp>
#include <string>

#include "camera.h"
#include "mappoint.h"
#include "utils.h"

/**
 * @todo Jaafar: Add those as global setting in yaml file.
 */
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

/**
 * @todo Jaafar: Add parent class for Frames, and inherit mono, stereo, and
 * multi from it
 */
class Frame {
public:
  Frame();
  Frame(int frame_id, bool pose_fixed, CameraPtr camera, double timestamp,
        int id_ts = -1);
  Frame &operator=(const Frame &other);

  void SetFrameId(int frame_id);
  int GetFrameId();
  int GetFrameId_ts();
  double GetTimestamp();
  void SetPoseFixed(bool pose_fixed);
  bool PoseFixed();
  void SetPose(const Eigen::Matrix4d &pose);
  Eigen::Matrix4d &GetPose();

  // point features
  bool FindGrid(double &x, double &y, int &grid_x, int &grid_y);
  void AddFeatures(Eigen::Matrix<double, 259, Eigen::Dynamic> &features);

  void AddFeatures(Eigen::Matrix<double, 259, Eigen::Dynamic> &features_left,
                   Eigen::Matrix<double, 259, Eigen::Dynamic> &features_right,
                   std::vector<cv::DMatch> &stereo_matches);

  void
  AddLeftFeatures(Eigen::Matrix<double, 259, Eigen::Dynamic> &features_left);
  int AddRightFeatures(
      Eigen::Matrix<double, 259, Eigen::Dynamic> &features_right,
      std::vector<cv::DMatch> &stereo_matches);

  Eigen::Matrix<double, 259, Eigen::Dynamic> &GetAllFeatures();

  size_t FeatureNum();
  bool GetKeypointPosition(size_t idx, Eigen::Vector3d &keypoint_pos);
  bool GetKeypointPosition(size_t idx, Eigen::Vector2d &keypoint_pos);
  std::vector<cv::KeyPoint> &GetAllKeypoints();
  cv::KeyPoint &GetKeypoint(size_t idx);
  int GetInlierFlag(std::vector<bool> &inliers_feature_message);

  double GetRightPosition(size_t idx);
  std::vector<double> &GetAllRightPosition();

  bool GetDescriptor(size_t idx,
                     Eigen::Matrix<double, 256, 1> &descriptor) const;

  double GetDepth(size_t idx);
  std::vector<double> &GetAllDepth();
  void SetDepth(size_t idx, double depth);

  void SetTrackIds(std::vector<int> &track_ids);
  std::vector<int> &GetAllTrackIds();
  void SetTrackId(size_t idx, int track_id);
  int GetTrackId(size_t idx);

  MappointPtr GetMappoint(size_t idx);
  std::vector<MappointPtr> &GetAllMappoints();
  void InsertMappoint(size_t idx, MappointPtr mappoint);
  bool BackProjectPoint(size_t idx, Eigen::Vector3d &p3D);
  bool BackProjectPointDepth(size_t idx, Eigen::Vector3d &p3D);
  bool BackProjectPointStereo(size_t idx, Eigen::Vector3d &p3D);
  CameraPtr GetCamera();
  void FindNeighborKeypoints(Eigen::Vector2d &p2D, std::vector<int> &indices,
                             double r, bool filter = true) const;
  void FindNeighborKeypoints(Eigen::Vector3d &p2D, std::vector<int> &indices,
                             double r, bool filter = true) const;

  // covisibility graph
  void AddConnection(std::shared_ptr<Frame> frame, int weight);
  void
  AddConnection(std::set<std::pair<int, std::shared_ptr<Frame>>> connections);
  void SetParent(std::shared_ptr<Frame> parent);
  std::shared_ptr<Frame> GetParent();
  void SetChild(std::shared_ptr<Frame> child);
  std::shared_ptr<Frame> GetChild();

  void RemoveConnection(std::shared_ptr<Frame> frame);
  void RemoveMappoint(MappointPtr mappoint);
  void RemoveMappoint(int idx);
  void DecreaseWeight(std::shared_ptr<Frame> frame, int weight);

  std::vector<std::pair<int, std::shared_ptr<Frame>>>
  GetOrderedConnections(int number);

  void SetPreviousFrame(const std::shared_ptr<Frame> previous_frame);
  std::shared_ptr<Frame> PreviousFrame();

  void remove();

  void SetDepthImg(const cv::Mat &depth_img);
  cv::Mat &GetDepthImg();

public:
  int tracking_frame_id;
  int local_map_optimization_frame_id;
  int local_map_optimization_fix_frame_id;

  std::vector<std::map<int, double>> relation_left;
  std::vector<std::map<int, double>> relation_right;

  cv::Mat debug_img;

private:
  int _frame_id;
  int _id_ts; // specific for data pipeline
  double _timestamp;
  bool _pose_fixed;
  Eigen::Matrix4d _pose;

  cv::Mat _depth_img;

  // point features
  Eigen::Matrix<double, 259, Eigen::Dynamic> _features;
  std::vector<cv::KeyPoint> _keypoints;
  std::vector<int> _feature_grid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
  double _grid_width_inv;
  double _grid_height_inv;
  std::vector<double> _u_right;
  std::vector<double> _depth;
  std::vector<int> _track_ids;
  std::vector<MappointPtr> _mappoints;

  CameraPtr _camera;

  // covisibility graph
  std::map<std::shared_ptr<Frame>, int> _connections;
  std::set<std::pair<int, std::shared_ptr<Frame>>> _ordered_connections;
  std::shared_ptr<Frame> _parent;
  std::shared_ptr<Frame> _child;
  std::shared_ptr<Frame> _previous_frame;
};

typedef std::shared_ptr<Frame> FramePtr;

#endif // FRAME_H_