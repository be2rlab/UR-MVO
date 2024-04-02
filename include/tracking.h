#ifndef TRACKING_H_
#define TRACKING_H_

#include <Eigen/Core>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "camera.h"
#include "dataset.h"
#include "epipolar_geometry.h"
#include "frame.h"
#include "types.h"
#include "mapping.h"
#include "point_matching.h"
#include "read_configs.h"
#include "ros2_publisher.h"
#include "super_glue.h"
#include "super_point.h"

struct TrackingData {
  FramePtr frame;
  FramePtr ref_keyframe;
  std::vector<cv::DMatch> matches;
  InputDataPtr input_data;

  TrackingData() {}

  TrackingData &operator=(TrackingData &other) {
    frame = other.frame;
    ref_keyframe = other.ref_keyframe;
    matches = other.matches;
    input_data = other.input_data;
    return *this;
  }
};

typedef std::shared_ptr<TrackingData> TrackingDataPtr;

class Tracking {
public:
  Tracking(Configs &configs);

  void AddInput(InputDataPtr data);

  void ExtractFeatureThread();

  void TrackingThread();

  void ExtractFeatrue(const cv::Mat &image, const cv::Mat &mask,
                      Eigen::Matrix<double, 259, Eigen::Dynamic> &points);

  void ExtractFeatureAndMatch(
      const cv::Mat &image, const cv::Mat &mask,
      const Eigen::Matrix<double, 259, Eigen::Dynamic> &points0,
      Eigen::Matrix<double, 259, Eigen::Dynamic> &points1,
      std::vector<cv::DMatch> &matches);

  bool Init(FramePtr frame, const cv::Mat &image, const cv::Mat &mask,
            const cv::Mat &depth);
  bool InitStereo(FramePtr frame, const cv::Mat &image, const cv::Mat &mask,
                  const cv::Mat &image_right);

  int TrackFrame(FramePtr frame0, FramePtr frame1,
                 std::vector<cv::DMatch> &matches);

  // pose_init = 0 : opencv pnp, pose_init = 1 : last frame pose, pose_init = 2
  // : original pose
  int FramePoseOptimization(FramePtr frame, std::vector<MappointPtr> &mappoints,
                            std::vector<int> &inliers, int pose_init = 0);

  bool AddKeyframe(FramePtr last_keyframe, FramePtr current_frame,
                   int num_match);

  void InsertKeyframe(FramePtr frame, const cv::Mat &mask,
                      const cv::Mat &image_right);
  void InsertKeyframe(FramePtr frame);

  void KeyframeCulling();

  // for tracking local map
  void UpdateReferenceFrame(FramePtr frame);

  void UpdateLocalKeyframes(FramePtr frame);

  void UpdateLocalMappoints(FramePtr frame);

  void
  SearchLocalPoints(FramePtr frame,
                    std::vector<std::pair<int, MappointPtr>> &good_projections);

  int TrackLocalMap(FramePtr frame, int num_inlier_thr);

  void PublishFrame(FramePtr frame, cv::Mat &image);

  void SaveTrajectory();

  void SaveTrajectory(std::string file_path);

  void SaveMap(const std::string &map_root);

  void ShutDown();

  bool use_mask() { return _use_mask; }

  SensorSetup sensor_setup() { return _sensor_setup; }

  CameraType getCameraType() { return _camera_type; }

  bool gotResult() { return _pose_buffer.size() > 0; }

  Eigen::MatrixXd getPose() {
    Eigen::MatrixXd pose = Eigen::MatrixXd::Identity(4, 4);
    if (!_pose_buffer.empty()) {
      pose = _pose_buffer.front();
      _pose_buffer.pop();
    }
    return pose;
  }
  void reset();

private:
  // left feature extraction and tracking thread
  std::mutex _buffer_mutex;
  std::queue<InputDataPtr> _data_buffer;
  std::thread _feature_thread;

  std::queue<Eigen::MatrixXd> _pose_buffer;

  // pose estimation thread
  std::mutex _tracking_mutex;
  std::queue<TrackingDataPtr> _tracking_data_buffer;
  std::thread _tracking_thread;

  // gpu mutex
  std::mutex _gpu_mutex;

  bool _shutdown;

  // tmp
  bool _init;
  int _track_id;
  FramePtr _last_frame;
  FramePtr _last_keyframe;
  int _num_since_last_keyframe;
  bool _last_frame_track_well;

  cv::Mat _last_image;
  cv::Mat _last_right_image;
  cv::Mat _last_keyimage;

  Pose3d _last_pose;

  // for tracking local map
  bool _to_update_local_map;
  FramePtr _ref_keyframe;
  FramePtr _init_frame;
  cv::Mat _init_depth;
  std::vector<MappointPtr> _local_mappoints;
  std::vector<FramePtr> _local_keyframes;

  // class
  Configs _configs;
  CameraPtr _camera;
  SuperPointPtr _superpoint;
  PointMatchingPtr _point_matching;
  Ros2PublisherPtr _ros_publisher;
  std::shared_ptr<EpipolarGeometry> _2_view_reconstruct;
  MapPtr _map;

  bool _use_mask;
  SensorSetup _sensor_setup;
  CameraType _camera_type;
};

#endif