#ifndef MAPPING_H_
#define MAPPING_H_

#include <opencv2/highgui/highgui.hpp>

#include "camera.h"
#include "frame.h"
#include "types.h"
#include "mappoint.h"
#include "read_configs.h"
#include "ros2_publisher.h"

class Mapping {
public:
  Mapping(OptimizationConfig &backend_optimization_config, CameraPtr camera,
      Ros2PublisherPtr ros_publisher, const std::string &seq_name = "");
  Eigen::MatrixXd InsertKeyframe(FramePtr frame);
  void InsertMappoint(MappointPtr mappoint);

  FramePtr GetFramePtr(int frame_id);
  MappointPtr GetMappointPtr(int mappoint_id);

  bool TriangulateMappoint(MappointPtr mappoint);
  bool UpdateMappointDescriptor(MappointPtr mappoint);
  void SearchNeighborFrames(FramePtr frame,
                            std::vector<FramePtr> &neighbor_frames);
  void AddFrameVertex(FramePtr frame, MapOfPoses &poses, bool fix_this_frame);
  void LocalMapOptimization(FramePtr new_frame);
  std::pair<FramePtr, FramePtr> MakeFramePair(FramePtr frame0, FramePtr frame1);
  void
  RemoveOutliers(const std::vector<std::pair<FramePtr, MappointPtr>> &outliers);
  void UpdateFrameConnection(FramePtr frame);
  void PrintConnection();
  void SearchByProjection(
      FramePtr frame, std::vector<MappointPtr> &mappoints, int thr,
      std::vector<std::pair<int, MappointPtr>> &good_projections);
  void SaveKeyframeTrajectory(std::string save_root);
  void KeyFrameCulling();
  void reset() {
    std::cout << "Map reset" << std::endl;
    for (auto &mappoint : _mappoints)
      mappoint.second.reset();
    std::cout << "Map reset 1" << std::endl;
    for (auto &keyframe : _keyframes)
      keyframe.second.reset();
    std::cout << "Map reset 2" << std::endl;

    _mappoints.clear();
    _keyframes.clear();
    _keyframe_ids.clear();
    _camera.reset();
    _ros_publisher->reset();
    // _ros_publisher.reset();
  }

private:
  OptimizationConfig _backend_optimization_config;
  std::string _sequence_name;
  CameraPtr _camera;
  std::map<int, MappointPtr> _mappoints;
  std::map<int, FramePtr> _keyframes;
  std::vector<int> _keyframe_ids;
  Ros2PublisherPtr _ros_publisher;
};

typedef std::shared_ptr<Mapping> MapPtr;

#endif // MAPPING_H_