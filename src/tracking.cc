#include "tracking.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <assert.h>
#include <iostream>
#include <opencv2/core/eigen.hpp>

#include "camera.h"
#include "dataset.h"
#include "debug.h"
#include "frame.h"
#include "g2o_optimization.h"
#include "mapping.h"
#include "point_matching.h"
#include "read_configs.h"
#include "super_glue.h"
#include "super_point.h"
#include "timer.h"
#include <algorithm>
#include <stdexcept>

Tracking::Tracking(Configs &configs)
    : _shutdown(false), _init(false), _track_id(0), _to_update_local_map(false),
      _configs(configs) {
  if (configs.sensor_setup == SensorSetup::RGBD ||
      configs.sensor_setup == SensorSetup::Mono) {
    _camera_type = CameraType::MONO;
  } else if (configs.sensor_setup == SensorSetup::Stereo) {
    _camera_type = CameraType::STEREO;
  } else {
    std::cout << "Error in Sensor Setup" << std::endl;
    exit(0);
  }

  _camera = std::shared_ptr<Camera>(
      new Camera(configs.camera_config_path, _camera_type));
  _superpoint =
      std::shared_ptr<SuperPoint>(new SuperPoint(configs.superpoint_config));
  if (!_superpoint->build()) {
    std::cout << "Error in SuperPoint building" << std::endl;
    exit(0);
  }

  _point_matching = std::shared_ptr<PointMatching>(
      new PointMatching(configs.superglue_config));
  _ros_publisher = std::shared_ptr<Ros2Publisher>(
      new Ros2Publisher(configs.ros_publisher_config));
  _map = std::shared_ptr<Mapping>(new Mapping(_configs.backend_optimization_config,
                                      _camera, _ros_publisher,
                                      configs.sequence_name));
  if (_camera_type == CameraType::MONO) {
    _2_view_reconstruct = std::shared_ptr<EpipolarGeometry>(
        new EpipolarGeometry(_camera->K(), 1.0f, 200));
  }

  _feature_thread =
      std::thread(boost::bind(&Tracking::ExtractFeatureThread, this));
  _tracking_thread = std::thread(boost::bind(&Tracking::TrackingThread, this));
  _use_mask = configs.use_mask;
  _sensor_setup = configs.sensor_setup;
}

void Tracking::AddInput(InputDataPtr data) {

  cv::Mat image_undistorted;
  cv::Mat depth_undistorted;

  if (_sensor_setup == SensorSetup::RGBD or
      _sensor_setup == SensorSetup::Mono) {
    _camera->UndistortImage(data->image, image_undistorted);
    data->image = image_undistorted;

    if (_sensor_setup == SensorSetup::RGBD) {
      std::cout << "undistort depth image!!" << std::endl;
      _camera->UndistortImage(data->depth, depth_undistorted);
      data->depth = depth_undistorted;
    }
  } else {
    if (_sensor_setup == SensorSetup::Stereo) {
      cv::Mat image_left_rect, image_right_rect;
      _camera->UndistortImage(data->image, data->image_right, image_left_rect,
                              image_right_rect);
      data->image = image_left_rect;
      data->image_right = image_right_rect;
    }
  }

  cv::Mat mask_undistorted;
  if (_use_mask) {
    std::cout << "undistort mask image!!" << std::endl;
    _camera->UndistortImage(data->mask, mask_undistorted);
    data->mask = mask_undistorted;
  }

  while (_data_buffer.size() >= 3 && !_shutdown) {
    usleep(2000);
  }
  // std::cout << "Input added -> " << data->id_ts << std::endl;
  _buffer_mutex.lock();
  _data_buffer.push(data);
  _buffer_mutex.unlock();
}

void Tracking::reset() {

  _last_frame.reset();
  _last_keyframe.reset();
  _ref_keyframe.reset();
  _init_frame.reset();
  _camera.reset();
  _gpu_mutex.lock();
  _superpoint.reset();
  _point_matching.reset();

  _gpu_mutex.unlock();
  _ros_publisher.reset();
  _2_view_reconstruct.reset();
  _map->reset();
  // _map.reset();
}

void Tracking::ExtractFeatureThread() {
  while (!_shutdown) {
    if (_data_buffer.empty()) {
      usleep(2000);
      continue;
    }

    InputDataPtr input_data;
    _buffer_mutex.lock();

    input_data = _data_buffer.front();
    _data_buffer.pop();
    _buffer_mutex.unlock();

    int frame_id = input_data->index;
    double timestamp = input_data->time;
    cv::Mat image_undistort = input_data->image.clone();
    cv::Mat mask, depth, image_right_undistorted;
    if (_use_mask) {
      mask = input_data->mask.clone();
    }
    if (_sensor_setup == SensorSetup::RGBD) {
      depth = input_data->depth.clone();
    }
    if (_sensor_setup == SensorSetup::Stereo) {
      image_right_undistorted = input_data->image_right.clone();
    }

    // construct frame
    FramePtr frame = std::shared_ptr<Frame>(
        new Frame(frame_id, false, _camera, timestamp, input_data->id_ts));

    if (_sensor_setup == SensorSetup::RGBD or
        _sensor_setup == SensorSetup::Mono) {

      // Monocular initialization
      if (!_init) {
        _init = Init(frame, image_undistort, mask, depth);
        _last_frame_track_well = _init;
        // std::cout << "Init !!" << std::endl;

        if (_init) {
          _last_frame = frame;
          _last_keyframe = frame;
          _last_image = image_undistort;
          _last_keyimage = image_undistort;
          PublishFrame(frame, image_undistort);
        }
        continue;
      }
      // std::cout << "after Init !!" << std::endl;
    } else {
      if (_sensor_setup == SensorSetup::Stereo) {
        if (!_init) {
          _init =
              InitStereo(frame, image_undistort, mask, image_right_undistorted);
          _last_frame_track_well = _init;
          if (_init) {
            _last_frame = frame;
            _last_image = image_undistort;
            _last_right_image = image_right_undistorted;
            _last_keyimage = image_undistort;
          }
          PublishFrame(frame, image_undistort);
          continue;
        }
      }
    }
    // extract features and track last keyframe
    FramePtr last_keyframe = _last_keyframe;
    const Eigen::Matrix<double, 259, Eigen::Dynamic> features_last_keyframe =
        last_keyframe->GetAllFeatures();

    std::vector<cv::DMatch> matches;
    Eigen::Matrix<double, 259, Eigen::Dynamic> features;

    ExtractFeatureAndMatch(image_undistort, mask, features_last_keyframe,
                           features, matches);
    frame->AddFeatures(features);

    TrackingDataPtr tracking_data =
        std::shared_ptr<TrackingData>(new TrackingData());
    tracking_data->frame = frame;
    tracking_data->ref_keyframe = last_keyframe;
    tracking_data->matches = matches;
    tracking_data->input_data = input_data;

    while (_tracking_data_buffer.size() >= 2) {
      usleep(2000);
    }

    _tracking_mutex.lock();
    _tracking_data_buffer.push(tracking_data);
    _tracking_mutex.unlock();
  }
}

void Tracking::TrackingThread() {
  while (!_shutdown) {
    if (_tracking_data_buffer.empty()) {
      usleep(2000);
      continue;
    }

    TrackingDataPtr tracking_data;
    _tracking_mutex.lock();
    tracking_data = _tracking_data_buffer.front();
    _tracking_data_buffer.pop();
    _tracking_mutex.unlock();

    FramePtr frame = tracking_data->frame;
    FramePtr ref_keyframe = tracking_data->ref_keyframe;
    InputDataPtr input_data = tracking_data->input_data;
    std::vector<cv::DMatch> matches = tracking_data->matches;

    double timestamp = input_data->time;
    cv::Mat image_undistort = input_data->image.clone();
    frame->debug_img = image_undistort.clone();

    cv::Mat mask, depth, image_right;
    if (_use_mask)
      mask = input_data->mask.clone();

    if (_sensor_setup == SensorSetup::RGBD)
      depth = input_data->depth.clone();

    if (_sensor_setup == SensorSetup::Stereo)
      image_right = input_data->image_right.clone();
    // track

    frame->SetPose(_last_frame->GetPose());
    std::function<int()> track_last_frame = [&]() {
      if (_num_since_last_keyframe < 1 || !_last_frame_track_well)
        return -1;
      if (_sensor_setup == SensorSetup::Stereo)
        InsertKeyframe(_last_frame, mask, _last_right_image);
      else
        InsertKeyframe(_last_frame);

      _last_keyimage = _last_image;
      matches.clear();
      ref_keyframe = _last_frame;
      return TrackFrame(_last_frame, frame, matches);
    };

    int num_match = matches.size();
    if (num_match < _configs.keyframe_config.min_num_match) {
      num_match = track_last_frame();
    } else {
      num_match = TrackFrame(ref_keyframe, frame, matches);
      if (num_match < _configs.keyframe_config.min_num_match) {
        num_match = track_last_frame();
      }
    }

    // std::ofstream out(_configs.sequence_name, std::ios::out | std::ios::app);
    Eigen::Matrix4d &pose = frame->GetPose();
    Eigen::Vector3d t = pose.block<3, 1>(0, 3);
    Eigen::Quaterniond q(pose.block<3, 3>(0, 0));

    _last_frame_track_well =
        (num_match >= _configs.keyframe_config.min_num_match);

    // out << std::setprecision(9) << input_data->id_ts <<"
    // "<<frame->GetTimestamp() << " "
    // << std::setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " "
    // << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " " <<
    // _last_frame_track_well <<std::endl;

    PublishFrame(frame, image_undistort);

    if (!_last_frame_track_well) {
      // frame->SetPose(_last_frame->GetPose());
      continue;
    };

    frame->SetPreviousFrame(ref_keyframe);
    _last_frame_track_well = true;

    if (AddKeyframe(ref_keyframe, frame, num_match) &&
        ref_keyframe->GetFrameId() == _last_keyframe->GetFrameId()) {
      if (_sensor_setup == SensorSetup::RGBD)
        frame->SetDepthImg(depth);
      if (_sensor_setup == SensorSetup::Stereo)
        InsertKeyframe(frame, mask, image_right);
      else
        InsertKeyframe(frame);
      _last_keyimage = image_undistort;
      _ref_keyframe = frame;
    }

    _last_frame = frame;
    _last_image = image_undistort;
    _last_right_image = image_right;
    // KeyframeCulling();
  }
}

void Tracking::ExtractFeatrue(
    const cv::Mat &image, const cv::Mat &mask,
    Eigen::Matrix<double, 259, Eigen::Dynamic> &points) {
  std::function<void()> extract_point = [&]() {
    _gpu_mutex.lock();
    bool good_infer = _superpoint->infer(image, mask, points);
    _gpu_mutex.unlock();
    if (!good_infer) {
      std::cout << "Failed when extracting point features !" << std::endl;
      return;
    }
  };

  std::thread point_ectraction_thread(extract_point);
  point_ectraction_thread.join();
}

void Tracking::ExtractFeatureAndMatch(
    const cv::Mat &image, const cv::Mat &mask,
    const Eigen::Matrix<double, 259, Eigen::Dynamic> &points0,
    Eigen::Matrix<double, 259, Eigen::Dynamic> &points1,
    std::vector<cv::DMatch> &matches) {
  std::function<void()> extract_point_and_match = [&]() {
    auto point0 = std::chrono::steady_clock::now();
    _gpu_mutex.lock();
    if (!_superpoint->infer(image, mask, points1)) {
      _gpu_mutex.unlock();
      std::cout << "Failed when extracting point features !" << std::endl;
      return;
    }
    auto point1 = std::chrono::steady_clock::now();

    matches.clear();
    _point_matching->MatchingPoints(points0, points1, matches, true);
    _gpu_mutex.unlock();
    auto point2 = std::chrono::steady_clock::now();
    auto point_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(point1 - point0)
            .count();
    auto point_match_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(point2 - point1)
            .count();
    // std::cout << "One Frame point Time: " << point_time << " ms." <<
    // std::endl; std::cout << "One Frame point match Time: " <<
    // point_match_time << " ms." << std::endl;
  };

  auto feature1 = std::chrono::steady_clock::now();
  std::thread point_ectraction_thread(extract_point_and_match);

  point_ectraction_thread.join();

  auto feature2 = std::chrono::steady_clock::now();
  auto feature_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(feature2 - feature1)
          .count();
}

bool Tracking::Init(FramePtr frame, const cv::Mat &image, const cv::Mat &mask,
                    const cv::Mat &depth) {

  auto computeMedianDepth = [](const std::vector<cv::Point3f> &v3DPoints,
                               const Eigen::Matrix3d &Rcw,
                               const Eigen::Vector3d &tcw) {
    std::vector<double> vDepths;
    int N = v3DPoints.size();
    if (N == 0)
      return (double)1.0f;
    vDepths.reserve(N);
    Eigen::Matrix<double, 1, 3> Rcw2 = Rcw.row(2);
    double zcw = tcw(2);

    for (int i = 0; i < N; i++) {
      Eigen::Vector3d pnt(v3DPoints[i].x, v3DPoints[i].y, v3DPoints[i].z);
      float z = Rcw2.dot(pnt) + zcw;
      vDepths.push_back(z);
    }
    std::sort(vDepths.begin(), vDepths.end());
    return vDepths[(vDepths.size() - 1) / 2];
  };
  // helpful lambda functions
  auto reset = [](FramePtr f) {
    f->remove();
    f.reset();
  };

  auto setIdentity = [](FramePtr f) {
    f->SetPose(Eigen::Matrix4d::Identity());
    f->SetPoseFixed(true);
  };

  if (!depth.empty()) {
    Eigen::Matrix<double, 259, Eigen::Dynamic> features;

    ExtractFeatrue(image, mask, features);
    int feature_num = features.cols();
    if (feature_num < 250) {
      return false;
    }
    frame->AddFeatures(features);

    if (!depth.empty())
      _init_depth = depth.clone();

    setIdentity(frame);
    frame->debug_img = image.clone();

    cv::Mat depth_debug = _init_depth.clone();

    auto init_pose = frame->GetPose();
    Eigen::Matrix3d Rwc = init_pose.block<3, 3>(0, 0);
    Eigen::Vector3d twc = init_pose.block<3, 1>(0, 3);
    // construct mappoints
    std::vector<int> track_ids(feature_num, -1);
    int frame_id = frame->GetFrameId();
    Eigen::Vector3d tmp_position;
    std::vector<MappointPtr> new_mappoints;
    int good_point_num = 0;
    for (size_t i = 0; i < feature_num; i++) {
      if (frame->BackProjectPointDepth(i, tmp_position)) {
        Eigen::Vector2d p2D;
        frame->GetKeypointPosition(i, p2D);
        int col = p2D.x();
        int row = p2D.y();
        if (depth.at<u_int8_t>(row, col) < 50 ||
            depth.at<u_int8_t>(row, col) > 200)
          continue;
        float d = 100.0 / (float(depth.at<u_int8_t>(row, col)) + 0.00001);
        tmp_position = tmp_position * d;
        tmp_position = Rwc * tmp_position + twc;

        track_ids[i] = _track_id++;
        Eigen::Matrix<double, 256, 1> descriptor;
        if (!frame->GetDescriptor(i, descriptor))
          continue;
        MappointPtr mappoint = std::shared_ptr<Mappoint>(
            new Mappoint(track_ids[i], tmp_position, descriptor));
        mappoint->AddObverser(frame_id, i);
        frame->InsertMappoint(i, mappoint);
        new_mappoints.push_back(mappoint);
        good_point_num++;
      }
    }
    frame->SetTrackIds(track_ids);

    if (good_point_num < 100)
      return false;

    // add frame and mappoints to map
    InsertKeyframe(frame);
    for (MappointPtr mappoint : new_mappoints) {
      _map->InsertMappoint(mappoint);
    }
    _ref_keyframe = frame;
    _last_frame = frame;
    return true;
  }

  // Set Initial Frame
  if (_init_frame == nullptr) {
    // extract features
    Eigen::Matrix<double, 259, Eigen::Dynamic> features;

    ExtractFeatrue(image, mask, features);
    int feature_num = features.cols();
    if (feature_num < 200) {
      return false;
    }
    frame->AddFeatures(features);
    _init_frame = frame;
    if (!depth.empty())
      _init_depth = depth.clone();
    setIdentity(_init_frame);
    _init_frame->debug_img = image.clone();
    return false;
  }

  // Check that current frame and initial frame are close enough!

  if (frame->GetTimestamp() - _init_frame->GetTimestamp() > 3.0) {
    reset(_init_frame);
    Eigen::Matrix<double, 259, Eigen::Dynamic> features;
    ExtractFeatrue(image, mask, features);
    int feature_num = features.cols();
    if (feature_num < 300) {
      return false;
    }
    frame->AddFeatures(features);
    _init_frame = frame;
    setIdentity(_init_frame);
    _init_frame->debug_img = image.clone();
    return false;
  }

  // This point means that we have init frame and we are gonna try to intialize
  //   extract features and track init keyframe
  const Eigen::Matrix<double, 259, Eigen::Dynamic> features_init_frame =
      _init_frame->GetAllFeatures();
  std::vector<cv::DMatch> matches;
  Eigen::Matrix<double, 259, Eigen::Dynamic> features;
  ExtractFeatureAndMatch(image, mask, features_init_frame, features, matches);
  frame->AddFeatures(features);
  frame->debug_img = image.clone();

  //    /** Visualization & debugging */
  //     cv::Mat debug;
  //     cv::hconcat(_init_frame->debug_img, frame->debug_img, debug);
  //     cv::cvtColor(debug, debug, cv::COLOR_GRAY2BGR);
  //     for(int i=0;i<matches.size();i++){
  //         int idx0 = matches[i].queryIdx;
  //         int idx1 = matches[i].trainIdx;
  //         auto kp1 = _init_frame->GetKeypoint(idx0).pt;
  //         auto kp2 = frame->GetKeypoint(idx1).pt;
  //         kp2.x = kp2.x + debug.cols/2.0;
  //         auto color = GenerateColor(idx0);
  //         cv::line(debug, kp1, kp2, color, 2);
  //         cv::circle(debug, kp1, 3, color, -1);
  //         cv::circle(debug, kp2 , 3, color, -1);
  //     }
  //     cv::imshow("debug_init", debug);
  //     cv::waitKey(2);

  // Epipolar Geometry
  std::vector<cv::KeyPoint> vKeys1, vKeys2;
  std::vector<int> vMatches12;
  Eigen::Matrix4f T21 = Eigen::Matrix4f::Identity();
  std::vector<cv::Point3f> vP3D;
  std::vector<bool> v3DInliers;
  for (int i = 0; i < matches.size(); i++) {
    int idx0 = matches[i].queryIdx;
    int idx1 = matches[i].trainIdx;
    vMatches12.push_back(i);
    vKeys1.push_back(_init_frame->GetKeypoint(idx0));
    vKeys2.push_back(frame->GetKeypoint(idx1));
  }

  // std::cout << vKeys1.size() << ' ' << vKeys2.size() << std::endl;
  // std::cout << vMatches12.size() << std::endl;
  bool success = _2_view_reconstruct->reconstruct(vKeys1, vKeys2, vMatches12,
                                                  T21, vP3D, v3DInliers);
  size_t triangulated_point_num = 0;
  for (auto x : v3DInliers)
    if (x)
      triangulated_point_num++;
  if (!success || triangulated_point_num < 150) {
    std::cout << "Still trying to initialize! "
                 "Please move a little bit to provide enough parallax!"
              << std::endl;
    return false;
  }

  _init_frame->SetFrameId(0);
  frame->SetFrameId(1);

  Eigen::Matrix4d T12 = T21.cast<double>().inverse();
  Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d Tcw = Twc.inverse();
  double scale = 4.0f / computeMedianDepth(vP3D, Tcw.block<3, 3>(0, 0),
                                           Tcw.block<3, 1>(0, 3));

  Eigen::Matrix3d Rwc = Twc.block<3, 3>(0, 0);
  Eigen::Vector3d twc = Twc.block<3, 1>(0, 3) * scale;
  Twc.block<3, 1>(0, 3) = twc;

  Eigen::Matrix4d Twc2 = Twc * T12;
  Eigen::Matrix3d Rwc2 = Twc2.block<3, 3>(0, 0);
  Eigen::Vector3d twc2 = Twc2.block<3, 1>(0, 3) * scale;
  Twc2.block<3, 1>(0, 3) = twc2;
  frame->SetPose(Twc2);
  int N = matches.size();
  // construct mappoints
  std::vector<int> track_ids(N, -1);

  int frame_id = frame->GetFrameId();
  int init_frame_id = _init_frame->GetFrameId();
  Eigen::Vector3d tmp_position;
  std::vector<MappointPtr> new_mappoints;
  for (size_t i = 0; i < N; i++) {
    if (v3DInliers.at(i)) { // if inlier
      tmp_position = Eigen::Vector3d(vP3D[i].x, vP3D[i].y, vP3D[i].z) * scale;
      tmp_position = Rwc * tmp_position + twc;
      track_ids[i] = _track_id++;
      Eigen::Matrix<double, 256, 1> descriptor;
      int idx0 = matches[i].queryIdx;
      int idx1 = matches[i].trainIdx;
      if (!_init_frame->GetDescriptor(idx0, descriptor))
        continue;
      MappointPtr mappoint = std::shared_ptr<Mappoint>(
          new Mappoint(track_ids[i], tmp_position, descriptor));
      mappoint->AddObverser(init_frame_id, idx0);
      mappoint->AddObverser(frame_id, idx1);
      frame->InsertMappoint(idx1, mappoint);
      _init_frame->InsertMappoint(idx0, mappoint);
      new_mappoints.push_back(mappoint);
    }
  }

  frame->SetTrackIds(track_ids);
  _init_frame->SetTrackIds(track_ids);

  Eigen::Matrix<double, 259, Eigen::Dynamic> &features0 =
      _init_frame->GetAllFeatures();
  Eigen::Matrix<double, 259, Eigen::Dynamic> &features1 =
      frame->GetAllFeatures();

  // add frame and mappoints to map
  for (MappointPtr mappoint : new_mappoints) {
    _map->InsertMappoint(mappoint);
  }

  InsertKeyframe(_init_frame);
  InsertKeyframe(frame);

  _ref_keyframe = frame;
  _last_frame = frame;
  _last_keyframe = frame;
  std::cout << "# Initialization is done!!\n"
               "Keyframes added to map are: { "
            << _init_frame->GetFrameId() << ", " << frame->GetFrameId()
            << "}\n"
               "with: "
            << new_mappoints.size() << " triangulated map points!\n"
            << std::endl;
  std::cout << "# Initial relative pose: \n" << std::endl;
  std::cout << Twc2 << std::endl;

  return true;
}

bool Tracking::InitStereo(FramePtr frame, const cv::Mat &image_left,
                          const cv::Mat &mask, const cv::Mat &image_right) {
  // extract features
  Eigen::Matrix<double, 259, Eigen::Dynamic> features_left, features_right;

  std::vector<cv::DMatch> stereo_matches;
  ExtractFeatrue(image_left, mask, features_left);
  int feature_num = features_left.cols();
  if (feature_num < 150)
    return false;
  ExtractFeatureAndMatch(image_right, mask, features_left, features_right,
                         stereo_matches);
  frame->AddLeftFeatures(features_left);
  int stereo_point_num =
      frame->AddRightFeatures(features_right, stereo_matches);
  if (stereo_point_num < 100)
    return false;

  Eigen::Matrix4d init_pose = Eigen::Matrix4d::Identity();
  // Eigen::Matrix4d init_pose;
  // init_pose << 1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1, 0, 0, 0, 1;
  frame->SetPose(init_pose);
  frame->SetPoseFixed(true);

  Eigen::Matrix3d Rwc = init_pose.block<3, 3>(0, 0);
  Eigen::Vector3d twc = init_pose.block<3, 1>(0, 3);
  // construct mappoints
  std::vector<int> track_ids(feature_num, -1);
  int frame_id = frame->GetFrameId();
  Eigen::Vector3d tmp_position;
  std::vector<MappointPtr> new_mappoints;
  for (size_t i = 0; i < feature_num; i++) {
    if (frame->BackProjectPointStereo(i, tmp_position)) {
      tmp_position = Rwc * tmp_position + twc;
      stereo_point_num++;
      track_ids[i] = _track_id++;
      Eigen::Matrix<double, 256, 1> descriptor;
      if (!frame->GetDescriptor(i, descriptor))
        continue;
      MappointPtr mappoint = std::shared_ptr<Mappoint>(
          new Mappoint(track_ids[i], tmp_position, descriptor));
      mappoint->AddObverser(frame_id, i);
      frame->InsertMappoint(i, mappoint);
      new_mappoints.push_back(mappoint);
    }
  }
  frame->SetTrackIds(track_ids);
  if (stereo_point_num < 100)
    return false;

  // add frame and mappoints to map
  InsertKeyframe(frame);
  for (MappointPtr mappoint : new_mappoints) {
    _map->InsertMappoint(mappoint);
  }

  _ref_keyframe = frame;
  _last_frame = frame;
  return true;
}

int Tracking::TrackFrame(FramePtr frame0, FramePtr frame1,
                         std::vector<cv::DMatch> &matches) {

  Eigen::Matrix<double, 259, Eigen::Dynamic> &features0 =
      frame0->GetAllFeatures();
  Eigen::Matrix<double, 259, Eigen::Dynamic> &features1 =
      frame1->GetAllFeatures();

  std::vector<int> inliers(frame1->FeatureNum(), -1);
  std::vector<MappointPtr> matched_mappoints(features1.cols(), nullptr);
  std::vector<MappointPtr> &frame0_mappoints = frame0->GetAllMappoints();

  for (auto &match : matches) {
    int idx0 = match.queryIdx;
    int idx1 = match.trainIdx;
    matched_mappoints[idx1] = frame0_mappoints[idx0];
    inliers[idx1] = frame0->GetTrackId(idx0);
  }

  int num_inliers = FramePoseOptimization(frame1, matched_mappoints, inliers);

  cv::Mat debug_img = frame1->debug_img.clone();
  cv::cvtColor(debug_img, debug_img, cv::COLOR_GRAY2BGR);
  for (int i = 0; i < matched_mappoints.size(); i++) {
    if (inliers[i] < 0)
      continue;
    auto mpt = matched_mappoints[i];
    if (!mpt || !mpt->IsValid())
      continue;

    auto kp = frame1->GetKeypoint(i);
    cv::circle(debug_img, kp.pt, 8, cv::Scalar(0, 255, 0), 2);

    Eigen::Matrix4d pose = frame1->GetPose();
    Eigen::Matrix3d Rwc = pose.block<3, 3>(0, 0);
    Eigen::Vector3d twc = pose.block<3, 1>(0, 3);

    const Eigen::Vector3d &pw = mpt->GetPosition();
    Eigen::Vector3d pc = Rwc.transpose() * (pw - twc);
    if (pc(2) <= 0)
      continue;
    // check whether mappoint can project on the image
    Eigen::Vector2d p2D;
    frame1->GetCamera()->Project(p2D, pc);
    cv::Point cvPt(int(p2D.x()), int(p2D.y()));
    cv::circle(debug_img, cvPt, 4, cv::Scalar(0, 0, 255), -1);
    cv::line(debug_img, cvPt, kp.pt, cv::Scalar(0, 0, 255), 2);
  }
  // cv::imshow("debug", debug_img);
  // cv::waitKey(30);
  DebugMessagePtr debug_message =
      std::shared_ptr<DebugMessage>(new DebugMessage);

  debug_message->image = debug_img;
  debug_message->time = frame1->GetTimestamp();

  _ros_publisher->PublishDebug(debug_message);

  // update track id
  int RM = 0;
  if (num_inliers > _configs.keyframe_config.min_num_match) {
    for (std::vector<cv::DMatch>::iterator it = matches.begin();
         it != matches.end();) {
      int idx0 = (*it).queryIdx;
      int idx1 = (*it).trainIdx;
      if (inliers[idx1] > 0) {
        frame1->SetTrackId(idx1, frame0->GetTrackId(idx0));
        frame1->InsertMappoint(idx1, frame0_mappoints[idx0]);
      }

      if (inliers[idx1] > 0) {
        it++;
      } else {
        it = matches.erase(it);
        RM++;
      }
    }
  }

  return num_inliers;
}

int Tracking::FramePoseOptimization(FramePtr frame,
                                    std::vector<MappointPtr> &mappoints,
                                    std::vector<int> &inliers, int pose_init) {
  // solve PnP using opencv to get initial pose
  Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
  std::vector<int> cv_inliers;
  int num_cv_inliers = SolvePnPWithCV(frame, mappoints, Twc, cv_inliers);
  Eigen::Vector3d check_dp =
      Twc.block<3, 1>(0, 3) - _last_frame->GetPose().block<3, 1>(0, 3);
  if (_sensor_setup == SensorSetup::Mono) {
    if (/*check_dp.norm() > 0.5 || */ num_cv_inliers <
        _configs.keyframe_config.min_num_match) {
      Twc = _last_frame->GetPose();
      // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      // InsertKeyframe(_last_frame);
      // TODO: check here in which cases this happens
      // std::cout<<"inliers: "<<num_cv_inliers<<"\n";
      // _init=false;
      // throw std::underflow_error("No points for tracking! Ending!!!");
    }
  } else {
    if (check_dp.norm() > 0.5 ||
        num_cv_inliers < _configs.keyframe_config.min_num_match) {
      Twc = _last_frame->GetPose();
    }
  }

  // Second, optimization
  MapOfPoses poses;
  MapOfPoints3d points;
  std::vector<CameraPtr> camera_list;
  VectorOfMonoPointConstraints mono_point_constraints;
  VectorOfStereoPointConstraints stereo_point_constraints;

  camera_list.emplace_back(_camera);

  // map of poses
  Pose3d pose;
  pose.p = Twc.block<3, 1>(0, 3);
  pose.q = Twc.block<3, 3>(0, 0);
  int frame_id = frame->GetFrameId();
  poses.insert(std::pair<int, Pose3d>(frame_id, pose));

  // visual constraint construction
  std::vector<size_t> mono_indexes;
  std::vector<size_t> stereo_indexes;
  for (size_t i = 0; i < mappoints.size(); i++) {
    // points
    MappointPtr mpt = mappoints[i];
    if (mpt == nullptr || !mpt->IsValid())
      continue;
    Eigen::Vector2d keypoint;
    Eigen::Vector3d keyPointwithR = Eigen::Vector3d::Zero();
    if (SensorSetup::Stereo == _sensor_setup) {
      frame->GetKeypointPosition(i, keyPointwithR);
      keypoint = keyPointwithR.head(2);
    } else if (!frame->GetKeypointPosition(i, keypoint))
      continue;

    int mpt_id = mpt->GetId();
    Position3d point;
    point.p = mpt->GetPosition();
    point.fixed = true;
    points.insert(std::pair<int, Position3d>(mpt_id, point));

    // visual constraint
    if (keyPointwithR(2) > 0) {
      StereoPointConstraintPtr stereo_constraint =
          std::shared_ptr<StereoPointConstraint>(new StereoPointConstraint());
      stereo_constraint->id_pose = frame_id;
      stereo_constraint->id_point = mpt_id;
      stereo_constraint->id_camera = 0;
      stereo_constraint->inlier = true;
      stereo_constraint->keypoint = keyPointwithR;
      stereo_constraint->pixel_sigma = 0.8;
      stereo_point_constraints.push_back(stereo_constraint);
      stereo_indexes.push_back(i);
    } else {
      MonoPointConstraintPtr mono_constraint =
          std::shared_ptr<MonoPointConstraint>(new MonoPointConstraint());
      mono_constraint->id_pose = frame_id;
      mono_constraint->id_point = mpt_id;
      mono_constraint->id_camera = 0;
      mono_constraint->inlier = true;
      mono_constraint->keypoint = keypoint.head(2);
      mono_constraint->pixel_sigma = 0.8;
      mono_point_constraints.push_back(mono_constraint);
      mono_indexes.push_back(i);
    }
  }
  int num_inliers = FrameOptimization(
      poses, points, camera_list, mono_point_constraints,
      stereo_point_constraints, _configs.tracking_optimization_config);

  if (num_inliers > _configs.keyframe_config.min_num_match) {
    // set frame pose
    Eigen::Matrix4d frame_pose = Eigen::Matrix4d::Identity();
    frame_pose.block<3, 3>(0, 0) = poses.begin()->second.q.matrix();
    frame_pose.block<3, 1>(0, 3) = poses.begin()->second.p;
    frame->SetPose(frame_pose);

    // update tracked mappoints
    for (size_t i = 0; i < mono_point_constraints.size(); i++) {
      size_t idx = mono_indexes[i];
      if (!mono_point_constraints[i]->inlier) {
        inliers[idx] = -1;
      }
    }

    for (size_t i = 0; i < stereo_point_constraints.size(); i++) {
      size_t idx = stereo_indexes[i];
      if (!stereo_point_constraints[i]->inlier) {
        inliers[idx] = -1;
      }
    }
  }

  return num_inliers;
}

bool Tracking::AddKeyframe(FramePtr last_keyframe, FramePtr current_frame,
                           int num_match) {
  Eigen::Matrix4d frame_pose = current_frame->GetPose();

  Eigen::Matrix4d &last_keyframe_pose = _last_keyframe->GetPose();
  Eigen::Matrix3d last_R = last_keyframe_pose.block<3, 3>(0, 0);
  Eigen::Vector3d last_t = last_keyframe_pose.block<3, 1>(0, 3);
  Eigen::Matrix3d current_R = frame_pose.block<3, 3>(0, 0);
  Eigen::Vector3d current_t = frame_pose.block<3, 1>(0, 3);

  Eigen::Matrix3d delta_R = last_R.transpose() * current_R;
  Eigen::AngleAxisd angle_axis(delta_R);
  double delta_angle = angle_axis.angle();
  double delta_distance = (current_t - last_t).norm();
  int passed_frame_num =
      current_frame->GetFrameId() - _last_keyframe->GetFrameId();

  bool not_enough_match = (num_match < _configs.keyframe_config.max_num_match);
  bool large_delta_angle = (delta_angle > _configs.keyframe_config.max_angle);
  /**
   * Large Distance is not correct measure as monocular odometry doesn't observe
   * scale
   */
  bool large_distance =
      (delta_distance > _configs.keyframe_config.max_distance);
  double delte_time =
      current_frame->GetTimestamp() - last_keyframe->GetTimestamp();
  bool long_time_interval = delte_time > 0 /* 0.5f */;
  bool enough_passed_frame =
      (passed_frame_num >= _configs.keyframe_config.max_num_passed_frame);

  bool keyframe = not_enough_match || large_delta_angle || large_distance ||
                  enough_passed_frame;
  // if (keyframe) {
  //   std::cout << "Keyframe: " << num_match << " " << delta_angle << " "
  //             << delta_distance << " " << delte_time << " " << passed_frame_num
  //             << std::endl;
  // } else {
  //   std::cout << "Not Keyframe: " << num_match << " " << delta_angle << " "
  //             << delta_distance << " " << delte_time << " " << passed_frame_num
  //             << std::endl;
  // }

  return (keyframe);
  // return true;
}

void Tracking::KeyframeCulling() { _map->KeyFrameCulling(); }

void Tracking::InsertKeyframe(FramePtr frame, const cv::Mat &mask,
                              const cv::Mat &image_right) {
  _last_keyframe = frame;

  Eigen::Matrix<double, 259, Eigen::Dynamic> features_right;

  std::vector<cv::DMatch> stereo_matches;
  /** Here the mask is not correct, we need to mask the right image as well!*/
  ExtractFeatureAndMatch(image_right, mask, frame->GetAllFeatures(),
                         features_right, stereo_matches);
  frame->AddRightFeatures(features_right, stereo_matches);
  InsertKeyframe(frame);
}

void Tracking::InsertKeyframe(FramePtr frame) {
  _last_keyframe = frame;

  // create new track id
  std::vector<int> &track_ids = frame->GetAllTrackIds();
  for (size_t i = 0; i < track_ids.size(); i++) {
    if (track_ids[i] < 0) {
      frame->SetTrackId(i, _track_id++);
    }
  }

  // insert keyframe to map
  auto pose = _map->InsertKeyframe(frame);
  _pose_buffer.push(pose);

  // update last keyframe
  _num_since_last_keyframe = 1;
  _ref_keyframe = frame;
  _to_update_local_map = true;
}

void Tracking::UpdateReferenceFrame(FramePtr frame) {
  int current_frame_id = frame->GetFrameId();
  std::vector<MappointPtr> &mappoints = frame->GetAllMappoints();
  std::map<FramePtr, int> keyframes;
  for (MappointPtr mpt : mappoints) {
    if (!mpt || mpt->IsBad())
      continue;
    const std::map<int, int> obversers = mpt->GetAllObversers();
    for (auto &kv : obversers) {
      int observer_id = kv.first;
      if (observer_id == current_frame_id)
        continue;
      FramePtr keyframe = _map->GetFramePtr(observer_id);
      if (!keyframe)
        continue;
      keyframes[keyframe]++;
    }
  }
  if (keyframes.empty())
    return;

  std::pair<FramePtr, int> max_covi = std::pair<FramePtr, int>(nullptr, -1);
  for (auto &kv : keyframes) {
    if (kv.second > max_covi.second) {
      max_covi = kv;
    }
  }

  if (max_covi.first->GetFrameId() != _ref_keyframe->GetFrameId()) {
    _ref_keyframe = max_covi.first;
    _to_update_local_map = true;
  }
}

void Tracking::UpdateLocalKeyframes(FramePtr frame) {
  _local_keyframes.clear();
  std::vector<std::pair<int, FramePtr>> neighbor_frames =
      _ref_keyframe->GetOrderedConnections(-1);
  for (auto &kv : neighbor_frames) {
    _local_keyframes.push_back(kv.second);
  }
}

void Tracking::UpdateLocalMappoints(FramePtr frame) {
  _local_mappoints.clear();
  int current_frame_id = frame->GetFrameId();
  for (auto &kf : _local_keyframes) {
    const std::vector<MappointPtr> &mpts = kf->GetAllMappoints();
    for (auto &mpt : mpts) {
      if (mpt && mpt->IsValid() && mpt->tracking_frame_id != current_frame_id) {
        mpt->tracking_frame_id = current_frame_id;
        _local_mappoints.push_back(mpt);
      }
    }
  }
}

void Tracking::SearchLocalPoints(
    FramePtr frame,
    std::vector<std::pair<int, MappointPtr>> &good_projections) {
  int current_frame_id = frame->GetFrameId();
  std::vector<MappointPtr> &mpts = frame->GetAllMappoints();
  for (auto &mpt : mpts) {
    if (mpt && !mpt->IsBad())
      mpt->last_frame_seen = current_frame_id;
  }

  std::vector<MappointPtr> selected_mappoints;
  for (auto &mpt : _local_mappoints) {
    if (mpt && mpt->IsValid() && mpt->last_frame_seen != current_frame_id) {
      selected_mappoints.push_back(mpt);
    }
  }

  _map->SearchByProjection(frame, selected_mappoints, 1, good_projections);
}

int Tracking::TrackLocalMap(FramePtr frame, int num_inlier_thr) {
  if (_to_update_local_map) {
    UpdateLocalKeyframes(frame);
    UpdateLocalMappoints(frame);
  }

  std::vector<std::pair<int, MappointPtr>> good_projections;
  SearchLocalPoints(frame, good_projections);
  if (good_projections.size() < 3)
    return -1;

  std::vector<MappointPtr> mappoints = frame->GetAllMappoints();
  for (auto &good_projection : good_projections) {
    int idx = good_projection.first;
    if (mappoints[idx] && !mappoints[idx]->IsBad())
      continue;
    mappoints[idx] = good_projection.second;
  }

  std::vector<int> inliers(mappoints.size(), -1);
  int num_inliers = FramePoseOptimization(frame, mappoints, inliers, 2);

  // update track id
  if (num_inliers > _configs.keyframe_config.min_num_match &&
      num_inliers > num_inlier_thr) {
    for (size_t i = 0; i < mappoints.size(); i++) {
      if (inliers[i] > 0) {
        frame->SetTrackId(i, mappoints[i]->GetId());
        frame->InsertMappoint(i, mappoints[i]);
      }
    }
  } else {
    num_inliers = -1;
  }
  return num_inliers;
}

void Tracking::PublishFrame(FramePtr frame, cv::Mat &image) {
  FeatureMessagePtr feature_message =
      std::shared_ptr<FeatureMessage>(new FeatureMessage);
  FramePoseMessagePtr frame_pose_message =
      std::shared_ptr<FramePoseMessage>(new FramePoseMessage);

  feature_message->image = image;
  feature_message->time = frame->GetTimestamp();
  feature_message->keypoints = frame->GetAllKeypoints();

  std::vector<bool> inliers_feature_message;
  frame->GetInlierFlag(inliers_feature_message);
  feature_message->inliers = inliers_feature_message;
  frame_pose_message->pose = frame->GetPose();
  frame_pose_message->time = frame->GetTimestamp();

  _ros_publisher->PublishFeature(feature_message);
  _ros_publisher->PublishFramePose(frame_pose_message);
}

void Tracking::SaveTrajectory() {
  // Todo(Jaafar): Save trajectory to file for evaluation!
  // std::string file_path = ConcatenateFolderAndFileName(_configs.saving_dir,
  // "keyframe_trajectory.txt"); _map->SaveKeyframeTrajectory(file_path);
}

void Tracking::SaveTrajectory(std::string file_path) {
  _map->SaveKeyframeTrajectory(file_path);
}


void Tracking::ShutDown() {
  _shutdown = true;
  _feature_thread.join();
  _tracking_thread.join();
}
