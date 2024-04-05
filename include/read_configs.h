#ifndef READ_CONFIGS_H_
#define READ_CONFIGS_H_

#include <iostream>
#include <yaml-cpp/yaml.h>

#include "utils.h"

struct SuperPointConfig {
  int max_keypoints;
  double keypoint_threshold;
  int remove_borders;
  int dla_core;
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};

struct SuperGlueConfig {
  int image_width;
  int image_height;
  int dla_core;
  double matching_threshold;
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};

struct KeyframeConfig {
  int min_num_match;
  int max_num_match;
  double max_distance;
  double max_angle;
  int max_num_passed_frame;
};

struct OptimizationConfig {
  double mono_point;
  double stereo_point;
  double rate;
};

struct RosPublisherConfig {
  std::string publisher_name;
  int feature;
  std::string feature_topic;
  int frame_pose;
  std::string frame_pose_topic;
  int keyframe;
  std::string keyframe_topic;
  std::string path_topic;
  int map;
  std::string map_topic;
};

enum class SensorSetup { Mono, Stereo, RGBD };

struct Configs {
  std::string dataroot;
  std::string camera_config_path;
  bool use_mask;
  SensorSetup sensor_setup;
  std::string sequence_name;
  std::string model_dir = "/workspace/ros2_ws/src/ur_mvo/output";
  std::string save_trajectory =
      "/workspace/ros2_ws/src/ur_mvo/results/";

  std::string data_path;

  SuperPointConfig superpoint_config;
  SuperGlueConfig superglue_config;
  KeyframeConfig keyframe_config;
  OptimizationConfig tracking_optimization_config;
  OptimizationConfig backend_optimization_config;
  RosPublisherConfig ros_publisher_config;

  Configs(const std::string &config_file, const std::string &setup) {
    std::cout << "config_file = " << config_file << std::endl;
    if (!FileExists(config_file)) {
      std::cout << "config file: " << config_file << " doesn't exist"
                << std::endl;
      return;
    }
    YAML::Node file_node = YAML::LoadFile(config_file);
    YAML::Node input_node = file_node["input"];
    sequence_name = save_trajectory + "_" + setup + ".txt";

    camera_config_path = "/workspace/ros2_ws/src/ur_mvo/configs/" +
                         input_node["camera_config_path"].as<std::string>();
    use_mask = input_node["use_mask"].as<int>() == 1;
    if (setup == "mono")
      sensor_setup = SensorSetup::Mono;
    else if (setup == "stereo")
      sensor_setup = SensorSetup::Stereo;
    else if (setup == "rgbd")
      sensor_setup = SensorSetup::RGBD;
    else
      std::cout << "setup: " << setup << " is not supported" << std::endl;

    YAML::Node superpoint_node = file_node["superpoint"];
    superpoint_config.max_keypoints =
        superpoint_node["max_keypoints"].as<int>();
    superpoint_config.keypoint_threshold =
        superpoint_node["keypoint_threshold"].as<double>();
    superpoint_config.remove_borders =
        superpoint_node["remove_borders"].as<int>();
    superpoint_config.dla_core = superpoint_node["dla_core"].as<int>();
    YAML::Node superpoint_input_tensor_names_node =
        superpoint_node["input_tensor_names"];
    size_t superpoint_num_input_tensor_names =
        superpoint_input_tensor_names_node.size();
    for (size_t i = 0; i < superpoint_num_input_tensor_names; i++) {
      superpoint_config.input_tensor_names.push_back(
          superpoint_input_tensor_names_node[i].as<std::string>());
    }
    YAML::Node superpoint_output_tensor_names_node =
        superpoint_node["output_tensor_names"];
    size_t superpoint_num_output_tensor_names =
        superpoint_output_tensor_names_node.size();
    for (size_t i = 0; i < superpoint_num_output_tensor_names; i++) {
      superpoint_config.output_tensor_names.push_back(
          superpoint_output_tensor_names_node[i].as<std::string>());
    }
    std::string superpoint_onnx_file =
        superpoint_node["onnx_file"].as<std::string>();
    std::string superpoint_engine_file =
        superpoint_node["engine_file"].as<std::string>();
    superpoint_config.onnx_file =
        ConcatenateFolderAndFileName(model_dir, superpoint_onnx_file);
    superpoint_config.engine_file =
        ConcatenateFolderAndFileName(model_dir, superpoint_engine_file);

    YAML::Node superglue_node = file_node["superglue"];
    superglue_config.image_width = superglue_node["image_width"].as<int>();
    superglue_config.image_height = superglue_node["image_height"].as<int>();
    superglue_config.matching_threshold =
        superglue_node["matching_threshold"].as<double>();
    superglue_config.dla_core = superglue_node["dla_core"].as<int>();
    YAML::Node superglue_input_tensor_names_node =
        superglue_node["input_tensor_names"];
    size_t superglue_num_input_tensor_names =
        superglue_input_tensor_names_node.size();
    for (size_t i = 0; i < superglue_num_input_tensor_names; i++) {
      superglue_config.input_tensor_names.push_back(
          superglue_input_tensor_names_node[i].as<std::string>());
    }
    YAML::Node superglue_output_tensor_names_node =
        superglue_node["output_tensor_names"];
    size_t superglue_num_output_tensor_names =
        superglue_output_tensor_names_node.size();
    for (size_t i = 0; i < superglue_num_output_tensor_names; i++) {
      superglue_config.output_tensor_names.push_back(
          superglue_output_tensor_names_node[i].as<std::string>());
    }
    std::string superglue_onnx_file =
        superglue_node["onnx_file"].as<std::string>();
    std::string superglue_engine_file =
        superglue_node["engine_file"].as<std::string>();
    superglue_config.onnx_file =
        ConcatenateFolderAndFileName(model_dir, superglue_onnx_file);
    superglue_config.engine_file =
        ConcatenateFolderAndFileName(model_dir, superglue_engine_file);

    YAML::Node keyframe_node = file_node["keyframe"];
    keyframe_config.min_num_match = keyframe_node["min_num_match"].as<int>();
    keyframe_config.max_num_match = keyframe_node["max_num_match"].as<int>();
    keyframe_config.max_distance = keyframe_node["max_distance"].as<double>();
    keyframe_config.max_angle = keyframe_node["max_angle"].as<double>();
    keyframe_config.max_num_passed_frame =
        keyframe_node["max_num_passed_frame"].as<int>();

    YAML::Node tracking_optimization_node =
        file_node["optimization"]["tracking"];
    tracking_optimization_config.mono_point =
        tracking_optimization_node["mono_point"].as<double>();
    tracking_optimization_config.stereo_point =
        tracking_optimization_node["stereo_point"].as<double>();
    tracking_optimization_config.rate =
        tracking_optimization_node["rate"].as<double>();

    YAML::Node backend_optimization_node = file_node["optimization"]["backend"];
    backend_optimization_config.mono_point =
        backend_optimization_node["mono_point"].as<double>();
    backend_optimization_config.stereo_point =
        backend_optimization_node["stereo_point"].as<double>();
    backend_optimization_config.rate =
        backend_optimization_node["rate"].as<double>();

    YAML::Node ros_publisher_node = file_node["ros2_publisher"];
    ros_publisher_config.publisher_name =
        ros_publisher_node["publisher_name"].as<std::string>();
    ros_publisher_config.feature = ros_publisher_node["feature"].as<int>();
    ros_publisher_config.feature_topic =
        ros_publisher_node["feature_topic"].as<std::string>();
    ros_publisher_config.frame_pose =
        ros_publisher_node["frame_pose"].as<int>();
    ros_publisher_config.frame_pose_topic =
        ros_publisher_node["frame_pose_topic"].as<std::string>();
    ros_publisher_config.keyframe = ros_publisher_node["keyframe"].as<int>();
    ros_publisher_config.keyframe_topic =
        ros_publisher_node["keyframe_topic"].as<std::string>();
    ros_publisher_config.path_topic =
        ros_publisher_node["path_topic"].as<std::string>();
    ros_publisher_config.map = ros_publisher_node["map"].as<int>();
    ros_publisher_config.map_topic =
        ros_publisher_node["map_topic"].as<std::string>();
  }
};

#endif // READ_CONFIGS_H_
