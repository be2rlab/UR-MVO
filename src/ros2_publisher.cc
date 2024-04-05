#include "ros2_publisher.h"

#include <Eigen/Geometry>
#include <chrono>

#include "utils.h"

double GetCurrentTime() {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::duration<double>>(duration)
      .count();
}

Ros2Publisher::Ros2Publisher(const RosPublisherConfig &ros_publisher_config)
    : Node(ros_publisher_config.publisher_name), _config(ros_publisher_config) {
  if (_config.feature) {
    _ros_feature_pub = this->create_publisher<sensor_msgs::msg::Image>(
        _config.feature_topic, 10);
    std::function<void(const FeatureMessageConstPtr &)>
        publish_feature_function =
            [&](const FeatureMessageConstPtr &feature_message) {
              cv::Mat drawed_image = DrawFeatures(feature_message->image,
                                                  feature_message->keypoints,
                                                  feature_message->inliers);
              sensor_msgs::msg::Image::SharedPtr ros_feature_msg =
                  cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8",
                                     drawed_image)
                      .toImageMsg();
              _ros_feature_pub->publish(*ros_feature_msg.get());
            };
    _feature_publisher.Register(publish_feature_function);
    _feature_publisher.Start();
  }
  if (_config.debug) {
    _ros_debug_pub = this->create_publisher<sensor_msgs::msg::Image>(
        _config.debug_topic, 10);
    std::function<void(const DebugMessageConstPtr &)>
        publish_debug_function =
            [&](const DebugMessageConstPtr &debug_message) {
              cv::Mat drawed_image = debug_message->image;
              sensor_msgs::msg::Image::SharedPtr ros_debug_msg =
                  cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8",
                                     drawed_image)
                      .toImageMsg();
              _ros_debug_pub->publish(*ros_debug_msg.get());
            };
    _debug_publisher.Register(publish_debug_function);
    _debug_publisher.Start();
  }


  if (_config.frame_pose) {
    _ros_frame_pose_pub =
        this->create_publisher<geometry_msgs::msg::PoseStamped>(
            _config.frame_pose_topic, 10);
    std::function<void(const FramePoseMessageConstPtr &)>
        publish_frame_pose_function =
            [&](const FramePoseMessageConstPtr &frame_pose_message) {
              geometry_msgs::msg::PoseStamped pose_stamped;
              double timestamp = frame_pose_message->time;
              int32_t secs = floor(timestamp);
              int32_t nsecs = floor((timestamp - secs) * pow(10, 9));
              pose_stamped.header.stamp = rclcpp::Time(secs, nsecs);
              pose_stamped.header.frame_id = "map";
              pose_stamped.pose.position.x = frame_pose_message->pose(0, 3);
              pose_stamped.pose.position.y = frame_pose_message->pose(1, 3);
              pose_stamped.pose.position.z = frame_pose_message->pose(2, 3);
              Eigen::Quaterniond q(frame_pose_message->pose.block<3, 3>(0, 0));
              pose_stamped.pose.orientation.x = q.x();
              pose_stamped.pose.orientation.y = q.y();
              pose_stamped.pose.orientation.z = q.z();
              pose_stamped.pose.orientation.w = q.w();
              _ros_frame_pose_pub->publish(pose_stamped);
            };
    _frame_pose_publisher.Register(publish_frame_pose_function);
    _frame_pose_publisher.Start();
  }

  if (_config.keyframe) {
    _ros_keyframe_pub = this->create_publisher<geometry_msgs::msg::PoseArray>(
        _config.keyframe_topic, 10);
    _ros_keyframe_array.header.stamp = this->now();
    _ros_keyframe_array.header.frame_id = "map";

    _ros_path_pub =
        this->create_publisher<nav_msgs::msg::Path>(_config.path_topic, 10);
    _ros_path.header.stamp = this->now();
    _ros_path.header.frame_id = "map";

    std::function<void(const KeyframeMessageConstPtr &)>
        publish_keyframe_function =
            [&](const KeyframeMessageConstPtr &keyframe_message) {
              std::map<int, int>::iterator it;
              for (int i = 0; i < keyframe_message->ids.size(); i++) {
                int keyframe_id = keyframe_message->ids[i];

                geometry_msgs::msg::Pose pose;
                pose.position.x = keyframe_message->poses[i](0, 3);
                pose.position.y = keyframe_message->poses[i](1, 3);
                pose.position.z = keyframe_message->poses[i](2, 3);
                Eigen::Quaterniond q(
                    keyframe_message->poses[i].block<3, 3>(0, 0));
                pose.orientation.x = q.x();
                pose.orientation.y = q.y();
                pose.orientation.z = q.z();
                pose.orientation.w = q.w();

                geometry_msgs::msg::PoseStamped pose_stamped;
                pose_stamped.header.stamp = this->now();
                pose_stamped.pose = pose;

                it = _keyframe_id_to_index.find(keyframe_id);
                if (it == _keyframe_id_to_index.end()) {
                  _ros_keyframe_array.poses.push_back(pose);
                  _ros_path.poses.push_back(pose_stamped);
                  _keyframe_id_to_index[keyframe_id] =
                      _ros_keyframe_array.poses.size() - 1;
                } else {
                  int idx = it->second;
                  _ros_keyframe_array.poses[idx] = pose;
                  _ros_path.poses[idx] = pose_stamped;
                }
              }
              _ros_keyframe_pub->publish(_ros_keyframe_array);
              _ros_path_pub->publish(_ros_path);
            };
    _keyframe_publisher.Register(publish_keyframe_function);
    _keyframe_publisher.Start();
  }

  if (_config.map) {
    // for mappoints
    _ros_map_pub = this->create_publisher<sensor_msgs::msg::PointCloud>(
        _config.map_topic, 1);
    _ros_mappoints.header.stamp = this->now();
    _ros_mappoints.header.frame_id = "map";

    std::function<void(const MapMessageConstPtr &)> publish_map_function =
        [&](const MapMessageConstPtr &map_message) {
          std::unordered_map<int, int>::iterator it;
          for (int i = 0; i < map_message->ids.size(); i++) {
            int mappoint_id = map_message->ids[i];
            it = _mappoint_id_to_index.find(mappoint_id);
            if (it == _mappoint_id_to_index.end()) {
              geometry_msgs::msg::Point32 point;
              point.x = map_message->points[i](0);
              point.y = map_message->points[i](1);
              point.z = map_message->points[i](2);
              _ros_mappoints.points.push_back(point);
              _mappoint_id_to_index[mappoint_id] =
                  _ros_mappoints.points.size() - 1;
            } else {
              int idx = it->second;
              _ros_mappoints.points[idx].x = map_message->points[i](0);
              _ros_mappoints.points[idx].y = map_message->points[i](1);
              _ros_mappoints.points[idx].z = map_message->points[i](2);
            }
          }
          _ros_map_pub->publish(_ros_mappoints);
        };
    _map_publisher.Register(publish_map_function);
    _map_publisher.Start();
  }
}

void Ros2Publisher::PublishFeature(FeatureMessagePtr feature_message) {
  _feature_publisher.Publish(feature_message);
}
void Ros2Publisher::PublishDebug(DebugMessagePtr debug_message) {
  _debug_publisher.Publish(debug_message);
}

void Ros2Publisher::PublishFramePose(FramePoseMessagePtr frame_pose_message) {
  _frame_pose_publisher.Publish(frame_pose_message);
}

void Ros2Publisher::PublishKeyframe(KeyframeMessagePtr keyframe_message) {
  _keyframe_publisher.Publish(keyframe_message);
}

void Ros2Publisher::PublishMap(MapMessagePtr map_message) {
  _map_publisher.Publish(map_message);
}

void Ros2Publisher::ShutDown() {
  _frame_pose_publisher.ShutDown();
  _keyframe_publisher.ShutDown();
  _keyframe_publisher.ShutDown();
  _map_publisher.ShutDown();
}
