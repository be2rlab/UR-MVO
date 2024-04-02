#include <yaml-cpp/yaml.h>

#include "camera.h"
#include "utils.h"

Camera::Camera() {}

Camera::Camera(const std::string &camera_file, CameraType camType) {
  _camera_type = camType;
  cv::FileStorage camera_configs(camera_file, cv::FileStorage::READ);
  if (!camera_configs.isOpened()) {
    std::cerr << "ERROR: Wrong path to settings" << std::endl;
    exit(-1);
  }

  _image_height = camera_configs["image_height"];
  _image_width = camera_configs["image_width"];
  _depth_lower_thr = camera_configs["depth_lower_thr"];
  _depth_upper_thr = camera_configs["depth_upper_thr"];

  cv::Mat K, P, D, R;
  camera_configs["LEFT_K"] >> K;
  camera_configs["LEFT_P"] >> P;
  camera_configs["LEFT_D"] >> D;
  camera_configs["LEFT_R"] >> R;

  if (K.empty() || P.empty() || D.empty() || R.empty() || _image_height == 0 ||
      _image_width == 0) {
    std::cout
        << "ERROR: Calibration parameters to monocular camera are missing!"
        << std::endl;
    exit(0);
  }

  _fx = P.at<double>(0, 0);
  _fy = P.at<double>(1, 1);
  _cx = P.at<double>(0, 2);
  _cy = P.at<double>(1, 2);
  _fx_inv = 1.0 / _fx;
  _fy_inv = 1.0 / _fy;
  int distortion_type = camera_configs["distortion_type"];
  if (_camera_type == CameraType::STEREO) {
    cv::Mat K_r, P_r, R_r, D_r;
    _bf = camera_configs["bf"];
    _max_x_diff = _bf / _depth_lower_thr;
    _min_x_diff = _bf / _depth_upper_thr;
    _max_y_diff = camera_configs["max_y_diff"];
    camera_configs["RIGHT_K"] >> K_r;
    camera_configs["RIGHT_P"] >> P_r;
    camera_configs["RIGHT_R"] >> R_r;
    camera_configs["RIGHT_D"] >> D_r;

    if (K_r.empty() || P_r.empty() || R_r.empty() || D_r.empty() ||
        _image_height == 0 || _image_width == 0) {
      std::cout
          << "ERROR: Calibration parameters to rectify stereo are missing!"
          << std::endl;
      exit(0);
    }

    if (distortion_type == 0) {
      cv::initUndistortRectifyMap(K, D, R, P.rowRange(0, 3).colRange(0, 3),
                                  cv::Size(_image_width, _image_height), CV_32F,
                                  _map1, _map2);
      cv::initUndistortRectifyMap(
          K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3),
          cv::Size(_image_width, _image_height), CV_32F, _mapr1, _mapr2);
    } else {
      cv::fisheye::initUndistortRectifyMap(
          K, D, R, P.rowRange(0, 3).colRange(0, 3),
          cv::Size(_image_width, _image_height), CV_32F, _map1, _map2);
      cv::fisheye::initUndistortRectifyMap(
          K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3),
          cv::Size(_image_width, _image_height), CV_32F, _mapr1, _mapr2);
    }
  } else {
    if (distortion_type == 0) {
      cv::initUndistortRectifyMap(
          K, D, cv::Mat::eye(3, 3, CV_32F), P.rowRange(0, 3).colRange(0, 3),
          cv::Size(_image_width, _image_height), CV_32F, _map1, _map2);
    } else {
      cv::fisheye::initUndistortRectifyMap(
          K, D, cv::Mat::eye(3, 3, CV_32F), P.rowRange(0, 3).colRange(0, 3),
          cv::Size(_image_width, _image_height), CV_32F, _map1, _map2);
    }
  }
}

Camera &Camera::operator=(const Camera &camera) {
  _camera_type = camera._camera_type;
  _image_height = camera._image_height;
  _image_width = camera._image_width;
  if (_camera_type == CameraType::STEREO) {
    _bf = camera._bf;
    _max_x_diff = _bf / _depth_lower_thr;
    _min_x_diff = _bf / _depth_upper_thr;
    _max_y_diff = camera._max_y_diff;
  }
  _depth_lower_thr = camera._depth_lower_thr;
  _depth_upper_thr = camera._depth_upper_thr;
  _fx = camera._fx;
  _fy = camera._fy;
  _cx = camera._cx;
  _cy = camera._cy;
  _fx_inv = camera._fx_inv;
  _fy_inv = camera._fy_inv;
  _map1 = camera._map1.clone();
  _map2 = camera._map2.clone();
  if (_camera_type == CameraType::STEREO) {
    _mapr1 = camera._mapr1.clone();
    _mapr2 = camera._mapr2.clone();
  }
  return *this;
}

void Camera::UndistortImage(cv::Mat &image, cv::Mat &image_undisorted) {
  cv::remap(image, image_undisorted, _map1, _map2, cv::INTER_LINEAR);
}

void Camera::UndistortImage(cv::Mat &image_left, cv::Mat &image_right,
                            cv::Mat &image_left_rect,
                            cv::Mat &image_right_rect) {
  cv::remap(image_left, image_left_rect, _map1, _map2, cv::INTER_LINEAR);
  cv::remap(image_right, image_right_rect, _mapr1, _mapr2, cv::INTER_LINEAR);
}

double Camera::ImageHeight() { return _image_height; }

double Camera::ImageWidth() { return _image_width; }

double Camera::BF() { return _bf; }

double Camera::Fx() { return _fx; }

double Camera::Fy() { return _fy; }

double Camera::Cx() { return _cx; }

double Camera::Cy() { return _cy; }

Eigen::Matrix3f Camera::K() {
  Eigen::Matrix3f k = Eigen::Matrix3f::Identity();
  k(0, 0) = _fx;
  k(1, 1) = _fy;
  k(0, 2) = _cx;
  k(1, 2) = _cy;
  return k;
}
double Camera::DepthLowerThr() { return _depth_lower_thr; }

double Camera::DepthUpperThr() { return _depth_upper_thr; }

double Camera::MaxXDiff() { return _max_x_diff; }

double Camera::MinXDiff() { return _min_x_diff; }

double Camera::MaxYDiff() { return _max_y_diff; }

void Camera::GetCamerMatrix(cv::Mat &camera_matrix) {
  camera_matrix =
      (cv::Mat_<double>(3, 3) << _fx, 0.0, _cx, 0.0, _fy, _cy, 0.0, 0.0, 1.0);
}

void Camera::GetDistCoeffs(cv::Mat &dist_coeffs) {
  dist_coeffs = (cv::Mat_<double>(5, 1) << 0.0, 0.0, 0.0, 0.0, 0.0);
}

bool Camera::BackProjectMono(const Eigen::Vector2d &keypoint,
                             Eigen::Vector3d &output) {
  output(0) = (keypoint(0) - _cx) * _fx_inv;
  output(1) = (keypoint(1) - _cy) * _fy_inv;
  output(2) = 1.0;
  return true;
}

bool Camera::BackProjectStereo(const Eigen::Vector3d &keypoint,
                               Eigen::Vector3d &output) {
  BackProjectMono(keypoint.head(2), output);
  double d = _bf / (keypoint(0) - keypoint(2));
  output = output * d;
  return true;
}