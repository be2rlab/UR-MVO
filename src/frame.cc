#include "frame.h"
#include <assert.h>

Frame::Frame() {}

Frame::Frame(int frame_id, bool pose_fixed, CameraPtr camera, double timestamp,
             int id_ts)
    : tracking_frame_id(-1), local_map_optimization_frame_id(-1),
      local_map_optimization_fix_frame_id(-1), _frame_id(frame_id),
      _pose_fixed(pose_fixed), _camera(camera), _timestamp(timestamp) {
  _grid_width_inv = static_cast<double>(FRAME_GRID_COLS) /
                    static_cast<double>(_camera->ImageWidth());
  _grid_height_inv = static_cast<double>(FRAME_GRID_ROWS) /
                     static_cast<double>(_camera->ImageHeight());
  _id_ts = id_ts;
}

Frame &Frame::operator=(const Frame &other) {
  tracking_frame_id = other.tracking_frame_id;
  local_map_optimization_frame_id = other.local_map_optimization_frame_id;
  local_map_optimization_fix_frame_id =
      other.local_map_optimization_fix_frame_id;
  _frame_id = other._frame_id;
  _timestamp = other._timestamp;
  _pose_fixed = other._pose_fixed;
  _pose = other._pose;
  _id_ts = other._id_ts;

  _features = other._features;
  _keypoints = other._keypoints;
  for (int i = 0; i < FRAME_GRID_COLS; i++) {
    for (int j = 0; j < FRAME_GRID_ROWS; j++) {
      _feature_grid[i][j] = other._feature_grid[i][j];
    }
  }
  _grid_width_inv = other._grid_width_inv;
  _grid_height_inv = other._grid_height_inv;
  if (_camera->GetCameraType() == CameraType::STEREO) {
    _u_right = other._u_right;
  }
  _depth = other._depth;
  _track_ids = other._track_ids;
  _mappoints = other._mappoints;
  _camera = other._camera;
  _depth_img = other._depth_img.clone();
  return *this;
}

void Frame::SetDepthImg(const cv::Mat &depth_img) {
  _depth_img = depth_img.clone();
}

cv::Mat &Frame::GetDepthImg() { return _depth_img; }

void Frame::SetFrameId(int frame_id) { _frame_id = frame_id; }

int Frame::GetFrameId() { return _frame_id; }

int Frame::GetFrameId_ts() { return _id_ts; }

double Frame::GetTimestamp() { return _timestamp; }

void Frame::SetPoseFixed(bool pose_fixed) { _pose_fixed = pose_fixed; }

bool Frame::PoseFixed() { return _pose_fixed; }

void Frame::SetPose(const Eigen::Matrix4d &pose) { _pose = pose; }

Eigen::Matrix4d &Frame::GetPose() { return _pose; }

bool Frame::FindGrid(double &x, double &y, int &grid_x, int &grid_y) {
  grid_x = std::round(x * _grid_width_inv);
  grid_y = std::round(y * _grid_height_inv);

  grid_x = std::min(std::max(0, grid_x), (FRAME_GRID_COLS - 1));
  grid_y = std::min(std::max(0, grid_y), (FRAME_GRID_ROWS - 1));

  return !(grid_x < 0 || grid_x >= FRAME_GRID_COLS || grid_y < 0 ||
           grid_y >= FRAME_GRID_ROWS);
}

void Frame::AddFeatures(
    Eigen::Matrix<double, 259, Eigen::Dynamic> &features_left,
    Eigen::Matrix<double, 259, Eigen::Dynamic> &features_right,
    std::vector<cv::DMatch> &stereo_matches) {
  AddLeftFeatures(features_left);
  AddRightFeatures(features_right, stereo_matches);
}

void Frame::AddLeftFeatures(
    Eigen::Matrix<double, 259, Eigen::Dynamic> &features) {
  AddFeatures(features);
}

int Frame::AddRightFeatures(
    Eigen::Matrix<double, 259, Eigen::Dynamic> &features_right,
    std::vector<cv::DMatch> &stereo_matches) {
  // filter matches from superglue
  std::vector<cv::DMatch> matches;
  double min_x_diff = _camera->MinXDiff();
  double max_x_diff = _camera->MaxXDiff();
  const double max_y_diff = _camera->MaxYDiff();
  for (cv::DMatch &match : stereo_matches) {
    int idx_left = match.queryIdx;
    int idx_right = match.trainIdx;

    double dx = std::abs(_features(1, idx_left) - features_right(1, idx_right));
    double dy = std::abs(_features(2, idx_left) - features_right(2, idx_right));

    if (dx > min_x_diff && dx < max_x_diff && dy <= max_y_diff) {
      matches.emplace_back(match);
    }
  }

  // trianguate stereo points
  for (cv::DMatch &match : matches) {
    int idx_left = match.queryIdx;
    int idx_right = match.trainIdx;

    assert(idx_left < _u_right.size());
    _u_right[idx_left] = features_right(1, idx_right);
    _depth[idx_left] =
        _camera->BF() / (_features(1, idx_left) - features_right(1, idx_right));
  }
  return matches.size();
}

void Frame::AddFeatures(Eigen::Matrix<double, 259, Eigen::Dynamic> &features) {
  _features = features;

  // fill in keypoints and assign features to grids
  size_t features_size = _features.cols();
  for (size_t i = 0; i < features_size; ++i) {
    double score = _features(0, i);
    double x = _features(1, i);
    double y = _features(2, i);
    _keypoints.emplace_back(x, y, 8, -1, score);

    int grid_x, grid_y;
    bool found = FindGrid(x, y, grid_x, grid_y);
    assert(found);
    _feature_grid[grid_x][grid_y].push_back(i);
  }

  // initialize depth
  // initialize u_right and depth
  _u_right = std::vector<double>(features_size, -1);
  _depth = std::vector<double>(features_size, -1);

  // initialize track_ids and mappoints
  std::vector<int> track_ids(features_size, -1);
  SetTrackIds(track_ids);
  std::vector<MappointPtr> mappoints(features_size, nullptr);
  _mappoints = mappoints;
}

Eigen::Matrix<double, 259, Eigen::Dynamic> &Frame::GetAllFeatures() {
  return _features;
}

size_t Frame::FeatureNum() { return _features.cols(); }
bool Frame::GetKeypointPosition(size_t idx, Eigen::Vector3d &keypoint_pos) {
  if (idx > _features.cols())
    return false;
  keypoint_pos.head(2) = _features.block<2, 1>(1, idx);
  keypoint_pos(2) = _u_right[idx];
  return true;
}

double Frame::GetRightPosition(size_t idx) {
  assert(idx < _u_right.size());
  return _u_right[idx];
}

std::vector<double> &Frame::GetAllRightPosition() { return _u_right; }

bool Frame::GetKeypointPosition(size_t idx, Eigen::Vector2d &keypoint_pos) {
  if (idx > _features.cols())
    return false;
  keypoint_pos.head(2) = _features.block<2, 1>(1, idx);
  return true;
}

std::vector<cv::KeyPoint> &Frame::GetAllKeypoints() { return _keypoints; }

cv::KeyPoint &Frame::GetKeypoint(size_t idx) {
  assert(idx < _keypoints.size());
  return _keypoints[idx];
}

int Frame::GetInlierFlag(std::vector<bool> &inliers_feature_message) {
  int num_inliers = 0;
  inliers_feature_message.resize(_mappoints.size());
  for (size_t i = 0; i < _mappoints.size(); i++) {
    if (_mappoints[i] && !_mappoints[i]->IsBad()) {
      inliers_feature_message[i] = true;
      num_inliers++;
    } else {
      inliers_feature_message[i] = false;
    }
  }
  return num_inliers;
}

bool Frame::GetDescriptor(size_t idx,
                          Eigen::Matrix<double, 256, 1> &descriptor) const {
  if (idx > _features.cols())
    return false;
  descriptor = _features.block<256, 1>(3, idx);
  return true;
}

double Frame::GetDepth(size_t idx) {
  assert(idx < _depth.size());
  return _depth[idx];
}

std::vector<double> &Frame::GetAllDepth() { return _depth; }

void Frame::SetDepth(size_t idx, double depth) {
  assert(idx < _depth.size());
  _depth[idx] = depth;
}

void Frame::SetTrackIds(std::vector<int> &track_ids) { _track_ids = track_ids; }

std::vector<int> &Frame::GetAllTrackIds() { return _track_ids; }

void Frame::SetTrackId(size_t idx, int track_id) { _track_ids[idx] = track_id; }

int Frame::GetTrackId(size_t idx) {
  assert(idx < _track_ids.size());
  return _track_ids[idx];
}

MappointPtr Frame::GetMappoint(size_t idx) {
  assert(idx < _mappoints.size());
  return _mappoints[idx];
}

std::vector<MappointPtr> &Frame::GetAllMappoints() { return _mappoints; }

void Frame::InsertMappoint(size_t idx, MappointPtr mappoint) {
  assert(idx < FeatureNum());
  _mappoints[idx] = mappoint;
}

bool Frame::BackProjectPointStereo(size_t idx, Eigen::Vector3d &p3D) {
  if (idx >= _depth.size() || _depth[idx] <= 0)
    return false;
  Eigen::Vector3d p2D;
  if (!GetKeypointPosition(idx, p2D))
    return false;
  if (!_camera->BackProjectStereo(p2D, p3D))
    return false;
  return true;
}

bool Frame::BackProjectPoint(size_t idx, Eigen::Vector3d &p3D) {
  if (idx >= _depth.size() || _depth[idx] <= 0)
    return false;
  Eigen::Vector2d p2D;
  if (!GetKeypointPosition(idx, p2D))
    return false;
  if (!_camera->BackProjectMono(p2D, p3D))
    return false;
  return true;
}

bool Frame::BackProjectPointDepth(size_t idx, Eigen::Vector3d &p3D) {
  if (idx >= _depth.size())
    return false;
  Eigen::Vector2d p2D;
  if (!GetKeypointPosition(idx, p2D))
    return false;
  if (!_camera->BackProjectMono(p2D, p3D))
    return false;
  return true;
}

CameraPtr Frame::GetCamera() { return _camera; }

void Frame::FindNeighborKeypoints(Eigen::Vector3d &p2D,
                                  std::vector<int> &indices, double r,
                                  bool filter) const {
  double x = p2D(0);
  double y = p2D(1);
  double xr = p2D(2);
  const int min_grid_x =
      std::max(0, (int)std::floor((x - r) * _grid_width_inv));
  const int max_grid_x = std::min((int)(FRAME_GRID_COLS - 1),
                                  (int)std::ceil((x + r) * _grid_width_inv));
  const int min_grid_y =
      std::max(0, (int)std::floor((y - r) * _grid_height_inv));
  const int max_grid_y = std::min((int)(FRAME_GRID_ROWS - 1),
                                  (int)std::ceil((y + r) * _grid_height_inv));
  if (min_grid_x >= FRAME_GRID_COLS || max_grid_x < 0 ||
      min_grid_y >= FRAME_GRID_ROWS || max_grid_y < 0)
    return;

  for (int gx = min_grid_x; gx <= max_grid_x; gx++) {
    for (int gy = min_grid_y; gy <= max_grid_y; gy++) {
      if (_feature_grid[gx][gy].empty())
        continue;
      for (auto &idx : _feature_grid[gx][gy]) {
        if (filter && _mappoints[idx] && !_mappoints[idx]->IsBad())
          continue;

        const double dx = _keypoints[idx].pt.x - x;
        const double dy = _keypoints[idx].pt.y - y;
        const double dxr = (xr > 0) ? (_u_right[idx] - xr) : 0;
        if (std::abs(dx) < r && std::abs(dy) < r && std::abs(dxr) < r) {
          indices.push_back(idx);
        }
      }
    }
  }
}

void Frame::FindNeighborKeypoints(Eigen::Vector2d &p2D,
                                  std::vector<int> &indices, double r,
                                  bool filter) const {
  double x = p2D(0);
  double y = p2D(1);
  const int min_grid_x =
      std::max(0, (int)std::floor((x - r) * _grid_width_inv));
  const int max_grid_x = std::min((int)(FRAME_GRID_COLS - 1),
                                  (int)std::ceil((x + r) * _grid_width_inv));
  const int min_grid_y =
      std::max(0, (int)std::floor((y - r) * _grid_height_inv));
  const int max_grid_y = std::min((int)(FRAME_GRID_ROWS - 1),
                                  (int)std::ceil((y + r) * _grid_height_inv));
  if (min_grid_x >= FRAME_GRID_COLS || max_grid_x < 0 ||
      min_grid_y >= FRAME_GRID_ROWS || max_grid_y < 0)
    return;

  for (int gx = min_grid_x; gx <= max_grid_x; gx++) {
    for (int gy = min_grid_y; gy <= max_grid_y; gy++) {
      if (_feature_grid[gx][gy].empty())
        continue;
      for (auto &idx : _feature_grid[gx][gy]) {
        if (filter && _mappoints[idx] && !_mappoints[idx]->IsBad())
          continue;

        const double dx = _keypoints[idx].pt.x - x;
        const double dy = _keypoints[idx].pt.y - y;
        if (std::abs(dx) < r && std::abs(dy) < r) {
          indices.push_back(idx);
        }
      }
    }
  }
}

void Frame::AddConnection(std::shared_ptr<Frame> frame, int weight) {
  std::map<std::shared_ptr<Frame>, int>::iterator it = _connections.find(frame);
  bool add_connection = (it == _connections.end());
  bool change_connection = (!add_connection && _connections[frame] != weight);
  if (add_connection || change_connection) {
    if (change_connection) {
      _ordered_connections.erase(
          std::pair<int, std::shared_ptr<Frame>>(it->second, frame));
    }
    _connections[frame] = weight;
    _ordered_connections.insert(
        std::pair<int, std::shared_ptr<Frame>>(weight, frame));
  }
}

void Frame::AddConnection(
    std::set<std::pair<int, std::shared_ptr<Frame>>> connections) {
  _ordered_connections = connections;
  _connections.clear();
  for (auto &kv : connections) {
    _connections[kv.second] = kv.first;
  }
}

void Frame::SetParent(std::shared_ptr<Frame> parent) { _parent = parent; }

std::shared_ptr<Frame> Frame::GetParent() { return _parent; }

void Frame::SetChild(std::shared_ptr<Frame> child) { _child = child; }

std::shared_ptr<Frame> Frame::GetChild() { return _child; }

void Frame::RemoveConnection(std::shared_ptr<Frame> frame) {
  std::map<std::shared_ptr<Frame>, int>::iterator it = _connections.find(frame);
  if (it != _connections.end()) {
    _ordered_connections.erase(
        std::pair<int, std::shared_ptr<Frame>>(it->second, it->first));
    _connections.erase(it);
  }
}

void Frame::RemoveMappoint(MappointPtr mappoint) {
  RemoveMappoint(mappoint->GetKeypointIdx(_frame_id));
}

void Frame::RemoveMappoint(int idx) {
  if (idx < _mappoints.size() && idx >= 0) {
    _mappoints[idx] = nullptr;
  }
}

void Frame::DecreaseWeight(std::shared_ptr<Frame> frame, int weight) {
  std::map<std::shared_ptr<Frame>, int>::iterator it = _connections.find(frame);
  if (it != _connections.end()) {
    _ordered_connections.erase(
        std::pair<int, std::shared_ptr<Frame>>(it->second, it->first));
    int original_weight = it->second;
    bool to_remove =
        (original_weight < (weight + 5) && _connections.size() >= 2) ||
        (original_weight <= weight);
    if (to_remove) {
      _connections.erase(it);
    } else {
      it->second = original_weight - weight;
      _ordered_connections.insert(
          std::pair<int, std::shared_ptr<Frame>>(it->second, it->first));
    }
  }
}

std::vector<std::pair<int, std::shared_ptr<Frame>>>
Frame::GetOrderedConnections(int number) {
  int n = (number > 0 && number < _ordered_connections.size())
              ? number
              : _ordered_connections.size();
  return std::vector<std::pair<int, std::shared_ptr<Frame>>>(
      _ordered_connections.begin(), std::next(_ordered_connections.begin(), n));
}

void Frame::SetPreviousFrame(std::shared_ptr<Frame> previous_frame) {
  _previous_frame = previous_frame;
}

std::shared_ptr<Frame> Frame::PreviousFrame() { return _previous_frame; }

void Frame::remove() {

  // Key Features
  _keypoints.clear();
  for (int i = 0; i < FRAME_GRID_COLS; i++)
    for (int j = 0; j < FRAME_GRID_ROWS; j++)
      _feature_grid[i][j].clear();
  _depth.clear();
  _track_ids.clear();
  _u_right.clear();

  // MapPoints
  for (auto &mp : _mappoints) {
    if (mp != nullptr) {
      mp->RemoveObverser(_frame_id);
      RemoveMappoint(mp);
    }
  }
  _mappoints.clear();

  // Camera
  _camera.reset();

  // covisibility graph
  _connections.clear();
  _ordered_connections.clear();
  _parent.reset();
  _child.reset();
  _previous_frame.reset();

  debug_img.release();
}