
#include "utils.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <dirent.h>

void ConvertVectorToRt(Eigen::Matrix<double, 7, 1> &m, Eigen::Matrix3d &R,
                       Eigen::Vector3d &t) {
  Eigen::Quaterniond q(m(0, 0), m(1, 0), m(2, 0), m(3, 0));
  R = q.matrix();
  t = m.block<3, 1>(4, 0);
}

// (f1 - f2) * (f1 - f2) = f1 * f1 + f2 * f2 - 2 * f1 *f2 = 2 - 2 * f1 * f2 ->
// [0, 4]
double DescriptorDistance(const Eigen::Matrix<double, 256, 1> &f1,
                          const Eigen::Matrix<double, 256, 1> &f2) {
  return 2 * (1.0 - f1.transpose() * f2);
}

cv::Scalar GenerateColor(int id) {
  id++;
  int red = (id * 23) % 255;
  int green = (id * 53) % 255;
  int blue = (id * 79) % 255;
  return cv::Scalar(blue, green, red);
}

void GenerateColor(int id, Eigen::Vector3d color) {
  id++;
  int red = (id * 23) % 255;
  int green = (id * 53) % 255;
  int blue = (id * 79) % 255;
  color << red, green, blue;
  color *= (1.0 / 255.0);
}

cv::Mat DrawFeatures(const cv::Mat &image,
                     const std::vector<cv::KeyPoint> &keypoints,
                     const std::vector<bool> &inliers) {
  cv::Mat img_color;
  cv::cvtColor(image, img_color, cv::COLOR_GRAY2RGB);

  size_t point_num = keypoints.size();
  std::vector<cv::Scalar> colors(point_num, cv::Scalar(0, 255, 0));
  std::vector<int> radii(point_num, 2);

  // draw points
  for (size_t j = 0; j < point_num; j++) {
    cv::circle(img_color, keypoints[j].pt, radii[j], colors[j], 1, cv::LINE_AA);
  }
  return img_color;
}

void GetFileNames(std::string path, std::vector<std::string> &filenames) {
  DIR *pDir;
  struct dirent *ptr;
  std::cout << "path = " << path << std::endl;
  if (!(pDir = opendir(path.c_str()))) {
    std::cout << "Folder doesn't Exist!" << std::endl;
    return;
  }
  while ((ptr = readdir(pDir)) != 0) {
    if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
      filenames.push_back(ptr->d_name);
    }
  }
  closedir(pDir);
}

bool FileExists(const std::string &file) {
  struct stat file_status;
  if (stat(file.c_str(), &file_status) == 0 &&
      (file_status.st_mode & S_IFREG)) {
    return true;
  }
  return false;
}

bool PathExists(const std::string &path) {
  struct stat file_status;
  if (stat(path.c_str(), &file_status) == 0 &&
      (file_status.st_mode & S_IFDIR)) {
    return true;
  }
  return false;
}

void ConcatenateFolderAndFileName(const std::string &folder,
                                  const std::string &file_name,
                                  std::string *path) {
  *path = folder;
  if (path->back() != '/') {
    *path += '/';
  }
  *path = *path + file_name;
}

std::string ConcatenateFolderAndFileName(const std::string &folder,
                                         const std::string &file_name) {
  std::string path;
  ConcatenateFolderAndFileName(folder, file_name, &path);
  return path;
}

void MakeDir(const std::string &path) {
  if (!PathExists(path)) {
    mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }
}
