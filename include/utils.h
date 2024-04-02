#ifndef UTILS_H_
#define UTILS_H_

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits.h>
#include <map>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam3d_addons/types_slam3d_addons.h>

struct InputData {
  size_t index;
  size_t id_ts;
  double time;
  cv::Mat image;
  cv::Mat image_right;

  cv::Mat mask;
  cv::Mat depth;

  InputData() {}
  InputData &operator=(InputData &other) {
    index = other.index;
    time = other.time;
    image = other.image.clone();
    if (!other.mask.empty())
      mask = other.mask.clone();
    if (!other.depth.empty())
      depth = other.depth.clone();
    if (!other.image_right.empty())
      image_right = other.image_right.clone();
    return *this;
  }
};
typedef std::shared_ptr<InputData> InputDataPtr;


// Eigen type
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 8, 1> Vector8d;
typedef Eigen::Matrix<double, 8, 8> Matrix8d;

template <template <typename, typename> class Container, typename Type>
using Aligned = Container<Type, Eigen::aligned_allocator<Type>>;

template <typename KeyType, typename ValueType>
using AlignedMap =
    std::map<KeyType, ValueType, std::less<KeyType>,
             Eigen::aligned_allocator<std::pair<const KeyType, ValueType>>>;

template <typename KeyType, typename ValueType>
using AlignedUnorderedMap = std::unordered_map<
    KeyType, ValueType, std::hash<KeyType>, std::equal_to<KeyType>,
    Eigen::aligned_allocator<std::pair<const KeyType, ValueType>>>;

template <typename KeyType, typename ValueType>
using AlignedUnorderedMultimap = std::unordered_multimap<
    KeyType, ValueType, std::hash<KeyType>, std::equal_to<KeyType>,
    Eigen::aligned_allocator<std::pair<const KeyType, ValueType>>>;

template <typename Type>
using AlignedUnorderedSet =
    std::unordered_set<Type, std::hash<Type>, std::equal_to<Type>,
                       Eigen::aligned_allocator<Type>>;

void ConvertVectorToRt(Eigen::Matrix<double, 7, 1> &m, Eigen::Matrix3d &R,
                       Eigen::Vector3d &t);
double DescriptorDistance(const Eigen::Matrix<double, 256, 1> &f1,
                          const Eigen::Matrix<double, 256, 1> &f2);
cv::Scalar GenerateColor(int id);
void GenerateColor(int id, Eigen::Vector3d color);
cv::Mat DrawFeatures(const cv::Mat &image,
                     const std::vector<cv::KeyPoint> &keypoints,
                     const std::vector<bool> &inliers);

// files
void GetFileNames(std::string path, std::vector<std::string> &filenames);
bool FileExists(const std::string &file);
bool PathExists(const std::string &path);
void ConcatenateFolderAndFileName(const std::string &folder,
                                  const std::string &file_name,
                                  std::string *path);

std::string ConcatenateFolderAndFileName(const std::string &folder,
                                         const std::string &file_name);

void MakeDir(const std::string &path);


#endif // UTILS_H_