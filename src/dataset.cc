#include <fstream>
#include <math.h>

#include "dataset.h"
#include "ros2_publisher.h"
#include "utils.h"
#include <iostream>

Dataset::Dataset(const std::string &dataroot) {
  if (!PathExists(dataroot)) {
    std::cout << "dataroot : " << dataroot << " doesn't exist" << std::endl;
    exit(0);
  }

  std::string image_dir = ConcatenateFolderAndFileName(dataroot, "cam0/data");

  std::vector<std::string> image_names;
  GetFileNames(image_dir, image_names);
  if (image_names.size() < 1)
    return;
  std::sort(image_names.begin(), image_names.end());
  std::cout << "Images size : " << image_names.size() << std::endl;

  bool use_current_time = (image_names[0].size() < 18);
  for (std::string &image_name : image_names) {
    _images.emplace_back(ConcatenateFolderAndFileName(image_dir, image_name));
    if (!use_current_time) {
      double timestamp = atof(image_name.substr(0, 10).c_str()) +
                         atof(image_name.substr(10, 18).c_str()) / 1e9;
      _timestamps.emplace_back(timestamp);
    }
  }
}

size_t Dataset::GetDatasetLength() { return _images.size(); }

InputDataPtr Dataset::GetData(size_t idx) {
  if (idx >= _images.size())
    return nullptr;
  if (!FileExists(_images[idx]))
    return nullptr;

  InputDataPtr data = std::shared_ptr<InputData>(new InputData());
  data->index = idx;
  data->image = cv::imread(_images[idx], 0);
  if (_timestamps.empty()) {
    data->time = GetCurrentTime();
  } else {
    data->time = _timestamps[idx];
  }
  return data;
}
