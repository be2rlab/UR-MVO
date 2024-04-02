#ifndef G2O_OPTIMIZATION_H_
#define G2O_OPTIMIZATION_H_

#include <vector>

#include "read_configs.h"
#include "camera.h"
#include "frame.h"
#include "mappoint.h"
#include "types.h"


void LocalmapOptimization(MapOfPoses& poses, MapOfPoints3d& points, std::vector<CameraPtr>& camera_list, 
    VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints,
    const OptimizationConfig& cfg);

int FrameOptimization(MapOfPoses& poses, MapOfPoints3d& points, std::vector<CameraPtr>& camera_list, 
    VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints,
    const OptimizationConfig& cfg);

int SolvePnPWithCV(FramePtr frame, std::vector<MappointPtr>& mappoints, Eigen::Matrix4d& pose, std::vector<int>& inliers);

#endif  // G2O_OPTIMIZATION_H_