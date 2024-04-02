#ifndef EPIPOLAR_GEOMETRY_H
#define EPIPOLAR_GEOMETRY_H

#include <Eigen/Core>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <unordered_set>

class EpipolarGeometry {
  typedef std::pair<int, int> Match;

public:
  /** @todo(Jaafar): Maybe add EIGEN_MAKE_ALIGNED_OPERATOR_NEW */
  /**
   * Constructor for Epipolar Geometry class
   * @param k : Camera Intrinsics
   * @param sigma
   * @param iterations : Number of iterations for Ransac.
   */
  EpipolarGeometry(const Eigen::Matrix3f &k, float sigma = 1.0,
                   int iterations = 200);

  /**
   * Reconstruct function
   * @param vKeys1 : Keypoints of reference frame
   * @param vKeys2 : Keypoints of second frame
   * @param vMatches12 : Matches between reference and candidate frame
   * @param T21 : Result of homogenious transformation
   * @param vP3D : Triangulated 3D points
   * @param vbTriangulated : Mask for points triangulated
   * @return Success boolean
   */
  bool reconstruct(const std::vector<cv::KeyPoint> &vKeys1,
                   const std::vector<cv::KeyPoint> &vKeys2,
                   const std::vector<int> vMatches12, Eigen::Matrix4f &T21,
                   std::vector<cv::Point3f> &vP3D,
                   std::vector<bool> &vbTriangulated);

  class Random {
  public:
    static bool already_seeded;

    static void seed_rand(int seed);

    static void seed_rand_once(int seed);

    static int RandomInt(int min, int max);
  };

private:
  /**
   * Find Homography
   * @param vbInliers : Mask for inliers for the model
   * @param score : score for the homography found
   * @param H21 : 3x3 homography
   */
  void _find_H(std::vector<bool> &vbInliers, float &score,
               Eigen::Matrix3f &H21);

  /**
   * Find Fundamental
   * @param vbInliers : Mask for inliers for the model
   * @param score : score for the fundamental found
   * @param F21 : 3x3 fundamental matrix
   */
  void _find_F(std::vector<bool> &vbInliers, float &score,
               Eigen::Matrix3f &F21);

  /**
   * Calculate Homography from Matches
   * @param vP1 : points of first scene
   * @param vP2 : points of second scene
   * @return Homography model
   */
  Eigen::Matrix3f _compute_H21(const std::vector<cv::Point2f> &vP1,
                               const std::vector<cv::Point2f> &vP2);

  /**
   * Calculate Fundamental from Matches
   * @param vP1 : points of first scene
   * @param vP2 : points of second scene
   * @return fundamental model
   */
  Eigen::Matrix3f _compute_F21(const std::vector<cv::Point2f> &vP1,
                               const std::vector<cv::Point2f> &vP2);

  /**
   * Score for Homography matrix
   * @param H21 Homography from scene 1 to 2
   * @param H12 Homography from scene 2 to 1
   * @param vbMatchesInliers Inliers mask
   * @param sigma tuned parameter
   * @return score
   */
  float _check_H(const Eigen::Matrix3f &H21, const Eigen::Matrix3f &H12,
                 std::vector<bool> &vbMatchesInliers, float sigma);

  /**
   * Score for Fundamental matrix
   * @param F21 fundamental from scene 1 to 2
   * @param vbMatchesInliers Inliers mask
   * @param sigma tuned parameter
   * @return score
   */
  float _check_F(const Eigen::Matrix3f &F21,
                 std::vector<bool> &vbMatchesInliers, float sigma);

  /**
   * Create homogenious transformation and triangulated points from Fundamental
   * matrix
   * @param vbMatchesInliers Matches inliers
   * @param F21 Fundametnal matrix
   * @param K Camera Intrinsics
   * @param T21 4x4 Transformation
   * @param vP3D 3D points
   * @param vbTriangulated mask for triangulated points
   * @param minParallax min parallax
   * @param minTriangulated threshold for triangulated points
   * @return success boolean
   */
  bool _reconstruct_F(std::vector<bool> &vbMatchesInliers, Eigen::Matrix3f &F21,
                      Eigen::Matrix3f &K, Eigen::Matrix4f &T21,
                      std::vector<cv::Point3f> &vP3D,
                      std::vector<bool> &vbTriangulated, float minParallax,
                      int minTriangulated);

  /**
   * Create homogenious transformation and triangulated points from Homography
   * matrix
   * @param vbMatchesInliers Matches inliers
   * @param H21 Homography matrix
   * @param K Camera Intrinsics
   * @param T21 4x4 Transformation
   * @param vP3D 3D points
   * @param vbTriangulated mask for triangulated points
   * @param minParallax min parallax
   * @param minTriangulated threshold for triangulated points
   * @return success boolean
   */
  bool _reconstruct_H(std::vector<bool> &vbMatchesInliers, Eigen::Matrix3f &H21,
                      Eigen::Matrix3f &K, Eigen::Matrix4f &T21,
                      std::vector<cv::Point3f> &vP3D,
                      std::vector<bool> &vbTriangulated, float minParallax,
                      int minTriangulated);

  /**
   * Normalize key points
   * @param vKeys Keypoints
   * @param vNormalizedPoints Normalized key points
   * @param T scale matrix
   */
  void _normalize(const std::vector<cv::KeyPoint> &vKeys,
                  std::vector<cv::Point2f> &vNormalizedPoints,
                  Eigen::Matrix3f &T);

  /**
   * Check Rotation and translation
   * @param R Rotation Matrix
   * @param t Translation matrix
   * @param vKeys1 first Vector of key points
   * @param vKeys2 scond Vector of key points
   * @param vMatches12 mathces
   * @param vbMatchesInliers Mask for matches inliers
   * @param K Camera intrinsics
   * @param vP3D Vector of 3D points
   * @param th2 threshold
   * @param vbGood mask for good points
   * @param parallax desired parallax
   * @return number of inliers
   */
  int _check_R_T(const Eigen::Matrix3f &R, const Eigen::Vector3f &t,
                 const std::vector<cv::KeyPoint> &vKeys1,
                 const std::vector<cv::KeyPoint> &vKeys2,
                 const std::vector<Match> &vMatches12,
                 std::vector<bool> &vbMatchesInliers, const Eigen::Matrix3f &K,
                 std::vector<cv::Point3f> &vP3D, float th2,
                 std::vector<bool> &vbGood, float &parallax);

  /**
   * Decomposition of essential matrix
   * @param E Essential Matrix
   * @param R1 First Rotation matrix
   * @param R2 Second Rotation matrix
   * @param t translation
   */
  void _decompose_E(const Eigen::Matrix3f &E, Eigen::Matrix3f &R1,
                    Eigen::Matrix3f &R2, Eigen::Vector3f &t);

  /**
   * Triangulation function
   * @param x_c1 point in camera 1
   * @param x_c2 point in camera 2
   * @param Tc1w pose of camera 1
   * @param Tc2w pose of camera 2
   * @param x3D 3D point
   * @return success Boolean
   */
  bool _triangulate(Eigen::Vector3f &x_c1, Eigen::Vector3f &x_c2,
                    Eigen::Matrix<float, 3, 4> &Tc1w,
                    Eigen::Matrix<float, 3, 4> &Tc2w, Eigen::Vector3f &x3D);

  // Keypoints from Reference Frame (Frame 1)
  std::vector<cv::KeyPoint> _vKeys1;

  // Keypoints from Current Frame (Frame 2)
  std::vector<cv::KeyPoint> _vKeys2;

  // Current Matches from Reference to Current
  std::vector<Match> _vMatches12;
  std::vector<bool> _vbMatched1;

  // Calibration
  Eigen::Matrix3f _K;

  // Standard Deviation and Variance
  float _Sigma, _Sigma2;

  // Ransac max iterations
  int _MaxIterations;

  // Ransac sets
  std::vector<std::vector<size_t>> _vSets;
};

#endif // EPIPOLAR_GEOMETRY_H
