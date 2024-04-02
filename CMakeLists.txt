cmake_minimum_required(VERSION 3.14)
project(ur_mvo)

set(CMAKE_CXX_STANDARD 17)

option(USE_ROS "COMPILE WITH ROS2" ON)

set(CMAKE_BUILD_TYPE "release")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
add_definitions(-w)

add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/TensorRTBuffer)

find_package(OpenCV 4.2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Boost REQUIRED)
find_package(G2O REQUIRED)
find_package(Gflags REQUIRED)
find_package(Glog REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(PythonLibs REQUIRED)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

if(USE_ROS)
  find_package(ament_cmake REQUIRED)
  find_package(ament_cmake_python REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(visualization_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(cv_bridge REQUIRED)
  find_package(image_transport REQUIRED)
  find_package(message_filters REQUIRED)
endif(USE_ROS)


include_directories(
  include
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIR}
  ${GFLAGS_INCLUDE_DIRS} 
  ${GLOG_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/Thirdparty/pybind11/include
)

if(USE_ROS)
  SET(ROS2_LIB
    rclcpp
    std_msgs
    sensor_msgs
    nav_msgs
    geometry_msgs
    cv_bridge
    image_transport
    message_filters
    visualization_msgs
    Eigen3
  )
endif(USE_ROS)

SET(UR_MVO_DEP
  nvinfer
  nvonnxparser
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${Boost_LIBRARIES}
  ${G2O_LIBRARIES}
  ${GFLAGS_LIBRARIES} 
  ${GLOG_LIBRARIES}
  yaml-cpp
  TensorRTBuffer
  pybind11::module
)

add_library(${PROJECT_NAME}_lib SHARED
  src/g2o_optimization.cc
  src/super_point.cpp
  src/super_glue.cpp
  src/utils.cc
  src/camera.cc
  src/dataset.cc
  src/frame.cc
  src/point_matching.cc
  src/mappoint.cc
  src/ros2_publisher.cc
  src/mapping.cc
  src/tracking.cc
  src/timer.cc
  src/epipolar_geometry.cc
)

target_include_directories(${PROJECT_NAME}_lib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME}_lib ${UR_MVO_DEP})

if(USE_ROS)
  ament_target_dependencies(${PROJECT_NAME}_lib ${ROS2_LIB})

  ament_export_dependencies(${ROS2_LIB})
  ament_export_include_directories(include)
  ament_export_targets(${PROJECT_NAME})
  ament_export_libraries(${PROJECT_NAME}_lib)

  install(
    DIRECTORY "include/"
    DESTINATION include
  )


  install(TARGETS
    ${PROJECT_NAME}_lib TensorRTBuffer
    EXPORT ${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
  

endif(USE_ROS)


if (Python_FOUND)
    ament_python_install_package(${PROJECT_NAME})
    pybind11_add_module(pyvo main_py.cpp)
    target_link_libraries(pyvo PRIVATE ${PROJECT_NAME}_lib)
    install(TARGETS pyvo DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME})
endif()


if(USE_ROS)
  ament_package()
endif(USE_ROS)
unset(USE_ROS CACHE)

