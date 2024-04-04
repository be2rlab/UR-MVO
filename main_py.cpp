#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include "read_configs.h"
#include "dataset.h"
#include "tracking.h"
#include <Eigen/Dense>
#include <vector>

namespace py = pybind11;
// enum class SensorSetup {Mono, Stereo, RGBD};

class UR_MVO{
public:
    UR_MVO(const py::dict& config_dict, const std::string& setup_) {
        setup = setup_;
        rclcpp::init(0, nullptr);
        processConfig(config_dict);
        Configs configs(config_path, setup);
        p_map_builder.push_back(std::unique_ptr<Tracking>(new Tracking(configs)));
    }

    Eigen::MatrixXd processMono(py::array_t<uint8_t>& img){
        Eigen::MatrixXd pose = Eigen::MatrixXd::Zero(4, 4);
        static size_t id = 0;
        auto t0 = std::chrono::steady_clock::now();

        auto rows = img.shape(0);
        auto cols = img.shape(1);
        auto type = CV_8UC1;
        
        cv::Mat image(rows, cols, type, (unsigned char*)img.data());
        
        InputDataPtr p_input_data = std::make_shared<InputData>();
        p_input_data->index = id;
        p_input_data->image = image;

        p_input_data->id_ts = id;
        p_input_data->time = id;

        if (p_input_data == nullptr) return pose;
        p_map_builder.back()->AddInput(p_input_data);

        auto t1 = std::chrono::steady_clock::now();
        auto t01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
        usleep(30000);
        if(p_map_builder.back()->gotResult()){
            pose = Eigen::Matrix4d::Identity();
            pose = p_map_builder.back()->getPose();
        }
        id++;
        return pose;
    }

    Eigen::MatrixXd processMonoWithMask(py::array_t<uint8_t>& img, py::array_t<uint8_t>& mask_){
        Eigen::MatrixXd pose = Eigen::MatrixXd::Zero(4, 4);
        static size_t id = 0;
        auto t0 = std::chrono::steady_clock::now();

        auto rows = img.shape(0);
        auto cols = img.shape(1);
        auto type = CV_8UC1;
        
        cv::Mat image(rows, cols, type, (unsigned char*)img.data());
        std::cout<<mask_.shape(0)<<std::endl;
        auto rows_mask = mask_.shape(0);
        auto cols_mask = mask_.shape(1);
        auto type_mask = CV_8UC1;
        std::cout << "rows_mask: " << rows_mask << std::endl;
        
        cv::Mat mask(rows_mask, cols_mask, type_mask, (unsigned char*)mask_.data());

        InputDataPtr p_input_data = std::make_shared<InputData>();
        p_input_data->index = id;
        p_input_data->image = image;

        
        if(p_map_builder.back()->use_mask()){
            p_input_data->mask = mask;
        }

        p_input_data->id_ts = id;
        p_input_data->time = id;

        if (p_input_data == nullptr) return pose;
        p_map_builder.back()->AddInput(p_input_data);

        auto t1 = std::chrono::steady_clock::now();
        auto t01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
        usleep(30000);
        if(p_map_builder.back()->gotResult()){
            pose = Eigen::Matrix4d::Identity();
            pose = p_map_builder.back()->getPose();
        }
        id++;
        return pose;
    }
    Eigen::MatrixXd processStereo(py::array_t<uint8_t>& img, py::array_t<uint8_t>& img_r){
        Eigen::MatrixXd pose = Eigen::MatrixXd::Zero(4, 4);
        static size_t id = 0;
        auto t0 = std::chrono::steady_clock::now();

        auto rows = img.shape(0);
        auto cols = img.shape(1);
        auto type = CV_8UC1;
        
        cv::Mat image(rows, cols, type, (unsigned char*)img.data());
        cv::Mat image_right(rows, cols, type, (unsigned char*)img_r.data());
        
        InputDataPtr p_input_data = std::make_shared<InputData>();
        p_input_data->index = id;
        p_input_data->image = image;
        p_input_data->image_right = image_right;

        p_input_data->id_ts = id;
        p_input_data->time = id;

        if (p_input_data == nullptr) return pose;
        p_map_builder.back()->AddInput(p_input_data);

        auto t1 = std::chrono::steady_clock::now();
        auto t01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
        std::cout << "Processing Time for Frame id " << id << ": " << t01 << " msec." << std::endl;
        usleep(30000);
        if(p_map_builder.back()->gotResult()){
            pose = Eigen::Matrix4d::Identity();
            pose = p_map_builder.back()->getPose();
        }
        id++;
        return pose;
    }

    Eigen::MatrixXd processStereoWithMask(py::array_t<uint8_t>& img, py::array_t<uint8_t>& img_r, py::array_t<uint8_t>& mask_){
        Eigen::MatrixXd pose = Eigen::MatrixXd::Zero(4, 4);
        static size_t id = 0;
        auto t0 = std::chrono::steady_clock::now();

        auto rows = img.shape(0);
        auto cols = img.shape(1);
        auto type = CV_8UC1;
        
        cv::Mat image(rows, cols, type, (unsigned char*)img.data());
        cv::Mat image_right(rows, cols, type, (unsigned char*)img_r.data());
        cv::Mat mask(rows, cols, type, (unsigned char*)mask_.data());
        
        InputDataPtr p_input_data = std::make_shared<InputData>();
        p_input_data->index = id;
        p_input_data->image = image;
        p_input_data->image_right = image_right;
        
        if(p_map_builder.back()->use_mask()){
            p_input_data->mask = mask;
        }

        p_input_data->id_ts = id;
        p_input_data->time = id;

        if (p_input_data == nullptr) return pose;
        p_map_builder.back()->AddInput(p_input_data);

        auto t1 = std::chrono::steady_clock::now();
        auto t01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
        std::cout << "Processing Time for Frame id " << id << ": " << t01 << " msec." << std::endl;
        usleep(30000);
        if(p_map_builder.back()->gotResult()){
            pose = Eigen::Matrix4d::Identity();
            pose = p_map_builder.back()->getPose();
        }
        id++;
        return pose;
    }

    void shutdown() {
    }
    // Todo(Jaafar): change depth to float
    Eigen::MatrixXd processRGBD(py::array_t<uint8_t>& img, py::array_t<uint8_t>& depth_){
        Eigen::MatrixXd pose = Eigen::MatrixXd::Zero(4, 4);
        static size_t id = 0;
        auto t0 = std::chrono::steady_clock::now();

        auto rows = img.shape(0);
        auto cols = img.shape(1);
        auto type = CV_8UC1;
        
        cv::Mat image(rows, cols, type, (unsigned char*)img.data());
        cv::Mat depth(rows, cols, type, (unsigned char*)depth_.data());
        
        InputDataPtr p_input_data = std::make_shared<InputData>();
        p_input_data->index = id;
        p_input_data->image = image;
        p_input_data->depth = depth;

        p_input_data->id_ts = id;
        p_input_data->time = id;

        if (p_input_data == nullptr) return pose;
        p_map_builder.back()->AddInput(p_input_data);

        auto t1 = std::chrono::steady_clock::now();
        auto t01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
        std::cout << "Processing Time for Frame id " << id << ": " << t01 << " msec." << std::endl;
        usleep(30000);
        if(p_map_builder.back()->gotResult()){
            pose = Eigen::Matrix4d::Identity();
            pose = p_map_builder.back()->getPose();
        }
        id++;
        return pose;
    }

    // Todo(Jaafar): change depth to float
    Eigen::MatrixXd processRGBDWithMask(py::array_t<uint8_t>& img, py::array_t<uint8_t>& depth_, py::array_t<uint8_t>& mask_){
        Eigen::MatrixXd pose = Eigen::MatrixXd::Zero(4, 4);
        static size_t id = 0;
        auto t0 = std::chrono::steady_clock::now();

        auto rows = img.shape(0);
        auto cols = img.shape(1);
        auto type = CV_8UC1;
        
        cv::Mat image(rows, cols, type, (unsigned char*)img.data());
        cv::Mat depth(rows, cols, type, (unsigned char*)depth_.data());
        cv::Mat mask(rows, cols, type, (unsigned char*)mask_.data());
        
        InputDataPtr p_input_data = std::make_shared<InputData>();
        p_input_data->index = id;
        p_input_data->image = image;
        p_input_data->depth = depth;
        
        if(p_map_builder.back()->use_mask()){
            p_input_data->mask = mask;
        }

        p_input_data->id_ts = id;
        p_input_data->time = id;

        if (p_input_data == nullptr) return pose;
        p_map_builder.back()->AddInput(p_input_data);

        auto t1 = std::chrono::steady_clock::now();
        auto t01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
        std::cout << "Processing Time for Frame id " << id << ": " << t01 << " msec." << std::endl;
        usleep(30000);
        if(p_map_builder.back()->gotResult()){
            pose = Eigen::Matrix4d::Identity();
            pose = p_map_builder.back()->getPose();
        }
        id++;
        return pose;
    }
    void reset(const py::dict& config_dict, const std::string& setup_) {
        setup = setup_;
        processConfig(config_dict);
        static size_t id_publisher = 0;
        
        Configs configs(config_path, setup);
        configs.ros_publisher_config.publisher_name = "ur_mvo_" + std::to_string(id_publisher);
        id_publisher++;
        p_map_builder.push_back(std::unique_ptr<Tracking>(new Tracking(configs)));
        
    }

private:
    void processConfig(const py::dict& config_dict){
        
        auto split = [](const std::string A){
            std::stringstream ss(A);
            std::vector<std::string> tokens;
            std::string token;
            while (std::getline(ss, token, '.')) 
                tokens.push_back(token);
            return tokens;
        };
        if(config_dict.contains("config_path")){
            std::cout<<"config_path found"<<std::endl;
            config_path = "/workspace/ros2_ws/src/ur_mvo/configs/" + config_dict["config_path"].cast<std::string>();
        }
        YAML::Node config = YAML::LoadFile(config_path);
        for (auto update : config_dict) {
            std::string key = update.first.cast<std::string>();
            std::string value = update.second.cast<std::string>();
            
            auto v = split(key);
            if(v.size() == 2){
                if (config[v[0]][v[1]]) {
                    config[v[0]][v[1]] = value;    
                } else {
                    std::cout << "Key '" << key << "' not found in the YAML file.\n";
                }
            } else {
                if(v.size()==3){
                    if (config[v[0]][v[1]][v[2]]) {
                        config[v[0]][v[1]][v[2]] = value;    
                    } else {
                        std::cout << "Key '" << key << "' not found in the YAML file.\n";
                    }
                } else {
                    std::cout << "Key '" << key << "' not found in the YAML file.\n";
                }
            }
        }
        std::ofstream file(config_path);
        file << config;
        file.close();        
    }
    std::string config_path = "/workspace/ros2_ws/src/ur_mvo/configs/configs_default.yaml";
    std::vector<std::shared_ptr<Tracking>> p_map_builder;
    std::string traj_path;
    std::string setup;
};

PYBIND11_MODULE(py_ur_mvo, m) {
    py::class_<UR_MVO>(m, "UR_MVO")
            .def(py::init<const py::dict&, const std::string&>())
            .def("processMono", &UR_MVO::processMono)
            .def("processMonoWithMask", &UR_MVO::processMonoWithMask)
            .def("processStereo", &UR_MVO::processStereo)
            .def("processStereoWithMask", &UR_MVO::processStereoWithMask)
            .def("processRGBD", &UR_MVO::processRGBD)
            .def("processRGBDWithMask", &UR_MVO::processRGBDWithMask)
            .def("shutdown", &UR_MVO::shutdown)
            .def("reset", &UR_MVO::reset);
}
