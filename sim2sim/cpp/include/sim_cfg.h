#pragma once
#include <deque>
#include <iostream>
#include <string>
#include <vector>
class ENV_CFG {
public:
  int num_actions = 6;
  int history_length = 5;
  // int num_obs = 174;
  // int num_slice_obs = 29;
  int num_commands = 4;
  std::vector<float> default_joint_angles;
  std::vector<float> min_joint_angles;
  std::vector<float> max_joint_angles;
  std::vector<std::string> dof_names;
  ENV_CFG()
      : default_joint_angles({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
        min_joint_angles({-1.0472,0,-1.0472,0,-100,-100}),
        max_joint_angles({0.5236,1.3963,0.5236,1.3963,100,100}), dof_names({"left_thigh_joint", "left_calf_joint", "right_thigh_joint",
                   "right_calf_joint", "left_wheel_joint",
                   "right_wheel_joint"}) {}
};

class OBS_SCALE {
public:
  float lin_vel = 2.0;
  float ang_vel = 0.5;
  float dof_pos = 1.0;
  float dof_vel = 0.05;
  // command scales
  float height_measurements = 5.0;
  std::vector<float> command_scale;
  OBS_SCALE()
      : command_scale({lin_vel, lin_vel, ang_vel, height_measurements}) {}
};

class ACTION_CFG {
public:
  float joint_action_scale = 0.5; //0.5
  float wheel_action_scale = 1.5; //10
  float clip_actions = 100;
  float kp = 30.0;
  float kd = 1.2;
};

class CircularBuffer {
public:
  CircularBuffer(size_t max_size) : max_size(max_size) {}

  void push_back(const std::vector<float> &value) {
    if (buffer.size() == max_size) {
      buffer.pop_front(); // 如果超过容量，从前端删除最早的元素
    }
    buffer.push_back(value);
  }

  // 获取所有元素并将它们合并为一个vector
  std::vector<float> get_all() const {
    std::vector<float> all_elements;
    for (const auto &vec : buffer) {
      all_elements.insert(all_elements.end(), vec.begin(), vec.end());
    }
    return all_elements;
  }

private:
  std::deque<std::vector<float>> buffer; // 仅支持 std::vector<doubfloatle>
  size_t max_size;
};