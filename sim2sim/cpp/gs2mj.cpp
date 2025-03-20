// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>
#include <cstring>

#include <GLFW/glfw3.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mujoco/mujoco.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "sim_cfg.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <gamepad.h>

#include <chrono>
#include <thread>

auto device = torch::kCUDA;

// model input & output
ENV_CFG env_cfg;
OBS_SCALE obs_sacle;
ACTION_CFG action_cfg;
CircularBuffer history_and_now_obs_buf(env_cfg.history_length + 1);
std::vector<float> actions(env_cfg.num_actions, 0);
std::vector<float> obs_actions(env_cfg.num_actions, 0);
std::vector<float> commands(env_cfg.num_commands, 0);

// MuJoCo data structures
mjModel *m = NULL; // MuJoCo model
mjData *d = NULL;  // MuJoCo data
mjvCamera cam;     // abstract cam
mjvOption opt;     // visualization options
mjvScene scn;      // abstract scene
mjrContext con;    // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

// keyboard callback
void keyboard(GLFWwindow *window, int key, int scancode, int act, int mods) {
  // backspace: reset simulation
  if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
    mj_resetData(m, d);
    mj_forward(m, d);
  }
}

// mouse button callback
void mouse_button(GLFWwindow *window, int button, int act, int mods) {
  // update button state
  button_left =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
  button_middle =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
  button_right =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

  // update mouse position
  glfwGetCursorPos(window, &lastx, &lasty);
}

// mouse move callback
void mouse_move(GLFWwindow *window, double xpos, double ypos) {
  // no buttons down: nothing to do
  if (!button_left && !button_middle && !button_right) {
    return;
  }

  // compute mouse displacement, save
  double dx = xpos - lastx;
  double dy = ypos - lasty;
  lastx = xpos;
  lasty = ypos;

  // get current window size
  int width, height;
  glfwGetWindowSize(window, &width, &height);

  // get shift key state
  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

  // determine action based on mouse button
  mjtMouse action;
  if (button_right) {
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  } else if (button_left) {
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  } else {
    action = mjMOUSE_ZOOM;
  }

  // move cam
  mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}

// scroll callback
void scroll(GLFWwindow *window, double xoffset, double yoffset) {
  // emulate vertical mouse motion = 5% of window height
  mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

std::vector<float> get_sensor_data(const mjModel *model, const mjData *data,
                                   const std::string &sensor_name) {
  int sensor_id = mj_name2id(model, mjOBJ_SENSOR, sensor_name.c_str());
  if (sensor_id == -1) {
    std::cout << "no found sensor" << std::endl;
    return std::vector<float>();
  }
  int data_pos = 0;
  for (int i = 0; i < sensor_id; i++) {
    data_pos += model->sensor_dim[i];
  }
  std::vector<float> sensor_data(model->sensor_dim[sensor_id]);
  for (int i = 0; i < sensor_data.size(); i++) {
    sensor_data[i] = data->sensordata[data_pos + i];
  }
  return sensor_data;
}

// v'=q^-1*v*q
std::vector<float> world2self(std::vector<float> &quat, std::vector<float> v) {
  Eigen::Quaterniond q(quat[0], quat[1], quat[2], quat[3]);
  Eigen::Vector3d v_vec(v[0], v[1], v[2]);
  float q_w = q.w();
  Eigen::Vector3d q_vec = q.vec();
  Eigen::Vector3d a = v_vec * (2.0 * q_w * q_w - 1.0);
  Eigen::Vector3d b = q_vec.cross(v_vec) * q_w * 2.0;
  Eigen::Vector3d c = q_vec * (q_vec.dot(v_vec)) * 2.0;
  Eigen::Vector3d result = a - b + c;
  std::vector<float> world_angle_speed = {static_cast<float>(result.x()),
                                          static_cast<float>(result.y()),
                                          static_cast<float>(result.z())};
  return world_angle_speed;
}

enum class Color {
  Red = 31,
  Green = 32,
  Yellow = 33,
  Blue = 34,
  Magenta = 35,
  Cyan = 36,
  White = 37,
  Reset = 0
};

// 打印向量，支持选择颜色
template <typename T>
void cout_vector(const T &data, const std::string &name,
                 Color color = Color::Reset) {
  // 选择颜色并打印
  std::cout << "\033[" << static_cast<int>(color) << "m" << name << ": ";
  std::cout << std::fixed << std::setprecision(6) << std::endl;

  for (const auto &i : data) {
    std::cout << i << " ";
  }

  // 重置颜色
  std::cout << "\033[" << static_cast<int>(Color::Reset) << "m" << std::endl;
}

void compute_ctrl(std::vector<float> act) {
  // 缩放
  actions[0] = act[0] * action_cfg.joint_action_scale;
  actions[1] = act[1] * action_cfg.joint_action_scale;
  actions[2] = act[2] * action_cfg.joint_action_scale;
  actions[3] = act[3] * action_cfg.joint_action_scale;
  actions[4] = act[4] * action_cfg.wheel_action_scale;
  actions[5] = act[5] * action_cfg.wheel_action_scale;
  // 裁减
  for (int i = 0; i < env_cfg.min_joint_angles.size(); i++) {
    if (actions[i] < env_cfg.min_joint_angles[i])
      actions[i] = env_cfg.min_joint_angles[i];
  }
  for (int i = 0; i < env_cfg.max_joint_angles.size(); i++) {
    if (actions[i] > env_cfg.max_joint_angles[i])
      actions[i] = env_cfg.max_joint_angles[i];
  }
}

std::vector<float> compute_observations(std::vector<float> commands) {
  std::vector<float> obs;

  auto base_quat = get_sensor_data(m, d, "orientation");

  // num 3
  auto base_lin_vel = get_sensor_data(m, d, "base_lin_vel");
  // base_lin_vel = world2self(base_quat, base_lin_vel);
  // for (int i = 0; i < base_lin_vel.size(); i++) {
  //   base_lin_vel[i] *= obs_sacle.lin_vel;
  //   obs.push_back(base_lin_vel[i]);
  // }
  // cout_vector(base_lin_vel, "base_lin_vel", Color::Green);

  // num 3
  auto base_ang_vel = get_sensor_data(m, d, "base_ang_vel");
  for (int i = 0; i < (int)base_ang_vel.size(); i++) {
    base_ang_vel[i] *= obs_sacle.ang_vel;
    obs.push_back(base_ang_vel[i]);
  }
  cout_vector(base_ang_vel, "base_ang_vel", Color::Blue);

  // 并非加速度计 num 3
  std::vector<float> gravity_vec = {0.0, 0.0, -1.0};
  auto projected_gravity = world2self(base_quat, gravity_vec);
  for (auto &i : projected_gravity) {
    obs.push_back(i);
  }

  // command---------- num 4
  for (int i = 0; i < (int)obs_sacle.command_scale.size(); i++) {
    obs.push_back(commands[i] * obs_sacle.command_scale[i]);
  }

  // dof_pos num 4
  std::vector<float> dof_pos;
  for (int i = 0; i < env_cfg.num_actions - 2; i++) {
    auto dof_p = get_sensor_data(m, d, env_cfg.dof_names[i] + "_p");
    dof_pos.push_back(dof_p[0]);
    obs.push_back((dof_p[0] - env_cfg.default_joint_angles[i]) *
                  obs_sacle.dof_pos);
  }
  cout_vector(dof_pos, "dof_pos", Color::Blue);

  // dof_vel num 6
  std::vector<float> dof_vel;
  for (int i = 0; i < env_cfg.num_actions; i++) {
    auto dof_v = get_sensor_data(m, d, env_cfg.dof_names[i] + "_v");
    dof_vel.push_back(dof_v[0]);
    obs.push_back(dof_v[0] * obs_sacle.dof_vel);
  }
  cout_vector(dof_vel, "dof_vel", Color::Blue);

  // action num 6
  for (auto &i : obs_actions)
    obs.push_back(i);

  return obs;
}

// main function
int main(int argc, const char **argv) {

  // load and compile model
  char error[1000] = "Could not load binary model";
  m = mj_loadXML("../../scence.xml", 0, error, 1000);

  // make data
  d = mj_makeData(m);

  // init GLFW
  if (!glfwInit()) {
    mju_error("Could not initialize GLFW");
  }

  // create window, make OpenGL context current, request v-sync
  GLFWwindow *window = glfwCreateWindow(640, 480, "Demo", NULL, NULL);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  // initialize visualization data structures
  mjv_defaultCamera(&cam);
  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn);
  mjr_defaultContext(&con);

  // create scene and context
  mjv_makeScene(m, &scn, 4000);
  mjr_makeContext(m, &con, mjFONTSCALE_150);

  // install GLFW mouse and keyboard callbacks
  glfwSetKeyCallback(window, keyboard);
  glfwSetCursorPosCallback(window, mouse_move);
  glfwSetMouseButtonCallback(window, mouse_button);
  glfwSetScrollCallback(window, scroll);

  //---------------------------jit-------------------------------
  mj_step(m, d);
  for (int i = 0; i < env_cfg.history_length; i++) {
    history_and_now_obs_buf.push_back(compute_observations(commands));
    mj_step(m, d);
  }
  std::string log_name = "wheel-legged-walking";
  ;
  std::string model_path = "../../../logs/" + log_name + "/policy.pt";
  torch::jit::script::Module module =
      torch::jit::load(model_path.c_str(), device);

  //---------------------------gamepad-------------------------------
  GamePad pad;
  pad.showGamePads();
  if (pad.GamePadpads.empty()) {
    std::cout << "No gamepads connected" << std::endl;
    return 0;
  }
  std::vector<float> gamepad_scale = {1.0, 1.0, 3.14, 0.05};
  pad.bindGamePadValues([&](GamePadValues map) {
    // 前ly为- 左lx为- 左转rx为-
    commands[0] = -(float)map.ly / 32767.0 * gamepad_scale[0];
    commands[1] = 0;
    commands[2] = -(float)map.rx / 32767.0 * gamepad_scale[2];
    commands[3] += (float)(map.lt + 32767) / 65535.0 * gamepad_scale[3];
    commands[3] -= (float)(map.rt + 32767) / 65535.0 * gamepad_scale[3];
    if (commands[3] > 0.32) {
      commands[3] = 0.32;
    } else if (commands[3] < 0.16) {
      commands[3] = 0.16;
    }
    if (map.x) {
      mj_resetData(m, d);
      // module = torch::jit::load(model_path.c_str(), device);
    }
  });
  int is;
  std::string opid = pad.GamePadpads.begin()->first;
  std::cout << "first gamepad id is " << opid << std::endl;
  if (pad.GamePadpads.size() > 1) {
    std::cout << "more than one gamepad" << std::endl;
    while (true) {
      std::cout << "please input the gamepad id" << std::endl;
      std::cin >> opid;
      is = pad.openGamePad(opid);
      if (is >= 0) {
        break;
      }
    }
  } else {
    is = pad.openGamePad(opid);
    if (is < 0) {
      std::cout << "open gamepad fail" << std::endl;
      return 0;
    }
  }
  pad.readGamePad();
  //---------------------------固定相机视角-------------------------------
  int target_body_id = mj_name2id(m, mjOBJ_BODY, "nz");
  if (target_body_id == -1) {
    std::cerr << "Body not found!" << std::endl;
    return -1;
  }
  // 摄像头的背后距离和高度
  double camera_distance = 3.0; // 摄像头离目标物体的距离
  double camera_height = -30.0; // 摄像头的高度，通常可以略高于目标物体
  double cam_azimuth = 0.0;

  while (!glfwWindowShouldClose(window)) {
    auto slice_obs_buf = compute_observations(commands);
    history_and_now_obs_buf.push_back(slice_obs_buf);

    auto obs_buf = history_and_now_obs_buf.get_all();
    torch::Tensor obs =
        torch::from_blob(obs_buf.data(), {static_cast<long>(obs_buf.size())},
                         torch::kFloat32)
            .to(device);
    torch::jit::Stack inputs;
    inputs.push_back(obs);

    // auto test_obs = torch::zeros(87).to(device);
    // inputs.push_back(test_obs);

    // std::cout << obs << std::endl;
    // std::cout << "obs_buf size: " << obs_buf.size() << std::endl;
    // std::cout << "Tensor shape: " << obs.sizes() << std::endl;
    // cout_vector(obs_buf, "obs_buf");

    auto start = std::chrono::high_resolution_clock::now();
    auto output_tensor = module.forward(std::move(inputs)).toTensor().cpu();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;
    // std::cout << "Time taken: " << duration_ms.count() << " ms." <<
    // std::endl;

    // std::cout<<output_tensor<<std::endl;
    // cout_vector(slice_obs_buf, "slice_obs_buf",Color::Green);

    // 裁减action 观测和输出前
    output_tensor = torch::clip(output_tensor, -action_cfg.clip_actions,
                                action_cfg.clip_actions);
    std::vector<float> vec(output_tensor.data_ptr<float>(),
                           output_tensor.data_ptr<float>() +
                               output_tensor.numel());

    // vec = {-0.1131,  1.3038, -0.1887,  1.2092,  0.0219,  0.0375};
    // vec = {1.0,1.0,1.0,1.0,1.0,1.0};
    obs_actions = vec;
    compute_ctrl(vec);
    // actions={-0.5,0.7,-0.5,0.7,0.0,0.0};
    for (int i = 0; i < env_cfg.num_actions; i++) {
      d->ctrl[i] = actions[i];
    }
    cout_vector(commands, "commands", Color::Red);
    cout_vector(actions, "actions", Color::Green);
    // cout_vector(commands, "commands",Color::Red);

    //同步时间
    auto step_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5; i++) // timestep 当前设置0.002
      mj_step(m, d);

    // ---------------------look at ------------------------
    // 获取目标物体的位置和朝向
    auto target_position = get_sensor_data(m, d, "base_pos");
    auto target_orientation = get_sensor_data(m, d, "orientation");
    // 计算yaw角度
    double yaw =
        atan2(2.0 * (target_orientation[0] * target_orientation[3] +
                     target_orientation[1] * target_orientation[2]),
              1.0 - 2.0 * (target_orientation[1] * target_orientation[1] +
                           target_orientation[2] * target_orientation[2]));
    cam_azimuth = yaw / M_PI * 180.0;
    // std::cout << "azimuth: " << cam.azimuth << std::endl;
    // 设置摄像头的目标位置（lookat）
    cam.lookat[0] = target_position[0];
    cam.lookat[1] = target_position[1];
    cam.lookat[2] = target_position[2];
    // cam.elevation = target_position[2] + camera_height; // 相机海拔
    // cam.azimuth = cam_azimuth;                          // 相机方位角
    // cam.distance = camera_distance;                     //
    // 设置相机和物体的距离
    // ---------------------look at ------------------------

    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    // update scene and render
    mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);
    // swap OpenGL buffers (blocking call due to v-sync)
    glfwSwapBuffers(window);
    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();

    //同步时间
    auto current_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec =
        std::chrono::duration<double>(current_time - step_start).count();
    double time_until_next_step = m->opt.timestep*5 - elapsed_sec;
    if (time_until_next_step > 0.0) {
      auto sleep_duration = std::chrono::duration<double>(time_until_next_step);
      std::this_thread::sleep_for(sleep_duration);
    }
  }

  // free visualization storage
  mjv_freeScene(&scn);
  mjr_freeContext(&con);

  // free MuJoCo model and data
  mj_deleteData(d);
  mj_deleteModel(m);

  // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
  glfwTerminate();
#endif

  return 1;
}
