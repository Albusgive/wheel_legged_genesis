#pragma once
#include "magic_enum/magic_enum.hpp"
#include <iostream>
#include <string>

#define RESET "\033[0m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define BOLD "\033[1m"

enum class MotorWarning {
  Normal = 0,
  HighTemperature,
  Overcurrent,
};

class Motor {
public:
  int id;
  float angle;
  float number_laps; // 圈数
  float ang_vel;
  float torque;
  float current;
  float temperature; // 摄氏度
  MotorWarning warning = MotorWarning::Normal;
  void print() {
    std::cout << "ID: " << id << std::endl;
    std::cout << "Angle: " << angle << std::endl;
    std::cout << "Angular Velocity: " << ang_vel << std::endl;
    std::cout << "Torque: " << torque << std::endl;
    std::cout << "Current: " << current << std::endl;
    std::cout << "Temperature: " << temperature << std::endl;
    if (warning == MotorWarning::Normal) {
      std::cout<< GREEN;
    } else {
        std::cout<< YELLOW;
    }
    std::cout << "Motor: " << std::string(magic_enum::enum_name(warning))<<RESET << std::endl;
  }
};