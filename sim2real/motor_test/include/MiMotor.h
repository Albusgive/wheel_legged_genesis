#pragma once
#include "Motor.hpp"
#include "PCANBasic.h"
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <cstdint>
#include <math.h>
#define P_MIN -12.5f
#define P_MAX 12.5f
#define V_MIN -30.0f
#define V_MAX 30.0f
#define KP_MIN 0.0f
#define KP_MAX 500.0f
#define KD_MIN 0.0f
#define KD_MAX 5.0f
#define T_MIN -12.0f
#define T_MAX 12.0f

enum class com_type {
  Loc = 1, // 运控模式
  MotorCallBack = 2,
  Enable = 3,            // 使能
  Stop = 4,              // 停止
  SetZero = 6,           // 设置电机零点
  SetCANID = 7,          // 设置电机ID
  SetFixParameter = 8,   // 掉电不消失
  SetParameter = 18,     // 掉电消失
  RequestParameter = 17, // 请求参数
}; // 通讯类型

// 通讯类型 SetParameter
enum class motor_indexs {
  /*-----------读写-----------*/
  run_mode = 0X7005,      // 0: 运控模式1: 位置模式 2: 速度模式 3: 电流模式
  iq_ref = 0X7006,        // 电流模式Iq指令
  spd_ref = 0X700A,       // 转速模式转速指令
  imit_torque = 0X700B,   // 转矩限制
  cur_kp = 0X7010,        // 电流的 Kp
  cur_ki = 0X7011,        // 电流的 Ki
  cur_filt_gain = 0X7014, // 电流滤波系数filt_gain
  loc_ref = 0X7016,       // 位置模式角度指令
  limit_spd = 0X7017,     // 位置模式速度设置
  limit_cur = 0X7018,     // 速度位置模式电流设置
  mechPos = 0x7019,       // 负载端计圈机械角度
  iqf = 0x701A,           // iq 滤波值
  mechVel = 0x701B,       // 负载端转速
  VBUS = 0x701C,          // 母线电压
  rotation = 0x701D,      // 圈数

}; // 电机功能码

// 通讯类型 SetFixParameter
enum class fix_parameter_indexs {
  /*-----------读写-----------*/
  Name = 0X0000,
  BarCode = 0X0001,
  MechPos_init = 0X2006,  // 初始多圈时的参考角度
  limit_torque = 0X2007,  // 转矩限制
  I_FW_MAX = 0X2008,      // 弱磁电流值，默认 0
  CAN_TIMEOUT = 0X200c,   // can 超时阈值，默认 0
  motorOverTemp = 0X200d, // 电机保护温度值，temp（度）*10
  overTempTime = 0X200e,  // 过温保护时间，time（s）
  GearRatio = 0X200f,     // 传动比 原装7.75
  cur_filt_gain = 0X2011, // 电流滤波系数filt_gain
  cur_kp = 0X2012,        // 电流 kp 0.025
  cur_ki = 0X2013,        // 电流 ki 0.0258
  spd_kp = 0X2014,        // 速度 kp 2
  spd_ki = 0X2015,        // 速度 ki 0.021
  loc_kp = 0X2016,        // 位置 kp 30
  spd_filt_gain = 0X2017, // 速度滤波参数 0.1
  limit_spd = 0X2018,     // 位置模式速度限制 2
  limit_cur = 0X2019,     // 位置、速度模式电流限制 23
}; // 电机功能码

// 电机反馈解码
struct MI_BACK {
  uint16_t motor_id : 8;
  uint16_t undervoltage : 1;
  uint16_t overcurrent : 1;
  uint16_t overtemperature : 1;
  uint16_t magnetic_encoding : 1;
  uint16_t HALL_encoding : 1;
  uint16_t not_calibrated : 1;
  uint16_t mode_state : 2;
};

// 扩展ID
struct MI_EXT_ID {
  uint32_t device_id : 8;
  uint32_t data : 16;
  uint32_t com_type : 5;
  uint32_t res : 3;
  void print() {
    std::cout << "  Device ID: " << device_id << std::endl;
    std::cout << "  Data: " << data << std::endl;
    std::cout << "  ComType: " << com_type << std::endl;
  }
  DWORD toEXTID() {
    return (com_type << 24) | (data << 8) | device_id;
    ;
  }
  MI_BACK *toMI_BACK() {
    static MI_BACK mi_back;
    mi_back.motor_id = data & 0xFF;
    mi_back.undervoltage = (data >> 8) & 0x01;
    mi_back.overcurrent = (data >> 9) & 0x01;
    mi_back.overtemperature = (data >> 10) & 0x01;
    mi_back.magnetic_encoding = (data >> 11) & 0x01;
    mi_back.HALL_encoding = (data >> 12) & 0x01;
    mi_back.not_calibrated = (data >> 13) & 0x01;
    mi_back.mode_state = (data >> 14) & 0x03;
    return &mi_back;
  }
};

// CAN数据
class MiCANMsg : public TPCANMsg {
public:
  MiCANMsg() = default;
  MiCANMsg(const TPCANMsg &base_msg) : TPCANMsg(base_msg) {}
  void print() {
    struct MI_EXT_ID *id_info = (struct MI_EXT_ID *)&this->ID;
    id_info->print();
    std::cout << "Data: ";
    for (size_t i = 0; i < 8; i++) {
      std::cout << " 0x" << std::hex << std::uppercase << std::setw(2)
                << std::setfill('0') << static_cast<int>(this->DATA[i]);
    }
    std::cout << std::dec << std::endl;
  }
  MI_EXT_ID *get_ext_id() { return (struct MI_EXT_ID *)&this->ID; }
};

class MiMotor : public Motor {
public:
  MiMotor();
  ~MiMotor();

  MiCANMsg *enableMotor(uint8_t motor_id, bool enable,
                        bool clear_fault = false) override;
  MotorBack decode(TPCANMsg msg) override;

  // 运动控制
  MiCANMsg *locomotion(uint8_t motor_id, float torque, float pos, float ang_vel,
                       float kp, float kd) override;
  // 位置PD
  MiCANMsg *setPosKP(uint8_t motor_id, float kp) override;
  MiCANMsg *setPosKD(uint8_t motor_id, float kd) override;
  // 速度PI
  MiCANMsg *setVelKP(uint8_t motor_id, float kp) override;
  MiCANMsg *setVelKI(uint8_t motor_id, float ki) override;
  // 扭矩/电流PI
  MiCANMsg *setTorqueKP(uint8_t motor_id, float kp) override;
  MiCANMsg *setTorqueKI(uint8_t motor_id, float ki) override;
  // 设置安全扭矩/电流
  MiCANMsg *setSafeTorque(uint8_t motor_id, float torque) override;
  MiCANMsg *setSafeVel(uint8_t motor_id, float vel) override;

  // 一般控制也在这里 iq_ref/spd_ref/loc_ref
  MiCANMsg *set_parameter(uint8_t motor_id, motor_indexs index,
                          float parameter);
  // 设置固定参数之后要确认参数，全部设置完之后确定就可以了
  MiCANMsg *set_fix_parameter(uint8_t motor_id, fix_parameter_indexs index,
                              float parameter);
  MiCANMsg *ok_fix_parameter(uint8_t motor_id);

  MiCANMsg *request_parameter(uint8_t motor_id, motor_indexs index);
  /*当前不可用 没写完*/
  // 设置id
  MiCANMsg *set_can_id(uint8_t motor_id, uint8_t id);
  MiCANMsg *set_zero_point(uint8_t motor_id);

  MiCANMsg *set_runmode(uint8_t motor_id, uint8_t runmode);

private:
  int float_to_uint(float x, float x_min, float x_max, int bits);
  MiCANMsg mi_can_msg;
};
