#include "MiMotor.h"
#include "PCAN.hpp"
int main() {
  PCAN pcan;
  auto channel = CAN1;
  if (!pcan.initPCAN(channel, BAUD_1MBPS))
    std::cout << "init pcan false" << std::endl;

  MiMotor mi;
  int motor_id = 5;
  // pcan.send(channel, mi.set_can_id(15, motor_id));

  TPCANMsg motor_enable = *mi.enableMotor(motor_id, true);
  pcan.send(channel, motor_enable);

  // auto loc = *mi.locomotion(motor_id, 0.0, 0.0, 1.0, 0.0, 0.1);
  // pcan.send(channel, loc);

  auto send_msg = *mi.set_runmode(motor_id, 2);
  pcan.send(channel, send_msg);

  auto send_speed = *mi.set_parameter(motor_id, motor_indexs::spd_ref, -2.0);
  send_speed.print();
  pcan.send(channel, send_speed);

  //   send_msg = mi.set_parameter(motor_id, motor_indexs::limit_cur, 23);
  //   pcan.send(channel, send_msg);
  //   std::this_thread::sleep_for(std::chrono::milliseconds(2000));

  for (int i = 0; i < 50; i++) {
    auto [is_read, can_msg] = pcan.read(channel);
    if (is_read) {
      // MiCANMsg(can_msg).print();
      auto decode = mi.decode(can_msg);
      decode.print();
      decode.invertMotor()->print();
    }
    // pcan.send(channel, loc);
    pcan.send(channel, send_speed);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  pcan.send(channel, *mi.enableMotor(motor_id, false));
  return 0;
}