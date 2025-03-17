#include "DMMotor.h"
#include "PCAN.hpp"
int main() {
  PCAN pcan;
  auto channel = CAN1;
  if (!pcan.initPCAN(channel, BAUD_1MBPS))
    std::cout << "init pcan false" << std::endl;
  DMMotor dm;
  int motor_id = 0x03;
  
  pcan.send(channel, *dm.enableMotor(motor_id, true));

  auto loc = *dm.locomotion(motor_id, 0.0, 0, 0.0, 0.0, 0.0);
  for (int i = 0; i < 100; i++) {
    auto [is_read, can_msg] = pcan.read(channel);
    if (is_read) {
      // DMCANMsg(can_msg).print();
      auto decode = dm.decode(can_msg);
      decode.print();
    }
    pcan.send(channel, loc);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  pcan.send(channel, *dm.enableMotor(motor_id, false));

  return 0;
}
