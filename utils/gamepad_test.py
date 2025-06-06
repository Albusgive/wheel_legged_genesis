import gamepad
import time
command_cfg = {
        "num_commands": 5,
        "base_range": 0.3,  #基础范围
        "lin_vel_x_range": [-1.2, 1.2], #修改范围要调整奖励权重
        "lin_vel_y_range": [-0.5, 0.5], 
        "ang_vel_range": [-6.28, 6.28],   #修改范围要调整奖励权重
        "leg_length_range": [0.0, 1.0],   #两条腿
}
pad = gamepad.control_gamepad(command_cfg,[1.2,0.5,6.0,0.03,0.03])

while True:
    commands, reset_flag = pad.get_commands()
    print(f"commands: {commands}")
    time.sleep(0.1)