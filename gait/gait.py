import sys
import os
import torch
import mujoco
import mujoco.viewer
import time
import argparse
import pickle
import numpy as np
import math

# 加载 mujoco 模型
m = mujoco.MjModel.from_xml_path('../assets/mjcf/CJ-003/scence.xml')
d = mujoco.MjData(m)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils import gamepad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="wheel-legged-walking")
    args = parser.parse_args()

    # 拼接到 logs 文件夹的路径
    log_dir = os.path.join('../logs', args.exp_name)
    cfg_path = os.path.join(log_dir, 'cfgs.pkl')

    # 读取配置文件
    if os.path.exists(cfg_path):
        print("文件存在:", cfg_path)
        env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))
    else:
        print("文件不存在:", cfg_path)
        exit()

    cnt = 0
    T = 0.5 #周期
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            
            if cnt >= T:
                cnt = 0
            
            pos_in_cycle = cnt
            
            # 纯二次函数步态有点依托了
            # d.ctrl[0] = (-math.cos(math.pi * 2 * pos_in_cycle / T) + 1.0) * 0.2
            # d.ctrl[3] = (-math.cos(math.pi * 2 * pos_in_cycle / T) + 1.0) * -0.2
            
            # if pos_in_cycle > T/2:
            #     d.ctrl[1] = (-math.cos(math.pi * 4 / T * pos_in_cycle) + 1) * 0.25
            #     d.ctrl[2] = (-math.cos(math.pi * 4 / T * pos_in_cycle) + 1) * -0.5
            # else:
            #     d.ctrl[4] = (-math.cos(math.pi * 4 / T * pos_in_cycle) + 1) * 0.25
            #     d.ctrl[5] = (-math.cos(math.pi * 4 / T * pos_in_cycle) + 1) * -0.5
            
            #摆线轨迹
            
            
            cnt +=0.01

            # 执行一步模拟
            step_start = time.time()
            for i in range(5):
                mujoco.mj_step(m, d)
            # 更新渲染
            viewer.sync()
            # 同步时间
            time_until_next_step = m.opt.timestep*5 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
