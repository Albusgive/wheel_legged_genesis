import argparse
import os
import pickle

import torch
from wheel_legged_env import WheelLeggedEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import gamepad
import copy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="wheel-legged-walking")
    parser.add_argument("--ckpt", type=int, default=7300)
    args = parser.parse_args()
    
    gs.init(backend=gs.gpu,logging_level="warning")
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    # env_cfg["simulate_action_latency"] = False
    terrain_cfg["terrain"] = True
    terrain_cfg["eval"] = "agent_eval_gym" #agent_eval_gym/agent_train_gym/circular
    # env_cfg["kp"] = 40
    # env_cfg["wheel_action_scale"] = 5
    # env_cfg["joint_damping"] = 0
    # env_cfg["wheel_damping"] = 0
    env = WheelLeggedEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        curriculum_cfg=curriculum_cfg,
        domain_rand_cfg=domain_rand_cfg,
        terrain_cfg=terrain_cfg,
        robot_morphs="urdf",
        show_viewer=True,
        train_mode=False
    )
    print(reward_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")
    #jit
    model = copy.deepcopy(runner.alg.actor_critic.actor).to('cpu')
    torch.jit.script(model).save(log_dir+"/policy.pt")
    # 加载模型进行测试
    print("\n--- 模型加载测试 ---")
    try:
        loaded_policy = torch.jit.load(log_dir + "/policy.pt")
        loaded_policy.eval() # 设置为评估模式
        loaded_policy.to('cuda')
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit()
    obs, _ = env.reset()
    pad = gamepad.control_gamepad(command_cfg,[1.0,1.0,3.14,0.005])
    with torch.no_grad():
        while True:
            # actions = policy(obs)
            actions = loaded_policy(obs)
            obs, _, rews, dones, infos = env.step(actions)
            comands,reset_flag = pad.get_commands()
            print(f"comands: {comands}")
            env.set_commands(0,comands)
            if reset_flag:
                env.reset()
            


if __name__ == "__main__":
    main()
