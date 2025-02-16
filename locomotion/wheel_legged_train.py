import argparse
import os
import pickle
import shutil

from wheel_legged_env import WheelLeggedEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.995,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,    #每轮仿真多少step
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 6,
        # joint/link names
        "link_names":[
            "base_link",
            "left_calf_link",
            "left_hip_link",
            "left_thigh_link",
            "left_wheel_link",
            "right_calf_link",
            "right_hip_link",
            "right_thigh_link",
            "right_wheel_link",
        ],
        "default_joint_angles": {  # [rad]
            # "left_hip_joint":0.0,
            "left_thigh_joint": 0.0,
            "left_calf_joint": 0.0,
            # "right_hip_joint":0.0,
            "right_thigh_joint": 0.0,
            "right_calf_joint": 0.0,
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
        },
        "dof_names": [
            # "left_hip_joint",
            "left_thigh_joint",
            "left_calf_joint",
            # "right_hip_joint",
            "right_thigh_joint",
            "right_calf_joint",
            "left_wheel_joint",
            "right_wheel_joint",
        ],
        # lower upper
        "dof_limit": {
            # "left_hip_joint":[-0.31416, 0.31416],
            "left_thigh_joint": [-1.0472, 0.5236],
            "left_calf_joint": [0, 1.3963],   #[0.0, 1.3963]
            # "right_hip_joint":[-0.31416, 0.31416],
            "right_thigh_joint": [-1.0472, 0.5236],
            "right_calf_joint": [0, 1.3963],
            "left_wheel_joint": [0.0, 0.0],
            "right_wheel_joint": [0.0, 0.0],
        },
        "safe_force": {
            # "left_hip_joint":23.6,
            "left_thigh_joint": 40.0,
            "left_calf_joint": 40.0,
            # "right_hip_joint":23.6,
            "right_thigh_joint": 40.0,
            "right_calf_joint": 40.0,
            "left_wheel_joint": 12.0,
            "right_wheel_joint": 12.0,
        },
        # PD
        "kp": 30.0,
        "kd": 1.2,
        # termination 角度制    obs的angv弧度制
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 20, #15度以内都摆烂，会导致episode太短难以学习
        # "termination_if_base_height_greater_than": 0.1,
        # "termination_if_knee_height_greater_than": 0.00,
        "termination_if_base_connect_plane_than": True, #base触地重置
        # base pose
        "base_init_pos": [0.0, 0.0, 0.15],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 30.0,
        "resampling_time_s": 4.0,
        "joint_action_scale": 0.5,
        "wheel_action_scale": 10,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        # num_obs = num_slice_obs + history_num * num_slice_obs
        "num_obs": 87, #在rsl-rl中使用的变量为num_obs表示state数量
        "num_slice_obs": 29,
        "history_length": 2,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "dof_acc": 0.0025,
            "height_measurements": 5.0,
        },
    }
    # 名字和奖励函数名一一对应
    reward_cfg = {
        "tracking_lin_sigma": 0.25, 
        "tracking_lin_sigma2": 0.01, 
        "tracking_ang_sigma": 0.25, 
        "tracking_ang_sigma2": 0.01, 
        "tracking_height_sigma": 0.0015,
        "tracking_similar_legged_sigma": 0.5,
        "tracking_gravity_sigma": 0.01,
        "reward_scales": {
            "tracking_lin_vel": 0.5,
            "tracking_ang_vel": 0.5,
            "tracking_base_height": 1.0,
            "lin_vel_z": -0.02, #大了影响高度变换速度
            "joint_action_rate": -0.005,
            "wheel_action_rate": -0.00001,
            "similar_to_default": 0.0,
            "projected_gravity": 5.0,
            "similar_legged": 0.5, 
            "dof_vel": -5e-5,
            "dof_acc": -1.25e-8,
            "dof_force": -0.0001,
            "knee_height": -0.6,    #相当有效，和similar_legged结合可以抑制劈岔和跪地重启，稳定运行
            "ang_vel_xy": -0.05,
            "collision": -1.0,  #base接触地面碰撞力越大越惩罚
        },
    }
    command_cfg = {
        "num_commands": 4,
        "lin_vel_x_range": [-3.0, 3.0], #修改范围要调整奖励权重
        "lin_vel_y_range": [-0.0, 0.0],
        "ang_vel_range": [-6.28, 6.28],   #修改范围要调整奖励权重
        "height_target_range": [0.22 , 0.32],
    }
    # 课程学习，奖励循序渐进 待优化
    curriculum_cfg = {
        "curriculum_reward": {
            "tracking_base_height",
            "projected_gravity",
            "similar_legged", 
        },
        "curriculum_lin_vel_step":0.05,   #百分比
        "curriculum_ang_vel_step":0.05,   #百分比
    }
    #域随机化 friction_ratio是范围波动 mass和com是偏移波动
    domain_rand_cfg = { 
        "friction_ratio_range":[0.8 , 1.2],
        "random_mass_shift":0.2,
        "random_com_shift":0.05,
        "dof_damping_range":[0.0 , 0.0], # genesis bug
        "dof_stiffness_range":[0.0 , 0.0], # genesis bug
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="wheel-legged-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=10000)
    args = parser.parse_args()

    gs.init(logging_level="warning",backend=gs.vulkan)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = WheelLeggedEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, 
        command_cfg=command_cfg, curriculum_cfg=curriculum_cfg, domain_rand_cfg=domain_rand_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
