import argparse
import os
import pickle
import shutil

from wheel_legged_env import WheelLeggedEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs # type: ignore


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.005,
            "entropy_coef": 0.002,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 3,
            "num_mini_batches": 5,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128, 64],
            "critic_hidden_dims": [256, 128, 64],
            "init_noise_std": 1.5,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 30,    #每轮仿真多少step
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
        # joint names
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
        "joint_init_angles": {  # [rad]
            # "left_hip_joint":0.0,
            "left_thigh_joint": 0.6,
            "left_calf_joint": 0.0,
            # "right_hip_joint":0.0,
            "right_thigh_joint": 0.6,
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
            "left_thigh_joint": [-0.436332313, 1.221730476],
            "left_calf_joint": [0, 1.3],   #[0.0, 1.3963]
            # "right_hip_joint":[-0.31416, 0.31416],
            "right_thigh_joint": [-0.436332313, 1.221730476],
            "right_calf_joint": [0, 1.3],
            "left_wheel_joint": [0.0, 0.0],
            "right_wheel_joint": [0.0, 0.0],
        },
        "safe_force": {
            # "left_hip_joint":23.6,
            "left_thigh_joint": 20.0,
            "left_calf_joint": 20.0,
            # "right_hip_joint":23.6,
            "right_thigh_joint": 20.0,
            "right_calf_joint": 20.0,
            "left_wheel_joint": 8.0, #给大点，测试的时候可以小
            "right_wheel_joint": 8.0,
        },
        # PD
        "joint_kp": 20.0,
        "joint_kv": 1.0,
        "wheel_kv": 1.0,
        "damping": 0.005,
        "stiffness":0.0, #不包含轮
        "armature":0.004,
        # termination 角度制    obs的angv弧度制
        "termination_if_roll_greater_than": 20,  # degree
        "termination_if_pitch_greater_than": 25, #15度以内都摆烂，会导致episode太短难以学习
        # "termination_if_base_height_greater_than": 0.1,
        # "termination_if_knee_height_greater_than": 0.00,
        "termination_if_base_connect_plane_than": True, #触地重置
        "connect_plane_links":[ #触地重置link
            "base_link",
            "left_calf_link",
            "left_thigh_link",
            "left_knee_link",
            "right_calf_link",
            "right_thigh_link",
            "right_knee_link",
                ],
        # base pose
        "base_init_pos":{
            "urdf":[0.0, 0.0, 0.2],#稍微高一点点
            "mjcf":[0.0, 0.0, 0.285],
            },
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],#0.996195, 0, 0.0871557, 0
        "episode_length_s": 20.0,
        "resampling_time_s": 3.0,
        "joint_action_scale": 0.5,
        "wheel_action_scale": 10,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        # num_obs = num_slice_obs + history_num * num_slice_obs
        "num_obs": 156, #在rsl-rl中使用的变量为num_obs表示state数量
        "num_slice_obs": 26,
        "history_length": 5,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.5,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "height_measurements": 5.0,
        },
        "noise":{
            "use": True,
            #[高斯,随机游走]
            "ang_vel": [0.01,0.1],
            "dof_pos": [0.02,0.01],
            "dof_vel": [0.01,0.1],
            "gravity": [0.02,0.02],
        }
    }
    # 名字和奖励函数名一一对应
    reward_cfg = {
        "tracking_lin_sigma": 0.5, 
        "tracking_ang_sigma": 1.0, 
        "tracking_height_sigma": 0.008,
        "tracking_similar_legged_sigma": 0.2,
        "tracking_gravity_sigma": 0.005,
        "reward_scales": {
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 2.0,
            "tracking_base_height": 2.0,    #和similar_legged对抗，similar_legged先提升会促进此项
            "lin_vel_z": -0.2, #大了影响高度变换速度
            "joint_action_rate": -0.005,
            "wheel_action_rate": -0.002,
            "similar_to_default": 0.0,
            "projected_gravity": 6.0,
            "similar_legged": 0.6,  #tracking_base_height和knee_height对抗
            "dof_vel": -0.005,
            "dof_acc": -0.5e-9,
            "dof_force": -0.0001,
            "knee_height": -0.0,    #相当有效，和similar_legged结合可以抑制劈岔和跪地重启，稳定运行
            "ang_vel_xy": -0.02,
            "collision": -0.0008,  #base接触地面碰撞力越大越惩罚，数值太大会摆烂
            # "terrain":0.1,
        },
    }
    command_cfg = {
        "num_commands": 4,
        "base_range": 0.3,  #基础范围
        "lin_vel_x_range": [-1.2, 1.2], #修改范围要调整奖励权重
        "lin_vel_y_range": [-0.0, 0.0],
        "ang_vel_range": [-6.28, 6.28],   #修改范围要调整奖励权重
        "height_target_range": [0.22, 0.32],   #lower会导致跪地
        #旋转和线速度范围会互冲，所以下面参数是降低互冲的（大概就行，轮不能完全代表整机），纯腿的用不上这个
        "limit_cmd_random":False,
        "wheel_spacing":0.37, #轮间距 m
        "wheel_radius":0.75, #轮半径 m
        "wheel_max_w":20, #轮最大转速 rad/s
    }
    # 课程学习，奖励循序渐进 待优化
    curriculum_cfg = {
        "curriculum_lin_vel_step":0.008,   #比例
        "curriculum_ang_vel_step":0.0001,   #比例
        "curriculum_height_target_step":0.0003,   #高度
        "curriculum_lin_vel_min_range":0.3,   #比例
        "curriculum_ang_vel_min_range":0.3,   #比例
        "lin_vel_err_range":[0.05,0.5],  #课程误差阈值
        "ang_vel_err_range":[0.25,0.5],  #课程误差阈值 连续曲线>方波>不波动
        "damping_descent":True,
        "dof_damping_descent":[0.2, 0.005, 0.001, 0.4],#[damping_max,damping_min,damping_step（比例）,damping_threshold（存活步数比例）]
    }
    #域随机化 friction_ratio是范围波动 mass和com是偏移波动
    domain_rand_cfg = { 
        "friction_ratio_range":[0.2 , 2.0],
        "random_base_mass_shift_range":[-2 , 2], #质量偏移量
        "random_other_mass_shift_range":[-0.1, 0.1],  #质量偏移量
        "random_base_com_shift":0.1, #位置偏移量 xyz
        "random_other_com_shift":0.01, #位置偏移量 xyz
        "random_KP":[0.8, 1.2], #比例
        "random_KV":[0.8, 1.2], #比例
        "random_default_joint_angles":[-0.1,0.1], #rad
        "damping_range":[0.9, 1.1], #比例
        "dof_stiffness_range":[0.0 , 0.0], #范围 不包含轮 [0.0 , 0.0]就是关闭，关闭的时候把初始值也调0
        "dof_stiffness_descent":[0.0 , 0.8], #刚度下降[max_stiffness，仿真步数占比阈值]，和域随机化冲突，二选一
        "dof_armature_range":[0.0 , 0.008], #范围 额外惯性 类似电机减速器惯性 有助于仿真稳定性
    }
    #地形配置
    terrain_cfg = {
        "terrain":True, #是否开启地形
        "train":"agent_train_gym",
        "eval":"agent_eval_gym",    # agent_eval_gym/circular
        "num_respawn_points":3,
        "respawn_points":[
            [-5.0, -5.0, 0.0],    #plane地形坐标，一定要有，为了远离其他地形
            [5.0, 5.0, 0.0],
            [15.0, 5.0, 0.08],
        ],
        "horizontal_scale":0.1,
        "vertical_scale":0.001,
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="wheel-legged-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=6144)
    parser.add_argument("--max_iterations", type=int, default=30000)
    args = parser.parse_args()

    gs.init(logging_level="warning",backend=gs.gpu)
    gs.device="cuda:0"
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = WheelLeggedEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, 
        command_cfg=command_cfg, curriculum_cfg=curriculum_cfg, 
        domain_rand_cfg=domain_rand_cfg, terrain_cfg=terrain_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""