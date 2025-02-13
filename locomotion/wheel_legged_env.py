import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import numpy as np

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower
    
class WheelLeggedEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, class_cfg, domain_rand_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_slice_obs = obs_cfg["num_slice_obs"]
        self.history_length = obs_cfg["history_length"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.01  # control frequency on real robot is 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        
        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="assets/urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="assets/urdf/nz2/urdf/nz2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )
        # self.base_init_pos = torch.tensor((0.0, 0.0, 0.265),device=self.device)
        # self.robot = self.scene.add_entity(
        #     gs.morphs.MJCF(file="assets/mjcf/nz/nz.xml",
        #     pos=self.base_init_pos.cpu().numpy()),
        #     vis_mode='collision'
        # )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        #dof limits
        lower = [self.env_cfg["dof_limit"][name][0] for name in self.env_cfg["dof_names"]]
        upper = [self.env_cfg["dof_limit"][name][1] for name in self.env_cfg["dof_names"]]
        self.dof_pos_lower = torch.tensor(lower).to(self.device)
        self.dof_pos_upper = torch.tensor(upper).to(self.device)

        # set safe force
        self.robot.set_dofs_force_range(
            lower          = [-self.env_cfg["safe_force"][name] for name in self.env_cfg["dof_names"]],
            upper          = [self.env_cfg["safe_force"][name] for name in self.env_cfg["dof_names"]],
            dofs_idx_local = self.motor_dofs,
        )

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # self.class_value = torch.zeros(1, device=self.device, dtype=gs.tc_float)
        # self.class1 = 0.8 * (self.reward_scales["projected_gravity"] + self.reward_scales["tracking_base_height"] + self.reward_scales["similar_legged"])

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_lin_acc = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_acc = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
        
        self.slice_obs_buf = torch.zeros((self.num_envs, self.num_slice_obs), device=self.device, dtype=gs.tc_float)
        self.history_obs_buf = torch.zeros((self.num_envs, self.history_length, self.num_slice_obs), device=self.device, dtype=gs.tc_float)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.history_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"], self.obs_scales["height_measurements"]], 
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.dof_force = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.left_knee = self.robot.get_joint("left_calf_joint")
        self.right_knee = self.robot.get_joint("right_calf_joint")
        self.left_knee_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.right_knee_pos = torch.zeros_like(self.left_knee_pos)
        self.extras = dict()  # extra information for logging
        
        print("self.obs_buf.size(): ",self.obs_buf.size())

    def _resample_commands(self, envs_idx):
        # if(self.class_value > self.class1):
        #     self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        #     self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        #     self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)
        # else:
        #     self.commands[envs_idx] = torch.zeros(self.num_commands, device=self.device, dtype=gs.tc_float)
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 3] = gs_rand_float(*self.command_cfg["height_target_range"], (len(envs_idx),), self.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions[:,0:4] * self.env_cfg["joint_action_scale"] + self.default_dof_pos[0:4]
        target_dof_vel = exec_actions[:,4:6] * self.env_cfg["wheel_action_scale"]
        #dof limits
        target_dof_pos = torch.clamp(target_dof_pos, min=self.dof_pos_lower[0:4], max=self.dof_pos_upper[0:4])
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs[0:4])
        self.robot.control_dofs_velocity(target_dof_vel, self.motor_dofs[4:6])
        
        self.scene.step()
        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_lin_acc[:] = (self.base_lin_vel[:] - self.last_base_lin_vel[:])/ self.dt
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.base_ang_acc[:] = (self.base_ang_vel[:] - self.last_base_ang_vel[:]) / self.dt
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.dof_force[:] = self.robot.get_dofs_force(self.motor_dofs)
        #获取膝关节高度
        self.left_knee_pos[:] = self.left_knee.get_pos()
        self.right_knee_pos[:] = self.right_knee.get_pos()
        
        # update last
        self.last_base_lin_vel[:] = self.base_lin_vel[:]
        self.last_base_ang_vel[:] = self.base_ang_vel[:]

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.check_termination()

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        # self.class_value = self.rew_buf.mean()
        
        # self.commands[0, 0] = torch.abs(self.commands[0, 0])
        # self.commands[0,1] = 0.0
        # self.commands[0,2] = 0.0
        # self.commands[0, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(0),), self.device)
        # print("commands",self.commands)
        # print("base_lin_vel",self.base_lin_vel[:,0])
        # compute observations
        self.slice_obs_buf = torch.cat(
            [
                self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 4
                (self.dof_pos[:,0:4] - self.default_dof_pos[0:4]) * self.obs_scales["dof_pos"],  # 4
                self.dof_vel * self.obs_scales["dof_vel"],  # 6
                self.actions,  # 6
            ],
            axis=-1,
        )
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # Update history buffer
        self.history_obs_buf[:, self.history_idx] = self.slice_obs_buf  # Store the current observation in the history buffer
        self.history_idx = (self.history_idx + 1) % self.history_length  # Increment and wrap around the index

        # Combine the current observation with historical observations (e.g., along the time axis)
        self.obs_buf = torch.cat([self.history_obs_buf, self.slice_obs_buf.unsqueeze(1)], dim=1).view(self.num_envs, -1)
        
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        # print("\033[31m Reset Reset Reset Reset Reset Reset\033[0m")
        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    def check_termination(self):
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= torch.abs(self.base_pos[:, 2]) < self.env_cfg["termination_if_base_height_greater_than"]
        #特殊姿态重置
        self.reset_buf |= torch.abs(self.left_knee_pos[:,2]) < self.env_cfg["termination_if_knee_height_greater_than"]
        self.reset_buf |= torch.abs(self.right_knee_pos[:,2]) < self.env_cfg["termination_if_knee_height_greater_than"]

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # if(self.class_value > self.class1):
        #     lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        #     return torch.exp(-lin_vel_error / self.reward_cfg["tracking_lin_sigma"])
        # return torch.zeros(1, device=self.device, dtype=gs.tc_float)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_lin_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        # if(self.class_value > self.class1):
        #     ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        #     return torch.exp(-ang_vel_error / self.reward_cfg["tracking_ang_sigma"])
        # return torch.zeros(1, device=self.device, dtype=gs.tc_float)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_ang_sigma"])

    def _reward_tracking_base_height(self):
        # Penalize base height away from target
        base_height_error = torch.square(self.base_pos[:, 2] - self.commands[:, 3])
        # return torch.exp(-base_height_error / self.reward_cfg["tracking_height_sigma"])
        return base_height_error
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_xy_vel(self):
        # Penalize z axis base linear velocity
        return torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
    
    def _reward_joint_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:,0:4] - self.actions[:,0:4]), dim=1)

    def _reward_wheel_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:,4:6] - self.actions[:,4:6]), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        #个人认为为了灵活性这个作用性不大
        return torch.sum(torch.abs(self.dof_pos[:,0:4] - self.default_dof_pos[0:4]), dim=1)
    
    def _reward_projected_gravity(self):
        # 这是奖励函数，并非观测，所以保持水平奖励使用重力投影
        #使用e^(-x^2)效果不是很好
        # projected_gravity_error = 1 + self.projected_gravity[:, 2]
        projected_gravity_error = torch.square(self.projected_gravity[:, :2])
        # return torch.exp(-projected_gravity_error / self.reward_cfg["tracking_gravity_sigma"])
        return torch.sum(projected_gravity_error, dim=1)

    def _reward_similar_legged(self):
        # 两侧腿相似 适合使用轮子行走 抑制劈岔，要和_reward_knee_height结合使用
        legged_error = torch.sum(torch.square(self.dof_pos[:,0:2] - self.dof_pos[:,2:4]), dim=1)
        # return torch.exp(-legged_error / self.reward_cfg["tracking_similar_legged_sigma"])
        return legged_error

    def _reward_knee_height(self):
        #关节处于某个范围惩罚，避免总跪着
        below_threshold = (torch.abs(self.left_knee_pos[:,2]) < 0.1).float()
        below_threshold+=(torch.abs(self.right_knee_pos[:,2]) < 0.1).float()
        return below_threshold

    # def _reward_lin_acc(self):
    #     # Penalize z axis base linear velocity
    #     return torch.sum(torch.square(self.base_lin_acc))

    def _reward_ang_acc(self):
        # Penalize z axis base linear velocity
        return torch.sum(torch.square(self.base_ang_acc[:,:2]))

    def _reward_dof_acc(self):
        # Penalize z axis base linear velocity
        return torch.sum(torch.square((self.dof_vel - self.last_dof_vel)/self.dt))

    def _reward_dof_force(self):
        # Penalize z axis base linear velocity
        return torch.sum(torch.square(self.dof_force ), dim=1)
    
    def _reward_action_smooth(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.actions[:, :2]
                - 2 * self.last_actions[:, :2, 0]
                + self.last_actions[:, :2, 1]
            ),
            dim=1,
        ) + torch.sum(
            torch.square(
                self.actions[:, 3:5]
                - 2 * self.last_actions[:, 3:5, 0]
                + self.last_actions[:, 3:5, 1]
            ),
            dim=1,
        )