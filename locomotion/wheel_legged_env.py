import torch
import math
import genesis as gs # type: ignore
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat # type: ignore
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
import numpy as np
import cv2

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

    
class WheelLeggedEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, 
                 domain_rand_cfg, terrain_cfg, robot_morphs="urdf", show_viewer=False, device="cuda", train_mode=True):
        self.device = torch.device(device)

        self.mode = train_mode   #True训练模式开启
        
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_slice_obs = obs_cfg["num_slice_obs"]
        self.history_length = obs_cfg["history_length"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.curriculum_cfg = curriculum_cfg
        self.domain_rand_cfg = domain_rand_cfg
        self.terrain_cfg = terrain_cfg
        self.num_respawn_points = self.terrain_cfg["num_respawn_points"]
        self.respawn_points = self.terrain_cfg["respawn_points"]

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
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=5),
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
                batch_dofs_info=True,
                # batch_links_info=True,
            ),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.URDF(file="assets/terrain/plane/plane.urdf", fixed=True))
        # init roboot quat and pos
        match robot_morphs:
            case "urdf":
                self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"]["urdf"], device=self.device)
            case "mjcf":
                self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"]["mjcf"], device=self.device)
            case _:
                self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"]["urdf"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        # add terrain 只能有一个Terrain(genesis v0.2.1)
        self.horizontal_scale = self.terrain_cfg["horizontal_scale"]
        self.vertical_scale = self.terrain_cfg["vertical_scale"]
        self.height_field = cv2.imread("assets/terrain/png/"+self.terrain_cfg["train"]+".png", cv2.IMREAD_GRAYSCALE)
        self.terrain_height = torch.tensor(self.height_field, device=self.device) * self.vertical_scale
        if self.terrain_cfg["terrain"]:
            print("\033[1;35m open terrain\033[0m")
            if self.mode:
                self.terrain = self.scene.add_entity(
                morph=gs.morphs.Terrain(
                height_field = self.height_field,
                horizontal_scale=self.horizontal_scale, 
                vertical_scale=self.vertical_scale,
                ),)
                self.base_terrain_pos = torch.zeros((self.num_respawn_points, 3), device=self.device)
                for i in range(self.num_respawn_points):
                    self.base_terrain_pos[i] = self.base_init_pos + torch.tensor(self.respawn_points[i], device=self.device)
                print("\033[1;34m respawn_points: \033[0m",self.base_terrain_pos)
            else:
                height_field = cv2.imread("assets/terrain/png/"+self.terrain_cfg["eval"]+".png", cv2.IMREAD_GRAYSCALE)
                self.terrain = self.scene.add_entity(
                morph=gs.morphs.Terrain(
                pos = (1.0,1.0,0.0),
                height_field = height_field,
                horizontal_scale=self.horizontal_scale, 
                vertical_scale=self.vertical_scale,
                ),)     
                print("\033[1;34m respawn_points: \033[0m",self.base_init_pos)

        # add robot
        base_init_pos = self.base_init_pos.cpu().numpy()
        if self.terrain_cfg["terrain"]:
            if self.mode:
                base_init_pos = self.base_terrain_pos[0].cpu().numpy()

        match robot_morphs:
            case "urdf":
                self.robot = self.scene.add_entity(
                    gs.morphs.URDF(
                        file="assets/urdf/nz/urdf/nz.urdf",
                        pos = base_init_pos,
                        quat=self.base_init_quat.cpu().numpy(),
                    ),
                )
            case "mjcf":
                self.robot = self.scene.add_entity(
                    gs.morphs.MJCF(file="assets/mjcf/nz/nz_view.xml",
                    pos=base_init_pos),
                    vis_mode='collision'
                )
            case _:
                self.robot = self.scene.add_entity(
                    gs.morphs.URDF(
                        file="assets/urdf/nz/urdf/nz.urdf",
                        pos = base_init_pos,
                        quat=self.base_init_quat.cpu().numpy(),
                    ),
                )
        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # PD control parameters
        self.kp = np.full((self.num_envs, self.num_actions), self.env_cfg["joint_kp"])
        self.kd = np.full((self.num_envs, self.num_actions), self.env_cfg["joint_kd"])
        self.kp[:,4:6] = 0
        self.kd[:,4:6] = self.env_cfg["wheel_kd"]
        self.robot.set_dofs_kp(self.kp, self.motor_dofs)
        self.robot.set_dofs_kv(self.kd, self.motor_dofs)
        
        
        damping = np.full((self.num_envs,self.robot.n_dofs), self.env_cfg["joint_damping"])
        damping[:,:6] = 0
        damping[:,self.motor_dofs[5]] = self.env_cfg["wheel_damping"]
        damping[:,self.motor_dofs[4]] = self.env_cfg["wheel_damping"]
        self.robot.set_dofs_damping(damping=damping, 
                                   dofs_idx_local=np.arange(0,self.robot.n_dofs)
                                   )
        stiffness = np.full((self.num_envs,self.robot.n_dofs), self.env_cfg["stiffness"])
        stiffness[:,:6] = 0
        stiffness[:,self.motor_dofs[5]] = 0
        stiffness[:,self.motor_dofs[4]] = 0
        self.robot.set_dofs_stiffness(stiffness=stiffness, 
                                   dofs_idx_local=np.arange(0,self.robot.n_dofs)
                                   )
        # from IPython import embed; embed()
        armature = np.full((self.num_envs, self.robot.n_dofs), self.env_cfg["armature"])
        armature[:,:6] = 0
        self.robot.set_dofs_armature(armature=armature, 
                                   dofs_idx_local=np.arange(0, self.robot.n_dofs))
        

        #dof limits
        lower = [self.env_cfg["dof_limit"][name][0] for name in self.env_cfg["dof_names"]]
        upper = [self.env_cfg["dof_limit"][name][1] for name in self.env_cfg["dof_names"]]
        self.dof_pos_lower = torch.tensor(lower).to(self.device)
        self.dof_pos_upper = torch.tensor(upper).to(self.device)

        # set safe force
        lower = np.array([[-self.env_cfg["safe_force"][name] for name in self.env_cfg["dof_names"]] for _ in range(num_envs)])
        upper = np.array([[self.env_cfg["safe_force"][name] for name in self.env_cfg["dof_names"]] for _ in range(num_envs)])
        self.robot.set_dofs_force_range(
            lower          = torch.tensor(lower, device=self.device, dtype=torch.float32),
            upper          = torch.tensor(upper, device=self.device, dtype=torch.float32),
            dofs_idx_local = self.motor_dofs,
        )

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)


        # prepare command_ranges lin_vel_x lin_vel_y ang_vel height_target
        self.command_ranges = torch.zeros((self.num_envs, self.num_commands,2),device=self.device,dtype=gs.tc_float)
        self.command_ranges[:,0,0] = self.command_cfg["lin_vel_x_range"][0] * self.command_cfg["base_range"]
        self.command_ranges[:,0,1] = self.command_cfg["lin_vel_x_range"][1] * self.command_cfg["base_range"]
        self.command_ranges[:,1,0] = self.command_cfg["lin_vel_y_range"][0] * self.command_cfg["base_range"]
        self.command_ranges[:,1,1] = self.command_cfg["lin_vel_y_range"][1] * self.command_cfg["base_range"]
        self.command_ranges[:,2,0] = self.command_cfg["ang_vel_range"][0] * self.command_cfg["base_range"]
        self.command_ranges[:,2,1] = self.command_cfg["ang_vel_range"][1] * self.command_cfg["base_range"]
        self.height_range = self.command_cfg["height_target_range"][1]-self.command_cfg["height_target_range"][0]
        self.command_ranges[:,3,0] = self.command_cfg["height_target_range"][0] + self.height_range * (1-self.command_cfg["base_range"])
        self.command_ranges[:,3,1] = self.command_cfg["height_target_range"][1]
        self.lin_vel_error = torch.zeros((self.num_envs,1), device=self.device, dtype=gs.tc_float)
        self.ang_vel_error = torch.zeros((self.num_envs,1), device=self.device, dtype=gs.tc_float)
        self.height_error = torch.zeros((self.num_envs,1), device=self.device, dtype=gs.tc_float)
        self.curriculum_lin_vel_scale = torch.zeros((self.num_envs,1), device=self.device, dtype=gs.tc_float)
        self.curriculum_ang_vel_scale = torch.zeros((self.num_envs,1), device=self.device, dtype=gs.tc_float)

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
        self.curriculum_rew_buf = torch.zeros_like(self.rew_buf)
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
        self.basic_default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        
        default_dof_pos_list = [[self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]]] * self.num_envs
        self.default_dof_pos = torch.tensor(default_dof_pos_list,device=self.device,dtype=gs.tc_float,)
        init_dof_pos_list = [[self.env_cfg["joint_init_angles"][name] for name in self.env_cfg["dof_names"]]] * self.num_envs
        self.init_dof_pos = torch.tensor(init_dof_pos_list,device=self.device,dtype=gs.tc_float,)
        #膝关节
        self.left_knee = self.robot.get_joint("left_calf_joint")
        self.right_knee = self.robot.get_joint("right_calf_joint")
        self.left_knee_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.right_knee_pos = torch.zeros_like(self.left_knee_pos)
        self.connect_force = torch.zeros((self.num_envs,self.robot.n_links, 3), device=self.device, dtype=gs.tc_float)
        self.extras = dict()  # extra information for logging
        
        #跪地重启   注意是idx_local不需要减去base_idx
        if(self.env_cfg["termination_if_base_connect_plane_than"]&self.mode):
            self.reset_links = [(self.robot.get_link(name).idx_local) for name in self.env_cfg["connect_plane_links"]]
            
        #域随机化 domain_rand_cfg
        self.friction_ratio_low = self.domain_rand_cfg["friction_ratio_range"][0]
        self.friction_ratio_range = self.domain_rand_cfg["friction_ratio_range"][1] - self.friction_ratio_low
        self.base_mass_low = self.domain_rand_cfg["random_base_mass_shift_range"][0]
        self.base_mass_range = self.domain_rand_cfg["random_base_mass_shift_range"][1] - self.base_mass_low  
        self.other_mass_low = self.domain_rand_cfg["random_other_mass_shift_range"][0]
        self.other_mass_range = self.domain_rand_cfg["random_other_mass_shift_range"][1] - self.other_mass_low            
        self.joint_damping_low = self.domain_rand_cfg["joint_damping_range"][0]
        self.joint_damping_range = self.domain_rand_cfg["joint_damping_range"][1] - self.joint_damping_low
        self.wheel_damping_low = self.domain_rand_cfg["wheel_damping_range"][0]
        self.wheel_damping_range = self.domain_rand_cfg["wheel_damping_range"][1] - self.wheel_damping_low
        self.dof_stiffness_low = self.domain_rand_cfg["dof_stiffness_range"][0]
        self.dof_stiffness_range = self.domain_rand_cfg["dof_stiffness_range"][1] - self.dof_stiffness_low
        self.dof_armature_low = self.domain_rand_cfg["dof_armature_range"][0]
        self.dof_armature_range = self.domain_rand_cfg["dof_armature_range"][1] - self.dof_armature_low
        self.kp_low = self.domain_rand_cfg["random_KP"][0]
        self.kp_range = self.domain_rand_cfg["random_KP"][1] - self.kp_low
        self.kd_low = self.domain_rand_cfg["random_KD"][0]
        self.kd_range = self.domain_rand_cfg["random_KD"][1] - self.kd_low
        self.joint_angle_low = self.domain_rand_cfg["random_default_joint_angles"][0]
        self.joint_angle_range = self.domain_rand_cfg["random_default_joint_angles"][1] - self.joint_angle_low
        #init joint
        damping = torch.full(
            size=(self.num_envs, self.robot.n_dofs),fill_value=env_cfg["joint_damping"],
            dtype=gs.tc_float,device=self.device)
        damping[:, :6] = 0  # 前6列设为0
        damping[:, self.motor_dofs[5]] = env_cfg["wheel_damping"]  # 直接赋标量值
        damping[:, self.motor_dofs[4]] = env_cfg["wheel_damping"]
        self.robot.set_dofs_damping(damping=damping, dofs_idx_local=np.arange(0, self.robot.n_dofs))
        stiffness = torch.full(
            size=(self.num_envs, self.robot.n_dofs),fill_value=env_cfg["stiffness"],
            dtype=gs.tc_float,device=self.device)
        stiffness[:, self.robot.n_dofs-6:] = 0  # 确保 n_dofs >= 6
        stiffness[:, self.motor_dofs[5]] = 0
        stiffness[:, self.motor_dofs[4]] = 0
        self.robot.set_dofs_stiffness(stiffness=stiffness, dofs_idx_local=np.arange(0, self.robot.n_dofs))
        armature = torch.full(
            size=(self.num_envs, self.robot.n_dofs),fill_value=env_cfg["armature"],
            dtype=gs.tc_float,device=self.device)
        armature[:, :6] = 0
        self.robot.set_dofs_armature(armature=armature, dofs_idx_local=np.arange(0, self.robot.n_dofs))
        #地形训练索引
        self.terrain_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        # print("self.obs_buf.size(): ",self.obs_buf.size())
        
        #外部力
        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            rigid_solver = solver

        print("self.init_dof_pos",self.init_dof_pos)
        #初始化角度
        self.dof_pos[:] = self.init_dof_pos[:]
        self.dof_vel[:] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[:],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
        )
        
    def _resample_commands(self, envs_idx):
        for idx in envs_idx:
            for command_idx in range(self.num_commands):
                low = self.command_ranges[idx, command_idx, 0]
                high = self.command_ranges[idx, command_idx, 1]
                self.commands[idx, command_idx] = gs_rand_float(low, high, (1,), self.device)

    def set_commands(self,envs_idx,commands):
        self.commands[envs_idx]=torch.tensor(commands,device=self.device, dtype=gs.tc_float)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions[:,0:4] * self.env_cfg["joint_action_scale"] + self.default_dof_pos[:,0:4]
        target_dof_vel = exec_actions[:,4:6] * self.env_cfg["wheel_action_scale"]
        #dof limits
        target_dof_pos = torch.clamp(target_dof_pos, min=self.dof_pos_lower[0:4], max=self.dof_pos_upper[0:4])
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs[0:4])
        self.robot.control_dofs_velocity(target_dof_vel, self.motor_dofs[4:6])

        self.scene.step()
        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.get_relative_terrain_pos(self.robot.get_pos())
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
        #碰撞力
        self.connect_force = self.robot.get_links_net_contact_force()

        # update last
        self.last_base_lin_vel[:] = self.base_lin_vel[:]
        self.last_base_ang_vel[:] = self.base_ang_vel[:]

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )

        # check terrain_buf
        # 线速度达到预设的90%范围，角速度达到90%以上去其他地形(建议高一点)
        self.terrain_buf = self.command_ranges[:, 0, 1] > self.command_cfg["lin_vel_x_range"][1] * 0.9
        self.terrain_buf &= self.command_ranges[:, 2, 1] > self.command_cfg["ang_vel_range"][1] * 0.9
        #固定一部分去地形
        self.terrain_buf[:int(self.num_envs*0.4)] = 1
        
        # check termination and reset
        if(self.mode):
            self.check_termination()

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        if(self.mode):
            self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        if(self.mode):
            self.rew_buf[:] = 0.0
            for name, reward_func in self.reward_functions.items():
                rew = reward_func() * self.reward_scales[name]
                self.rew_buf += rew
                self.episode_sums[name] += rew
            
        # compute curriculum reward
        self.lin_vel_error += torch.abs(self.commands[:, :2] - self.base_lin_vel[:, :2]).mean(dim=1, keepdim=True)
        self.ang_vel_error += torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2]).mean()
        self.height_error += torch.abs(self.commands[:, 3] - self.base_pos[:, 2]).mean()

        if(self.mode):
            self._resample_commands(envs_idx)
            # self.curriculum_commands()
        # else:
        #     print("base_lin_vel: ",self.base_lin_vel[0,:])
            
        # compute observations
        self.slice_obs_buf = torch.cat(
            [
                # self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 4
                (self.dof_pos[:,0:4] - self.default_dof_pos[:,0:4]) * self.obs_scales["dof_pos"],  # 4
                self.dof_vel * self.obs_scales["dof_vel"],  # 6
                self.actions,  # 6
            ],
            axis=-1,
        )
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # print("slice_obs_buf: ",self.slice_obs_buf)
        
        # Combine the current observation with historical observations (e.g., along the time axis)
        self.obs_buf = torch.cat([self.history_obs_buf, self.slice_obs_buf.unsqueeze(1)], dim=1).view(self.num_envs, -1)
        # Update history buffer
        if self.history_length > 1:
            self.history_obs_buf[:, :-1, :] = self.history_obs_buf[:, 1:, :].clone() # 移位操作
        self.history_obs_buf[:, -1, :] = self.slice_obs_buf 
        
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
        self.dof_pos[envs_idx] = self.init_dof_pos[envs_idx]
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        if self.terrain_cfg["terrain"]:
            if self.mode:
                terrain_buf = self.terrain_buf[envs_idx]
                terrain_idx = envs_idx[terrain_buf.nonzero(as_tuple=False).flatten()] # 获取 envs_idx 中满足条件的索引
                non_terrain_idx = envs_idx[(~terrain_buf).nonzero(as_tuple=False).flatten()] # 获取 envs_idx 中不满足条件的索引
                # 设置地形位置
                if len(terrain_idx) > 0: # 只有当有满足地形重置条件的环境时才执行
                    #目前认为坡路和崎岖路面是相同难度，所以reset随机选取一个环境去复活
                    n = len(terrain_idx)
                    random_idx = torch.randint(1, self.num_respawn_points, (n,)) # 注意从 1 开始，避免使用 base_terrain_pos[0] 作为随机位置
                    selected_pos = self.base_terrain_pos[random_idx]
                    self.base_pos[terrain_idx] = selected_pos
                # 设置非地形位置 (默认位置)
                if len(non_terrain_idx) > 0:
                    self.base_pos[non_terrain_idx] = self.base_terrain_pos[0]
            else:
                self.base_pos[envs_idx] = self.base_init_pos
        else:
            self.base_pos[envs_idx] = self.base_init_pos   #没开地形就基础
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
        if self.mode:
            self.domain_rand(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    def check_termination(self):
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        # self.reset_buf |= torch.abs(self.base_pos[:, 2]) < self.env_cfg["termination_if_base_height_greater_than"]
        #特殊姿态重置
        # self.reset_buf |= torch.abs(self.left_knee_pos[:,2]) < self.env_cfg["termination_if_knee_height_greater_than"]
        # self.reset_buf |= torch.abs(self.right_knee_pos[:,2]) < self.env_cfg["termination_if_knee_height_greater_than"]
        if(self.env_cfg["termination_if_base_connect_plane_than"]):
            for idx in self.reset_links:
                self.reset_buf |= torch.abs(self.connect_force[:,idx,:]).sum(dim=1) > 0
        
    def domain_rand(self, envs_idx):
        friction_ratio = self.friction_ratio_low + self.friction_ratio_range * torch.rand(len(envs_idx), self.robot.n_links)
        self.robot.set_friction_ratio(friction_ratio=friction_ratio,
                                      ls_idx_local=np.arange(0, self.robot.n_links),
                                      envs_idx = envs_idx)

        base_mass_shift = self.base_mass_low + self.base_mass_range * torch.rand(len(envs_idx), 1, device=self.device)
        other_mass_shift =-self.other_mass_low + self.other_mass_range * torch.rand(len(envs_idx), self.robot.n_links - 1, device=self.device)
        mass_shift = torch.cat((base_mass_shift, other_mass_shift), dim=1)
        self.robot.set_mass_shift(mass_shift=mass_shift,
                                  ls_idx_local=np.arange(0, self.robot.n_links),
                                  envs_idx = envs_idx)

        base_com_shift = -self.domain_rand_cfg["random_base_com_shift"] / 2 + self.domain_rand_cfg["random_base_com_shift"] * torch.rand(len(envs_idx), 1, 3, device=self.device)
        other_com_shift = -self.domain_rand_cfg["random_other_com_shift"] / 2 + self.domain_rand_cfg["random_other_com_shift"] * torch.rand(len(envs_idx), self.robot.n_links - 1, 3, device=self.device)
        com_shift = torch.cat((base_com_shift, other_com_shift), dim=1)
        self.robot.set_COM_shift(com_shift=com_shift,
                                 ls_idx_local=np.arange(0, self.robot.n_links),
                                 envs_idx = envs_idx)

        kp_shift = (self.kp_low + self.kp_range * torch.rand(len(envs_idx), self.num_actions)) * self.kp[0]
        self.robot.set_dofs_kp(kp_shift, self.motor_dofs, envs_idx=envs_idx)

        kd_shift = (self.kd_low + self.kd_range * torch.rand(len(envs_idx), self.num_actions)) * self.kd[0]
        self.robot.set_dofs_kv(kd_shift, self.motor_dofs, envs_idx = envs_idx)

        #random_default_joint_angles
        dof_pos_shift = self.joint_angle_low + self.joint_angle_range * torch.rand(len(envs_idx),self.num_actions,device=self.device,dtype=gs.tc_float)
        self.default_dof_pos[envs_idx] = dof_pos_shift + self.basic_default_dof_pos

        damping = (self.joint_damping_low+self.joint_damping_range * torch.rand(len(envs_idx), self.robot.n_dofs))
        damping[:,:6] = 0
        damping[:,self.motor_dofs[5]] = (self.wheel_damping_low+self.wheel_damping_range * torch.rand(len(envs_idx), ))
        damping[:,self.motor_dofs[4]] = (self.wheel_damping_low+self.wheel_damping_range * torch.rand(len(envs_idx), ))
        self.robot.set_dofs_damping(damping=damping, 
                                   dofs_idx_local=np.arange(0, self.robot.n_dofs), 
                                   envs_idx=envs_idx)

        stiffness = (self.dof_stiffness_low+self.dof_stiffness_range * torch.rand(len(envs_idx), self.robot.n_dofs))
        stiffness[:,self.robot.n_dofs-6:] = 0
        stiffness[:,self.motor_dofs[5]] = 0
        stiffness[:,self.motor_dofs[4]] = 0
        self.robot.set_dofs_stiffness(stiffness=stiffness, 
                                   dofs_idx_local=np.arange(0, self.robot.n_dofs), 
                                   envs_idx=envs_idx)

        armature = (self.dof_armature_low+self.dof_armature_range * torch.rand(len(envs_idx), self.robot.n_dofs))
        armature[:,:6] = 0
        self.robot.set_dofs_armature(armature=armature, 
                                   dofs_idx_local=np.arange(0, self.robot.n_dofs), 
                                   envs_idx=envs_idx)

    def adjust_scale(self, error,lower_err,upper_err, scale, scale_step, min_range, range_cfg):
        # 计算误差范围
        min_condition = error < lower_err
        max_condition = error > upper_err
        # 调整 scale
        scale[min_condition] += scale_step
        scale[max_condition] -= scale_step * 2
        scale.clip_(min_range, 1)
        # 更新 command_ranges
        range_min, range_max = range_cfg
        return scale * range_min, scale * range_max

    def curriculum_commands(self, num):
        # 更新误差
        self.lin_vel_error /= num
        self.ang_vel_error /= num
        self.height_error /= num
        # 调整线速度
        lin_min_range, lin_max_range = self.adjust_scale(
            self.lin_vel_error, 
            self.curriculum_cfg["lin_vel_err_range"][0],   #误差反馈更新
            self.curriculum_cfg["lin_vel_err_range"][1],    #err back update
            self.curriculum_lin_vel_scale, 
            self.curriculum_cfg["curriculum_lin_vel_step"], 
            self.curriculum_cfg["curriculum_lin_vel_min_range"], 
            self.command_cfg["lin_vel_x_range"]
        )
        self.command_ranges[:, 0, 0] = lin_min_range.squeeze()
        self.command_ranges[:, 0, 1] = lin_max_range.squeeze()
        # 调整角速度    角速度误差可以大一些，因为comand范围更大
        ang_min_range, ang_max_range = self.adjust_scale(
            self.ang_vel_error, 
            self.curriculum_cfg["ang_vel_err_range"][0],
            self.curriculum_cfg["ang_vel_err_range"][1],
            self.curriculum_ang_vel_scale, 
            self.curriculum_cfg["curriculum_ang_vel_step"], 
            self.curriculum_cfg["curriculum_ang_vel_min_range"], 
            self.command_cfg["ang_vel_range"]
        )
        self.command_ranges[:, 2, 0] = ang_min_range.squeeze()
        self.command_ranges[:, 2, 1] = ang_max_range.squeeze()
        #调整高度
        add_height = self.height_error.squeeze() > 0.05
        self.command_ranges[add_height,3,0] += self.curriculum_cfg["curriculum_ang_vel_step"]
        cut_height = self.height_error.squeeze() < 0.1
        self.command_ranges[cut_height,3,0] -= self.curriculum_cfg["curriculum_ang_vel_step"]
        self.command_ranges[:,3,0].clip_(self.command_cfg["height_target_range"][0],
                                         self.command_cfg["height_target_range"][0] + self.height_range * (1-self.command_cfg["base_range"]))
        # 重置误差
        self.lin_vel_error = torch.zeros((self.num_envs, 1), device=self.device, dtype=gs.tc_float)
        self.ang_vel_error = torch.zeros((self.num_envs, 1), device=self.device, dtype=gs.tc_float)
        self.height_error = torch.zeros((self.num_envs, 1), device=self.device, dtype=gs.tc_float)

    def get_relative_terrain_pos(self, base_pos):
        if not self.terrain_cfg["terrain"]:
            return base_pos
        #对多个 (x, y) 坐标进行双线性插值计算地形高度
        # 提取x和y坐标
        x = base_pos[:, 0]
        y = base_pos[:, 1]
        # 转换为浮点数索引
        fx = x / self.horizontal_scale
        fy = y / self.horizontal_scale
        # 获取四个最近的整数网格点，确保在有效范围内
        x0 = torch.floor(fx).int()
        x1 = torch.min(x0 + 1, torch.full_like(x0, self.terrain_height.shape[1] - 1))
        y0 = torch.floor(fy).int()
        y1 = torch.min(y0 + 1, torch.full_like(y0, self.terrain_height.shape[0] - 1))
        # 确保x0, x1, y0, y1在有效范围内
        x0 = torch.clamp(x0, 0, self.terrain_height.shape[1] - 1)
        x1 = torch.clamp(x1, 0, self.terrain_height.shape[1] - 1)
        y0 = torch.clamp(y0, 0, self.terrain_height.shape[0] - 1)
        y1 = torch.clamp(y1, 0, self.terrain_height.shape[0] - 1)
        # 获取四个点的高度值
        # 使用广播机制处理批量数据
        Q11 = self.terrain_height[y0, x0]
        Q21 = self.terrain_height[y0, x1]
        Q12 = self.terrain_height[y1, x0]
        Q22 = self.terrain_height[y1, x1]
        # 计算双线性插值
        wx = fx - x0
        wy = fy - y0
        height = (
            (1 - wx) * (1 - wy) * Q11 +
            wx * (1 - wy) * Q21 +
            (1 - wx) * wy * Q12 +
            wx * wy * Q22
        )
        base_pos[:,2] -= height
        return base_pos


    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]),dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_lin_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_ang_sigma"])

    def _reward_tracking_base_height(self):
        # Penalize base height away from target
        base_height_error = torch.square(self.base_pos[:, 2] - self.commands[:, 3])
        return torch.exp(-base_height_error / self.reward_cfg["tracking_height_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_joint_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:,0:4] - self.actions[:,0:4]), dim=1)

    def _reward_wheel_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:,4:6] - self.actions[:,4:6]), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        #个人认为为了灵活性这个作用性不大
        return torch.sum(torch.abs(self.dof_pos[:,0:4] - self.default_dof_pos[:,0:4]), dim=1)

    def _reward_projected_gravity(self):
        #保持水平奖励使用重力投影 0 0 -1
        #使用e^(-x^2)效果不是很好
        projected_gravity_error = 1 + self.projected_gravity[:, 2] #[0, 0.2]
        projected_gravity_error = torch.square(projected_gravity_error)
        # projected_gravity_error = torch.square(self.projected_gravity[:,2])
        return torch.exp(-projected_gravity_error / self.reward_cfg["tracking_gravity_sigma"])
        # return torch.sum(projected_gravity_error)

    def _reward_similar_legged(self):
        # 两侧腿相似 适合使用轮子行走 抑制劈岔，要和_reward_knee_height结合使用
        legged_error = torch.sum(torch.square(self.dof_pos[:,0:2] - self.dof_pos[:,2:4]), dim=1)
        return torch.exp(-legged_error / self.reward_cfg["tracking_similar_legged_sigma"])
        # return legged_error

    def _reward_knee_height(self):
        # 关节处于某个范围惩罚，避免总跪着
        left_knee_idx = torch.abs(self.left_knee_pos[:, 2]) < 0.08
        right_knee_idx = torch.abs(self.right_knee_pos[:, 2]) < 0.08
        knee_rew = torch.sum(torch.square(self.left_knee_pos[left_knee_idx, 2] - 0.08)) if left_knee_idx.any() else 0
        knee_rew += torch.sum(torch.square(self.right_knee_pos[right_knee_idx, 2] - 0.08)) if right_knee_idx.any() else 0
        return knee_rew


    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel[:, :4]), dim=1)

    def _reward_dof_acc(self):
        # Penalize z axis base linear velocity
        return torch.sum(torch.square((self.dof_vel - self.last_dof_vel)/self.dt))

    def _reward_dof_force(self):
        # Penalize z axis base linear velocity
        return torch.sum(torch.square(self.dof_force), dim=1)

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_collision(self):
        # 接触地面惩罚 力越大惩罚越大
        collision = torch.zeros(self.num_envs,device=self.device,dtype=gs.tc_float)
        for idx in self.reset_links:
            collision += torch.square(self.connect_force[:,idx,:]).sum(dim=1)
        return collision

    # def _reward_terrain(self):
    #     # extra_lin_vel = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]),dim=1)
    #     # extra_ang_vel = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    #     # extra_terrain_rew = torch.exp(-extra_lin_vel / self.reward_cfg["tracking_lin_sigma"]) 
    #     # + torch.exp(-extra_ang_vel / self.reward_cfg["tracking_ang_sigma"])
    #     # return extra_terrain_rew
        
    #     return self.terrain_buf