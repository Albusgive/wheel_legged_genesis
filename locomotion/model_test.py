import genesis as gs # type: ignore
import numpy as np
import time
import cv2
import math
import torch
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
gs.init(backend=gs.cuda)

scene = gs.Scene(
    show_viewer = True,
    viewer_options = gs.options.ViewerOptions(
        res           = (1280, 960),
        camera_pos    = (3.5, 0.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 120,
        run_in_thread = False,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = True,
        world_frame_size = 1.0,
        show_link_frame  = False,
        show_cameras     = False,
        plane_reflection = True,
        ambient_light    = (0.1, 0.1, 0.1),
    ),
    renderer=gs.renderers.Rasterizer(),
)

# plane = scene.add_entity(
#     gs.morphs.Plane(),
# )

# mesh = scene.add_entity(
#         morph=gs.morphs.Mesh(
#             file = "/home/albusgive2/wheel_legged_genesis/assets/terrain/stair.stl",
#             fixed = True,
#             convexify=True,
#             decimate_aggressiveness=0,
#         ),
#     )

# horizontal_scale = 0.1
# vertical_scale = 0.001
# terrain = scene.add_entity(
#         morph=gs.morphs.Terrain(
#             pos = (0.0, 0.0, 0.0),
#             n_subterrains=(1, 2),
#             subterrain_size=(12.0, 12.0),
#             horizontal_scale=horizontal_scale,
#             vertical_scale=vertical_scale,
#             subterrain_types=[
#                 ["pyramid_stairs_terrain", "stairs_terrain"],
#             ],
#         ),
#     )

# v_stairs_height = 0.1  # 阶梯高度
# v_stairs_width = 0.2  # 阶梯宽度
# plane_size = 0.5
# v_stairs_num = 10       # 阶梯数量
# point = [0,0,0]  # 阶梯位置
# stairs = []

# '''正金字塔'''
# def add_pyramid(point):
#     max_z_pos = v_stairs_num * v_stairs_height - v_stairs_height / 2 + point[2]
#     current_z = max_z_pos
#     box_size = plane_size
#     box_pos = point
#     for _ in range(v_stairs_num):
#         box_pos[2] = current_z
#         box = scene.add_entity(
#             morph=gs.morphs.Box(
#                 pos=tuple(box_pos),
#                 size=(box_size, box_size, v_stairs_height),
#                 fixed=True
#             )
#         )
#         stairs.append(box)
#         box_size += v_stairs_width*2
#         current_z -= v_stairs_height
# print("max_size:", plane_size + v_stairs_width*2* (v_stairs_num-1))
# max_size = plane_size + v_stairs_width*2* (v_stairs_num-1)   

# '''倒金字塔'''
# def add_inverted_pyramid(point):
#     min_z_pos = v_stairs_height / 2 + point[2]
#     box_offset = (plane_size + v_stairs_width)/2
#     box_length = plane_size + v_stairs_width*2
#     box_pos = [[point[0]+box_offset, point[1], point[2]+v_stairs_height/2],
#             [point[0]-box_offset, point[1], point[2]+v_stairs_height/2],
#             [point[0], point[1]+box_offset, point[2]+v_stairs_height/2],
#             [point[0], point[1]-box_offset, point[2]+v_stairs_height/2]]
#     box_size = [[v_stairs_width, box_length, v_stairs_height],
#                 [v_stairs_width, box_length, v_stairs_height],
#                 [box_length, v_stairs_width, v_stairs_height],
#                 [box_length, v_stairs_width, v_stairs_height]]
#     box = scene.add_entity(
#                 morph=gs.morphs.Box(
#                     pos=(point[0], point[1], min_z_pos),
#                     size=(plane_size,plane_size,v_stairs_height),
#                     fixed=True
#                 )
#             )
#     stairs.append(box)
#     for num_stairs in range(v_stairs_num*2):
#         for i in range(4):
#             if num_stairs<=v_stairs_num-1:
#                 box_pos[i][2] += v_stairs_height
#             else:
#                 box_pos[i][2] -= v_stairs_height
#             box = scene.add_entity(
#                 morph=gs.morphs.Box(
#                     pos=tuple(box_pos[i]),
#                     size=tuple(box_size[i]),
#                     fixed=True
#                 )
#             )
#             stairs.append(box)
#         box_pos[0][0] += v_stairs_width
#         box_pos[1][0] -= v_stairs_width
#         box_pos[2][1] += v_stairs_width
#         box_pos[3][1] -= v_stairs_width
#         box_size[3][0] += v_stairs_width * 2
#         box_size[0][1] = box_size[1][1] =box_size[2][0]= box_size[3][0]

# max_size = plane_size + v_stairs_width * 2 * (v_stairs_num*2)  
# add_inverted_pyramid([0,0,0])
# add_inverted_pyramid([0,max_size,0])
# add_inverted_pyramid([-max_size,0,0])
# add_inverted_pyramid([0,-max_size,0])


# vs = scene.add_entity(
#     gs.morphs.MJCF(file="assets/terrain/el.xml",
#     pos=(0.0, 1.0, 0.0)
#     ),
#     vis_mode='collision'
# )

# robot = scene.add_entity(
#     gs.morphs.URDF(file="assets/urdf/CJ-003/urdf/CJ-003-wheelfoot.urdf",
#     pos=(0.0, 0.0, 0.7),
#     quat=(1, 0, 0, 0)
#     ),
#     # gs.morphs.MJCF(file="assets/mjcf/point_foot2/point_foot2.xml",
#     # pos=(0.0, 0.0, 0.65)
#     # ),
#     # vis_mode='collision'
# )

# height_field = cv2.imread("assets/terrain/png/stairs.png", cv2.IMREAD_GRAYSCALE)
# terrain_height = torch.tensor(height_field) * 0.1
# print(terrain_height.size())
# terrain = scene.add_entity(
#         morph=gs.morphs.Terrain(
#         # pos = (0.0,0.0,0.0),
#         height_field = height_field,
#         horizontal_scale=0.1, 
#         vertical_scale=0.001,
#         ),
#     )

# cam = scene.add_camera(
#     res    = (640, 480),
#     pos    = (3.5, 0.0, 2.5),
#     lookat = (0, 0, 0.5),
#     fov    = 30,
#     GUI    = False,
# )
scene.build(n_envs=1)

# for solver in scene.sim.solvers:
#     if not isinstance(solver, RigidSolver):
#         continue
#     rigid_solver = solver

# jnt_names = [
#     "left_hip_joint",
#     "left_thigh_joint",
#     "left_calf_joint",
#     "right_hip_joint",
#     "right_thigh_joint",
#     "right_calf_joint",
#     "left_wheel_joint",
#     "right_wheel_joint",
# ]
# dofs_idx = [robot.get_joint(name).dof_idx_local for name in jnt_names]
# robot.set_dofs_kp(
#     kp = np.array([40,40,40,40,40,40,40,40]),
#     dofs_idx_local = dofs_idx,
# )
# robot.set_dofs_kv(
#     kv = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]),
#     dofs_idx_local = dofs_idx,
# )
# left_knee = robot.get_joint("left_calf_joint")

# print(robot.n_links)
# link = robot.get_link("left_calf_Link")
# print(link.idx)
# link = robot.get_link("right_calf_Link")
# print(link.idx)


# 渲染rgb、深度、分割掩码和法线图
# rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)

# cam.start_recording()
cnt=2
while True:
    # link_pos = rigid_solver.get_links_pos([1,])
    # force = 100 * link_pos
    # rigid_solver.apply_links_external_force(
    #     force=force,
    #     links_idx=[1,],
    # )
    # robot.control_dofs_position(
    #         np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0 ,0.0 ,0.0]),
    #         dofs_idx,
    #     )
    scene.step()
    scene.clear_debug_objects()
    # scene.draw_debug_arrow(pos=torch.tensor([cnt,0,0]), vec=torch.tensor([cnt,1,1]).cpu(), radius=0.01, color=(1.0, 0.0, 1.0, 0.5))
    scene.draw_debug_sphere(pos=torch.tensor([cnt,0,0]), radius=torch.tensor([cnt,1,1]).sum().cpu(), color=(1.0, 0.0, 1.0, 0.5))
    cnt += 0.01
    if cnt>5:
        cnt = 2
    # print(robot.get_pos())
    # left_knee_pos = left_knee.get_pos()
    # print("left_knee_pos    ",left_knee_pos)
    # force = robot.get_links_net_contact_force()
    # dof_vel = robot.get_dofs_velocity()
    # print("dof_pos:",robot.get_dofs_position(dofs_idx))
    # time.sleep(0.1)
    # cam.render()
# cam.stop_recording(save_to_filename='video.mp4', fps=60)