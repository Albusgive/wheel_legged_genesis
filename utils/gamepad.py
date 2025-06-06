import pygame
import numpy as np
from pathlib import Path
class control_gamepad:
    def __init__(self,command_cfg,command_scale=None):
        pygame.init()
        pygame.joystick.init()
        self.use_gamepad = True

        # 获取连接的游戏手柄数量
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            print("no gamepad,open keyboard window")
            self.use_gamepad = False
            screen_width = 400
            screen_height = 580
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            image_path1 = "../picture/keyboard_key.png" #sim2sim
            image_path2 = "picture/keyboard_key.png"
            image_center = (200, 280)
            pygame.display.set_caption("请用此窗口进行键盘控制(This use your keyboard)")
            image_surface = pygame.Surface((800, 600), pygame.SRCALPHA)
            image_surface.fill((255, 255, 255, 0))  # 透明初始化
            try:
                if Path(image_path1).exists():
                    image_surface = pygame.image.load(image_path1)
                elif Path(image_path2).exists():
                    image_surface = pygame.image.load(image_path2)
                else:
                    print(f"无法加载图片: picture/keyboard_key.png")
            except pygame.error as e:
                print(f"无法加载图片: picture/keyboard_key.png")
                print(e)
                pygame.quit()
                exit()
            image_rect = image_surface.get_rect()
            image_rect.center = image_center
            self.screen.fill((255, 255, 255)) # 背景
            self.screen.blit(image_surface, image_rect)
            pygame.display.flip() # 更新屏幕显示 (一次性) 
        else:
            # 选择第一个手柄
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"link gamepad: {self.joystick.get_name()}")
        self.num_commands = command_cfg["num_commands"]
        self.command_cfg = command_cfg
        self.commands = np.zeros(self.num_commands)
        self.command_scale = command_scale
        if self.command_scale is None:
            self.command_scale = [1.0, 1.0, 1.0, 0.05, 0.05, 1.0]
        # 控制腿flag
        self.leg_length_flag_1 = False
        self.leg_length_flag_2 = False
        #keyboard控制速度
        self.quiet_walking = 1.0
    
    def get_commands(self):
        pygame.event.pump()
        reset_flag = False
        if self.use_gamepad:
            for event in pygame.event.get():
                if event.type == pygame.JOYHATMOTION:
                    # print("event:", event)
                    match event.hat:
                        case 0: #xx value[0] yy value[1] 太空步
                            if event.value[1] == 1:
                                self.commands[3] = self.command_cfg["leg_length_range"][0]
                                self.commands[4] = self.command_cfg["leg_length_range"][1]
                            elif event.value[1] == -1:
                                self.commands[3] = self.command_cfg["leg_length_range"][1]
                                self.commands[4] = self.command_cfg["leg_length_range"][0]
                            # 铁山靠
                            if event.value[0] == 1:
                                self.commands[5] = self.command_cfg["tsk_range"][0]
                            elif event.value[0] == -1:
                                self.commands[5] = self.command_cfg["tsk_range"][1]
                elif event.type == pygame.JOYBUTTONDOWN:
                    # print(f"按钮 {event.button} 被按下。")
                    # X-0 A-1 B-2 Y-3 LB-4 RB-5
                    match event.button:
                        case 0:
                            reset_flag=True
                        case 4:
                            self.leg_length_flag_1 = True
                        case 5:
                            self.leg_length_flag_2 = True
                        case 2: # B 铁山靠回正
                            self.commands[5] = 0
                elif event.type == pygame.JOYBUTTONUP:
                    # print(f"按钮 {event.button} 被释放。")
                    match event.button:
                        case 4:
                            self.leg_length_flag_1 = False
                        case 5:
                            self.leg_length_flag_2 = False
                elif event.type == pygame.JOYAXISMOTION:
                    # print(f"轴 {event.axis},{event.value}")
                    match event.axis:
                        case 0: #lx 左正右负
                            self.commands[1] = -event.value * self.command_scale[1]
                        case 1: #ly 前正后负
                            self.commands[0] = -event.value * self.command_scale[0]
                        case 2: #rx 左正右负
                            self.commands[2] = -event.value * self.command_scale[2]
                        case 3: #lt 增加身高
                            if self.leg_length_flag_1:
                                self.commands[3] += (event.value+1) * self.command_scale[3]
                            elif self.leg_length_flag_2:
                                self.commands[4] += (event.value+1) * self.command_scale[4]
                            else:
                                self.commands[3] += (event.value+1) * self.command_scale[3]
                                self.commands[4] += (event.value+1) * self.command_scale[4]
                        case 4: #rt 减少身高
                            if self.leg_length_flag_1:
                                self.commands[3] -= (event.value+1) * self.command_scale[3]
                            elif self.leg_length_flag_2:
                                self.commands[4] -= (event.value+1) * self.command_scale[4]
                            else:
                                self.commands[3] -= (event.value+1) * self.command_scale[3]
                                self.commands[4] -= (event.value+1) * self.command_scale[4]
                        # case 5: #ry 铁山靠
                        #     self.commands[3] -= event.value * self.command_scale[3]
                        #     self.commands[4] += event.value * self.command_scale[4]
        else:
            for event in pygame.event.get():  # 获取事件队列中的所有事件
                if event.type == pygame.QUIT:  # 用户点击窗口关闭按钮
                    running = False
                elif event.type == pygame.KEYDOWN:  # 键盘按键按下事件
                    match event.key:
                        case pygame.K_w:
                            self.commands[0] = self.command_scale[0] * self.quiet_walking
                        case pygame.K_s:
                            self.commands[0] = -self.command_scale[0] * self.quiet_walking
                        case pygame.K_a:
                            self.commands[1] = self.command_scale[1] * self.quiet_walking
                        case pygame.K_d:
                            self.commands[1] = -self.command_scale[1] * self.quiet_walking
                        case pygame.K_q:
                            self.commands[2] = +self.command_scale[2] * self.quiet_walking
                        case pygame.K_e:
                            self.commands[2] = -self.command_scale[2] * self.quiet_walking
                        case pygame.K_SPACE:
                            if self.leg_length_flag_1:
                                self.commands[3] = +self.command_scale[3] * 100.0
                            elif self.leg_length_flag_2:
                                self.commands[4] = +self.command_scale[4] * 100.0
                            else:
                                self.commands[3] = +self.command_scale[3] * 100.0
                                self.commands[4] = +self.command_scale[4] * 100.0
                        case pygame.K_c:
                            if self.leg_length_flag_1:
                                self.commands[3] = -self.command_scale[3] * 100.0
                            elif self.leg_length_flag_2:
                                self.commands[4] = -self.command_scale[4] * 100.0
                            else:
                                self.commands[3] = -self.command_scale[3] * 100.0
                                self.commands[4] = -self.command_scale[4] * 100.0
                        case pygame.K_LCTRL:
                            if self.leg_length_flag_1:
                                self.commands[3] = -self.command_scale[3] * 100.0
                            elif self.leg_length_flag_2:
                                self.commands[4] = -self.command_scale[4] * 100.0
                            else:
                                self.commands[3] = -self.command_scale[3] * 100.0
                                self.commands[4] = -self.command_scale[4] * 100.0
                        case pygame.K_r:
                            reset_flag=True
                        case pygame.K_LSHIFT:
                            self.quiet_walking=0.5
                        case pygame.K_1:
                            self.leg_length_flag_1 = True
                        case pygame.K_2:
                            self.leg_length_flag_2 = True
                elif event.type == pygame.KEYUP:  # 键盘按键释放事件
                    match event.key:
                        case pygame.K_w:
                            self.commands[0] = 0
                        case pygame.K_s:
                            self.commands[0] = 0
                        case pygame.K_a:
                            self.commands[1] = 0
                        case pygame.K_d:
                            self.commands[1] = 0
                        case pygame.K_q:
                            self.commands[2] = 0
                        case pygame.K_e:
                            self.commands[2] = 0
                        case pygame.K_LSHIFT:
                            self.quiet_walking=1.0
                        case pygame.K_1:
                            self.leg_length_flag_1 = False
                        case pygame.K_2:
                            self.leg_length_flag_2 = False
        self.commands_clip()
        return self.commands,reset_flag
    
    def commands_clip(self):
        # lin_vel_x
        if self.commands[0] <= self.command_cfg["lin_vel_x_range"][0]:
            self.commands[0] = self.command_cfg["lin_vel_x_range"][0]
        elif self.commands[0] >= self.command_cfg["lin_vel_x_range"][1]:
            self.commands[0] = self.command_cfg["lin_vel_x_range"][1]

        #lin_vel_y
        if self.commands[1] <= self.command_cfg["lin_vel_y_range"][0]:
            self.commands[1] = self.command_cfg["lin_vel_y_range"][0]
        elif self.commands[1] >= self.command_cfg["lin_vel_y_range"][1]:
            self.commands[1] = self.command_cfg["lin_vel_y_range"][1]

        #ang_vel
        if self.commands[2] <= self.command_cfg["ang_vel_range"][0]:
            self.commands[2] = self.command_cfg["ang_vel_range"][0]
        elif self.commands[2] >= self.command_cfg["ang_vel_range"][1]:
            self.commands[2] = self.command_cfg["ang_vel_range"][1]

        #leg_length
        if self.commands[3] <= self.command_cfg["leg_length_range"][0]:
            self.commands[3] = self.command_cfg["leg_length_range"][0]
        elif self.commands[3] >= self.command_cfg["leg_length_range"][1]:
            self.commands[3] = self.command_cfg["leg_length_range"][1]
        
        if self.commands[4] <= self.command_cfg["leg_length_range"][0]:
            self.commands[4] = self.command_cfg["leg_length_range"][0]
        elif self.commands[4] >= self.command_cfg["leg_length_range"][1]:
            self.commands[4] = self.command_cfg["leg_length_range"][1]