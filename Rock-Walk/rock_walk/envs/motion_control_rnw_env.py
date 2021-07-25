import gym
import math
import time
import numpy as np
import pybullet as bullet
import matplotlib.pyplot as plt
import pybullet

from rock_walk.resources.plane import Plane
from rock_walk.resources.goal import Goal
from rock_walk.resources.motion_controlled_cone import MotionControlledCone

import pkgutil
# for rendering with bullet.ER_TINY_RENDERER
# egl = pkgutil.get_loader('eglRenderer')
# self._plugin = bullet.loadPlugin(egl.get_filename(), "_eglRendererPlugin")


class MotionControlRnwEnv(gym.Env):

    def __init__(self, bullet_connection, step_freq, frame_skip, episode_timeout=5.):

        self._bullet_connection = bullet_connection
        self._frame_skip = frame_skip
        self._ep_timeout = episode_timeout
        self._mu_min = 0.2
        self._mu_max = 2.0

        self._cam_dist = 4
        self._cam_yaw = 90
        self._cam_pitch = -30

        self._mu_cone_ground = 100
        self._action_scaling = 1

        self.done = False

        self.goal_position = np.array([0, 7.5])

        action_low = np.array([-1, -1, -1], dtype=np.float64)
        action_high = np.array([1, 1, 1], dtype=np.float64)
        self.action_space = gym.spaces.box.Box(low=action_low, high=action_high)

        obs_low = np.array([-100, -100, -100, -100, -np.pi, -np.pi, -np.pi, -10, -10, -10, -10, -10], dtype=np.float64)
        obs_high = np.array([100, 100, 100, 100, np.pi, np.pi, np.pi, 10, 10, 10, 10, 10], dtype=np.float64)
        self.observation_space = gym.spaces.box.Box(low=obs_low, high=obs_high)

        self.bullet_setup(bullet_connection)

        self.np_random, _ = gym.utils.seeding.np_random()
        self.reset()

    def step(self, action):
        if time.time() - self.start_time > self._ep_timeout:
            print("terminated: timout")
            self.done = True
        for _ in range(self._frame_skip):
            self.cone.apply_action(action)
            bullet.stepSimulation()

        p, q, v, w = self.cone.get_cone_odom()
        dist = np.linalg.norm(self.goal_position-p[:2])
        reward = 0
        if dist < self.min_dist:
            reward = self.min_dist - dist
            self.min_dist = dist

        if self._bullet_connection == 2:
            self.adjust_camera_pose()

        return self.observe(), reward, self.done, dict()

    def reset(self):
        bullet.resetSimulation(self.clientID)
        bullet.setGravity(0, 0, -9.8)
        self.initialize_physical_objects()
        self.done = False
        self.start_time = time.time()
        if self._bullet_connection == 2:
            self.adjust_camera_pose()
        p, q, v, w = self.cone.get_cone_odom()
        self.min_dist = np.linalg.norm(self.goal_position-p[:2])
        return self.observe()

    def render(self, mode='human'):
        pass

    def observe(self):
        """
        :return: observation := ( goal_position * 2, rnw_state * 10 )
        """
        return np.concatenate(
            (self.goal_position, np.array(self.cone.get_rnw_state()))
        )

    def close(self):
        bullet.disconnect(self.clientID)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def bullet_setup(self, bullet_connection):
        if bullet_connection == 0:
            self.clientID = bullet.connect(bullet.DIRECT)
            # bullet.setDefaultContactERP(0.9)
        elif bullet_connection == 1:
            self.clientID = bullet.connect(bullet.GUI)
        elif bullet_connection == 2:
            self.clientID = bullet.connect(bullet.SHARED_MEMORY)
            bullet.configureDebugVisualizer(bullet.COV_ENABLE_GUI,0)

    def initialize_physical_objects(self):
        self.goal = Goal(self.clientID, self.goal_position[0], self.goal_position[1])
        self.plane = Plane(self.clientID)
        self.cone = MotionControlledCone(self.clientID)
        self.cone.mass = 10
        self.cone.set_strength_mass_ratio(0.3)
        self.plane.lateral_friction = self._mu_cone_ground
        self.cone.lateral_friction = self._mu_cone_ground

    def adjust_camera_pose(self):
        base_pos, _ = bullet.getBasePositionAndOrientation(self.cone.bodyID, self.clientID)
        bullet.resetDebugVisualizerCamera(cameraDistance=self._cam_dist,
                                          cameraYaw=self._cam_yaw,
                                          cameraPitch=self._cam_pitch,
                                          cameraTargetPosition=base_pos)