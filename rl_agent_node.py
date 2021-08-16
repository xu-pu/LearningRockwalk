#!/usr/bin/env python3

import os
import copy
import rospy
import time
import math
import numpy as np
from stable_baselines3 import SAC
from sensor_msgs.msg import Joy


class RnwAgent:

    def __init__(self):
        rospy.init_node("rnw_rl_agent", anonymous=True)

        """
        Object shape: [ellipse_a, ellipse_b, apex_x, apex_y, apex_z]
        """
        self.ellipse_params = [0.35, 0.35]
        self.apex_coordinates = [0, -0.35, 1.5]
        self.object_param = self.ellipse_params + self.apex_coordinates

        self.freq = 10.
        self.agent = SAC.load("./save_real/500K_steps_current_reward_function/rw_model_260000_steps.zip", device="cpu")

        self.pub_action = rospy.Publisher('/rl_agent/action', Joy, queue_size=10)
        self.sub_observation = rospy.Subscriber("/rl_agent/observation", Joy, self.on_observation)

        self.rate = rospy.Rate(self.freq)
        self.dt = 1/10.
        self.prev_speed = 0.
        self.action_scale = 0.18

    def on_observation(self, msg):
        obs = np.array(msg.axes+self.object_param)
        action, _ = self.agent.predict(obs, deterministic=True)
        action = action * self.action_scale
        action_msg = Joy()
        action_msg.header = copy.deepcopy(msg.header)
        action_msg.axes = action
        self.pub_action.publish(action_msg)
        rospy.loginfo(f'obs: {{obs}}, action: {{action}}')


if __name__ == "__main__":
    rnw_rl_agent = RnwAgent()
    rospy.spin()
