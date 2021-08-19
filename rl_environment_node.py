#!/usr/bin/env python3
import math

import rospy
import gym
import numpy as np
from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry
import rock_walk


class RLEnv:

    def __init__(self):
        self.sim_freq = 240
        self.env = gym.make("RockWalk-v0", bullet_connection=0, step_freq=self.sim_freq, frame_skip=1, isTrain=False, episode_timout=math.inf)
        self.env.reset()
        self.action = np.zeros(self.env.action_space.shape[-1])
        self.pub_odom = rospy.Publisher('/rl_agent/odom', Odometry, queue_size=10)
        self.pub_obs = rospy.Publisher('/rl_agent/observation', Joy, queue_size=10)
        self.sub_action = rospy.Subscriber("/rl_agent/action", Joy, self.on_action)

    def on_action(self, msg):
        self.action = np.array(msg.axes)

    def sim_loop(self):
        rate = rospy.Rate(self.sim_freq)
        while not rospy.is_shutdown():
            observation, reward, done, info = self.env.step(self.action)
            p, q, v, w = info['odom']
            msg = Odometry()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'world'
            msg.pose.pose.position.x = p[0]
            msg.pose.pose.position.y = p[1]
            msg.pose.pose.position.z = p[2]
            msg.pose.pose.orientation.x = q[0]
            msg.pose.pose.orientation.y = q[1]
            msg.pose.pose.orientation.z = q[2]
            msg.pose.pose.orientation.w = q[3]
            msg.twist.twist.linear.x = v[0]
            msg.twist.twist.linear.y = v[1]
            msg.twist.twist.linear.z = v[2]
            msg.twist.twist.angular.x = w[0]
            msg.twist.twist.angular.y = w[1]
            msg.twist.twist.angular.z = w[2]
            self.pub_odom.publish(msg)
            joy_msg = Joy()
            joy_msg.header = msg.header
            joy_msg.axes = observation
            self.pub_obs.publish(joy_msg)
            rate.sleep()

def main():
    rospy.init_node("rl_env_node", anonymous=True)
    simulator = RLEnv()
    simulator.sim_loop()


if __name__ == "__main__":
    main()
