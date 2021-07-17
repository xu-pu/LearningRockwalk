import pygame
import sys
from time import sleep
import threading
import gym
import rock_walk
import numpy as np


class Joystick:

    def __init__(self):
        self.x = 0
        self.y = 0
        pygame.init()
        pygame.joystick.init()
        print(f'{pygame.joystick.get_count()} joysticks found')
        if pygame.joystick.get_count() == 0:
            print('joystick not found')
            sys.exit(1)
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()
        self.thread = threading.Thread(target=self.body)
        self.thread.start()

    def body(self):
        while True:
            sleep(1./50)
            pygame.event.get()
            self.x = self.joy.get_axis(3)
            self.y = self.joy.get_axis(4)


if __name__ == "__main__":
    js = Joystick()
    sim_hz = 5000
    joystick_scale = 10
    env = gym.make("RockWalk-v0", bullet_connection=1, step_freq=sim_hz, frame_skip=1, episode_timeout=1000)
    env.reset()
    while True:
        sleep(1./sim_hz)
        action = joystick_scale * np.array([js.y, js.x])
        obs, rewards, done, info = env.step(action)
        if done:
            env.reset()
