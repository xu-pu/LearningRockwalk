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
            self.y = -self.joy.get_axis(4)
            #print(f'Input x={self.x}, y={self.y}')


if __name__ == "__main__":
    js = Joystick()
    sim_hz = 5000
    env = gym.make("RockWalk-v0", bullet_connection=1, step_freq=sim_hz, frame_skip=1)
    env.reset()
    while True:
        action = 10 * np.array([js.x,js.y])
        env.step(action)
        sleep(1/sim_hz)
