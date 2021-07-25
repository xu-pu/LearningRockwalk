import sys
import pygame
import threading
from time import sleep


class Joystick:

    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
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
            self.z = self.joy.get_axis(1)
