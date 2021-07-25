import pybullet as bullet
import os


class Goal:
    def __init__(self, client, target_x, target_y):
        self.clientID = client
        f_name = os.path.join(os.path.dirname(__file__), 'models/goal_direction.urdf')
        self.goalID = bullet.loadURDF(
            fileName=f_name,
            basePosition=[target_x, target_y, -0.04],
            physicsClientId=self.clientID
        )
