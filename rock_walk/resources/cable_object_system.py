import pybullet as bullet
import numpy as np

import os
from rock_walk.resources.utils import *


class CableObjectSystem:

    def __init__(self, client, yaw_spawn):
        self.clientID = client
        f_name = os.path.join(os.path.dirname(__file__), 'models/obj_cable.urdf')
        base_position = [0.35*np.sin(yaw_spawn), -0.35*np.cos(yaw_spawn), 1.50]
        self.robotID = bullet.loadURDF(
            fileName=f_name, basePosition=base_position, useFixedBase=1, physicsClientId=client
        )
        self.link_idx_obj = 7

    def get_ids(self):
        return self.robotID, self.clientID

    def get_dynamics_info(self):
        print(bullet.getDynamicsInfo(self.robotID, -1, self.clientID))

    def apply_action(self, action):
        bullet.setJointMotorControl2(self.robotID,
                                     jointIndex=0,
                                     controlMode=bullet.VELOCITY_CONTROL,
                                     targetVelocity=action[0],
                                     physicsClientId=self.clientID)
        bullet.setJointMotorControl2(self.robotID,
                                     jointIndex=1,
                                     controlMode=bullet.VELOCITY_CONTROL,
                                     targetVelocity=action[1],
                                     physicsClientId=self.clientID)
        bullet.setJointMotorControl2(self.robotID,
                                     jointIndex=2,
                                     controlMode=bullet.VELOCITY_CONTROL,
                                     targetVelocity=action[2],
                                     physicsClientId=self.clientID)
        # passive joints
        bullet.setJointMotorControl2(self.robotID,
                                     jointIndex=3,
                                     controlMode=bullet.VELOCITY_CONTROL,
                                     force=0,
                                     physicsClientId=self.clientID)
        bullet.setJointMotorControl2(self.robotID,
                                     jointIndex=4,
                                     controlMode=bullet.VELOCITY_CONTROL,
                                     force=0,
                                     physicsClientId=self.clientID)
        bullet.setJointMotorControl2(self.robotID,
                                     jointIndex=5,
                                     controlMode=bullet.VELOCITY_CONTROL,
                                     force=0,
                                     physicsClientId=self.clientID)
        bullet.setJointMotorControl2(self.robotID,
                                     jointIndex=6,
                                     controlMode=bullet.VELOCITY_CONTROL,
                                     force=0,
                                     physicsClientId=self.clientID)

    def get_observation(self):
        link_pos_world = bullet.getLinkState(self.robotID, linkIndex=2, physicsClientId=self.clientID)[0]
        link_vel_world = bullet.getLinkState(self.robotID, linkIndex=2, computeLinkVelocity=1, physicsClientId=self.clientID)[-2]
        state = [link_pos_world[0], link_pos_world[1], link_pos_world[2],
                 link_vel_world[0], link_vel_world[1], link_vel_world[2]]
        return state

    def get_object_state(self):
        link_pos_world = bullet.getLinkState(self.robotID, linkIndex=2, physicsClientId=self.clientID)[0]
        link_vel_world = bullet.getLinkState(self.robotID, linkIndex=2, computeLinkVelocity=1, physicsClientId=self.clientID)[-2]
        state = [link_pos_world[0], link_pos_world[1], link_pos_world[2],
                 link_vel_world[0], link_vel_world[1], link_vel_world[2]]
        return state
        pass
