import os
import pybullet
from pprint import pprint
import numpy as np


class MotionDevice(object):

    def __init__(self, client):
        pybullet.setAdditionalSearchPath(
            os.path.join(os.path.dirname(__file__), 'models')
        )
        self.clientID = client
        self.bodyID = pybullet.loadURDF(
            'motion_device.urdf',
            [0, 0, 1],
            pybullet.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True
        )
        self.joint_info = [pybullet.getJointInfo(self.bodyID, i) for i in range(pybullet.getNumJoints(self.bodyID))]
        self.joint_name2id = dict([(item[1].decode("UTF-8"), item[0]) for item in self.joint_info])
        self.link_name2id = dict([(item[12].decode("UTF-8"), item[0]) for item in self.joint_info])
        pprint(self.joint_name2id)
        pprint(self.link_name2id)
        self.joint_id2name = dict([(item[0], item[1].decode("UTF-8")) for item in self.joint_info])
        self.link_id2name = dict([(item[0], item[12].decode("UTF-8")) for item in self.joint_info])

    def apply_action(self, action):
        self.apply_joint_vel('axis_x', action[0])
        self.apply_joint_vel('axis_y', action[1])
        self.apply_joint_vel('axis_z', action[2])
        self.set_joint_passive('hinge_x')
        self.set_joint_passive('hinge_y')
        self.set_joint_passive('hinge_z')

    def apply_joint_vel(self, name, vel):
        pybullet.setJointMotorControl2(
            self.bodyID,
            jointIndex=self.joint_name2id[name],
            controlMode=pybullet.VELOCITY_CONTROL,
            targetVelocity=vel,
            force=500,
            physicsClientId=self.clientID
        )

    def set_joint_passive(self, name):
        pybullet.setJointMotorControl2(
            self.bodyID,
            jointIndex=self.joint_name2id[name],
            controlMode=pybullet.VELOCITY_CONTROL,
            force=0,
            physicsClientId=self.clientID
        )

    @property
    def end_effector_link_id(self):
        return self.joint_info[-1][0]

    @property
    def end_effector_link_name(self):
        return self.link_id2name[self.end_effector_link_id]

