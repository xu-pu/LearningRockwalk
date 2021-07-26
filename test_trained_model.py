import os
import gym
import numpy as np
import rock_walk
import time
import pybullet as bullet
import matplotlib.pyplot as plt

from stable_baselines3 import TD3, SAC, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder

if __name__ == "__main__":
    freq = 240.
    frame_skip = 1
    env = gym.make("MotionControlRnw-v0", bullet_connection=1, step_freq=freq, frame_skip=frame_skip)
    trained_model = SAC.load("./save/rw_model_150000_steps", device="cpu")
    print("Trained model loaded")
    obs = env.reset()
    for count in range(5000):
        action, _states = trained_model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        time.sleep(1./240.)
