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
    freq = 240
    frame_skip = 24
    env = gym.make("MotionControlRnw-v0", bullet_connection=0, step_freq=freq, frame_skip=frame_skip)
    env = Monitor(env, "./log")
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=0 * np.ones(n_actions), sigma=1.5 * np.ones(n_actions))
    model = SAC("MlpPolicy",
                env,
                action_noise=action_noise,
                batch_size=128,
                train_freq=64,
                gradient_steps=64,
                learning_starts=20000,
                verbose=1,
                tensorboard_log="./rockwalk_tb/")
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./save/', name_prefix='rw_model')
    model.learn(total_timesteps=500000, log_interval=10, callback=checkpoint_callback)
    model.save_replay_buffer('./save/buffer')
    env.close()
