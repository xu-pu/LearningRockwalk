import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.sac import SACTrainer
from rock_walk.envs.motion_control_rnw_env import MotionControlRnwEnv
import ray
from ray import tune

if __name__ == "__main__":

    ray.init()
    tune.run(
        "SAC",
        stop={"timesteps_total": 1e6},
        config={
            "framework": "torch",
            "env": MotionControlRnwEnv,
            "rollout_fragment_length": 100,
            "learning_starts": 2000,
            "num_gpus": 1,
            "num_workers": 6
        },
    )
