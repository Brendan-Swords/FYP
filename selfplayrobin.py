import logging
import os
import numpy as np
import random
import diambra.arena
from stable_baselines3 import PPO
from diambra.arena import SpaceTypes, EnvironmentSettingsMultiAgent, WrappersSettings
from gymnasium.spaces import MultiDiscrete, Dict
from gymnasium import Wrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

#learning rate and clip range linear schedule
def linear_schedule(start, end):
    def schedule(progress):
        return end + progress * (start - end)
    return schedule


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Custom wrapper to flatten the action space
class FlattenObservationDictWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.observation_space, Dict):
            raise ValueError("FlattenObservationDictWrapper requires a Dict observation space.")
        self.observation_space = env.observation_space

    def observation(self, observation):
        return observation


# Wrapper to inject PPO agents into a multi-agent DIAMBRA environment
# Agent_0 = learning agent
# Agent_1 = opponent from past checkpoints
class SelfPlayActionWrapper(Wrapper):
    def __init__(self, env, opponent_agent):
        super().__init__(env)
        self.learning_agent = None  #assigned later after ppo instantiation
        self.opponent_agent = opponent_agent
        self.observation_space = env.observation_space
        self.action_space = MultiDiscrete([
            env.action_space.spaces['agent_0'].n,
            env.action_space.spaces['agent_1'].n
        ])

    def reset(self):
        obs = self.env.reset()
        self.current_obs = obs
        return obs

    def step(self, _):
        # use learning agent for P1 (agent_0), opponent agent for P2 (agent_1)
        action_p1, _ = self.learning_agent.predict(self.current_obs, deterministic=True)
        action_p2, _ = self.opponent_agent.predict(self.current_obs, deterministic=True)

        action_dict = {
            'agent_0': action_p1,
            'agent_1': action_p2
        }

        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        self.current_obs = obs
        return obs, reward, terminated, truncated, info


# load a random opponent agent
# if none exist, create a dummy agent
def load_opponent(opponent_dir, env):
    opponents = sorted([f for f in os.listdir(opponent_dir) if f.endswith(".zip")])
    if opponents:
        opponent_path = os.path.join(opponent_dir, random.choice(opponents))
        logging.info(f"Loading opponent from: {opponent_path}")
        return PPO.load(opponent_path, env=env)
    else:
        logging.warning("No opponents found. Using random dummy model.")
        return PPO("MultiInputPolicy", env)



def main():
    
    settings = EnvironmentSettingsMultiAgent()
    settings.characters = ("Urien", "Urien")
    settings.action_space = (SpaceTypes.DISCRETE, SpaceTypes.DISCRETE)

    wrappers_settings = WrappersSettings()
    wrappers_settings.flatten = True
    wrappers_settings.filter_keys = [
        "frame", "stage", "timer",
        "P1_character", "P1_health", "P1_side",
        "P2_character", "P2_health", "P2_side"
    ]
    # creates base env, shared accross selfplay/opponent loading
    base_env = diambra.arena.make("sfiii3n", settings, wrappers_settings, render_mode="rgb_array")
    base_env = FlattenObservationDictWrapper(base_env)

    # load opponent agent
    opponent_dir = "./opponents"
    os.makedirs(opponent_dir, exist_ok=True)
    opponent_env = DummyVecEnv([lambda: base_env])
    opponent_agent = load_opponent(opponent_dir, opponent_env)

    # create self-play environment with the opponent agent
    sp_env = SelfPlayActionWrapper(base_env, opponent_agent)
    train_env = DummyVecEnv([lambda: sp_env])

    # hyperparams
    policy_kwargs = dict(net_arch=[64, 64])
    learning_rate = linear_schedule(2.5e-4, 2.5e-6)
    clip_range = linear_schedule(0.15, 0.025)

    agent = PPO(
        "MultiInputPolicy",
        train_env,
        verbose=1,
        gamma=0.94,
        batch_size=512,
        n_epochs=4,
        n_steps=512,
        learning_rate=learning_rate,
        clip_range=clip_range,
        clip_range_vf=clip_range,
        policy_kwargs=policy_kwargs
    )

    # links agent to selfplay wrapper
    sp_env.learning_agent = agent

    # sets up video recording
    video_folder = "/mnt/c/Users/Brendan/Desktop/Final/3rdStrike/videos"
    os.makedirs(video_folder, exist_ok=True)

    train_env = VecVideoRecorder(
        train_env,
        video_folder,
        record_video_trigger=lambda step: step % 5000 == 0,
        video_length=1000,
        name_prefix="sfiii3n_selfplay"
    )

    # checkpoint dir
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)

    #training loop with round robin updates
    total_timesteps = 50000
    save_interval = 10000

    for step in range(0, total_timesteps, save_interval):
        logging.info(f"Training: {step} to {step + save_interval}")
        agent.learn(total_timesteps=save_interval, reset_num_timesteps=False)

        
        checkpoint_path = os.path.join(model_dir, f"ppo_selfplay_{step + save_interval}.zip")
        agent.save(checkpoint_path)
        logging.info(f"Saved checkpoint: {checkpoint_path}")

        #reload random opponent from dir
        opponent_agent = load_opponent(opponent_dir, opponent_env)
        sp_env.opponent_agent = opponent_agent

    train_env.close()
    logging.info("Training complete.")

if __name__ == "__main__":
    main()
