import logging
import os
import numpy as np
import diambra.arena
from stable_baselines3 import PPO
from diambra.arena import SpaceTypes, EnvironmentSettingsMultiAgent, WrappersSettings
from gymnasium import Wrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from diambra.arena.env_settings import Roles

#learning rate and clip range linear schedule
def linear_schedule(start, end):
    def schedule(progress):
        return end + progress * (start - end)
    return schedule

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Custom wrapper to flatten the action space
class FlattenActionWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.agent_0_action_space = env.action_space.spaces['agent_0']
        self.agent_1_action_space = env.action_space.spaces['agent_1']
        self.action_space = self.agent_0_action_space  

    def step(self, action_0):
        # Random action for agent_1, ppo controles agent_0
        action_1 = self.agent_1_action_space.sample()
        action_dict = {
            'agent_0': action_0,
            'agent_1': action_1
        }
        return self.env.step(action_dict)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def main():
    #multi-agent environment settings
    settings = EnvironmentSettingsMultiAgent()
    settings.characters = ("Urien", "Urien")
    settings.outfits = (1, 2)
    settings.action_space = (SpaceTypes.DISCRETE, SpaceTypes.DISCRETE)
    settings.role = (Roles.P1, Roles.P2) 

    wrappers_settings = WrappersSettings()
    wrappers_settings.flatten = True
    wrappers_settings.normalize_reward = True
    wrappers_settings.no_attack_buttons_combinations = True
    wrappers_settings.stack_frames = 4
    wrappers_settings.dilation = 1
    wrappers_settings.add_last_action = True
    wrappers_settings.stack_actions = 12
    wrappers_settings.scale = True
    wrappers_settings.exclude_image_scaling = True
    wrappers_settings.role_relative = True
    wrappers_settings.filter_keys = [
        "agent_0_action",
        "agent_0_own_health", "agent_0_own_side", "agent_0_own_character",
        "agent_0_opp_health", "agent_0_opp_side", "agent_0_opp_character",
        "stage", "timer", "frame"
    ]
    #create env and wrap
    env = diambra.arena.make("sfiii3n", settings, wrappers_settings, render_mode=None)
    env = FlattenActionWrapper(env) #flatten action space 
    env = DummyVecEnv([lambda: env]) #wrap in DummyVecEnv for stable baselines3

    model_path = "ppo_selfplay_agent"

    # train or load model
    if not os.path.exists(model_path + ".zip"):
        logging.info("No existing model found. Training a new one...")

        #ppo hyperparams
        policy_kwargs = dict(net_arch=[64, 64])
        learning_rate = linear_schedule(2.5e-4, 2.5e-6)
        clip_range = linear_schedule(0.15, 0.025)
        clip_range_vf = clip_range

        #save model every 5000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path="./checkpoints/",
            name_prefix="ppo_selfplay_agent"
        )

        #create ppo agent
        agent = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            gamma=0.94,
            batch_size=512,
            n_epochs=4,
            n_steps=512,
            learning_rate=learning_rate,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            policy_kwargs=policy_kwargs,
            tensorboard_log=None,
        )
        #train agent
        agent.learn(total_timesteps=1000, callback=checkpoint_callback)
        agent.save(model_path)
        logging.info(f"Model saved as '{model_path}.zip'.")
    else:
        logging.info("Loading existing model...")
        agent = PPO.load(model_path, env=env)

    env.close()

if __name__ == '__main__':
    main()
