
import diambra.arena
from diambra.arena import SpaceTypes, EnvironmentSettingsMultiAgent, RecordingSettings
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium import spaces, Env
import numpy as np
import os
from collections import deque
from os.path import expanduser

class FightingRewardWrapper(Env):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # tracking variables for rewards
        self.prev_health = {'P1': None, 'P2': None}
        self.inactivity_timer = {'P1': 0, 'P2': 0}
        self.last_position = None
        self.attacks_thrown = {'P1': 0, 'P2': 0}
        self.hits_landed = {'P1': 0, 'P2': 0}
        self.total_damage = {'P1': 0, 'P2': 0}
        self.total_distance = {'P1': 0, 'P2': 0}
        
        
        self.action_history = deque(maxlen=5)
        
        #prints keys for debugging
        obs, _ = self.env.reset()
        print("Available observation keys:", obs.keys())
        
    def _is_attack_action(self, action):
        attack_actions = [1, 2, 3, 4, 5, 6]  
        #attack button action indices
        return action in attack_actions
    
    def _calculate_distance(self, obs):
        #returns the distance between the two players
        if 'P1' in obs and 'P2' in obs:
            return abs(obs['P1']['side'] - obs['P2']['side'])
        return 0
    
    def _calculate_movement(self, obs):
        # calculates the movement of the players based on position changes
        if self.last_position is None:
            self.last_position = self._calculate_distance(obs)
            return 0
        current_position = self._calculate_distance(obs)
        movement = abs(current_position - self.last_position)
        self.last_position = current_position
        return movement
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        self.prev_health = {'P1': None, 'P2': None}
        self.inactivity_timer = {'P1': 0, 'P2': 0}
        self.last_position = None
        self.attacks_thrown = {'P1': 0, 'P2': 0}
        self.hits_landed = {'P1': 0, 'P2': 0}
        self.total_damage = {'P1': 0, 'P2': 0}
        self.total_distance = {'P1': 0, 'P2': 0}
        self.action_history.clear()
        return obs, info
    #resets wrapper per episode
    

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        agent_rewards = {'agent_0': 0, 'agent_1': 0}
        # rewards for attempting attack
        if self._is_attack_action(action):
            self.attacks_thrown['P1'] += 1
        
        #rewards for damage dealt, penalties for damage taken
        if self.prev_health['P1'] is not None and self.prev_health['P2'] is not None:
            health_p1 = obs['P1']['health'][0]
            health_p2 = obs['P2']['health'][0]
            damage_p1 = self.prev_health['P1'] - health_p1
            damage_p2 = self.prev_health['P2'] - health_p2
            if damage_p2 > 0:  
                self.hits_landed['P1'] += 1
                self.total_damage['P1'] += damage_p2
                agent_rewards['agent_0'] += damage_p2 * 2  
            if damage_p1 > 0:  
                agent_rewards['agent_0'] -= 1  
            
            
            if damage_p1 > 0:  
                self.hits_landed['P2'] += 1
                self.total_damage['P2'] += damage_p1
                agent_rewards['agent_1'] += damage_p1 * 2  
            if damage_p2 > 0:  
                agent_rewards['agent_1'] -= 1  
        
        
        health_p1 = obs['P1']['health'][0]
        health_p2 = obs['P2']['health'][0]
        self.prev_health['P1'] = health_p1
        self.prev_health['P2'] = health_p2
        
        # rewards movement to avoid idling
        movement = self._calculate_movement(obs)
        self.total_distance['P1'] += movement
        self.total_distance['P2'] += movement
        
        
        if movement < 0.01:  
            self.inactivity_timer['P1'] += 1
            self.inactivity_timer['P2'] += 1
            agent_rewards['agent_0'] -= 0.5 * self.inactivity_timer['P1']
            agent_rewards['agent_1'] -= 0.5 * self.inactivity_timer['P2']
        else:
            self.inactivity_timer['P1'] = 0
            self.inactivity_timer['P2'] = 0
            agent_rewards['agent_0'] += movement * 0.1
            agent_rewards['agent_1'] += movement * 0.1
        
        # bonus for winning
        if terminated:
            health_p1 = obs['P1']['health'][0]
            health_p2 = obs['P2']['health'][0]
            if health_p1 > health_p2:
                agent_rewards['agent_0'] += 100  
                agent_rewards['agent_1'] += 0    
            elif health_p2 > health_p1:
                agent_rewards['agent_1'] += 100  
                agent_rewards['agent_0'] += 0    
        
        #stores stats in info
        info['attacks_thrown'] = self.attacks_thrown
        info['hits_landed'] = self.hits_landed
        info['total_damage'] = self.total_damage
        info['total_distance'] = self.total_distance
        
        
        return obs, agent_rewards['agent_0'], terminated, truncated, info

class FlattenObservationWrapper(Env):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.observation_space = self._flatten_space(env.observation_space)
        
        # converts multi-agent action space to single discrete space
        if isinstance(env.action_space, spaces.Dict):
            self.action_space = spaces.Discrete(env.action_space.spaces['agent_0'].n * env.action_space.spaces['agent_1'].n)
        else:
            self.action_space = env.action_space
        
    #flatten observation space definitions
    def _flatten_space(self, space, prefix=''):
        if isinstance(space, spaces.Dict):
            new_spaces = {}
            for key, value in space.spaces.items():
                new_spaces.update(self._flatten_space(value, f"{prefix}{key}_"))
            return spaces.Dict(new_spaces)
        elif isinstance(space, spaces.Tuple):
            new_spaces = {}
            for i, value in enumerate(space.spaces):
                new_spaces.update(self._flatten_space(value, f"{prefix}{i}_"))
            return spaces.Dict(new_spaces)
        else:
            return {prefix: space}
    
    # flatten observation data
    def _flatten_obs(self, obs, prefix=''):
        if isinstance(obs, dict):
            flattened = {}
            for key, value in obs.items():
                flattened.update(self._flatten_obs(value, f"{prefix}{key}_"))
            return flattened
        elif isinstance(obs, (list, tuple)):
            flattened = {}
            for i, value in enumerate(obs):
                flattened.update(self._flatten_obs(value, f"{prefix}{i}_"))
            return flattened
        else:
            return {prefix: obs}
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs), info
    
    def step(self, action):
        # convert flat action index into agent_0 and agent_1 actions
        if isinstance(self.env.action_space, spaces.Dict):
            n_actions = self.env.action_space.spaces['agent_0'].n
            action_p1 = action % n_actions
            action_p2 = action // n_actions
            
            actions = {
                'agent_0': action_p1,
                'agent_1': action_p2
            }
        else:
            actions = action
            
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return self._flatten_obs(obs), reward, terminated, truncated, info
    def render(self):
        return self.env.render()
    def close(self):
        return self.env.close()

def make_env():
    # Set up the environment settings
    settings = EnvironmentSettingsMultiAgent()
    settings.action_space = (SpaceTypes.DISCRETE, SpaceTypes.DISCRETE)
    settings.n_players = 2  
    settings.step_ratio = 6  
    settings.splash_screen = True  

    
    settings.characters = ("Urien", "Urien")
    settings.outfits = (2, 2)
    settings.role = (None, None)  
    settings.continue_game = 0.0  
    settings.show_final = False  

    #set up video recording settings
    home_dir = expanduser("~")
    recording_settings = RecordingSettings()
    recording_settings.dataset_path = os.path.join(home_dir, "DIAMBRA/episode_recording", "sfiii3n")
    recording_settings.username = "training_agent"
    recording_settings.save_video = True  
    recording_settings.save_frequency = 50000  

    # make env, and wrap with custom wrappers
    env = diambra.arena.make("sfiii3n", settings, episode_recording_settings=recording_settings)
    env = FightingRewardWrapper(env)
    env = FlattenObservationWrapper(env)
    return env

class CheckpointAndVideoCallback:
    def __init__(self, save_freq, save_path, name_prefix):
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.n_calls = 0
        
    def _on_step(self) -> bool:
        self.n_calls += 1
        if self.n_calls % self.save_freq == 0:
            
            self.model.save(f"{self.save_path}/{self.name_prefix}_{self.n_calls}_steps")
            print(f"\nSaved checkpoint at {self.n_calls} steps")
            
            
            if hasattr(self.model.env, 'envs'):
                for env in self.model.env.envs:
                    if hasattr(env, 'env') and hasattr(env.env, 'env'):
                        env.env.env.recording_settings.save_video = True
        return True

def main():
    
    env = DummyVecEnv([make_env])

    
    checkpoint_callback = CheckpointAndVideoCallback(
        save_freq=50000,
        save_path="./checkpoints/",
        name_prefix="ppo_selfplay_agent"
    )

    
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=1e-4,  
        n_steps=2048,
        batch_size=128,  
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"  
    )

    
    total_timesteps = 2_000_000  
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=checkpoint_callback
    )

    
    model_path = os.path.join(os.path.dirname(__file__), "ppo_selfplay_agent")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    
    env.close()

if __name__ == '__main__':
    main() 