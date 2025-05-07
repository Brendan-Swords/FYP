import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from diambra.arena import load_settings_flat_dict, SpaceTypes
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage


CHECKPOINT_PATH = "/mnt/c/Users/Brendan/Desktop/Final/3rdStrike/results/sfiii3n/sf3_p1_vs_ai_ppo/model/0_autosave_1550000"
NUM_EPISODES = 5
VIDEO_FOLDER = "./videos/"


settings_dict = {
    "game_id": "sfiii3n",
    "step_ratio": 6,
    "frame_shape": (224, 384, 1),
    "continue_game": 0.0,
    "action_space": SpaceTypes.DISCRETE,
    "characters": "Urien",
    "outfits": 2,
    "role": 2,  
    "n_players": 1,
    "difficulty": 3,
    "disable_joystick": True,
    "disable_keyboard": True,
}

wrappers_dict = {
    "normalize_reward": True,
    "no_attack_buttons_combinations": True,
    "stack_frames": 1,
    "dilation": 1,
    "add_last_action": True,
    "stack_actions": 8,
    "scale": True,
    "exclude_image_scaling": True,
    "role_relative": True,
    "flatten": True,
    "filter_keys": [
        "action", "own_health", "opp_health", "own_side", "opp_side", "opp_character", "stage", "timer"
    ],
}


settings = load_settings_flat_dict(EnvironmentSettings, settings_dict)
wrappers = load_settings_flat_dict(WrappersSettings, wrappers_dict)
env, _ = make_sb3_env(settings.game_id, settings, wrappers, render_mode="rgb_array")  



env = VecVideoRecorder(
    env,
    video_folder=VIDEO_FOLDER,
    record_video_trigger=lambda step: True,
    video_length=5000,
    name_prefix="ppo_eval_ep"
)


model = PPO.load(CHECKPOINT_PATH, env=env)
print("Loaded model:", CHECKPOINT_PATH)


for ep in range(NUM_EPISODES):
    obs = env.reset()
    print("Obs keys:", obs.keys()) 
    print("Obs shape:", obs["observation"].shape if "observation" in obs else next(iter(obs.values())).shape)

    done = False
    ep_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True) 
        obs, reward, done, _ = env.step(action) 
        ep_reward += reward
        time.sleep(0.01)
    print(f"Episode {ep + 1} reward: {ep_reward}")

env.close()
