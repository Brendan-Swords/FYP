import diambra.arena
from diambra.arena import SpaceTypes, EnvironmentSettingsMultiAgent, WrappersSettings
import logging
import numpy as np
from pprint import pformat  


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


settings = EnvironmentSettingsMultiAgent()
settings.characters = ("Urien", "Urien")
settings.n_players = 2
settings.action_space = (SpaceTypes.DISCRETE, SpaceTypes.DISCRETE)

wrappers_settings = WrappersSettings()
wrappers_settings.flatten = True
wrappers_settings.filter_keys = [
    "frame", "stage", "timer",
    "P1_character", "P1_health", "P1_side",
    "P2_character", "P2_health", "P2_side"
]

env = diambra.arena.make("sfiii3n", settings, wrappers_settings, render_mode="rgb_array")

obs = env.reset()


logging.info("=== OBSERVATION KEYS ===")
logging.info(pformat(obs))

if isinstance(obs, dict):
    for key, value in obs.items():
        logging.info(f"{key}: {type(value)} - {value.shape if isinstance(value, np.ndarray) else value}")


for _ in range(10):  
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    
    logging.info(f"=== INFO STRUCTURE ===\n{pformat(info)}")

    if done:
        obs = env.reset()

env.close()
