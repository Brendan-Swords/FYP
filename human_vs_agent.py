
import diambra.arena
from diambra.arena import SpaceTypes, EnvironmentSettingsMultiAgent
from stable_baselines3 import PPO
import numpy as np
from pynput import keyboard
from collections import defaultdict
import sys
import time
import os
import grpc
import subprocess
import socket
import psutil


os.environ['DIAMBRA_ROM_PATH'] = '/mnt/c/Users/Brendan/Desktop/Final/3rdStrike'
os.environ['DIAMBRA_DEBUG'] = '1'  


def check_system_state():
    """Check system state and print diagnostic information."""
    print("\n=== System Diagnostics ===")
    
    try:
        docker_status = subprocess.run(['systemctl', 'is-active', 'docker'], 
                                    capture_output=True, text=True).stdout.strip()
        print(f"Docker service status: {docker_status}")
    except Exception as e:
        print(f"Could not check Docker status: {e}")
    
    docker_socket = '/var/run/docker.sock'
    if os.path.exists(docker_socket):
        print(f"Docker socket exists: {os.access(docker_socket, os.R_OK)}")
    else:
        print("Docker socket does not exist!")
    
    print("\nChecking port availability...")
    for port in range(32860, 32870):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('127.0.0.1', port))
            print(f"Port {port} is available")
            sock.close()
        except socket.error:
            print(f"Port {port} is in use")
        finally:
            sock.close()
    
    try:
        containers = subprocess.run(['docker', 'ps', '-a'], 
                                 capture_output=True, text=True).stdout
        print("\nRunning containers:")
        print(containers)
    except Exception as e:
        print(f"Could not list containers: {e}")
    
    print("\n=== End Diagnostics ===\n")

def cleanup_docker_containers():
    """Clean up any lingering DIAMBRA containers."""
    print("Starting container cleanup...")
    try:
        
        result = subprocess.run(['docker', 'ps', '-a', '--filter', 'name=diambra', '--format', '{{.ID}}'], 
                              capture_output=True, text=True)
        container_ids = result.stdout.strip().split('\n')
        
        if not container_ids or container_ids[0] == '':
            print("No DIAMBRA containers found")
            return
        
        print(f"Found {len(container_ids)} DIAMBRA containers")
        
        for container_id in container_ids:
            if container_id:
                try:
                    print(f"Stopping container {container_id}")
                    subprocess.run(['docker', 'stop', container_id], check=True)
                    print(f"Removing container {container_id}")
                    subprocess.run(['docker', 'rm', container_id], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Could not clean up container {container_id}: {e}")
    except Exception as e:
        print(f"Error during container cleanup: {e}")


def create_environment():
    print("\nCreating new environment...")
    
    cleanup_docker_containers()
    
    time.sleep(2)
    
    check_system_state()
    
    settings = EnvironmentSettingsMultiAgent()

    
    settings.action_space = (SpaceTypes.MULTI_DISCRETE, SpaceTypes.MULTI_DISCRETE) 
    settings.n_players = 2
    settings.step_ratio = 6 
    settings.splash_screen = True
    settings.frame_shape = (224, 384, 1)
    settings.frame_skip = 1  
    settings.disable_keyboard = False  
    settings.disable_joystick = True  

    
    settings.key_bindings = {
        'P1': {
            'left': 'a',
            'right': 'd',
            'up': 'space',
            'down': 's',
            'button1': 'y',  
            'button2': 'u',  
            'button3': 'i',  
            'button4': 'h',  
            'button5': 'j',  
            'button6': 'k'   
        }
    }

    
    settings.characters = ("Urien", "Urien") 
    settings.outfits = (2, 2)
    settings.continue_game = 0.0
    settings.show_final = False
    settings.difficulty = 1 

    try:
        print("Attempting to create DIAMBRA environment...")
        env = diambra.arena.make("sfiii3n", settings, render_mode="human")
        print("Environment created successfully")
        return env
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Checking system state after error...")
        check_system_state()
        raise

class KeyState:
    def __init__(self):
        self.pressed_keys = defaultdict(bool)
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            if key == keyboard.Key.space:
                self.pressed_keys[' '] = True
            else:
                self.pressed_keys[key.char.lower()] = True
        except AttributeError:
            if key == keyboard.Key.left:
                self.pressed_keys['left'] = True
            elif key == keyboard.Key.right:
                self.pressed_keys['right'] = True
            elif key == keyboard.Key.up:
                self.pressed_keys['up'] = True
            elif key == keyboard.Key.down:
                self.pressed_keys['down'] = True

    def on_release(self, key):
        try:
            if key == keyboard.Key.space:
                self.pressed_keys[' '] = False
            else:
                self.pressed_keys[key.char.lower()] = False
        except AttributeError:
            if key == keyboard.Key.left:
                self.pressed_keys['left'] = False
            elif key == keyboard.Key.right:
                self.pressed_keys['right'] = False
            elif key == keyboard.Key.up:
                self.pressed_keys['up'] = False
            elif key == keyboard.Key.down:
                self.pressed_keys['down'] = False

    def is_pressed(self, key):
        if isinstance(key, keyboard.Key):
            if key == keyboard.Key.left:
                return self.pressed_keys['left']
            elif key == keyboard.Key.right:
                return self.pressed_keys['right']
            elif key == keyboard.Key.up:
                return self.pressed_keys['up']
            elif key == keyboard.Key.down:
                return self.pressed_keys['down']
            elif key == keyboard.Key.space:
                return self.pressed_keys[' ']
        return self.pressed_keys[key.lower()]

    def cleanup(self):
        self.listener.stop()

def get_human_action(key_state):
    
    movement_action = 0
    attack_action = 0
    
    
    if key_state.is_pressed('a') and key_state.is_pressed('s'):
        movement_action = 8  
    elif key_state.is_pressed('d') and key_state.is_pressed('s'):
        movement_action = 6  
    elif key_state.is_pressed('a') and key_state.is_pressed(' '):
        movement_action = 2  
    elif key_state.is_pressed('d') and key_state.is_pressed(' '):
        movement_action = 4  
    elif key_state.is_pressed('a'):
        movement_action = 1  
    elif key_state.is_pressed('d'):
        movement_action = 5  
    elif key_state.is_pressed(' '):
        movement_action = 3  
    elif key_state.is_pressed('s'):
        movement_action = 7  
    
    
    if key_state.is_pressed('y') and key_state.is_pressed('h'):
        attack_action = 7  
    elif key_state.is_pressed('u') and key_state.is_pressed('j'):
        attack_action = 8  
    elif key_state.is_pressed('i') and key_state.is_pressed('k'):
        attack_action = 9  
    elif key_state.is_pressed('y'):
        attack_action = 1  
    elif key_state.is_pressed('u'):
        attack_action = 2  
    elif key_state.is_pressed('i'):
        attack_action = 3  
    elif key_state.is_pressed('h'):
        attack_action = 4  
    elif key_state.is_pressed('j'):
        attack_action = 5  
    elif key_state.is_pressed('k'):
        attack_action = 6  
    
   
    print("\nKey States:")
    print(f"Movement: A={key_state.is_pressed('a')}, D={key_state.is_pressed('d')}, Space={key_state.is_pressed(' ')}, S={key_state.is_pressed('s')}")
    print(f"Punches: Y={key_state.is_pressed('y')}, U={key_state.is_pressed('u')}, I={key_state.is_pressed('i')}")
    print(f"Kicks: H={key_state.is_pressed('h')}, J={key_state.is_pressed('j')}, K={key_state.is_pressed('k')}")
    print(f"Movement Action: {movement_action}")
    print(f"Attack Action: {attack_action}")
    
    return [movement_action, attack_action]

def validate_action(action):  
    valid_actions = {
        'movement': list(range(9)), 
        'attacks': list(range(10))   
    }
    
    for category, valid_range in valid_actions.items():
        if action in valid_range:
            return action
    
    print(f"Warning: Invalid action {action} detected, using no-op instead")
    return 0

def preprocess_observation(obs):
    
    if 'frame' in obs:
        frame = obs['frame']
        if len(frame.shape) == 3 and frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        frame = np.transpose(frame, (2, 0, 1))
        obs['frame'] = frame
    return obs

def flatten_observation(obs, player_key='P1'):
    
    obs = preprocess_observation(obs)
    
    flattened = {}
    
    if 'frame' in obs:
        flattened['frame_'] = obs['frame']
    
    if 'stage' in obs:
        flattened['stage_'] = np.array(obs['stage'], dtype=np.int32)
    if 'timer' in obs:
        flattened['timer_'] = np.array(obs['timer'], dtype=np.int32)
    
    for p_key in ['P1', 'P2']:
        if p_key in obs:
            player_data = obs[p_key]
            for key, value in player_data.items():
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        value = value[0]  
                    else:
                        value = value
                if np.isscalar(value):
                    value = np.array([value], dtype=np.int32)
                flattened[f'{p_key}_{key}_'] = value
    
    return flattened

def main():
    env = None
    key_state = None
    max_retries = 3
    retry_delay = 5

    try:
        print("Initializing environment settings...")
        
        check_system_state()
        
       
        cleanup_docker_containers()
        time.sleep(1)
        
        print("Creating environment...")
        env = create_environment()
        
     
        print("Loading model...")
        model_path = "/mnt/c/Users/Brendan/Desktop/Final/checkpoints/ppo_selfplay_agent_1000000_steps.zip"
        print(f"Attempting to load model from: {model_path}")
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}")
            print("Please ensure the model file exists at the specified path")
            return 1
            
        agent = PPO.load(model_path)
        print(f"Successfully loaded model from {model_path}")
        print(f"Model info: {agent}")
        
        print("\nControls:")
        print("Movement:")
        print("  A - Left")
        print("  D - Right")
        print("  S - Down")
        print("  Space - Up")
        print("  A+S - Down+Left")
        print("  D+S - Down+Right")
        print("  A+Space - Left+Up")
        print("  D+Space - Up+Right")
        print("Punches:")
        print("  Y - Low Punch")
        print("  U - Medium Punch")
        print("  I - High Punch")
        print("Kicks:")
        print("  H - Low Kick")
        print("  J - Medium Kick")
        print("  K - High Kick")
        print("Combos:")
        print("  Y+H - Low Punch + Low Kick")
        print("  U+J - Medium Punch + Medium Kick")
        print("  I+K - High Punch + High Kick")
        print("Press 'q' to quit\n")
        
        key_state = KeyState()
        
       
        print("Starting game...")
        observation, info = env.reset(seed=42)
        env.show_obs(observation)

        time.sleep(0.5)

        while True:
            try:
                env.render()

                action_p1 = get_human_action(key_state)
                action_p1 = [validate_action(a) for a in action_p1]

                observation_p2 = flatten_observation(observation, 'P2')
                print("\nAgent Observation:")
                print(f"Frame shape: {observation_p2['frame_'].shape}")
                print(f"Stage: {observation_p2['stage_']}")
                print(f"Timer: {observation_p2['timer_']}")
                print(f"P2 health: {observation_p2['P2_health_']}")
                
                action_p2, _ = agent.predict(observation_p2, deterministic=True)
                print(f"Raw agent action: {action_p2}")
                action_p2 = int(action_p2) if np.isscalar(action_p2) or action_p2.shape == () else int(action_p2[0])
                
                n_moves = 9
                move = action_p2 % n_moves
                attack = action_p2 // n_moves
                action_p2 = [move, attack]
                action_p2 = [validate_action(a) for a in action_p2]
                print(f"Validated agent action: {action_p2}")

                actions = {
                    'agent_0': action_p1, 
                    'agent_1': action_p2  
                }
                
                print(f"\nStep Info:")
                print(f"Actions: {actions}")
                print(f"Raw observation shape: {observation['frame'].shape}")
                print(f"Processed observation shape: {observation_p2['frame_'].shape}")
                print(f"P1 health: {observation['P1']['health']}")
                print(f"P2 health: {observation['P2']['health']}")
                
                time.sleep(0.01)
                
                observation, reward, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                env.show_obs(observation)
                
                if done:
                    print("Match finished, resetting...")
                    observation, info = env.reset()
                    env.show_obs(observation)
                    time.sleep(0.5)
                
                if key_state.is_pressed('q'):
                    print("Quit command received...")
                    break

                time.sleep(0.01)

            except grpc.RpcError as e:
                print(f"\nConnection error: {e}")
                print("Checking system state after connection error...")
                check_system_state()
                print("Attempting to reconnect...")
                
                for attempt in range(max_retries):
                    try:
                        if env is not None:
                            try:
                                env.close()
                            except:
                                pass
                        cleanup_docker_containers()
                        time.sleep(retry_delay)
                        env = create_environment()
                        observation, info = env.reset(seed=42)
                        env.show_obs(observation)
                        time.sleep(0.5)
                        print("Successfully reconnected!")
                        break
                    except Exception as reconnect_error:
                        print(f"Reconnection attempt {attempt + 1} failed: {reconnect_error}")
                        if attempt == max_retries - 1:
                            print("Max retries reached. Exiting...")
                            break
                        time.sleep(retry_delay)
                
                if attempt == max_retries - 1:
                    break

            except Exception as e:
                print(f"Error during game loop: {e}")
                print("Checking system state after error...")
                check_system_state()
                print("Attempting to clean up...")
                break

    except Exception as e:
        print(f"Error during setup: {e}")
        print("Checking system state after setup error...")
        check_system_state()
    finally:
        
        print("Cleaning up resources...")
        if env is not None:
            try:
                env.close()
            except Exception as e:
                print(f"Error closing environment: {e}")
        if key_state is not None:
            try:
                key_state.cleanup()
            except Exception as e:
                print(f"Error cleaning up key state: {e}")
        cleanup_docker_containers()
        print("Game ended.")

    return 0

if __name__ == '__main__':
    main() 