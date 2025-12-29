import gymnasium as gym
from stable_baselines3 import PPO
import time
import os

env_id = "BipedalWalker-v3"

def train_and_visualize():
   
    train_env = gym.make(env_id)
    model = PPO("MlpPolicy", train_env, verbose=1)

    print("Begun")
    model.learn(total_timesteps=1000000)
    train_env.close()

    print("Done")
    eval_env = gym.make(env_id, render_mode="human")
    
    obs, _ = eval_env.reset()
    try:
        while True: 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    eval_env.close()
                    return

            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            eval_env.render()
            time.sleep(0.01) 
            
            if terminated or truncated:
                obs, _ = eval_env.reset()
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        eval_env.close()

if __name__ == "__main__":
    train_and_visualize()

