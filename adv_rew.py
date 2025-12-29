import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import time
import pygame

class CraftedWalkerReward(gym.RewardWrapper):
    def reward(self, reward):
        hull_angle = self.env.unwrapped.hull.angle
        reward -= abs(hull_angle) * 0.5
        return reward

env_id = "BipedalWalker-v3"

def train_and_visualize():
    train_env = Monitor(CraftedWalkerReward(gym.make(env_id)))
    
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        gamma=0.999,      
        gae_lambda=0.95  
    )

    print("Training")
    model.learn(total_timesteps=2000000)
    train_env.close()

    print("\nBegun")
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
                print("\nit fell. Beginning again")
                time.sleep(0.5)
                obs, _ = eval_env.reset()
                
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        eval_env.close()

if __name__ == "__main__":
    train_and_visualize()
