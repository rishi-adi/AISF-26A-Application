import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
import time
import pygame
import os

class CraftedWalkerReward(gym.RewardWrapper):

    def reward(self, reward):
        hull_angle = self.env.unwrapped.hull.angle
        reward -= abs(hull_angle) * 0.5
        return reward

env_id = "BipedalWalker-v3"
stats_path = "vec_normalize_final.pkl"
model_name = "ppo_walker_combined"
total_timesteps = 1000000

params = {
    "learning_rate": 0.0002,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.999,
    "ent_coef": 0.01,
    "clip_range": 0.2,
    "policy_kwargs": dict(net_arch=[256, 256])
}

def train_and_visualize():
    if not os.path.exists("./logs/"):
        os.makedirs("./logs/")

    def make_env():
        env = gym.make(env_id)
        env = CraftedWalkerReward(env)
        return env

    train_env = DummyVecEnv([make_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path="./logs/",
        name_prefix=model_name
    )

    model = PPO("MlpPolicy", train_env, verbose=1, **params)
    
    print(f"Training started with Normalization and Reward Crafting...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    model.save(model_name)
    train_env.save(stats_path)
    train_env.close()

    print("\nTraining complete. Starting visualization...")
    
    eval_env = DummyVecEnv([lambda: CraftedWalkerReward(gym.make(env_id, render_mode="human"))])
    eval_env = VecNormalize.load(stats_path, eval_env)
    
    eval_env.training = False 
    eval_env.norm_reward = False 
    obs = eval_env.reset()
    total_reward = 0 

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    eval_env.close()
                    return

            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, terminated, infos = eval_env.step(action)

            if 'terminal_observation' not in infos[0]:
                total_reward += infos[0].get('r', rewards[0])

            eval_env.render()
            time.sleep(0.01)

            if terminated[0]:
                print(f"Episode finished. Total Raw Score: {total_reward:.2f}")
                total_reward = 0 
                time.sleep(0.5)
                obs = eval_env.reset()

    except KeyboardInterrupt:
        print("\nStopping Visualization...")
    finally:
        eval_env.close()

if __name__ == "__main__":
    train_and_visualize()
