import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize # New imports
from stable_baselines3.common.monitor import Monitor
import time

class CraftedWalkerReward(gym.RewardWrapper):
    def reward(self, reward):
        hull_angle = self.env.unwrapped.hull.angle
        reward -= abs(hull_angle) * 0.5
        return reward

env_id = "BipedalWalker-v3"

def train_and_visualize():
    def make_env():
        env = gym.make(env_id)
        env = CraftedWalkerReward(env)
        env = Monitor(env) 
        return env

    train_env = DummyVecEnv([make_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    model = PPO("MlpPolicy", train_env, verbose=1)

    model.learn(total_timesteps=1000000)
    
    train_env.save("vec_normalize.pkl")
    train_env.close()

    print("Begun")
    
    eval_env = DummyVecEnv([lambda: CraftedWalkerReward(gym.make(env_id, render_mode="human"))])
    eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
    
    eval_env.training = False 
    eval_env.norm_reward = False 

    obs = eval_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, dones, info = eval_env.step(action)
        
        eval_env.render()
        time.sleep(0.01) 
        
        if dones[0]:
            obs = eval_env.reset()

    eval_env.close()

if __name__ == "__main__":
    train_and_visualize()
