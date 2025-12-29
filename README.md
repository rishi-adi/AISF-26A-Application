# AISF Application w/ Bipedal Walker

## Files

initial_PPO.py = the PPO algorithm I generated from Google Gemini

reward_craft.py = PPO algorithm with posture-based reward crafting

obv_nor.py = PPO algorithm witn observation/normalization (used VecNormalize)

adv_rew.py = PPO algorithm with advantage vs. reward (modified gamma values)

exp.py = PPO algorithm with exploration/exploitation (modified ent_coef values)

final_PPO.py = PPO algorithm with tuned hyperparameters from previous files and all strategies implemented

## Video

walker_best.mp4 = Best performance of bipedal walker I achieved with final_PPO.py (Score = 247)

