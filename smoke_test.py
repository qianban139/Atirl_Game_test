"""5K-step training smoke test — minimal run to verify the training loop doesn't crash."""
from config import Config
Config.total_timesteps = 5000
Config.num_envs = 2
Config.rollout_steps = 32
Config.log_interval = 1
Config.save_interval = 1000000
from train import train
train()
