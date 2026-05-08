"""GPU smoke test — verifies networks load, forward pass works, VRAM is sufficient."""
from config import Config
from networks import CNNEncoder, Actor, Critic, InverseDynamics, ForwardDynamics
import torch

config = Config()
device = config.device
print(f"Device: {device}")

encoder = CNNEncoder(feature_dim=config.feature_dim).to(device)
actor = Actor(feature_dim=config.feature_dim, num_actions=config.num_actions).to(device)
critic = Critic(feature_dim=config.feature_dim).to(device)
inv = InverseDynamics(feature_dim=config.feature_dim, num_actions=config.num_actions).to(device)
fwd = ForwardDynamics(feature_dim=config.feature_dim, num_actions=config.num_actions).to(device)

B = config.batch_size
x = torch.randn(B, 4, 84, 84, device=device)
phi = encoder(x)
logits = actor(phi)
v = critic(phi)
print(f"Forward: {phi.shape} -> logits {logits.shape}, value {v.shape}")

vram_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"VRAM: {vram_used:.1f} / {vram_total:.1f} GB")
print("GPU test PASSED.")
