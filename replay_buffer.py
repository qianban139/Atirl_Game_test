"""Replay buffer with episode-boundary-safe sequence sampling."""
import random
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, latent_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.h = np.zeros((capacity, latent_dim), dtype=np.float32)
        self.z = np.zeros((capacity, 32, 16), dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, action, reward, done, h, z):
        idx = self.pos
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.h[idx] = h
        self.z[idx] = z
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.pos

    def sample_sequences(self, batch_size, seq_len, device):
        """Sample sequences that do NOT cross episode boundaries."""
        size = len(self)
        obs_batch = np.zeros((batch_size, seq_len, 1, 84, 84), dtype=np.float32)
        act_batch = np.zeros((batch_size, seq_len), dtype=np.int64)
        rew_batch = np.zeros((batch_size, seq_len), dtype=np.float32)
        done_batch = np.zeros((batch_size, seq_len), dtype=np.float32)

        for b in range(batch_size):
            while True:
                start = random.randint(0, size - seq_len - 1)
                valid = True
                for t in range(seq_len - 1):
                    if self.dones[(start + t) % self.capacity]:
                        valid = False
                        break
                if valid:
                    break

            for t in range(seq_len):
                idx = (start + t) % self.capacity
                obs_batch[b, t] = self.obs[idx].astype(np.float32) / 255.0
                act_batch[b, t] = self.actions[idx]
                rew_batch[b, t] = self.rewards[idx]
                done_batch[b, t] = self.dones[idx]

        return (torch.from_numpy(obs_batch).to(device),
                torch.from_numpy(act_batch).to(device),
                torch.from_numpy(rew_batch).to(device),
                torch.from_numpy(done_batch).to(device))

    def sample_starts(self, n, device):
        """Sample (h, z) pairs from non-terminal states for imagination."""
        size = len(self)
        indices = []
        while len(indices) < n:
            idx = random.randint(0, size - 1)
            if not self.dones[idx]:
                indices.append(idx)

        h_batch = torch.from_numpy(np.stack([self.h[i] for i in indices])).float().to(device)
        z_batch = torch.from_numpy(np.stack([self.z[i] for i in indices])).float().to(device)
        return h_batch, z_batch
