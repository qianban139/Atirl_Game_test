"""Generate publication-quality figures for the coursework report."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_curve(ppo_rewards, icm_rewards, save_path: str):
    """Training curve with rolling mean."""
    fig, ax = plt.subplots(figsize=(8, 5))

    def rolling(x, window=100):
        return np.convolve(x, np.ones(window) / window, mode="valid")

    ax.plot(rolling(ppo_rewards), label="PPO only", alpha=0.7)
    ax.plot(rolling(icm_rewards), label="PPO + ICM", alpha=0.7)

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Donkey Kong: PPO vs PPO+ICM")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")


def plot_entropy_decay(entropies, save_path: str):
    """Policy entropy over training."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(entropies, color="purple", alpha=0.8)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Policy Entropy")
    ax.set_title("Policy Entropy Decay")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")


def plot_stage_progress(stages: list[int], save_path: str):
    """Histogram of stages reached."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(stages, bins=range(1, max(stages) + 2), align="left",
            color="steelblue", edgecolor="white")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Stages Reached")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    Path("report/figures").mkdir(parents=True, exist_ok=True)
    print("Run after training completes with collected metric data.")
    print("Use TensorBoard CSV export or logged data to populate arrays.")
