from src.utils.replay_buffer import ReplayBuffer
import torch

dataset_path = "stockfish_dataset.pkl.gz"
buffer = ReplayBuffer.load(dataset_path, max_size=100000)
states, policies, values = buffer.sample_all()
print(f"States shape: {states.shape}, Policies shape: {policies.shape}, Values shape: {values.shape}")