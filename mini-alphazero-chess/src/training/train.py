# src/train/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.network.model import ChessNet
from src.utils.replay_buffer import ReplayBuffer


# -------------------------------
# Dataset wrapper for replay buffer
# -------------------------------
class ChessDataset(Dataset):
    def __init__(self, buffer: ReplayBuffer):
        self.buffer = buffer
        self.data = buffer.sample(len(buffer))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, policy, value = self.data[idx]
        state = torch.tensor(state, dtype=torch.float32)
        policy = torch.tensor(policy, dtype=torch.float32)
        value = torch.tensor(value, dtype=torch.float32)
        return state, policy, value


# -------------------------------
# Training Loop
# -------------------------------
def train(
    replay_buffer_path="replay_buffer.pkl",
    model_path="chess_model.pth",
    epochs=10,
    batch_size=64,
    lr=1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # Load replay buffer
    buffer = ReplayBuffer.load(replay_buffer_path)
    dataset = ChessDataset(buffer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model + optimizer
    model = ChessNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss function: policy loss + value loss
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for states, target_policies, target_values in dataloader:
            states = states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)

            optimizer.zero_grad()

            # Forward
            policy_logits, values = model(states)

            # Policy loss
            policy_loss = -(target_policies * torch.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()

            # Value loss (MSE)
            value_loss = mse_loss(values.squeeze(), target_values)

            # Total loss
            loss = policy_loss + value_loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}/{epochs}] Loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"[Train] Saved trained model to {model_path}")


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train(
        replay_buffer_path="replay_buffer.pkl.gz",
        model_path="checkpoints/chess_model.pth",
        epochs=10,
        batch_size=64,
        lr=1e-3,
    )