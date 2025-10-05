# src/train/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from src.network.model import ChessNet
from src.utils.replay_buffer import ReplayBuffer

writer = SummaryWriter(log_dir="logs")
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
    replay_buffer_path="replay_buffer.pkl.gz",
    model_path="checkpoints/candidate.pth",
    base_model_path="checkpoints/best.pth",
    epochs=15,
    batch_size=256,
    lr=1e-3,
):
    

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # Load replay buffer
    buffer = ReplayBuffer.load(replay_buffer_path)
    print("[Train] Loaded replay buffer, size =", len(buffer))
    dataset = ChessDataset(buffer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model + optimizer
    if(base_model_path is None):
        model = ChessNet().to(device)
        print("[Train] Initialized new model.")
    else:
        model = ChessNet().to(device)
        model.load_state_dict(torch.load(base_model_path, map_location=device))
        print(f"[Train] Loaded model from {base_model_path}.")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))
    # Loss function: policy loss + value loss
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    scaler = GradScaler(enabled=(device.type == 'cuda'))
    global_step = 0
    for epoch in range(epochs):
        model.train()
        
        for states, target_policies, target_values in dataloader:
            states = states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)

            optimizer.zero_grad(set_to_none=True) # Use set_to_none=True for better performance

            # Use autocast for the forward pass
            with autocast(device_type="cuda",enabled=(device.type == 'cuda')):
                policy_logits, values = model(states)

                # Policy loss (cross-entropy for soft labels)
                policy_loss = -(target_policies * torch.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()

                # Value loss (MSE)
                value_loss = mse_loss(values.squeeze(-1), target_values)

                # Total loss
                loss = policy_loss + value_loss

            # Backpropagation with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()

            
            if global_step % 50 == 0: # Log every 50 steps
                writer.add_scalar("Loss/Total", loss.item(), global_step)
                writer.add_scalar("Loss/Policy", policy_loss.item(), global_step)
                writer.add_scalar("Loss/Value", value_loss.item(), global_step)
                writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], global_step)
            
            global_step += 1

        print(f"[Epoch {epoch+1}/{epochs}] Last Batch Loss = {loss.item():.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"[Train] Saved trained model to {model_path}")
    writer.close()


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train(
        replay_buffer_path="replay_buffer.pkl.gz",
        model_path="checkpoints/chess_model_testing.pth",
        base_model_path="checkpoints/chess_model_old.pth",
        epochs=10,
        batch_size=64,
        lr=1e-3,
    )