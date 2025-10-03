import os
import argparse
import torch

from src.selfplay.selfplay import SelfPlay
from src.utils.replay_buffer import ReplayBuffer
from src.training.train import train
from src.evaluation.evaluate import evaluate_models
from src.mcts.mcts import MCTS
from src.mcts.mcts_action_indexer import UCIActionIndexer
from src.network.model import NeuralNet, ChessNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=str, default="replay_buffer.pkl.gz")
    parser.add_argument("--new", type=str, default="checkpoints/chess_model.pth")
    parser.add_argument("--old", type=str, default="checkpoints/chess_model_old.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--games", type=int, default=25)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--c_puct", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_moves", type=int, default=400)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")

    # ------------------------
    # Step 1: Load or init replay buffer
    # ------------------------
    if os.path.exists(args.buffer):
        buffer = ReplayBuffer.load(args.buffer)
        print(f"[Main] Loaded replay buffer from {args.buffer}, size={len(buffer)}")
    else:
        buffer = ReplayBuffer(max_size=1e6)
        print(f"[Main] Initialized empty replay buffer")

    # ------------------------
    # Step 2: Self-play
    # ------------------------
    print(f"[Main] Generating {args.games} self-play games...")
    action_indexer = UCIActionIndexer()
    model = NeuralNet(model_path=args.new, device=device)

    if os.path.exists(args.old):
        model.load(args.old)
        print(f"[Main] Loaded old model from {args.old}")
    else:
        print("[Main] No old model found, using random init")

    mcts = MCTS(
        network=model,
        action_indexer=action_indexer,
        num_simulations=args.sims,
        c_puct=args.c_puct,
    )

    selfplay = SelfPlay(
        mcts=mcts,
        buffer=buffer,
        num_games=args.games,
        action_indexer=action_indexer,
    )
    selfplay.generate()
    buffer.save(args.buffer)
    print(f"[Main] Saved replay buffer to {args.buffer} (size={len(buffer)})")

    # ------------------------
    # Step 3: Train model
    # ------------------------
    print("[Main] Starting training...")
    train(
        replay_buffer_path=args.buffer,
        model_path=args.new,
        epochs=args.epochs,
        batch_size=128,
        lr=3e-3,
    )

    # ------------------------
    # Step 4: Evaluate
    # ------------------------
    if os.path.exists(args.old):
        print("[Main] Evaluating new model vs old model...")
        results = evaluate_models(
            new_model_path=args.new,
            old_model_path=args.old,
            num_games=1,
            sims=args.sims,
            c_puct=args.c_puct,
            device=str(device),
            temperature=1e-3,
            add_noise_in_selfplay=False,
            swap_colors=True,
            max_moves=args.max_moves,
        )
        print("[Main] Evaluation results:", results)
    else:
        print("[Main] No old model found, skipping evaluation.")

    # ------------------------
    # Step 5: Promote
    # ------------------------
    if os.path.exists(args.new):
        os.makedirs(os.path.dirname(args.old), exist_ok=True)
        model_state = torch.load(args.new, map_location="cpu")
        torch.save(model_state, args.old)
        print(f"[Main] Promoted {args.new} to {args.old}")



if __name__ == "__main__":
    train_count = 0
    while True:
        main()
        train_count += 1
        print(f"[Main] Completed training iteration {train_count}")
        if train_count >= 100:
            break
    print("[Main] Training loop finished.")
