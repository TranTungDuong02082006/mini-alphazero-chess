# main.py

import os
import argparse
import torch
import multiprocessing
import time

# Import the NEW worker function, not the SelfPlay class
from src.selfplay.selfplay import self_play_worker 
from src.utils.replay_buffer import ReplayBuffer
from src.training.train import train
from src.evaluation.evaluate import evaluate_models
from src.network.model import NeuralNet

def main():
    parser = argparse.ArgumentParser()
    # General file paths
    parser.add_argument("--buffer", type=str, default="replay_buffer.pkl.gz")
    parser.add_argument("--new", type=str, default="checkpoints/chess_model.pth")
    parser.add_argument("--old", type=str, default="checkpoints/chess_model_old.pth")

    # Training loop args
    parser.add_argument("--num_workers", type=int, default=8, help="Number of self-play workers")
    parser.add_argument("--train_after_positions", type=int, default=20000, help="Trigger training after N new positions")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--max_moves", type=int, default=400)
    
    # NEW: Replay buffer max_size is now a configurable argument
    parser.add_argument("--buffer_size", type=int, default=500000, help="Maximum size of the replay buffer")

    # MCTS args
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--c_puct", type=float, default=1.0)
    
    # System args
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()

    # It's crucial to set the start method for CUDA safety
    multiprocessing.set_start_method('spawn', force=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device for training: {device}")

    # ------------------------
    # Step 1: Load or init replay buffer and model
    # ------------------------
    if os.path.exists(args.buffer):
        # CHANGE: Pass the configured max_size when loading the buffer
        buffer = ReplayBuffer.load(args.buffer, max_size=args.buffer_size)
        print(f"[Main] Loaded replay buffer from {args.buffer}, size={len(buffer)}")
    else:
        # CHANGE: Use the configured max_size for a new buffer
        buffer = ReplayBuffer(max_size=args.buffer_size) 
        print(f"[Main] Initialized empty replay buffer with max_size={args.buffer_size}")

    # Ensure the model exists before starting workers
    if not os.path.exists(args.new):
        print(f"[Main] No model found at {args.new}. Initializing a new one.")
        initial_model = NeuralNet(device=device)
        initial_model.save(args.new)
        print(f"[Main] Saved new initial model to {args.new}")

    # ------------------------
    # Step 2: Set up and start Self-play Workers
    # ------------------------
    data_queue = multiprocessing.Queue()
    
    workers = []
    for i in range(args.num_workers):
        process = multiprocessing.Process(
            target=self_play_worker,
            args=(args.new, data_queue, i, args) # Note: self_play_worker needs to be updated to accept args
        )
        process.daemon = True
        process.start()
        workers.append(process)

    print(f"[Main] Started {args.num_workers} self-play workers.")
    
    # ------------------------
    # Step 3: Main Asynchronous Loop
    # ------------------------
    new_positions_counter = 0
    iteration = 0
    while True:
        iteration += 1
        print(f"\n[Main] #################### Starting Iteration {iteration} ####################\n")

        # Collect data until the threshold is reached
        while new_positions_counter < args.train_after_positions:
            if not data_queue.empty():
                game_data = data_queue.get()
                # CHANGE: Use the .extend() method from your ReplayBuffer class
                buffer.extend(game_data)
                new_positions_counter += len(game_data)
                print(f"[Main] Data received. Buffer: {len(buffer)}. New positions: {new_positions_counter}/{args.train_after_positions}")
            else:
                print(f"[Main] Waiting for data... ({new_positions_counter}/{args.train_after_positions})")
                time.sleep(10)

        # When there is enough data -> Train
        print("[Main] Data threshold reached. Starting training...")
        if len(buffer) > 0:
            buffer.save(args.buffer) # Save buffer before training
            
            # Backup the old model for comparison
            if os.path.exists(args.new):
                if os.path.exists(args.old):
                    os.remove(args.old)
                os.rename(args.new, args.old)

            train(
                replay_buffer_path=args.buffer,
                model_path=args.new, 
                base_model_path=args.old, # Pass the old model path to continue training from it
                epochs=args.epochs,
                batch_size=256,
                lr=1e-4,
                device=str(device)
            )
            new_positions_counter = 0 # Reset the counter
        else:
            print("[Main] Buffer is empty, skipping training.")
            continue

        # After training, evaluate the new model against the old one
        if os.path.exists(args.old):
            print("[Main] Evaluating new model vs old model...")
            results = evaluate_models(
                new_model_path=args.new,
                old_model_path=args.old,
                num_games=100,
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
    main()