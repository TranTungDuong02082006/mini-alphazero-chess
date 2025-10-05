# main.py

import os
import argparse
import torch
import multiprocessing
import time
import shutil

from src.selfplay.selfplay import self_play_worker 
from src.utils.replay_buffer import ReplayBuffer
from src.training.train import train
from src.evaluation.evaluate import evaluate_models
from src.network.model import NeuralNet

def main():
    parser = argparse.ArgumentParser()
    # --- Model paths ---
    parser.add_argument("--model_best", type=str, default="checkpoints/best.pth", help="Path to the best, stable model for workers.")
    parser.add_argument("--model_candidate", type=str, default="checkpoints/candidate.pth", help="Path for the newly trained model to be evaluated.")
    
    # --- Other arguments ---
    parser.add_argument("--buffer", type=str, default="replay_buffer.pkl.gz")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--train_after_positions", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--max_moves", type=int, default=400)
    parser.add_argument("--buffer_size", type=int, default=500000)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--c_puct", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()

    multiprocessing.set_start_method('spawn', force=True)
    device = torch.device(args.device)
    print(f"[Main] Using device for training: {device}")

    # --- Initialization ---
    
    # Shared flag to notify workers of a new model version
    model_version = multiprocessing.Value('i', 0)

    # Ensure the 'best' model exists before starting workers
    if not os.path.exists(args.model_best):
        print(f"[Main] No best model found. Initializing a new one at {args.model_best}")
        initial_model = NeuralNet(device=device)
        initial_model.save(args.model_best)
        with model_version.get_lock():
            model_version.value += 1
        print(f"[Main] Saved initial model. Version: {model_version.value}")
    else:
        # If a model exists, set its version so workers load it
        with model_version.get_lock():
            model_version.value = 1 # Or read from a file if you want to persist versions
    
    # Load or init replay buffer
    buffer = ReplayBuffer(max_size=args.buffer_size)
    if os.path.exists(args.buffer):
        buffer = ReplayBuffer.load(args.buffer, max_size=args.buffer_size)
        print(f"[Main] Loaded replay buffer from {args.buffer}, size={len(buffer)}")

    # --- Start Self-play Workers ---
    data_queue = multiprocessing.Queue()
    workers = []
    for i in range(args.num_workers):
        process = multiprocessing.Process(
            target=self_play_worker,
            # Workers ONLY ever load the 'best' model
            args=(args.model_best, data_queue, i, args, model_version)
        )
        process.daemon = True
        process.start()
        workers.append(process)

    print(f"[Main] Started {args.num_workers} workers to generate data using '{args.model_best}'.")
    
    # --- Main Asynchronous Loop ---
    new_positions_counter = 0
    iteration = 0
    while True:
        iteration += 1
        print(f"\n[Main] #################### Starting Iteration {iteration} ####################\n")

        # 1. Collect data from workers until the threshold is reached
        # (Workers are continuously playing using the 'best' model in the background)
        while new_positions_counter < args.train_after_positions:
            if not data_queue.empty():
                game_data = data_queue.get()
                buffer.extend(game_data)
                new_positions_counter += len(game_data)
                print(f"[Main] Data received. Buffer: {len(buffer)}. New positions: {new_positions_counter}/{args.train_after_positions}")
            else:
                print(f"[Main] Waiting for data... ({new_positions_counter}/{args.train_after_positions})")
                time.sleep(10)

        # 2. Train a new 'candidate' model
        print("[Main] Data threshold reached. Starting training...")
        if len(buffer) > 0:
            buffer.save(args.buffer)
            
            # The 'train' function will load the 'best' model as a base,
            # and save the newly trained model as the 'candidate'.
            train(
                replay_buffer_path=args.buffer,
                model_path=args.model_candidate, # Output path for the new model
                base_model_path=args.model_best, # Starting point for training
                epochs=args.epochs,
                batch_size=256,
                lr=1e-4,
            )
            new_positions_counter = 0
        else:
            print("[Main] Buffer is empty, skipping training.")
            continue

        # 3. Evaluate the new 'candidate' model against the 'best' model
        print("[Main] Evaluating new candidate model vs best model...")
        results = evaluate_models(
            new_model_path=args.model_candidate,
            old_model_path=args.model_best,
            num_games=20, # More games for a reliable result
            sims=args.sims,
            device=str(device),
        )
        print("[Main] Evaluation results:", results['summary'])
        
        # 4. Conditional Promotion
        win_rate_threshold = 0.55
        new_model_win_rate = results['summary']['new_win_rate']
        
        if new_model_win_rate > win_rate_threshold:
            print(f"[Main] PROMOTION! New model won with {new_model_win_rate:.2%} > {win_rate_threshold:.0%}. Promoting candidate to best.")
            # Copy the successful candidate to become the new best model
            shutil.copy(args.model_candidate, args.model_best)
            # Notify workers that a new version of the 'best' model is available
            with model_version.get_lock():
                model_version.value += 1
            print(f"[Main] Model version is now {model_version.value}. Workers will update on their next game.")
        else:
            print(f"[Main] NO PROMOTION. New model win rate was {new_model_win_rate:.2%}. Keeping the old best model.")
            
if __name__ == "__main__":
    main()