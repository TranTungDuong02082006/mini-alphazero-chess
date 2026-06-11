"""
src/evaluation/evaluate.py

Evaluate a "new" model against an "old" model by playing a number of self-play games
using MCTS + NeuralNet. Alternate colors between games for fairness.

Outputs a summary of wins/draws/losses and an optional JSON report and optional checkpoint replacement.

Assumptions (matching your project layout):
- network.model.NeuralNet exists with from_checkpoint(...) or load(...)
- mcts.mcts.MCTS exists and provides run(root_game, temperature=..., add_noise=...) -> (probs, info)
- mcts.mcts_action_indexer.UCIActionIndexer exists with idx_to_action and action_to_idx and legal_mask_from_moves
- game.chess_game.ChessGame exists with clone(), play_move(), is_game_over(), get_result(), get_legal_moves()
"""

import argparse
import json
import logging
import os
import shutil
import time
from typing import Optional, Tuple

import numpy as np

# Adjust imports if your package layout differs; this matches `src/` as package root.
try:
    from network.model import NeuralNet
    from mcts.mcts import MCTS
    from mcts.mcts_action_indexer import UCIActionIndexer
    from game.chess_game import ChessGame
except Exception as e:
    # Try alternate import paths (if you run from src folder differently)
    # This fallback helps when running the script in different working directories.
    import sys
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from network.model import NeuralNet
    from mcts.mcts import MCTS
    from mcts.mcts_action_indexer import UCIActionIndexer
    from game.chess_game import ChessGame


def setup_logger(level=logging.INFO):
    fmt = "%(asctime)s %(levelname)s - %(message)s"
    logging.basicConfig(format=fmt, level=level)


def load_net(path: str, device: str = "cpu") -> NeuralNet:
    """
    Load a NeuralNet from checkpoint path. Uses NeuralNet.from_checkpoint if available,
    otherwise constructs a NeuralNet and loads weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    logging.info(f"Loading model from: {path} on device={device}")
    # Prefer classmethod from_checkpoint if implemented
    try:
        net = NeuralNet.from_checkpoint(path, device=device)
    except Exception:
        # Fallback: instantiate default and load
        net = NeuralNet(device=device)
        net.load(path)
    return net


def play_one_game(
    mcts_white: MCTS,
    mcts_black: MCTS,
    action_indexer: UCIActionIndexer,
    start_game: Optional[ChessGame] = None,
    max_moves: int = 800,
    temperature: float = 1e-3,
    add_noise: bool = False,
    temp_moves = 40,
) -> Tuple[int, int, str]:
    """
    Play a single game between two MCTS instances.
    Returns (outcome, moves_played)
      outcome: 1 if white wins, -1 if black wins, 0 draw
    mcts_white: MCTS instance that will play white
    mcts_black: MCTS instance that will play black
    """
    if start_game is None:
        game = ChessGame()
    else:
        game = start_game.clone()

    moves = 0
    while not game.is_game_over() and moves < max_moves:
        # determine which MCTS to use now based on game.get_turn()
        # get_turn() returns 1 for white, -1 for black (per your ChessGame)
        turn = game.get_turn()
        if turn == 1:
            engine = mcts_white
        else:
            engine = mcts_black

        if moves < temp_moves:
            temp = 1.0
            noise = True
        else:
            temp = temperature
            noise = add_noise

        # run MCTS to get probabilities & info (info should contain selected_action)
        probs, info = engine.run(game, temperature=temp, add_noise=noise)

        # robustly get selected action from info; fallback to argmax(probs)
        selected_action = None
        if isinstance(info, dict):
            selected_action = info.get("selected_action") or info.get("chosen_action") or info.get("action")
            # some implementations store selected_idx; convert to action
            if selected_action is None and "selected_idx" in info:
                try:
                    selected_action = action_indexer.idx_to_action(int(info["selected_idx"]))
                except Exception:
                    selected_action = None

        if selected_action is None:
            # fallback: choose highest prob idx
            idx = int(np.argmax(probs))
            selected_action = action_indexer.idx_to_action(idx)

        # play it on game (ChessGame.play_move accepts UCI string or chess.Move)
        try:
            game.play_move(selected_action)
        except Exception as ex:
            # If play_move fails, attempt alternative: try converting string/uci
            logging.exception("Failed to play selected_action; attempt to recover.")
            # try idx -> move object
            try:
                if isinstance(selected_action, (str,)):
                    # convert to chess.Move via indexer or ChessGame
                    game.play_move(selected_action)
                else:
                    # last resort: try apply idx
                    idx = int(np.argmax(probs))
                    move = action_indexer.idx_to_action(idx)
                    game.play_move(move)
            except Exception as ex2:
                logging.exception("Recovery failed. Aborting game as draw.")
                return 0, moves, "Illegal_move_error"

        moves += 1

    if game.is_game_over():
        try:
            reason = game.get_reason()
        except Exception:
            reason = "game_over"
    elif moves >= max_moves:
        reason = "max_moves_reached"
    else:
        reason = "unknown"

    # game ended or move limit
    result = game.get_result()  # 1 white, -1 black, 0 draw
    return result, moves, reason


def evaluate_models(
    new_model_path: str,
    old_model_path: str,
    num_games: int = 10,
    sims: int = 100,
    c_puct: float = 1.0,
    device: str = "cpu",
    temperature: float = 1e-3,
    add_noise_in_selfplay: bool = False,
    swap_colors: bool = True,
    max_moves: int = 800,
) -> dict:
    """
    Run evaluation between new_model and old_model with detailed logs.

    Returns dict with wins/draws/losses and per-game details.
    """
    # load nets
    net_new = load_net(new_model_path, device=device)
    net_old = load_net(old_model_path, device=device)

    # action indexer / shared across MCTS
    action_indexer = UCIActionIndexer()

    # create MCTS instances (one per engine)
    mcts_new = MCTS(network=net_new, action_indexer=action_indexer, num_simulations=sims, c_puct=c_puct)
    mcts_old = MCTS(network=net_old, action_indexer=action_indexer, num_simulations=sims, c_puct=c_puct)

    results = {"new_wins": 0, "old_wins": 0, "draws": 0, "games": []}

    logging.info(f"Starting evaluation: {num_games} games, sims={sims}, c_puct={c_puct}, device={device}")
    tic_all = time.time()

    for g in range(num_games):
        if swap_colors:
            new_is_white = (g % 2 == 0)
        else:
            new_is_white = True

        mcts_white = mcts_new if new_is_white else mcts_old
        mcts_black = mcts_old if new_is_white else mcts_new

        logging.info(f"[Game {g+1}/{num_games}] new_is_white={new_is_white}")
        t0 = time.time()

        # giả sử bạn chỉnh play_one_game để trả thêm reason
        outcome, moves, reason = play_one_game(
            mcts_white,
            mcts_black,
            action_indexer,
            start_game=None,
            max_moves=max_moves,
            temperature=temperature,
            add_noise=add_noise_in_selfplay,
        )
        dt = time.time() - t0

        if outcome == 0:
            results["draws"] += 1
            winner = None
        else:
            new_won = (outcome == 1 and new_is_white) or (outcome == -1 and not new_is_white)
            if new_won:
                results["new_wins"] += 1
                winner = "new"
            else:
                results["old_wins"] += 1
                winner = "old"

        game_info = {
            "game_index": g,
            "new_is_white": new_is_white,
            "outcome": int(outcome),
            "winner": winner,
            "moves": moves,
            "reason": reason,   # NEW
            "time_seconds": dt,
        }

        logging.info(
            f"[Game {g+1}] Finished. Winner={winner}, outcome={outcome}, "
            f"reason={reason}, num_moves={moves}, time={dt:.2f}s"
        )
        logging.debug(f"[Game {g+1}] Moves: {moves}")

        results["games"].append(game_info)

    tot = time.time() - tic_all
    results["summary"] = {
        "new_wins": results["new_wins"],
        "old_wins": results["old_wins"],
        "draws": results["draws"],
        "num_games": num_games,
        "new_win_rate": results["new_wins"] / num_games if num_games > 0 else 0.0,
        "elapsed_seconds": tot,
    }
    logging.info("Evaluation complete: %s", json.dumps(results["summary"], indent=2))
    return results



def main():
    setup_logger()
    parser = argparse.ArgumentParser(description="Evaluate new vs old chess models using MCTS.")
    parser.add_argument("--new", required=True, help="Path to new model checkpoint")
    parser.add_argument("--old", required=True, help="Path to old model checkpoint")
    parser.add_argument("--games", type=int, default=20, help="Number of games to play (default 20)")
    parser.add_argument("--sims", type=int, default=100, help="MCTS simulations per move (default 100)")
    parser.add_argument("--c_puct", type=float, default=1.0, help="PUCT exploration constant (default 1.0)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for model (cpu or cuda)")
    parser.add_argument("--temperature", type=float, default=1e-3, help="Temperature for pi calculation")
    parser.add_argument("--noise", action="store_true", help="Add Dirichlet noise during games (not typical for evaluation)")
    parser.add_argument("--swap", action="store_true", default=True, help="Alternate colors between games (default True)")
    parser.add_argument("--max_moves", type=int, default=1000, help="Max moves per game before force stop")
    parser.add_argument("--report", type=str, default=None, help="If provided, save JSON report to this path")
    parser.add_argument("--replace-if-better", action="store_true", help="If new model wins > 0.55, replace old checkpoint with new (copy file)")
    args = parser.parse_args()

    results = evaluate_models(
        new_model_path=args.new,
        old_model_path=args.old,
        num_games=args.games,
        sims=args.sims,
        c_puct=args.c_puct,
        device=args.device,
        temperature=args.temperature,
        add_noise_in_selfplay=args.noise,
        swap_colors=args.swap,
        max_moves=args.max_moves,
    )

    if args.report:
        report_dir = os.path.dirname(args.report)
        if report_dir and not os.path.exists(report_dir):
            os.makedirs(report_dir, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(results, f, indent=2)
        logging.info("Saved evaluation report to %s", args.report)

    new_win_rate = results["summary"]["new_win_rate"]
    logging.info("Final new model win rate = %.3f", new_win_rate)

    if args.replace_if_better:
        threshold = 0.55
        if new_win_rate > threshold:
            # backup old
            backup_path = args.old + ".backup"
            if os.path.exists(args.old) and not os.path.exists(backup_path):
                shutil.copy2(args.old, backup_path)
                logging.info("Backed up old model to %s", backup_path)
            # replace old with new
            shutil.copy2(args.new, args.old)
            logging.info("Replaced %s with %s (new was better: %.3f > %.3f)", args.old, args.new, new_win_rate, threshold)
        else:
            logging.info("New model did not pass replacement threshold (%.3f <= %.3f)", new_win_rate, threshold)

    logging.info("Evaluation finished.")


if __name__ == "__main__":
    main()