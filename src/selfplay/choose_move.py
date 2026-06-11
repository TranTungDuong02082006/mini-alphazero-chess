import numpy as np
import chess
from typing import Any, Tuple, Optional
from src.mcts.mcts import MCTS
from src.mcts.mcts_action_indexer import UCIActionIndexer

def choose_move(
    game,
    mcts: MCTS,
    action_indexer: UCIActionIndexer,
    temperature: float = 1e-5,
    add_noise: bool = False
) -> Tuple[Optional[chess.Move], np.ndarray, np.ndarray, float]:
    """
    Select the best move from the current game state using MCTS + Neural Network.
    Runs MCTS only once and returns all relevant information for API usage.

    Args:
        game: ChessGame instance representing current board state.
        mcts: MCTS instance initialized with a neural network.
        action_indexer: Maps moves <-> indices for MCTS.
        temperature: Temperature for computing final policy (low -> near-deterministic argmax).
        add_noise: Whether to add Dirichlet noise (usually False for real play).

    Returns:
        move: chess.Move object (best move selected) or None if game over
        probs: np.ndarray of shape (action_space,) with policy probabilities
        visit_counts: np.ndarray of shape (action_space,) with visit counts
        root_value: float, value of root node from NN
    """
    # Run MCTS from current state
    probs, info = mcts.run(game, temperature=temperature, add_noise=add_noise)

    # Convert probs to np.ndarray
    probs = np.asarray(probs, dtype=np.float64)

    # Get legal moves and mask
    legal_moves = game.get_legal_moves()
    if len(legal_moves) == 0:
        return None, probs, np.zeros_like(probs), 0.0  # Game over
    legal_mask = action_indexer.legal_mask_from_moves(legal_moves)
    probs = np.where(legal_mask, probs, 0.0)

    # Fallback if probs invalid
    if not np.all(np.isfinite(probs)) or probs.sum() <= 0:
        visit_counts = info.get("visit_counts", None)
        if visit_counts is not None:
            visit_counts = np.asarray(visit_counts, dtype=np.float64)
            visit_counts = np.where(legal_mask, visit_counts, 0.0)
            s = visit_counts.sum()
            if s > 0 and np.isfinite(s):
                probs = visit_counts / s
            else:
                probs = np.zeros_like(probs)
                legal_idx = np.where(legal_mask)[0]
                probs[legal_idx] = 1.0 / len(legal_idx)
        else:
            probs = np.zeros_like(probs)
            legal_idx = np.where(legal_mask)[0]
            probs[legal_idx] = 1.0 / len(legal_idx)

    # Normalize probs
    probs = probs / probs.sum()

    # Choose move: argmax over probs
    action_idx = int(np.argmax(probs))
    action_raw = action_indexer.idx_to_action(action_idx)
    move = chess.Move.from_uci(action_raw) if isinstance(action_raw, str) else action_raw

    # Defensive legality check
    legal_uci_set = {m.uci() if isinstance(m, chess.Move) else str(m) for m in legal_moves}
    chosen_uci = move.uci() if isinstance(move, chess.Move) else str(move)
    if chosen_uci not in legal_uci_set:
        # Deterministic fallback: pick legal move with highest visit_count
        visit_counts = info.get("visit_counts", None)
        if visit_counts is not None:
            visit_counts = np.asarray(visit_counts, dtype=np.float64)
            legal_idx = np.where(legal_mask)[0]
            rel = int(np.argmax(visit_counts[legal_idx]))
            action_idx = int(legal_idx[rel])
            action_raw = action_indexer.idx_to_action(action_idx)
            move = chess.Move.from_uci(action_raw) if isinstance(action_raw, str) else action_raw
        else:
            move = next(iter(legal_moves))

    visit_counts = np.asarray(info.get("visit_counts", []), dtype=np.float64)
    root_value = info.get("root_value", 0.0)

    return move, probs, visit_counts, root_value
