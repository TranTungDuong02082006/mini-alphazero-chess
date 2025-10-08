from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import torch
import chess
import numpy as np

from src.game.chess_game import ChessGame
from src.mcts.mcts import MCTS
from src.mcts.mcts_action_indexer import UCIActionIndexer
from src.network.model import NeuralNet
from src.selfplay import choose_move

app = FastAPI(title="Chess Move API", version="4.0")

class FenRequest(BaseModel):
    fen: str
    method: Optional[str] = "best"  # "best" or "random"

# --- Setup device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Initialize NN + MCTS once ---
model = NeuralNet(device=device)
# TODO: load checkpoint weights nếu có
action_indexer = UCIActionIndexer()
mcts = MCTS(network=model, action_indexer=action_indexer, num_simulations=800, c_puct=1.0)


@app.post("/move")
def get_move(request: FenRequest):
    """
    Return a move from a given FEN.

    method = "random" -> select random legal move
    method = "best"   -> select best move using MCTS + NN
    """
    game = ChessGame()
    try:
        game.board.set_fen(request.fen)
    except Exception as e:
        return {"status": "error", "reason": f"Invalid FEN: {e}"}

    if game.is_game_over():
        return {
            "status": "game_over",
            "reason": game.get_reason(),
            "result": game.get_result()
        }

    legal_moves = list(game.board.legal_moves)
    if len(legal_moves) == 0:
        return {"status": "game_over", "reason": "No legal moves", "result": None}

    if request.method == "random":
        move = np.random.choice(legal_moves)
        game.board.push(move)
        return {
            "status": "ok",
            "method": "random",
            "move": move.uci(),
            "fen_after": game.board.fen()
        }

    # --- method == "best" ---
    move, probs, visit_counts, root_value = choose_move(game, mcts, action_indexer)

    if move is not None:
        game.board.push(move)

    return {
        "status": "ok",
        "method": "best",
        "move": move.uci() if isinstance(move, chess.Move) else str(move),
        "fen_after": game.board.fen(),
        "policy_probs": probs.tolist(),
        "visit_counts": visit_counts.tolist(),
        "root_value": root_value
    }


@app.get("/")
def root():
    return {"message": "Chess Move API v4 is running", "device": device}
