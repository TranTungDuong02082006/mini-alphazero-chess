from fastapi import FastAPI
from pydantic import BaseModel
from src.game.chess_game import ChessGame
import random

app = FastAPI(title="Chess Move API", version="1.0")

class FenRequest(BaseModel):
    fen: str

@app.post("/move")
def get_random_move(request: FenRequest):
    """
    Trả về 1 nước đi hợp lệ từ FEN.
    Dùng để test API.
    """
    game = ChessGame()
    game.board.set_fen(request.fen)

    if game.is_game_over():
        return {
            "status": "game_over",
            "reason": game.get_reason(),
            "result": game.get_result()
        }

    legal_moves = list(game.board.legal_moves)
    move = random.choice(legal_moves)
    game.board.push(move)

    return {
        "status": "ok",
        "move": move.uci(),           
        "fen_after": game.board.fen() # FEN sau khi đi nước đó
    }

@app.get("/")
def root():
    return {"message": "Chess Move API is running"}
