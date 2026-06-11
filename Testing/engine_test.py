import sys
import os
import chess
import numpy as np

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.game.chess_game import ChessGame
    from src.mcts.mcts import MCTS
    from src.mcts.mcts_action_indexer import UCIActionIndexer
    from src.network.model import NeuralNet
    print("Imports successful!")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_chess_game_results():
    print("\n--- Testing ChessGame.get_result() ---")
    
    # 1. Active game
    game = ChessGame()
    res = game.get_result()
    print(f"Active game result: {res}")
    assert res == 0, f"Expected 0 for active game default fallback, got {res}"
    
    # 2. Checkmate (White wins)
    game.reset()
    game.play_move("e2e4")
    game.play_move("e7e5")
    game.play_move("d1h5")
    game.play_move("b8c6")
    game.play_move("f1c4")
    game.play_move("g8f6")
    game.play_move("h5f7")
    print(f"Board FEN: {game.board.fen()}")
    print(f"Is game over: {game.is_game_over()}")
    print(f"Is checkmate: {game.board.is_checkmate()}")
    res = game.get_result()
    print(f"Scholar's Mate result (White checkmates Black): {res}")
    assert res == 1, f"Expected 1 (White win), got {res}"

    # 3. Stalemate
    stalemate_fen = "k7/8/8/8/8/8/5q2/K7 w - - 0 1"
    game.board.set_fen(stalemate_fen)
    print(f"Stalemate Board FEN: {game.board.fen()}")
    print(f"Is game over: {game.is_game_over()}")
    print(f"Is stalemate: {game.board.is_stalemate()}")
    res = game.get_result()
    print(f"Stalemate result: {res}")
    assert res == 0, f"Expected 0 for stalemate, got {res}"

    print("ChessGame get_result tests passed successfully!")

def test_mcts_run():
    print("\n--- Testing MCTS.run() ---")
    device = "cpu"
    net = NeuralNet(device=device)
    indexer = UCIActionIndexer()
    mcts = MCTS(network=net, action_indexer=indexer, num_simulations=10)

    game = ChessGame()
    probs, info = mcts.run(game, temperature=1.0, add_noise=False)

    print(f"MCTS probs sum: {probs.sum():.4f}")
    assert abs(probs.sum() - 1.0) < 1e-4, f"MCTS probabilities must sum to 1.0, got {probs.sum()}"

    selected_idx = info["selected_idx"]
    selected_action = info["selected_action"]
    print(f"MCTS selected index: {selected_idx}, action: {selected_action}")
    assert selected_action is not None, "MCTS failed to select action"

    legal_moves = game.get_legal_moves()
    legal_mask = indexer.legal_mask_from_moves(legal_moves)
    illegal_indices = np.where(~legal_mask)[0]
    for idx in illegal_indices[:10]:
        assert probs[idx] == 0.0, f"Illegal move at index {idx} has non-zero probability {probs[idx]}"

    print("MCTS.run tests passed successfully!")

if __name__ == "__main__":
    test_chess_game_results()
    test_mcts_run()
    print("\nAll automated tests completed successfully!")
