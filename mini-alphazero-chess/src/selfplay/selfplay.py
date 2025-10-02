import numpy as np
import os
from typing import List, Tuple

from src.game.chess_game import ChessGame
from src.mcts.mcts import MCTS
from src.utils.replay_buffer import ReplayBuffer


class SelfPlay:
    def __init__(self, mcts: MCTS, buffer: ReplayBuffer, num_games: int = 1, action_indexer=None):
        """
        SelfPlay wrapper.

        Nếu action_indexer không truyền vào thì sẽ cố lấy từ mcts.action_indexer.
        """
        self.mcts = mcts
        self.buffer = buffer
        self.num_games = int(num_games)

        # Prefer explicit action_indexer argument, otherwise try to get from mcts.
        if action_indexer is not None:
            self.action_indexer = action_indexer
        else:
            self.action_indexer = getattr(mcts, "action_indexer", None)

        if self.action_indexer is None:
            raise RuntimeError(
                "SelfPlay requires an action_indexer. "
                "Pass action_indexer=... to SelfPlay(...) or ensure mcts.action_indexer exists."
            )

    def play_game(self, max_moves: int = 800) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        game = ChessGame()
        game.reset()
        game_data: List[Tuple[np.ndarray, np.ndarray, float]] = []
        turn_history: List[int] = []
        move_count = 0

        while not game.is_game_over() and move_count < max_moves:
            state_enc = game.encode_state()

            probs, info = self.mcts.run(game, temperature=1.0, add_noise=True)
            
            # Defensive sanity checks
            probs = np.asarray(probs, dtype=np.float64)
           
            action_space_size = len(self.action_indexer.all_actions)
            if probs.shape[0] != action_space_size:
                raise RuntimeError(f"probs length {probs.shape[0]} != action_space_size {action_space_size}")

            # compute legal mask and enforce legality on probs
            legal_moves = game.get_legal_moves()
            
            if len(legal_moves) == 0:
                break
            root_legal_mask = self.action_indexer.legal_mask_from_moves(legal_moves)
            
            probs = np.where(root_legal_mask, probs, 0.0)
            
            # fallback to visit_counts if probs invalid
            if (not np.all(np.isfinite(probs))) or (probs.sum() <= 0.0):
                visit_counts = info.get("visit_counts", None)
                if visit_counts is None:
                    legal_idx = np.where(root_legal_mask)[0]
                    if len(legal_idx) == 0:
                        break
                    probs = np.zeros(action_space_size, dtype=np.float64)
                    probs[legal_idx] = 1.0 / len(legal_idx)
                else:
                    visit_counts = np.asarray(visit_counts, dtype=np.float64)
                    visit_counts = np.where(root_legal_mask, visit_counts, 0.0)
                    s = visit_counts.sum()
                    if s > 0 and np.isfinite(s):
                        probs = visit_counts / s
                    else:
                        legal_idx = np.where(root_legal_mask)[0]
                        if len(legal_idx) == 0:
                            break
                        probs = np.zeros(action_space_size, dtype=np.float64)
                        probs[legal_idx] = 1.0 / len(legal_idx)

            probs = probs / probs.sum()

            # sample legal action
            action_idx = int(np.random.choice(action_space_size, p=probs))

            

            # map idx -> action (UCI string or chess.Move)
            action_raw = self.action_indexer.idx_to_action(action_idx)
            import chess
            action = chess.Move.from_uci(action_raw) if isinstance(action_raw, str) else action_raw

            # final legality assert
            legal_uci_set = {m.uci() if isinstance(m, chess.Move) else str(m) for m in game.get_legal_moves()}
            chosen_uci = action.uci() if isinstance(action, chess.Move) else str(action)
            if chosen_uci not in legal_uci_set:
                # deterministic fallback: pick argmax among legal visit_counts
                visit_counts = info.get("visit_counts", None)
                if visit_counts is not None:
                    visit_counts = np.asarray(visit_counts, dtype=np.float64)
                    legal_idx = np.where(root_legal_mask)[0]
                    rel = int(np.argmax(visit_counts[legal_idx]))
                    action_idx = int(legal_idx[rel])
                    action_raw = self.action_indexer.idx_to_action(action_idx)
                    action = chess.Move.from_uci(action_raw) if isinstance(action_raw, str) else action_raw
                else:
                    action = next(iter(game.get_legal_moves()))

            game_data.append((state_enc, probs.copy(), None))
            turn_history.append(game.get_turn())

            game.play_move(action)
            move_count += 1
        # backfill values
        if game.is_game_over():
            result = game.get_result()
            if result is None:
                print("Game over but result is None, treating as draw.")
                result = 0
        else:
            result = 0
        updated_data: List[Tuple[np.ndarray, np.ndarray, float]] = []
        for (state, policy, _), turn in zip(game_data, turn_history):
            value = result * turn
            updated_data.append((state, policy, float(value)))

        return updated_data

    def generate(self):
        for g in range(self.num_games):
            data = self.play_game()
            print(f"[SelfPlay] Game {g+1}/{self.num_games} generated, {len(data)} moves.")
            for s, p, v in data:
                self.buffer.add(s, p, v)
            print(f"[SelfPlay] Game {g+1}/{self.num_games} finished, {len(data)} moves added.")
            self.buffer.save("replay_buffer.pkl.gz")
            print(f"[SelfPlay] Buffer saved after game {g + 1}, size={len(self.buffer)}")


if __name__ == "__main__":
    from src.network.model import NeuralNet, ChessNet
    from src.mcts.mcts_action_indexer import UCIActionIndexer

    net = NeuralNet()
    indexer = UCIActionIndexer()
    mcts = MCTS(network=net, action_indexer=indexer, num_simulations=100)

    # Replay buffer
    buffer = ReplayBuffer(max_size=50000)

    # Self-play
    sp = SelfPlay(mcts, buffer, num_games=1, action_indexer=indexer)
    sp.generate()

    print(f"Replay buffer size: {len(buffer.buffer)}")