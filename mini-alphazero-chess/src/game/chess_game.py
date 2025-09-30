import chess
import numpy as np


class ChessGame:
    """
    interface of MCTS and NN
    """

    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset() # reset the board

    def clone(self):
        new_game = ChessGame()
        new_game.board = self.board.copy()
        return new_game

    def get_legal_moves(self):
        return list(self.board.legal_moves) # return list of legal moves

    def play_move(self, move):
        if isinstance(move, str):
            move = chess.Move.from_uci(move)

        if move in self.board.legal_moves:
            self.board.push(move)
        else:
            raise ValueError(f"Illegal move: {move}")

    def is_game_over(self):
        if self.board.is_game_over():
            return True
        if self.board.can_claim_threefold_repetition():
            return True
        if self.board.can_claim_fifty_moves():
            return True
        return False

    def get_result(self):
        """
        Return battle score
        1: White win
        -1: Black win
        0: draw
        """
        result = None

        if self.board.is_checkmate():
            result = 1 if self.board.turn == chess.BLACK else -1
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.can_claim_fifty_moves() or self.board.can_claim_threefold_repetition():
            result = 0
        return result

    def get_turn(self):
        return 1 if self.board.turn == chess.WHITE else -1

    def get_state_fen(self):
        return self.board.fen()

    def get_state(self):
        """
        Return matrix 8x8.
        """
        piece_map = {
            chess.PAWN: 1,
            chess.KNIGHT: 2,
            chess.BISHOP: 3,
            chess.ROOK: 4,
            chess.QUEEN: 5,
            chess.KING: 6
        }
        state = np.zeros((8, 8), dtype=np.int8)
        for square, piece in self.board.piece_map().items():
            row = 7 - (square // 8)
            col = square % 8
            val = piece_map[piece.piece_type]
            if piece.color == chess.BLACK:
                val = -val
            state[row, col] = val
        return state

    def encode_state(self):
        """
        Encode FEN to number tensor for Neural Net
        Return numpy.ndarray dtype=float32, shape = (18, 8, 8) (channels-first) follow convention:
        0..5: white P,N,B,R,Q,K
        6..11: black P,N,B,R,Q,K
        12: side to move (1 if white to move else 0)
        13: white can castle King side (K)
        14: white can castle Queen side (Q)
        15: black can castle King side (k)
        16: black can castle Queen side (q)
        17: en_passant square (one-hot on board)
        row 0 = rank 8
        """
        planes = np.zeros((18, 8, 8), dtype=np.float32)

        # Mapping piece_type -> base index
        piece_to_base = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }

        for square, piece in self.board.piece_map().items():
            row = 7 - (square // 8)
            col = square % 8
            base = piece_to_base[piece.piece_type]
            plane_idx = base + (0 if piece.color == chess.WHITE else 6)
            planes[plane_idx, row, col] = 1.0

        if self.board.turn == chess.WHITE:
            planes[12, :, :] = 1.0
        else:
            planes[12, :, :] = 0.0

        castling_field = self.board.fen().split()[2]
        planes[13, :, :] = 1.0 if 'K' in castling_field else 0.0
        planes[14, :, :] = 1.0 if 'Q' in castling_field else 0.0
        planes[15, :, :] = 1.0 if 'k' in castling_field else 0.0
        planes[16, :, :] = 1.0 if 'q' in castling_field else 0.0

        if self.board.ep_square is not None:
            row = 7 - (self.board.ep_square // 8)
            col = self.board.ep_square % 8
            planes[17, row, col] = 1.0

        return planes

    def render(self):
        print(self.board)

    def get_reason(self) -> str:
        """
        Trả về lý do ván cờ kết thúc, ví dụ:
        - "checkmate"
        - "stalemate"
        - "insufficient_material"
        - "seventyfive_moves"
        - "fivefold_repetition"
        - "fifty_moves"
        - "threefold_repetition"
        - "resignation" (nếu bạn tự implement)
        - "unknown"
        """
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return "not_over"

        term = outcome.termination
        if term == chess.Termination.CHECKMATE:
            return "checkmate"
        elif term == chess.Termination.STALEMATE:
            return "stalemate"
        elif term == chess.Termination.INSUFFICIENT_MATERIAL:
            return "insufficient_material"
        elif term == chess.Termination.SEVENTYFIVE_MOVES:
            return "seventyfive_moves"
        elif term == chess.Termination.FIVEFOLD_REPETITION:
            return "fivefold_repetition"
        elif term == chess.Termination.FIFTY_MOVES:
            return "fifty_moves"
        elif term == chess.Termination.THREEFOLD_REPETITION:
            return "threefold_repetition"
        elif term == chess.Termination.VARIANT_WIN:
            return "variant_win"
        elif term == chess.Termination.VARIANT_LOSS:
            return "variant_loss"
        elif term == chess.Termination.VARIANT_DRAW:
            return "variant_draw"
        else:
            return "unknown"