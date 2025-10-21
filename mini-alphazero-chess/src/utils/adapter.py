import chess
import numpy as np
from typing import Optional, Dict, List, Tuple
from queue import Queue
# ======================
# State encoder
# ======================
def board_to_tensor(board: chess.Board, board_size: int = 8):
    planes = np.zeros((18, board_size, board_size), dtype=np.float32)

    piece_to_plane = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    for square, piece in board.piece_map().items():
        row = chess.square_rank(square)
        col = chess.square_file(square)
        base = piece_to_plane[piece.piece_type]
        plane = base if piece.color == chess.WHITE else base + 6
        planes[plane, row, col] = 1.0

    # --- side-to-move ---
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else -1.0

    # --- castling rights ---
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16, :, :] = 1.0

    # --- 50-move rule (no-progress) ---
    planes[17, :, :] = board.halfmove_clock / 50.0

    return planes

def file_of(sq: int) -> int:
    return sq % 8

def rank_of(sq: int) -> int:
    return sq // 8

def on_board(sq: int) -> bool:
    return 0 <= sq < 64

# -----------------------
# Define directions used by sliding moves and knight
# -----------------------
SLIDING_DIRS = [
    +1,   # east
    -1,   # west
    +8,   # north
    -8,   # south
    +9,   # north-east
    -9,   # south-west
    +7,   # north-west
    -7,   # south-east
]

KNIGHT_DIRS = [15, 17, -15, -17, 10, -10, 6, -6]

# -----------------------
# Global action maps
# -----------------------
# =================================================================
# Global action maps to be populated
# =================================================================

# ALL_ACTION_SLOTS: A list of 4672 slots, where each slot contains a
# chess.Move object or None if the move is geometrically impossible (e.g., wraps around the board).
ALL_ACTION_SLOTS: List[Optional[chess.Move]] = [None] * 4672

# MOVE_TO_IDX: A dictionary mapping a move's UCI string representation to its unique index (0-4671).
MOVE_TO_IDX: Dict[str, int] = {}

# IDX_TO_MOVE: A dictionary mapping an index back to its corresponding chess.Move object.
IDX_TO_MOVE: Dict[int, Optional[chess.Move]] = {}
ACTION_SPACE_SIZE: int = 0  # to be set after building maps

# def build_action_maps():
#     """
#     Populates the global move mapping variables (ALL_ACTION_SLOTS, MOVE_TO_IDX, IDX_TO_MOVE)
#     based on the 64x73 action space used in AlphaZero for chess.
#     """
#     global ACTION_SPACE_SIZE
#     # Clear maps to ensure idempotency if called multiple times
#     ALL_ACTION_SLOTS[:] = [None] * 4672
#     MOVE_TO_IDX.clear()
#     IDX_TO_MOVE.clear()
    
#     # Define move directions from a square's index perspective
#     # N, NE, E, SE, S, SW, W, NW (clockwise)
#     queen_directions = [8, 9, 1, -7, -8, -9, -1, 7]
#     # NNE, ENE, ESE, SSE, SSW, WSW, WNW, NNW (clockwise)
#     knight_directions = [17, 10, -6, -15, -17, -10, 6, 15]
#     # Pawn moves from White's perspective: NW, N, NE
#     underpromotion_directions = [7, 8, 9]
#     underpromotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

#     action_index = 0
#     for from_sq in chess.SQUARES:
#         from_rank, from_file = chess.square_rank(from_sq), chess.square_file(from_sq)

#         # 1. Queen-like moves (56 planes)
#         for direction in queen_directions:
#             for distance in range(1, 8):  # 1 to 7 squares
#                 to_sq = from_sq + direction * distance

#                 move = None
#                 if 0 <= to_sq < 64:
#                     to_rank, to_file = chess.square_rank(to_sq), chess.square_file(to_sq)
#                     # Check for board wrap-around to ensure valid geometry
#                     is_valid = False
#                     if direction in [8, -8, 1, -1]:  # Rook moves
#                         if from_rank == to_rank or from_file == to_file:
#                             is_valid = True
#                     else:  # Bishop moves
#                         if abs(from_rank - to_rank) == abs(from_file - to_file):
#                             is_valid = True

#                     if is_valid:
#                         # By convention, pawn moves to the promotion rank become Queen promotions
#                         is_pawn_move = from_file == to_file
#                         if (from_rank == 6 and to_rank == 7 and is_pawn_move) or \
#                            (from_rank == 1 and to_rank == 0 and is_pawn_move):
#                             move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
#                         else:
#                             move = chess.Move(from_sq, to_sq)
                
#                 ALL_ACTION_SLOTS[action_index] = move
#                 action_index += 1

#         # 2. Knight moves (8 planes)
#         for direction in knight_directions:
#             to_sq = from_sq + direction
            
#             move = None
#             if 0 <= to_sq < 64:
#                 to_rank, to_file = chess.square_rank(to_sq), chess.square_file(to_sq)
#                 # Check for valid knight move shape
#                 if abs(from_rank - to_rank) * abs(from_file - to_file) == 2:
#                     move = chess.Move(from_sq, to_sq)

#             ALL_ACTION_SLOTS[action_index] = move
#             action_index += 1

#         # 3. Underpromotion moves (9 planes)
#         for direction in underpromotion_directions:
#             for piece in underpromotion_pieces:
#                 to_sq = from_sq + direction
                
#                 move = None
#                 if 0 <= to_sq < 64:
#                     to_rank, to_file = chess.square_rank(to_sq), chess.square_file(to_sq)
#                     # Must be a one-rank advance
#                     if abs(from_rank - to_rank) == 1:
#                         # Check forward vs. diagonal capture geometry
#                         if (direction == 8 and from_file == to_file) or \
#                            (direction in [7, 9] and abs(from_file - to_file) == 1):
#                             move = chess.Move(from_sq, to_sq, promotion=piece)

#                 ALL_ACTION_SLOTS[action_index] = move
#                 action_index += 1
    
#     # Populate the reverse lookup dictionaries from the generated list
#     for idx, move in enumerate(ALL_ACTION_SLOTS):
#         if move:
#             IDX_TO_MOVE[idx] = move
#             MOVE_TO_IDX[move.uci()] = idx

#     ACTION_SPACE_SIZE = len(ALL_ACTION_SLOTS)
#     assert ACTION_SPACE_SIZE == 4672, f"Expected action space size 4672, got {ACTION_SPACE_SIZE}"

def build_action_maps():
    """
    Tạo ra các bản đồ ánh xạ nước đi toàn cục theo cấu trúc 3 loại chuẩn AlphaZero:
    1. Queen-like (Trượt + Tốt Phong Hậu) - 56 planes (Offset 0-55)
    2. Knight - 8 planes (Offset 56-63)
    3. Underpromotion (Tốt Phong N, B, R) - 9 planes (Offset 64-72)
    Đã sửa các lỗi logic phong cấp.
    """
    global ACTION_SPACE_SIZE
    # Xóa các bản đồ
    ALL_ACTION_SLOTS[:] = [None] * 4672
    MOVE_TO_IDX.clear()
    IDX_TO_MOVE.clear()

    queen_directions_relative = [8, 9, 1, -7, -8, -9, -1, 7]
    knight_directions_relative = [17, 10, -6, -15, -17, -10, 6, 15]
    white_promo_directions = [7, 8, 9]
    black_promo_directions = [-9, -8, -7]
    underpromotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
    promotion_able_list = Queue()
    for idx in range(4672):
        move: Optional[chess.Move] = None
        move_sub: Optional[chess.Move] = None
        from_sq = idx // 73
        type_offset = idx % 73 # 0-72
        from_rank = chess.square_rank(from_sq)
        from_file = chess.square_file(from_sq)

        # --- Loại 1: Queen-like ---
        if 0 <= type_offset < 56:
            direction_index = type_offset // 7
            distance = (type_offset % 7) + 1
            direction = queen_directions_relative[direction_index]
            to_sq = from_sq + direction * distance

            if 0 <= to_sq < 64:
                to_rank = chess.square_rank(to_sq)
                to_file = chess.square_file(to_sq)
                is_valid_geometry = False
                if direction in [8, -8, 1, -1] and (from_rank == to_rank or from_file == to_file): is_valid_geometry = True
                elif direction in [9, -7, -9, 7] and (abs(from_rank - to_rank) == abs(from_file - to_file)): is_valid_geometry = True

                if is_valid_geometry:
                    move = chess.Move(from_sq, to_sq)
                    # Kiểm tra phong cấp Tốt thành Hậu
                    is_white_promo = (from_rank == 6 and to_rank == 7)
                    is_black_promo = (from_rank == 1 and to_rank == 0)
                    is_pawn_shape = (abs(from_file - to_file) <= 1)

                    if (is_white_promo or is_black_promo) and is_pawn_shape:
                        promotion_able_list.put(chess.Move(from_sq, to_sq, promotion=chess.QUEEN))

        # --- Loại 2: Knight (Offset 56-63) ---
        elif 56 <= type_offset < 64:
            direction_index = type_offset - 56
            direction = knight_directions_relative[direction_index]
            to_sq = from_sq + direction
            if 0 <= to_sq < 64:
                to_rank = chess.square_rank(to_sq)
                to_file = chess.square_file(to_sq)
                if abs(from_rank - to_rank) * abs(from_file - to_file) == 2:
                    move = chess.Move(from_sq, to_sq)

        # --- Loại 3: Underpromotion (N, B, R) (Offset 64-72) ---
        elif 64 <= type_offset < 73:
            underpromo_index = type_offset - 64 # Index 0..8
            # Xác định hướng và quân cờ từ index này
            # Index 0,1,2: NW (N, B, R)
            # Index 3,4,5: N  (N, B, R)
            # Index 6,7,8: NE (N, B, R) - cho Trắng
            direction_sub_index = underpromo_index // 3 # 0, 1, 2
            piece_index = underpromo_index % 3          # 0, 1, 2 (N, B, R)

            promotion_piece = underpromotion_pieces[piece_index]
            direction = 0
            target_rank = -1
            is_promo_rank = False

            if from_rank == 6: # Trắng
                direction = white_promo_directions[direction_sub_index]
                target_rank = 7
                is_promo_rank = True
            elif from_rank == 1: # Đen
                direction = black_promo_directions[direction_sub_index]
                target_rank = 0
                is_promo_rank = True

            if is_promo_rank:
                to_sq = from_sq + direction
                if 0 <= to_sq < 64 and chess.square_rank(to_sq) == target_rank:
                    to_file = chess.square_file(to_sq)
                    if (abs(direction) == 8 and from_file == to_file) or \
                       (abs(direction) in [7, 9] and abs(from_file - to_file) == 1):
                        move = chess.Move(from_sq, to_sq, promotion=promotion_piece)

        ALL_ACTION_SLOTS[idx] = move
    print(len(promotion_able_list.queue))
    for idx in range(4672):
        if ALL_ACTION_SLOTS[idx] is None:
            # Kiểm tra xem có thể lấy từ hàng đợi phong cấp Hậu không
            if not promotion_able_list.empty():
                move_sub = promotion_able_list.get()
                ALL_ACTION_SLOTS[idx] = move_sub


    # --- Kết thúc lặp qua các chỉ mục ---

    # Điền vào các bản đồ tra cứu cuối cùng
    successful_moves = 0
    for idx, move_obj in enumerate(ALL_ACTION_SLOTS):
        if move_obj:
            IDX_TO_MOVE[idx] = move_obj
            MOVE_TO_IDX[move_obj.uci()] = idx # Sửa lỗi gán giá trị
            successful_moves += 1

    ACTION_SPACE_SIZE = len(ALL_ACTION_SLOTS) # = 4672
    assert ACTION_SPACE_SIZE == 4672
    print(f"[build_action_maps] Hoàn thành. Tổng số khe: {ACTION_SPACE_SIZE}. Số nước đi hợp lệ hình học: {successful_moves}")
        

# -----------------------
# helper converters
# -----------------------
def move_to_policy(move: chess.Move) -> Optional[int]:
    """
    Given a chess.Move object (uci), return corresponding action index if it exists in MOVE_TO_IDX,
    otherwise None.
    """
    if move is None:
        return None
    return MOVE_TO_IDX.get(move.uci(), None)

def idx_to_move(idx: int) -> Optional[chess.Move]:
    """
    Return chess.Move for given idx (or None if slot has no concrete move).
    """
    return IDX_TO_MOVE.get(idx, None)



class NeuralNetAdapter:
    """
    Adapter exposing:
      - predict(state_encoding, return_logits=True) -> (logits, value)
      - mask_and_normalize(logits, mask) -> normalized probs over action space
    Internally delegates to your Neuralnet instance (from src.network.model).
    """
    def __init__(self, inner_net):
        self.net = inner_net

    def _to_np(self, arr):
        # accept torch tensors or numpy arrays
        try:
            import torch
            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().numpy()
        except Exception:
            pass
        return np.array(arr)

    def predict(self, state_encoding, return_logits: bool = True):
        """
        state_encoding: whatever ChessGame.encode_state() returns
        returns (logits, value)
        """
        # if your network expects batch dimension, call accordingly
        # many implementations: net.predict(state) -> (policy_probs, value) but we need logits
        res = self.net.predict(state_encoding, return_logits=True)
        # Expecting res to be (logits, value) OR (policy_probs, value) depending on implementation
        logits, value = res
        logits = self._to_np(logits).ravel()
        value = float(value)
        return logits, value

    def mask_and_normalize(self, logits, mask=None):
        """
        Default: apply -inf to illegal moves then softmax on remaining logits.
        If your model already provides a mask_and_normalize, you can call it instead.
        """
        logits = self._to_np(logits)
        if mask is None:
            # just softmax over all
            ex = np.exp(logits - np.max(logits))
            return ex / ex.sum()
        mask = np.array(mask, dtype=bool)
        # large negative for illegal
        neg_inf = -1e9
        mod = np.where(mask, logits, neg_inf)
        max_v = np.max(mod[mask]) if mask.any() else 0.0
        ex = np.exp(mod - max_v)
        ex = ex * mask  # zero out illegal
        s = ex.sum()
        if s == 0:
            # fallback: uniform over legal moves
            res = np.zeros_like(ex)
            if mask.any():
                res[mask] = 1.0 / mask.sum()
            return res
        return ex / s
