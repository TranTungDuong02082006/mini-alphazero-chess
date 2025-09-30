import chess
import numpy as np
from typing import Optional, Dict, List, Tuple

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
ALL_ACTION_SLOTS: List[Tuple[int, int]] = []  # list of (from_sq, move_type)
MOVE_TO_IDX: Dict[str, int] = {}
IDX_TO_MOVE: Dict[int, Optional[chess.Move]] = {}

# We'll fill these in _build_action_space()
def _build_action_space():
    """
    Build AlphaZero-style action space: 73 move-types per from-square => 4672 total.
    Move-types layout (per from-square):
      - 0..55   : sliding directions (8 dirs x steps 1..7) -> 56
      - 56..63  : knight moves (8)
      - 64..72  : pawn moves (9)  <-- we implement canonical 9 pawn-types
    """
    ALL_ACTION_SLOTS.clear()
    MOVE_TO_IDX.clear()
    IDX_TO_MOVE.clear()
    idx = 0

    for from_sq in range(64):
        from_file = file_of(from_sq)
        from_rank = rank_of(from_sq)

        # --- sliding moves: 8 directions x steps 1..7  -> 56 slots
        for d in SLIDING_DIRS:
            for step in range(1, 8):
                ALL_ACTION_SLOTS.append((from_sq, ("slide", d, step)))
                # compute to_sq; we won't create MOVE object yet until mapping phase below
                idx += 1

        # --- knight moves: 8 slots
        for d in KNIGHT_DIRS:
            ALL_ACTION_SLOTS.append((from_sq, ("knight", d)))
            idx += 1

        # --- pawn move types: 9 slots (we choose deterministic set)
        # We'll use the following 9 pawn move-types (common AZ-like breakdown):
        # 0: pawn single push (forward 1)
        # 1: pawn double push (forward 2)
        # 2: pawn capture to left (non-promo)
        # 3: pawn capture to right (non-promo)
        # 4: pawn single push promotion to QUEEN
        # 5: pawn single push promotion to ROOK
        # 6: pawn single push promotion to BISHOP
        # 7: pawn single push promotion to KNIGHT
        # 8: pawn capture promotion (we'll treat capture-right promotion and capture-left promotion slots combined by testing both captures)
        for i in range(9):
            ALL_ACTION_SLOTS.append((from_sq, ("pawn", i)))
            idx += 1

    # sanity
    total = len(ALL_ACTION_SLOTS)
    assert total == 64 * 73, f"Action slots count mismatch: {total} != 4672"

    # Now build IDX_TO_MOVE and MOVE_TO_IDX for the slots that correspond to actual legal UCI strings.
    # For each slot, compute the canonical target (if any). If invalid, set None.
    idx = 0
    for (from_sq, spec) in ALL_ACTION_SLOTS:
        slot_move = None
        kind = spec[0]
        if kind == "slide":
            d = spec[1]; step = spec[2]
            to_sq = from_sq + d * step
            # Must ensure step-by-step file/rank don't wrap: check intermediate
            valid = True
            prev = from_sq
            for _s in range(step):
                prev = prev + d
                if not on_board(prev):
                    valid = False
                    break
                # also ensure file wrap not occur for horizontal/diagonal steps:
                # If movement changes file by more than 1 per step it's invalid — but simpler check:
                # check that the absolute file diff between intermediate and from is <= step
            if valid and on_board(to_sq):
                # create move (no promotion here)
                slot_move = chess.Move(from_sq, to_sq)
                u = slot_move.uci()
                IDX_TO_MOVE[idx] = slot_move
                MOVE_TO_IDX[u] = idx
            else:
                IDX_TO_MOVE[idx] = None

        elif kind == "knight":
            d = spec[1]
            to_sq = from_sq + d
            if on_board(to_sq) and abs(file_of(to_sq) - file_of(from_sq)) <= 2:
                slot_move = chess.Move(from_sq, to_sq)
                u = slot_move.uci()
                IDX_TO_MOVE[idx] = slot_move
                MOVE_TO_IDX[u] = idx
            else:
                IDX_TO_MOVE[idx] = None

        elif kind == "pawn":
            ptype = spec[1]
            # handle from white perspective and black via the fact from_rank determines color of pawn's forward direction
            # We will generate both white and black canonical moves depending on from_rank:
            # White pawns move "north" (+8), black pawns move "south" (-8)
            # We produce a move only if the resulting to-square on board is valid; promotions handled by separate kinds.
            if from_rank in range(0, 8):  # always true; use from_rank to decide color
                # Determine forward direction based on potential pawn color:
                # Heuristic: if from_rank <= 1 => likely black pawn starting rank; if from_rank >=6 => white pawn promotion
                # We'll try both: create move only if it makes chess sense (non-wrapping).
                # Simpler deterministic rule: create both white-like and black-like candidates where possible.
                # For ptype meanings:
                # 0: forward1 (non-promo)
                # 1: forward2 (non-promo)
                # 2: capture left (non-promo)
                # 3: capture right (non-promo)
                # 4..7: forward1 promotions (Q,R,B,N)
                # 8: capture promotion (we will try left then right promotions and pick the first valid one)
                created = False

                # WHITE-like moves
                # white forward
                to_forward1 = from_sq + 8
                to_forward2 = from_sq + 16
                to_cap_left = from_sq + 7
                to_cap_right = from_sq + 9

                # BLACK-like moves
                to_b_forward1 = from_sq - 8
                to_b_forward2 = from_sq - 16
                to_b_cap_left = from_sq - 9
                to_b_cap_right = from_sq - 7

                # ptype mapping (try white variant first, then black)
                if ptype == 0:
                    # single push non-promo (either white or black)
                    if on_board(to_forward1):
                        m = chess.Move(from_sq, to_forward1)
                        IDX_TO_MOVE[idx] = m
                        MOVE_TO_IDX[m.uci()] = idx
                        created = True
                    elif on_board(to_b_forward1):
                        m = chess.Move(from_sq, to_b_forward1)
                        IDX_TO_MOVE[idx] = m
                        MOVE_TO_IDX[m.uci()] = idx
                        created = True
                    else:
                        IDX_TO_MOVE[idx] = None

                elif ptype == 1:
                    # double push (two squares forward) - only valid from starting rank typically
                    if on_board(to_forward2):
                        m = chess.Move(from_sq, to_forward2)
                        IDX_TO_MOVE[idx] = m
                        MOVE_TO_IDX[m.uci()] = idx
                        created = True
                    elif on_board(to_b_forward2):
                        m = chess.Move(from_sq, to_b_forward2)
                        IDX_TO_MOVE[idx] = m
                        MOVE_TO_IDX[m.uci()] = idx
                        created = True
                    else:
                        IDX_TO_MOVE[idx] = None

                elif ptype == 2:
                    # capture left
                    if on_board(to_cap_left) and abs(file_of(to_cap_left) - file_of(from_sq)) == 1:
                        m = chess.Move(from_sq, to_cap_left)
                        IDX_TO_MOVE[idx] = m
                        MOVE_TO_IDX[m.uci()] = idx
                        created = True
                    elif on_board(to_b_cap_left) and abs(file_of(to_b_cap_left) - file_of(from_sq)) == 1:
                        m = chess.Move(from_sq, to_b_cap_left)
                        IDX_TO_MOVE[idx] = m
                        MOVE_TO_IDX[m.uci()] = idx
                        created = True
                    else:
                        IDX_TO_MOVE[idx] = None

                elif ptype == 3:
                    # capture right
                    if on_board(to_cap_right) and abs(file_of(to_cap_right) - file_of(from_sq)) == 1:
                        m = chess.Move(from_sq, to_cap_right)
                        IDX_TO_MOVE[idx] = m
                        MOVE_TO_IDX[m.uci()] = idx
                        created = True
                    elif on_board(to_b_cap_right) and abs(file_of(to_b_cap_right) - file_of(from_sq)) == 1:
                        m = chess.Move(from_sq, to_b_cap_right)
                        IDX_TO_MOVE[idx] = m
                        MOVE_TO_IDX[m.uci()] = idx
                        created = True
                    else:
                        IDX_TO_MOVE[idx] = None

                elif 4 <= ptype <= 7:
                    # promotions on forward1 to Q/R/B/N
                    promo_map = {4: chess.QUEEN, 5: chess.ROOK, 6: chess.BISHOP, 7: chess.KNIGHT}
                    promo_piece = promo_map[ptype]
                    if on_board(to_forward1):
                        m = chess.Move(from_sq, to_forward1, promotion=promo_piece)
                        IDX_TO_MOVE[idx] = m
                        MOVE_TO_IDX[m.uci()] = idx
                        created = True
                    elif on_board(to_b_forward1):
                        m = chess.Move(from_sq, to_b_forward1, promotion=promo_piece)
                        IDX_TO_MOVE[idx] = m
                        MOVE_TO_IDX[m.uci()] = idx
                        created = True
                    else:
                        IDX_TO_MOVE[idx] = None

                elif ptype == 8:
                    # capture promotion - try left/right both with promotions - we will store only first valid
                    found = False
                    for promo in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                        if on_board(to_cap_left) and abs(file_of(to_cap_left) - file_of(from_sq)) == 1:
                            m = chess.Move(from_sq, to_cap_left, promotion=promo)
                            IDX_TO_MOVE[idx] = m
                            MOVE_TO_IDX[m.uci()] = idx
                            found = True
                            break
                        if on_board(to_cap_right) and abs(file_of(to_cap_right) - file_of(from_sq)) == 1:
                            m = chess.Move(from_sq, to_cap_right, promotion=promo)
                            IDX_TO_MOVE[idx] = m
                            MOVE_TO_IDX[m.uci()] = idx
                            found = True
                            break
                    if not found:
                        IDX_TO_MOVE[idx] = None

                else:
                    IDX_TO_MOVE[idx] = None

            else:
                IDX_TO_MOVE[idx] = None

        else:
            IDX_TO_MOVE[idx] = None

        # ensure key exists
        if idx not in IDX_TO_MOVE:
            IDX_TO_MOVE[idx] = None
        idx += 1

    # final sanity
    total_idx = len(IDX_TO_MOVE)
    if total_idx != 64 * 73:
        raise RuntimeError(f"Bad build: IDX_TO_MOVE size {total_idx} != 4672")
    # Print summary of how many real moves we registered (i.e. valid UCI -> idx)
    print(f"[Adapter] Built action-slot table with {total_idx} slots; known UCI mappings: {len(MOVE_TO_IDX)}")


# build on import
_build_action_space()
ACTION_SPACE_SIZE = 64 * 73  # 4672

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
