# src/mcts/mcts_action_indexer.py
from typing import List, Any
import chess
import logging


# Try both import styles so file works when run as package or as script
try:
    # when running as `python -m src...` or tests that add src to path
    from src.utils.adapter import ALL_ACTION_SLOTS, MOVE_TO_IDX, IDX_TO_MOVE, build_action_maps
    from src.utils import adapter
except Exception:
    # fallback when running directly with src on PYTHONPATH (common in dev)
    from src.utils.adapter import ALL_ACTION_SLOTS, MOVE_TO_IDX, IDX_TO_MOVE, build_action_maps
    from src.utils import adapter

logger = logging.getLogger(__name__)


class UCIActionIndexer:
    """
    Action indexer backed by the global adapter action table (ALL_MOVES).
    - self.all_actions: list of UCI strings (one per index)
    - self.action_to_index: mapping uci -> idx
    - idx_to_action returns chess.Move (from adapter) when available, else UCI string
    """

    def __init__(self):
        # Build list of UCI strings from adapter's ALL_MOVES (which may contain chess.Move)
        print(adapter.ACTION_SPACE_SIZE)
        build_action_maps()
        print(adapter.ACTION_SPACE_SIZE)
        actions = []
        for m in ALL_ACTION_SLOTS:
            if isinstance(m, chess.Move):
                actions.append(m.uci())
            else:
                # if adapter stores UCI strings already
                actions.append(str(m))

        # ensure length matches adapter constant if present
        if hasattr(adapter, 'ACTION_SPACE_SIZE'):
            expected = adapter.ACTION_SPACE_SIZE
            if len(actions) != expected:
                logger.warning(
                    "Adapter ACTION_SPACE_SIZE=%s but built %s UCI strings. Adapter content may be inconsistent.",
                    expected, len(actions)
                )

        self.all_actions: List[str] = actions
        self.action_to_index = {a: i for i, a in enumerate(self.all_actions)}

    def action_to_idx(self, action: Any) -> int:
        """Return index for chess.Move or UCI string"""
        if isinstance(action, chess.Move):
            action = action.uci()
        return self.action_to_index[action]

    def idx_to_action(self, idx: int) -> Any:
        """
        Return the canonical action object for an index.
        Prefer returning chess.Move from adapter's IDX_TO_MOVE if available,
        otherwise return UCI string from self.all_actions.
        """
        try:
            # IDX_TO_MOVE indexed by int -> chess.Move (adapter should provide)
            move_obj = IDX_TO_MOVE.get(idx)
            if move_obj is not None:
                return move_obj
        except Exception:
            # if IDX_TO_MOVE not a dict or not available, ignore
            pass

        # fallback: return UCI string
        return self.all_actions[idx]

    def legal_mask_from_moves(self, legal_moves: List[Any]):
        """
        Build boolean mask of length len(self.all_actions): True for legal uci moves.
        Accepts chess.Move objects or UCI strings in `legal_moves`.
        """
        import numpy as np
        mask = np.zeros(len(self.all_actions), dtype=np.bool_)
        for m in legal_moves:
            if isinstance(m, chess.Move):
                u = m.uci()
            else:
                u = str(m)
            idx = self.action_to_index.get(u)
            #print(f"[UCIActionIndexer] legal move {u} -> idx {idx}")
            if idx is not None:
                mask[idx] = True
        return mask