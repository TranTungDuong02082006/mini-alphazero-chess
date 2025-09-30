"""
MCTS implementation for Mini-AlphaZero chess project (mcts.py)

Assumptions & integration points (please ensure your project exposes these):
- src/game/chess_game.py -> class ChessGame
    - clone() -> ChessGame (deep copy of game state)
    - encode_state() -> np.ndarray (state encoding consumable by NeuralNet.predict)
    - get_legal_moves() -> List[any] (list of moves in whatever format your game uses)
    - play_move(move) -> None (apply the move on the game object)
    - is_game_over() -> bool
    - get_result() -> float  (from perspective of player who moved first; used only if you want terminal values)
    - get_turn() -> int (current player to move, e.g., +1 or -1 or 0/1 — this implementation treats values as from current player's perspective and flips during backprop)

- src/network/model.py -> class NeuralNet (or Neuralnet)
    - predict(state_encoding) -> Tuple[np.ndarray, float]
        * returns (policy_logits, value) where policy_logits is a 1D numpy array aligned with the action-space used by your project (size = action_space_size)
        * value is a scalar in [-1, 1] from the perspective of the player to move in the provided state
    - mask_and_normalize(logits, mask) -> np.ndarray
        * masks out illegal actions and returns a normalized probability distribution over legal actions

Because projects can differ on how the action-space is represented (UCI strings, move objects, action indices...), this file includes an ActionIndexer helper class you can subclass or replace to map between your game's moves and network action indices.

This MCTS follows the AlphaZero-style MCTS where leaf evaluation is performed by the neural network (policy + value) and no random playouts are used.

"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# typing for external classes (avoid circular import at runtime)
# from src.game.chess_game import ChessGame
# from src.network.model import NeuralNet


def safe_softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    """
    Stable softmax with temperature scaling.
    """
    if temperature <= 1e-6:  # quá nhỏ thì coi như argmax
        probs = np.zeros_like(x, dtype=np.float32)
        probs[np.argmax(x)] = 1.0
        return probs

    x = np.array(x, dtype=np.float64) / temperature
    x = x - np.max(x)  # tránh overflow
    exp_x = np.exp(x)
    if exp_x.sum() == 0:
        return np.ones_like(x, dtype=np.float32) / len(x)
    return exp_x / exp_x.sum()

def safe_softmax_counts(counts: np.ndarray, temperature: float) -> np.ndarray:
    """
    Convert visit counts -> probability distribution in a numerically stable way.
    If temperature is very small -> return one-hot on argmax.
    This avoids overflow when doing counts ** (1/temperature).
    """
    counts = np.asarray(counts, dtype=np.float64)
    if temperature <= 1e-6:
        probs = np.zeros_like(counts, dtype=np.float32)
        best = int(np.argmax(counts))
        probs[best] = 1.0
        return probs

    # use log-space-like trick: apply a stable softmax over log(counts)
    # but counts may contain zeros, use log(counts + eps) to avoid -inf
    eps = 1e-12
    log_counts = np.log(counts + eps) / float(temperature)
    # shift
    log_counts = log_counts - np.max(log_counts)
    exp_vals = np.exp(log_counts)
    s = exp_vals.sum()
    if s == 0 or not np.isfinite(s):
        # fallback: uniform over positive counts, else uniform full
        positive = counts > 0
        if positive.any():
            probs = positive.astype(np.float32) / positive.sum()
            return probs
        else:
            n = len(counts)
            return np.ones(n, dtype=np.float32) / n
    return (exp_vals / s).astype(np.float32)

@dataclass
class Node:
    prior: float = 0.0  # P(s,a)
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "Node"] = field(default_factory=dict)
    is_expanded: bool = False

    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class ActionIndexer:
    """
    ActionIndexer maps between your game's moves and the neural network's action indices.

    Default implementation assumes the network's action space equals a provided list of action identifiers
    (for example, UCI strings or other canonical move encodings). You must create and pass an instance
    of ActionIndexer matching your project's representation.
    """

    def __init__(self, all_actions: Optional[List[Any]] = None):
        # all_actions is a list of canonical action identifiers where index==action_index
        if all_actions is None:
            all_actions = []
        self.all_actions = list(all_actions)
        self.action_to_index = {a: i for i, a in enumerate(self.all_actions)}

    def action_to_idx(self, action: Any) -> int:
        """Convert a move representation (from ChessGame.get_legal_moves) to an action index."""
        return self.action_to_index[action]

    def idx_to_action(self, idx: int) -> Any:
        return self.all_actions[idx]

    def legal_mask_from_moves(self, legal_moves: List[Any]) -> np.ndarray:
        """Return a boolean mask (1 = legal) over the full action space for the given legal moves list."""
        mask = np.zeros(len(self.all_actions), dtype=np.bool_)
        for m in legal_moves:
            idx = self.action_to_index.get(m)
            if idx is not None:
                mask[idx] = True
        return mask


class MCTS:
    def __init__(
        self,
        network,  # instance of NeuralNet-like class with predict and mask_and_normalize
        action_indexer: ActionIndexer,
        num_simulations: int = 800,
        c_puct: float = 1.0,
        dirichlet_alpha: Optional[float] = 0.03,
        dirichlet_epsilon: float = 0.25,
        ):
        self.network = network
        self.action_indexer = action_indexer
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def _select(self, node: Node) -> Tuple[Node, int]:
        """Select child with highest PUCT score. Returns (parent_node, action_index_selected)."""
        # parent node must be expanded and have children
        total_visits = sum(child.visit_count for child in node.children.values())
        best_score = -float("inf")
        best_action = -1
        best_child = None
        for action_idx, child in node.children.items():
            # PUCT score
            q = child.q_value()
            u = self.c_puct * child.prior * math.sqrt(total_visits + 1e-8) / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action_idx
                best_child = child
        return best_child, best_action

    def _expand_and_eval(self, game, node: Node) -> float:
        """
        Expand the leaf node and evaluate with neural network.
        Returns the value (from current player's perspective).
        """
        # get encoding and legal moves
        state_enc = game.encode_state()
        # Neural network returns logits over the action space and a scalar value
        logits, value = self.network.predict(state_enc, return_logits=True)
        # Build legal mask
        legal_moves = game.get_legal_moves()
        legal_mask = self.action_indexer.legal_mask_from_moves(legal_moves)

        # Mask and normalize policy
        policy = self.network.mask_and_normalize(logits, legal_mask)

        # Create children for legal actions
        for idx, is_legal in enumerate(legal_mask.tolist()):
            if not is_legal:
                continue
            p = float(policy[idx])
            if p > 0:
                node.children[idx] = Node(prior=p)

        node.is_expanded = True
        return float(value)

    def _backpropagate(self, path: List[Tuple[Node, int]], value: float) -> None:
        """
        Backpropagate `value` up the path. `value` is from the perspective of the player to move at the leaf.
        We negate value at each step because players alternate.
        `path` is list of (node, action_idx) starting at root and going to leaf.
        """
        for node, action_idx in reversed(path):
            child = node.children[action_idx]
            child.visit_count += 1
            # add value to child's sum
            child.value_sum += value
            # next player's perspective -> flip
            value = -value

    def _add_dirichlet_noise(self, root: Node):
        """Add Dirichlet noise to root priors to encourage exploration (AlphaZero trick)."""
        if self.dirichlet_alpha is None or self.dirichlet_epsilon == 0:
            return
        priors = np.array([child.prior for child in root.children.values()], dtype=float)
        if priors.sum() == 0:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(priors))
        for (action_idx, child), n in zip(root.children.items(), noise):
            child.prior = child.prior * (1 - self.dirichlet_epsilon) + n * self.dirichlet_epsilon

    def run(self, root_game, temperature: float = 1e-3, add_noise: bool = True) -> Tuple[np.ndarray, Any]:
        """
        Run MCTS simulations from root_game and return action probabilities (over the full action space)
        and optionally the selected action (most visited) as an action object.

        Returns: (action_probs: np.ndarray(shape=(action_space_size,)), info)
        info contains a small dict with visit counts and optionally chosen action.
        """
        # build a fresh root
        root = Node()
        # expand root
        root_game_copy = root_game.clone()
        root_value = self._expand_and_eval(root_game_copy, root)  # expand root with NN

        if add_noise:
            self._add_dirichlet_noise(root)

        for sim in range(self.num_simulations):
            game_copy = root_game.clone()
            node = root
            path: List[Tuple[Node, int]] = []  # store (node, action_idx) pairs

            # selection
            while node.is_expanded and len(node.children) > 0:
                child, action_idx = self._select(node)
                if child is None:
                    break
                # play action on the game copy
                action = self.action_indexer.idx_to_action(action_idx)
                game_copy.play_move(action)
                path.append((node, action_idx))
                node = child
                # if game ended during selection, break and backpropagate terminal value
                if game_copy.is_game_over():
                    # get terminal value from game. We assume get_result() returns value from perspective of player who started the game,
                    # so convert it to value from perspective of current player to move.
                    result = game_copy.get_result()
                    # result should be in [-1, 1]
                    value = float(result)
                    # backpropagate
                    self._backpropagate(path, value)
                    break

            else:
                # if we didn't hit a terminal state during selection and node not expanded -> expand and evaluate
                if not game_copy.is_game_over():
                    leaf_value = self._expand_and_eval(game_copy, node)
                    # backpropagate the leaf value
                    self._backpropagate(path, leaf_value)
                else:
                    # terminal at the leaf
                    result = game_copy.get_result()
                    value = float(result)
                    self._backpropagate(path, value)

        # build action probability vector from visit counts
        action_space_size = len(self.action_indexer.all_actions)
        visit_counts = np.zeros(action_space_size, dtype=np.float32)
        for idx, child in root.children.items():
            visit_counts[idx] = child.visit_count

            # get legal mask at root (important!)
        root_legal_moves = root_game.get_legal_moves()
        root_legal_mask = self.action_indexer.legal_mask_from_moves(root_legal_moves)  # bool array

        # Build probs in a stable and legal-aware way
        probs = safe_softmax_counts(visit_counts, temperature)

        # Enforce legality: zero out illegal indices (defensive)
        probs = probs * root_legal_mask.astype(np.float32)

        # If all zero (could happen if numeric problems), fallback to visit_counts restricted to legal
        if probs.sum() <= 0 or not np.all(np.isfinite(probs)):
            # choose among legal indices according to visit counts
            legal_visits = visit_counts * root_legal_mask.astype(np.float32)
            if legal_visits.sum() > 0:
                # normalize legal_visits to produce probs
                probs = legal_visits.astype(np.float64)
                probs = probs / (probs.sum() + 1e-12)
                probs = probs.astype(np.float32)
            else:
                # extreme fallback (shouldn't happen): uniform over legal moves
                legal_idx = np.where(root_legal_mask)[0]
                if len(legal_idx) == 0:
                    # no legal moves? return uniform over all to avoid crash (shouldn't occur)
                    probs = np.ones(action_space_size, dtype=np.float32) / float(action_space_size)
                else:
                    probs = np.zeros(action_space_size, dtype=np.float32)
                    probs[legal_idx] = 1.0 / len(legal_idx)

        # selected action index: choose argmax over visit_counts but restricted to legal indices
        legal_indices = np.where(root_legal_mask)[0]
        if len(legal_indices) == 0:
            # no legal moves (should be game over), fallback to argmax overall
            selected_idx = int(np.argmax(visit_counts))
        else:
            # argmax among legal_indices
            # if many ties, np.argmax picks first; that's fine
            legal_visit_counts = visit_counts[legal_indices]
            selected_relative = int(np.argmax(legal_visit_counts))
            selected_idx = int(legal_indices[selected_relative])

        selected_action = self.action_indexer.idx_to_action(selected_idx)

        info = {
            "visit_counts": visit_counts,
            "selected_idx": selected_idx,
            "selected_action": selected_action,
            "root_value": root_value,
        }

        return probs, info

    def get_action(self, root_game, temperature: float = 1e-3, add_noise: bool = True) -> Any:
        probs, info = self.run(root_game, temperature=temperature, add_noise=add_noise)

        # defensive checks: probs must be finite and sum to 1
        if not np.all(np.isfinite(probs)) or probs.sum() <= 0:
            # fallback to selected_idx (guaranteed legal by run)
            return self.action_indexer.idx_to_action(info["selected_idx"])

        # choose according to probs (which is already masked & normalized)
        action_idx = int(np.random.choice(len(probs), p=probs))
        return self.action_indexer.idx_to_action(action_idx)