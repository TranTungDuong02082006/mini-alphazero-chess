"""
Replay buffer for AlphaZero-style self-play.

Stores tuples (state, policy, value) where:
- state: np.ndarray (board encoding from ChessGame.encode_state())
- policy: np.ndarray (probability vector over action space, shape (action_space_size,))
- value: float (game outcome from perspective of player at that state, in [-1, 1])

Features:
- fixed max size (circular)
- thread-safe add/sample (simple Lock)
- sample_batch can return numpy arrays or torch tensors
- save / load buffer to disk (pickle, with optional gzip)
"""

from collections import deque
from typing import Deque, List, Tuple, Optional, Iterable
import os
import pickle
import gzip
import threading

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


Sample = Tuple[np.ndarray, np.ndarray, float]  # (state, policy, value)


class ReplayBuffer:
    def __init__(self, max_size: int = 1e6, seed: Optional[int] = None):
        self.max_size = int(max_size)
        self.buffer: Deque[Sample] = deque(maxlen=self.max_size)
        self.lock = threading.Lock()
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        self._state_shape: Optional[Tuple[int, ...]] = None
        self._policy_shape: Optional[Tuple[int, ...]] = None

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, state: np.ndarray, policy: np.ndarray, value: float) -> None:
        state_a = np.asarray(state)
        policy_a = np.asarray(policy, dtype=np.float32)
        value_f = float(value)

        with self.lock:
            if self._state_shape is None:
                self._state_shape = state_a.shape
            if self._policy_shape is None:
                self._policy_shape = policy_a.shape
            self.buffer.append((state_a, policy_a, value_f))

    def extend(self, samples: Iterable[Sample]) -> None:
        samples = [(np.asarray(s),
                    np.asarray(p, dtype=np.float32),
                    float(v)) for s, p, v in samples]

        with self.lock:
            for s, p, v in samples:
                if self._state_shape is None:
                    self._state_shape = s.shape
                if self._policy_shape is None:
                    self._policy_shape = p.shape
                self.buffer.append((s, p, v))

    def clear(self) -> None:
        with self.lock:
            self.buffer.clear()
            self._state_shape = None
            self._policy_shape = None

    def sample(self, batch_size: int, replace: bool = False) -> List[Sample]:
        """
        Sample raw items from the buffer.

        If buffer size < batch_size and replace=False, will automatically sample with replacement.
        """
        with self.lock:
            size = len(self.buffer)
            if size == 0:
                raise ValueError("ReplayBuffer is empty.")
            if batch_size <= 0:
                raise ValueError("batch_size must be positive.")

            if size < batch_size and not replace:
                # fallback to sampling with replacement to satisfy batch size
                replace = True

            indices = self.rng.choice(size, size=batch_size, replace=replace)
            samples = [self.buffer[i] for i in indices]
        return samples

    def sample_batch(
        self,
        batch_size: int,
        replace: bool = False,
        to_torch: bool = False,
        device: Optional[str] = None,
        dtype=np.float32,
    ):
        """
        Sample a batch and return stacked arrays.

        Returns:
            states: np.ndarray or torch.Tensor, shape (batch, *state_shape)
            policies: np.ndarray or torch.Tensor, shape (batch, *policy_shape)
            values: np.ndarray or torch.Tensor, shape (batch, 1)
        """
        samples = self.sample(batch_size=batch_size, replace=replace)
        states = [s for s, _, _ in samples]
        policies = [p for _, p, _ in samples]
        values = [v for _, _, v in samples]

        try:
            states_arr = np.stack(states)  # (batch, ...)
        except Exception:
            # fallback: convert to object ndarray (less efficient)
            states_arr = np.array(states, dtype=object)

        policies_arr = np.stack(policies).astype(np.float32)  # (batch, action_space)
        values_arr = np.asarray(values, dtype=np.float32).reshape(-1, 1)  # (batch,1)

        if to_torch:
            if not _HAS_TORCH:
                raise ImportError("torch is not available but to_torch=True was requested.")
            device_t = torch.device(device) if device is not None else torch.device("cpu")
            states_t = torch.tensor(states_arr, dtype=_np2torch_dtype(dtype), device=device_t)
            policies_t = torch.tensor(policies_arr, dtype=torch.float32, device=device_t)
            values_t = torch.tensor(values_arr, dtype=torch.float32, device=device_t)
            return states_t, policies_t, values_t

        return states_arr, policies_arr, values_arr

    def sample_all(self):
        """
        Return all items in the buffer stacked as arrays.
        """
        with self.lock:
            if len(self.buffer) == 0:
                return np.array([]), np.array([]), np.array([])
            states = [s for s, _, _ in self.buffer]
            policies = [p for _, p, _ in self.buffer]
            values = [v for _, _, v in self.buffer]

        states_arr = np.stack(states)
        policies_arr = np.stack(policies).astype(np.float32)
        values_arr = np.asarray(values, dtype=np.float32).reshape(-1, 1)
        return states_arr, policies_arr, values_arr

    def save(self, path: str, use_gzip: bool = True) -> None:
        """
        Save buffer to disk (pickle).
        If use_gzip=True or path endswith '.gz', save gzipped.
        """
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        to_save = list(self.buffer)

        if use_gzip or path.endswith(".gz"):
            with gzip.open(path, "wb") as f:
                pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, "wb") as f:
                pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str, max_size: int = 1e6):
        """
        Load buffer from disk, return a ReplayBuffer instance.
        Supports both .pkl and .pkl.gz.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        # đọc file pkl hoặc pkl.gz
        if path.endswith(".gz"):
            with gzip.open(path, "rb") as f:
                data = pickle.load(f)
        else:
            with open(path, "rb") as f:
                data = pickle.load(f)

        if not isinstance(data, list):
            raise ValueError("Loaded data must be a list of (state, policy, value) tuples")

        buffer = cls(max_size=max_size)
        with buffer.lock:
            if len(data) > buffer.max_size:
                data = data[-buffer.max_size:]  # chỉ giữ lại mới nhất
            buffer.buffer = deque(data, maxlen=buffer.max_size)

            # update shapes nếu buffer không rỗng
            if len(buffer.buffer) > 0:
                s0, p0, _ = buffer.buffer[0]
                buffer._state_shape = np.asarray(s0).shape
                buffer._policy_shape = np.asarray(p0).shape
        
        return buffer

    def info(self) -> dict:
        """Return simple statistics about the buffer."""
        with self.lock:
            return {
                "size": len(self.buffer),
                "max_size": self.max_size,
                "state_shape": self._state_shape,
                "policy_shape": self._policy_shape,
            }


def _np_dtype_to_torch(dtype):
    """Map numpy dtype to torch dtype (limited)."""
    if not _HAS_TORCH:
        raise ImportError("torch not available")
    import torch
    if dtype == np.float32:
        return torch.float32
    if dtype == np.float64:
        return torch.float64
    if dtype == np.int64:
        return torch.int64
    return torch.float32


def _np2torch_dtype(dtype):
    # helper to convert numpy dtype or dtype-like to torch dtype
    if isinstance(dtype, str):
        # allow 'float32' etc.
        if "64" in dtype:
            return torch.float64
        return torch.float32
    if dtype == np.float64:
        return torch.float64
    return torch.float32


if __name__ == "__main__":
    # quick demo
    print("ReplayBuffer demo")
    rb = ReplayBuffer(max_size=100, seed=42)

    # create fake state (e.g., flattened 18x8x8 = 1152) and fake policy (4096)
    for i in range(10):
        s = np.random.randn(18, 8, 8).astype(np.float32)
        p = np.random.rand(4096).astype(np.float32)
        p /= p.sum()
        v = float(np.random.choice([-1.0, 0.0, 1.0]))
        rb.add(s, p, v)

    print("Info:", rb.info())
    states, policies, values = rb.sample_batch(4)
    print("States.shape:", states.shape)
    print("Policies.shape:", policies.shape)
    print("Values.shape:", values.shape)

    if _HAS_TORCH:
        st, pl, vl = rb.sample_batch(4, to_torch=True, device="cpu")
        print("Torch shapes:", st.shape, pl.shape, vl.shape)