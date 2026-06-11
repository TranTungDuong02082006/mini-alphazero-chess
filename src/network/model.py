import json
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple 2-layer residual block used in AlphaZero-like nets"""

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = F.relu(out)
        return out


class ChessNet(nn.Module):
    """
    AlphaZero-style neural network for chess.
    Input:
        - tensor shaped (B, board_channels=18, 8, 8)
    Output:
        - policy_logits: (B, policy_size) -- raw logits (do not softmax here)
        - value: (B, 1) -- tanh function activated in [-1, 1]
    """

    def __init__(
        self,
        board_channels: int = 18,
        board_size: int = 8,
        num_filters: int = 128,
        num_res_blocks: int = 15,
        policy_size: int = 4672
    ):
        super().__init__()
        self.board_channels = board_channels
        self.board_size = board_size
        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks
        self.policy_size = policy_size

        #Stem convolution
        self.conv = nn.Conv2d(board_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_filters)

        # Residual Tower
        self.res_layers = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_res_blocks)])

        # Policy head: conv 1x1 -> flatten -> fc -> logits
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, policy_size)

        # Value head: conv 1x1 -> flatten -> fc -> fc -> tanh
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

        # Initialize weights(basic)
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights with common strategies."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming normal for conv
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            policy_logits: (B, policy_size) -- raw logits (suitable for CrossEntropyLoss)
            value: (B, 1) -- values in [-1, 1]
        """
        # shared trunk
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)

        for block in self.res_layers:
            x = block(x)

        # policy head
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(p.size(0), -1)   # (B, 2*8*8)
        p = self.policy_fc(p)       # (B, policy_size)
        # Note: return raw logits (no softmax/log_softmax) -> suitable for nn.CrossEntropyLoss

        # value head
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(v.size(0), -1)           # (B, 64)
        v = F.relu(self.value_fc1(v))       # (B, 64)
        v = torch.tanh(self.value_fc2(v))   # (B, 1)

        return p, v


class NeuralNet:
    """
    Wrapper for ChessNet. Provides convenient methods:
    - predict(state) -> (policy_probs, value)
    - batch_predict(states) -> (policies_probs, values)
    - save(path) and load(path)
    - load_from_checkpoint(path) -> classMethod to restore easily
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None, **model_kwargs):
        # device is None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        # store config used to build model
        # fill defaults from ChessNet signature if not provided
        default_config = {
            "board_channels": 18,
            "board_size": 8,
            "num_filters": 128,
            "num_res_blocks": 15,
            "policy_size": 4672,
        }
        self.config: Dict[str, Any] = {**default_config, **model_kwargs}

        self.model = ChessNet(
            board_channels=self.config["board_channels"],
            board_size=self.config["board_size"],
            num_filters=self.config["num_filters"],
            num_res_blocks=self.config["num_res_blocks"],
            policy_size=self.config["policy_size"]
        ).to(self.device)

    def _to_tensor(self, state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Convert state to torch tensor on the correct device.
        Supports: list, numpy array, or torch tensor.
        Shapes allowed:
            (C, H, W)        -> single board
            (B, C, H, W)     -> batch of boards
        """
        if isinstance(state, list):
            state = np.stack(state, axis=0)
            tensor = torch.from_numpy(state).float()
        elif isinstance(state, np.ndarray):
            tensor = torch.from_numpy(state).float()
        elif isinstance(state, torch.Tensor):
            tensor = state.float()
        else:
            raise ValueError("state must be numpy array or torch tensor")

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0) # add batch dim
        elif tensor.dim() != 4:
            raise ValueError(f"Unexpected tensor shape {tensor.shape}, expected 3D or 4D")
        # expected shape (B, C, H, W)
        return tensor.to(self.device)

    def mask_and_normalize(self, logits: Union[np.ndarray, torch.Tensor], mask: Optional[Union[np.ndarray, torch.Tensor]] = None) -> np.ndarray:
        """
        Given logits (policy_size,) or (B, policy_size), optionally apply mask and return normalized probabilities
        mask: boolean or 0/1 array of same last-dim size where valid moves = 1/True
        """
        is_tensor = isinstance(logits, torch.Tensor)
        if not is_tensor:
            logits_t = torch.from_numpy(logits).float()
        else:
            logits_t = logits.clone().float()

        squeeze_single = False
        if logits_t.dim() == 1:
            logits_t = logits_t.unsqueeze(0)
            squeeze_single = True

        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask_t = torch.from_numpy(mask).bool()
            elif isinstance(mask, torch.Tensor):
                mask_t = mask.bool()
            else:
                raise ValueError("mask must be numpy or torch tensor")

            if mask_t.dim() == 1:
                mask_t = mask_t.unsqueeze(0).expand(logits_t.size(0), - 1)

            neg_inf = -1e9
            logits_t = torch.where(mask_t, logits_t, torch.full_like(logits_t, neg_inf))

        probs = F.softmax(logits_t, dim=-1)
        probs_np = probs.cpu().numpy()
        return probs_np[0] if squeeze_single else probs_np

    def predict(
        self,
        state_encoding: Union[np.ndarray, torch.Tensor],
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Predict on a single state.
        Args:
            state_encoding: np.ndarray shape (18, 8, 8) or torch.Tensor (18, 8, 8) or (1, 18, 8, 8)
            return_logits: if True, returns policy_logits (numpy). If False, returns softmax probabilities.

        Returns:
            policy (np.ndarray) shape (policy_size)
            value (float)
        """
        self.model.eval()
        # Normalize input types
        if isinstance(state_encoding, np.ndarray):
            x = torch.tensor(state_encoding, dtype=torch.float32, device=self.device)
        elif isinstance(state_encoding, torch.Tensor):
            x = state_encoding.to(self.device).float()
        else:
            raise ValueError("state_encoding must be np.ndarray or torch.Tensor")

        if x.ndim == 3:
            x = x.unsqueeze(0) # add batch dim

        with torch.no_grad():
            logits, v = self.model(x) # logits: (1, policy_size), v: (1,1)
            logits = logits.cpu().squeeze(0)
            v = v.cpu().squeeze(0).item()

            if return_logits:
                policy_out = logits.numpy()
            else:
                # softmax to probabilities (numerically stable)
                policy_out = F.softmax(logits, dim=0).numpy()

        return policy_out, float(v)

    def batch_predict(
        self,
        batch_states: Union[np.ndarray, torch.Tensor],
        legal_masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict for a batch of states

        Args:
            batch_states: np.ndarray shape (B, 18, 8, 8) or torch tensor
            legal_masks: optional np.ndarray shape (B, policy_size) or (policy_size,)
            return_logits: whether to return raw logits

        Returns:
            policies: np.ndarray shape (B, policy_size)
            values: np.ndarray shape (B,)
        """
        self.model.eval()
        x = self._to_tensor(batch_states) # (B, C, H, W)
        with torch.no_grad():
            logits_t, v_t = self.model(x)

        logits_np = logits_t.cpu().numpy() # (B, P)
        values_np = v_t.squeeze(-1).cpu().numpy() # (B,)

        if return_logits:
            return logits_np, values_np

        # handle masks
        if legal_masks is None:
            probs = F.softmax(torch.from_numpy(logits_np), dim=1).numpy()
            return probs, values_np
        else:
            # broadcast mask if single provided
            if isinstance(legal_masks, np.ndarray) and legal_masks.ndim == 1:
                masks = np.tile(legal_masks, (logits_np.shape[0], 1))
            else:
                masks = np.array(legal_masks)

            probs_list = []
            for row_logits, row_mask in zip(logits_np, masks):
                probs_list.append(self.mask_and_normalize(row_logits, mask=row_mask))
            probs = np.stack(probs_list, axis=0)
            return probs, values_np

    def predict_topk(
        self,
        state_encoding: Union[np.ndarray, torch.Tensor],
        k: int = 10,
        legal_mask: Optional[Union[np.ndarray, np.ndarray]] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Return top-k (indices and probabilities) for a single state.
        """
        probs, value = self.predict(state_encoding, return_logits=False)

        if legal_mask is not None:
            probs = self.mask_and_normalize(probs, legal_mask)

        topk_idx = np.argsort(probs)[-k:][::-1]
        topk_probs = probs[topk_idx]
        return np.stack([topk_idx, topk_probs], axis=1), value

    def save(self, path: str) -> None:
        """
        Save checkpoint containing:
            - model state_dict
            -config (architecture)
        """
        ckpt = {"state_dict": self.model.state_dict(), "config": self.config}
        torch.save(ckpt, path)

    @classmethod
    def from_checkpoint(cls, path: str, device: Optional[Union[str, torch.device]] = None) -> "NeuralNet":
        """
        Load a NeuralNet from a checkpoint saved by save().
        """
        ckpt = torch.load(path, map_location="cuda" if device is None else device)
        config = ckpt.get("config", {})
        wrapper = cls(device=device, **config)
        state = ckpt.get("state_dict", ckpt)
        wrapper.model.load_state_dict(state)
        wrapper.model.to(wrapper.device)
        return wrapper

    def load(self, path:str) -> None:
        """ Load weights into an existing instance (net must have compatible architecture). """
        ckpt = torch.load(path, map_location=self.device)
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

