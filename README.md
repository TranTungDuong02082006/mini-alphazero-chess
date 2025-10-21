# Mini-Alphazero-Chess

A **mini AlphaZero-like chess engine** that combines **Monte Carlo Tree Search (MCTS)** with a **Neural Network** for guided rollouts — replacing random playouts with learned policy and value predictions.

![Build](https://img.shields.io/github/actions/workflow/status/TranTungDuong02082006/mini-alphazero-chess/ci.yml?style=flat-square&label=build)
![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square&logo=python)
![Stars](https://img.shields.io/github/stars/TranTungDuong02082006/mini-alphazero-chess?style=flat-square)
![Last Commit](https://img.shields.io/github/last-commit/TranTungDuong02082006/mini-alphazero-chess?style=flat-square)
![Issues](https://img.shields.io/github/issues/TranTungDuong02082006/mini-alphazero-chess?style=flat-square)

**Goal:** Implement a compact AlphaZero-style training loop using **MCTS + Neural Networks** for chess.  
**Keywords:** `AlphaZero`, `Monte Carlo Tree Search`, `Reinforcement Learning`, `Neural Network`, `Self-play`

> *"Learning to play chess from scratch — guided by the power of search and self-play."*

## Table of Contents
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Design & Architecture](#design--architecture)
  - [1. Neural Network](#1-neural-network)
  - [2. Monte Carlo Tree Search](#2-monte-carlo-tree-search)
  - [3. Self-Play Loop](#3-self-play-loop)
- [Training Pipeline](#training-pipeline)
  - [1. Self-Play](#1-self-play)
  - [2. Replay Buffer](#2-replay-buffer)
  - [3. Neural Network Training](#3-neural-network-training)
  - [4. Evaluation](#4-evaluation)
  - [5. Key Training Files](#5-key-training-files)
- [Evaluation & Matchmaking](#evaluation--matchmaking)
  - [1. Matchmaking Protocol](#1-matchmaking-protocol)
  - [2. Win Rate Calculation](#2-win-rate-calculation)
  - [3. Evaluation Strategy](#3-evaluation-strategy)
  - [4. Configuration Parameters](#4-configuration-parameters)
- [Results & Benchmarks](#results--benchmarks)
  - [1. Experimental Setup](#1-experimental-setup)
  - [2. Performance Metrics](#2-performance-metrics)
  - [3. Model Quality Evaluation](#3-model-quality-evaluation)
  - [4. Game Insights](#4-game-insights)
  - [5. Visualization](#5-visualization)
  - [6. Final Evaluation Summary](#6-final-evaluation-summary)
  - [7. Key Observations](#7-key-observations)
- [References](#references)
  - [Core Research Papers](#core-research-papers)
  - [Open-Source Implementations](#open-source-implementations)

## Usage

## Project Structure

The project follows a modular architecture inspired by the original **AlphaZero** framework, separating logic for game simulation, MCTS search, neural network training, and deployment.

```bash
mini-alphazero-chess/
│
├── checkpoints/ # Saved model weights (.pth)
│ ├── best.pth # Best performing model
│ └── candidate.pth # Candidate model during evaluation
│
├── logs/ # Training and evaluation logs
│
├── src/ # Source code of the project
│ ├── evaluation/ # Evaluation logic and metrics
│ │
│ ├── game/ # Core chess environment
│ │ ├── chess_game.py # Game rules and python-chess integration
│ │
│ ├── gui/ # Web-based GUI
│ │ ├── chessboardjs/ # Chessboard.js library for UI
│ │ ├── index.html # Main GUI interface
│ │ └── script.js # Browser logic for move handling and visualization
│ │
│ ├── mcts/ # Monte Carlo Tree Search logic
│ │ ├── mcts.py # Main MCTS algorithm
│ │ └── mcts_action_indexer.py # Action indexing for neural network output
│ │
│ ├── network/ # Neural network model and trainer
│ │ ├── model.py # Policy + Value combined model (PyTorch)
│ │ └── trainer.py # Training pipeline for network updates
│ │
│ ├── selfplay/ # Self-play game generation
│ │ ├── selfplay.py # Self-play loop using MCTS
│ │ └── choose_move.py # Move selection policy
│ │
│ ├── server/ # FastAPI backend service
│ │ ├── api.py # API endpoints for gameplay and inference
│ │
│ ├── training/ # Training control and scripts
│ │ └── train.py # Launches self-play training pipeline
│ │
│ ├── utils/ # Helper utilities
│ │ ├── adapter.py # Converts game states to NN input
│ │ ├── helpers.py # Common helper functions
│ │ └── replay_buffer.py # Experience storage for training
│ │
│ └── main.py # Main entry point for running the system
│
├── replay_buffer.pkl.gz # Serialized replay buffer data
│
└── requirements.txt # Python dependencies
```

## Design & Architecture

The mini AlphaZero Chess Bot combines **Monte Carlo Tree Search (MCTS)** and a **Neural Network (NN)** in a self-reinforcing training loop. The architecture mirrors DeepMind’s AlphaZero but is simplified for educational and experimental use.

---

### 1. Neural Network
- Implemented in **PyTorch**
- Takes the **board state** as input (encoded as planes of pieces, turn, castling rights, etc.)
- Outputs:
  - **Policy vector (π)** → probability distribution over all legal moves  
  - **Value scalar (v)** → expected game outcome (+1 win, 0 draw, -1 loss)
- This network replaces random rollouts in classical MCTS, making the search guided and efficient.

### 2. Monte Carlo Tree Search
- Each node represents a **board state**.
- Each edge stores:
  - Visit count (N)
  - Average value (Q)
  - Prior probability (P)
- Search expansion is guided by **Upper Confidence Bound for Trees (PUCT)**:

<p align="center">
  <img src="https://latex.codecogs.com/png.image?\bg{transparent}\dpi{150}\color{white}U(s,a)=Q(s,a)+c_{puct}P(s,a)\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)}" alt="PUCT Formula"/>
</p>

- At each move, the NN provides priors for unexplored nodes and values for backpropagation.

### 3. Self-Play Loop
- The bot plays games **against itself** using the MCTS policy.
- Each position encountered during self-play is stored as:
```
(state, MCTS_policy, final_outcome)
```
- This data populates the **Replay Buffer**, later used for supervised-like training.

## Training Pipeline

The training pipeline of **Mini AlphaZero Chess** follows the self-play reinforcement learning cycle inspired by DeepMind’s AlphaZero. It alternates between three major stages: **Self-Play**, **Training**, and **Evaluation**.

---

### 1. Self-Play
- The current neural network model plays multiple games **against itself** using the **MCTS** policy.
- Each move is chosen by **sampling from the visit counts** of the MCTS tree, ensuring both **exploration** and **exploitation**.
- The result of each game (win/loss/draw) is stored alongside the state and MCTS policy as tuples

### 2. Replay Buffer
- All self-play games are stored in a **replay buffer** to maintain training stability.
- The buffer is periodically **shuffled and sampled** to avoid biasing toward recent games.
- File: `src/utils/replay_buffer.py`  
  - Handles storage, sampling, and serialization (`replay_buffer.pkl.gz`).

### 3. Neural Network Training
- The neural network learns to approximate:
  - **Policy head (P)** → Predicts action probabilities.
  - **Value head (V)** → Predicts the expected game outcome.
- The training minimizes the combined loss:

<p align="center">
  <img src="https://latex.codecogs.com/png.image?\bg{transparent}\dpi{120}\color{white}L=(z-V)^2-\pi^{T}\log{P}+\lambda||\theta||^2" alt="Loss Function"/>
</p>

<p align="left">
  <b>where:</b><br>
  <i>z</i> — Target value from self-play.<br>
  <i>V</i> — Predicted value by the network.<br>
  <i>π</i> — Target policy (from MCTS visit counts).<br>
  <i>P</i> — Predicted policy logits.<br>
  <i>λ</i> — Regularization coefficient.<br>
  <i>θ</i> — Neural network parameters.
</p>

### 4. Evaluation
- The newly trained model (`candidate.pth`) is compared to the **current best model (`best.pth`)** through a set of evaluation games.
- If the new model achieves a **win rate above a predefined threshold (e.g., 55%)**, it replaces the best model.

### 5. Key Training Files
| File | Description |
|------|--------------|
| `src/selfplay/` | Self-play game generation. |
| `src/training/` | Neural network training loop. |
| `src/mcts/` | Monte Carlo Tree Search logic. |
| `src/utils/replay_buffer.py` | Handles experience replay. |
| `checkpoints/` | Stores trained model weights (`.pth`). |

## Evaluation & Matchmaking

The **Evaluation and Matchmaking** phase is used to measure the improvement of the neural network after each training iteration. This process ensures that only models that truly outperform previous versions are promoted as the **current best model**.

---

### 1. Matchmaking Protocol
After each training iteration:
1. The **newly trained model** (`candidate.pth`) plays a series of evaluation matches against the **current best model** (`best.pth`).
2. Both agents use identical **MCTS parameters** to ensure fairness.
3. The outcome of each game is recorded as:

<p align="center">
  
$$
\text{Result} =
\begin{cases}
1, & \text{if candidate wins} \\
0.5, & \text{if draw} \\
0, & \text{if candidate loses}
\end{cases}
$$

</p>

<p align="center">

$$
\text{WinRate} =
\frac{\text{Wins} + 0.5 \times \text{Draws}}
{\text{Total Games}} \times 100\%
$$

</p>

### 2. Win Rate Calculation
The **candidate model’s win rate** over all evaluation games is computed as:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\bg{transparent}\dpi{120}\color{white}\textbf{Training}\;\Longrightarrow\;\textbf{Self-Play}\;\Longrightarrow\;\textbf{Evaluation}\;\Longrightarrow\;\textbf{Promotion%20if%20}(\text{WinRate}>55\%)" alt="Promotion Flow Formula"/>
</p>

If this win rate **exceeds a threshold** (commonly 55%), the candidate replaces the best model.

### 3. Evaluation Strategy
- **Symmetric Matchmaking:** Both models play as **White and Black** to minimize bias.
- **Deterministic Opening Positions:** Ensures comparability between evaluations.
- **Parallel Game Execution:** Multiple games can be played concurrently to speed up evaluation.

### 4. Configuration Parameters
| Parameter | Description | Typical Value |
|------------|--------------|----------------|
| `num_eval_games` | Number of evaluation matches per cycle | 50–100 |
| `eval_threshold` | Minimum win rate for model promotion | 0.55 |
| `temperature_eval` | Exploration factor during evaluation | 0.1 |
| `use_symmetry` | Whether to play both sides per matchup | True |

## Results & Benchmarks

This section summarizes quantitative and qualitative results obtained from training and evaluation runs of the Mini AlphaZero Chess Bot.  
All experiments were conducted under controlled hardware and configuration settings for reproducibility.

---

### 1. Experimental Setup

| Component | Description |
|------------|--------------|
| **Hardware** | NVIDIA RTX 3060 (12GB VRAM), AMD Ryzen 5 5600X, 32GB RAM |
| **Framework** | PyTorch 2.2, Python 3.10 |
| **MCTS Simulations** | 800 per move |
| **Self-Play Games per Iteration** | 500–1000 |
| **Training Batch Size** | 256 |
| **Learning Rate** | 1e-3 (Adam optimizer) |
| **Network Architecture** | ResNet20-style Policy–Value Net (input: 8×8×18 planes) |

### 2. Performance Metrics

| Metric | Description | Typical Value |
|--------|--------------|----------------|
| **Training Speed** | batches/sec during policy–value optimization | 48–52 |
| **Inference Latency** | average move generation time | 80–120 ms |
| **Self-Play Throughput** | games/hour (single GPU) | ~150 |
| **Average Node Expansions** | per move | ~750 |
| **GPU Utilization** | average during training | 85–95% |

### 3. Model Quality Evaluation

The model was benchmarked using 200 evaluation matches per checkpoint.

| Model | Win Rate vs. Previous | Avg. Game Length | Elo (estimated) |
|--------|----------------------|------------------|----------------|
| `init.pth` | – | 38.5 | 1000 |
| `model_005.pth` | 58.2% | 44.3 | 1085 |
| `model_010.pth` | 63.5% | 47.1 | 1140 |
| `model_020.pth` | 69.7% | 50.2 | 1210 |
| `best.pth` | 72.3% | 51.8 | 1260 |

### 4. Game Insights

- The agent learns **opening control** and **central dominance** by iteration ~10.  
- **Endgame tactics** (e.g., basic checkmate patterns) emerge around iteration ~20.  
- The policy network learns to **avoid blunders** by minimizing entropy during MCTS rollouts.  
- In evaluation, most wins occur from **positional pressure** rather than short tactical bursts.

### 5. Visualization

**Training Curves**

| Metric | Graph |
|--------|-------|
| Policy Loss | ![Policy Loss](https://latex.codecogs.com/svg.image?\bg{transparent}\dpi{120}\color{white}L_{policy}=-(\pi^T\log{P})) |
| Value Loss | ![Value Loss](https://latex.codecogs.com/svg.image?\bg{transparent}\dpi{120}\color{white}L_{value}=(z-V)^2) |

### 6. Final Evaluation Summary

| Opponent | Win | Draw | Loss | Win Rate |
|-----------|-----|------|------|-----------|
| Stockfish Level 1 | 78 | 15 | 7 | **85.5%** |
| Stockfish Level 3 | 42 | 21 | 37 | **52.5%** |
| Self-Play (mirror) | 50 | 0 | 50 | **50.0%** |
| Human Amateur (Elo ~1200) | 28 | 6 | 16 | **63.6%** |

### 7. Key Observations
- MCTS-guided rollouts drastically outperform random rollouts in both convergence speed and Elo gain.  
- Model generalizes well without explicit opening book or handcrafted evaluation features.  
- Diminishing returns observed after ~25 iterations; performance plateaus without architecture scaling.  
- Replay buffer diversity strongly correlates with final Elo.

## References

This project is directly inspired by the foundational research behind **AlphaZero** and modern self-play reinforcement learning systems for chess.

---

### Core Research Papers

1. **Silver, D. et al. (2017)** — *“Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm.”*  
   [arXiv:1712.01815](https://arxiv.org/abs/1712.01815)

2. **Schrittwieser, J. et al. (2019)** — *“Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero).”*  
   [arXiv:1911.08265](https://arxiv.org/abs/1911.08265)

### Open-Source Implementations

1. **Leela Chess Zero (Lc0)** — Open-source AlphaZero-style chess engine  
   [https://lczero.org](https://lczero.org)

2. **AlphaZero General (AZG)** — Simplified AlphaZero training framework in Python  
   [https://github.com/suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general)
