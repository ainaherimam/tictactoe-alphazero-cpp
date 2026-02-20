## ▶️ Run the project on colab via this [Link](https://colab.research.google.com/drive/10f90xLjmt0aWyRmx28Li5slZwutchoro?usp=sharing)

# TicTacToe with AlphaZero on C++20

A project that implements the AlphaZero algorithm for the classic game of Tic-Tac-Toe (4x4) using C++20 and Libtorch.

![img1](./images/tictactoe.png)

---

## 📖 About TicTacToe

Tic-Tac-Toe is a classic two-player strategy game played on a traditional 3x3 grid. On this implementation we choose 4x4 grid.

---

## 🚀 Features
- **Neural Network-Guided MCTS**: MCTS algorithm with neural network policy and value predictions
- **Self-Play Training**: AlphaZero selfplay loop
- **Vizualize Training with html/css/java/**: A web app to see training statistics and games generated
---

## 🛠️ Getting Started

## Prerequisites
* **C++20 Compiler** (GCC 10+, Clang 10+)
* **CMake 3.18+**
* **Python 3.11 or 3.12**
* **NVIDIA Triton Client** (auto-installed by script)

## Installation
```sh
# Clone repository
git clone https://github.com/yourusername/tictactoe-alphazero-cpp.git && cd tictactoe-alphazero-cpp

# install triton, creates venv, builds C++
./setup.sh
```
## Weights and Biases Loggin
Create a .env file the this on your main folder, where the folder src is.

```sh
WANDB_API_KEY=YOUR_API_KEY
```

## Run selfplay and training (run in this order)
```sh
# Launch inference first (one terminal)
./run_inference_server.sh

# Run training second (new terminal)
./run_train.sh

# Run selfplay (new terminal)
./AlphaZero_TTT
```

## Run selfplay and training with evaluations(run in this order)
```sh
# Launch inference first (one terminal)
./run_inference_server.sh

# Run training second (new terminal)
./run_train.sh

# Launch best model inference for evaluation (new terminal)
./run_best_server.sh

# Launch candidate model inference for evaluation (new terminal)
./run_candidate_server.sh

# Run selfplay (new terminal)
./AlphaZero_TTT

# Run eval
./AlphaZero_TTT_Eval
```


## View game data and play against AI

```sh
# Server that will load the lastest checkpoint
./run_triton_server.sh

# Launch a web page where you can play the game
python viewer/game.py
```