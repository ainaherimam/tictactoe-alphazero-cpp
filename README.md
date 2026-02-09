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
* **CMake 3.15+**
* **Python 3.11 or 3.12**
* **NVIDIA Triton Client** (auto-installed by script)

## Installation
```sh
# Clone repository
git clone https://github.com/yourusername/tictactoe-alphazero-cpp.git && cd tictactoe-alphazero-cpp

# install triton, creates venv, builds C++
./install_and_setup.sh

# Run inference server
./run_triton.sh 
or 
./run_inference_server.sh

# Run selfplay
./AZ_Triton_TTT
or 
./AlphaZero_TTT

# Run training
./run_train.sh
---

