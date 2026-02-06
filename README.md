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

---

## 🛠️ Getting Started

## Prerequisites
- **C++20 Compiler** (GCC 10+, Clang 10+, MSVC 2019+)
- **CMake 3.20+**

## Installation
```sh
# Clone repository
git clone https://github.com/yourusername/tictactoe-alphazero-cpp.git && cd tictactoe-alphazero-cpp

# Create build directory
mkdir build && cd build

# Configure and build
cmake - ..
make -j$(nproc)

# 1. Create a virtual environment (if you don’t have one yet)
python -m venv venv

# 2. Activate it
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate    # Windows

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run inference server
python inference_server.py

# 6. Run selfplay
python ./AlphaZero_TTT
---
