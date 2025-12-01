## ▶️ Run the project on colab via this [Link](https://colab.research.google.com/drive/10f90xLjmt0aWyRmx28Li5slZwutchoro?usp=sharing)

# TicTacToe with AlphaZero on C++20

A project that implements the AlphaZero algorithm for the classic game of Tic-Tac-Toe using C++20 and Libtorch.

![img1](./images/tictactoe.png)

---

## 📖 About TicTacToe

Tic-Tac-Toe is a classic two-player strategy game played on a 3x3 grid. 

---

## 🚀 Features
- **TicTacToe Game Logic**: Complete game implementation with move validation and win detection
- **Neural Network-Guided MCTS**: MCTS algorithm with neural network policy and value predictions
- **Self-Play Training**: AlphaZero-style self-play loop
- **Type of gameplay**: AI vs AI / Human vs Human / AI vs Human

---

## 🛠️ Getting Started

## Prerequisites
- **C++20 Compiler** (GCC 10+, Clang 10+, MSVC 2019+)
- **CMake 3.20+**
- **LibTorch** (PyTorch C++ API)

## Installation
```sh
# Clone repository
git clone https://github.com/yourusername/tictactoe-alphazero-cpp.git && cd tictactoe-alphazero-cpp

# Create build directory
mkdir build && cd build

# Configure and build
cmake -DCMAKE_PREFIX_PATH=${LIBTORCH_PATH} ..
make -j$(nproc)

---
