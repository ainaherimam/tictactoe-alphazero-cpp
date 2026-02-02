// constants.h
#pragma once
#include <vector>


//BOARDS
constexpr int BOARD_SIZE = 4;
constexpr int BOARD_CELLS = BOARD_SIZE * BOARD_SIZE;
constexpr int BOARD_HEIGHT = 4;
constexpr int BOARD_WIDTH = 4;
constexpr int INPUT_PLANES = 3;
constexpr size_t INPUT_CHANNELS = INPUT_PLANES;
constexpr int X_ = BOARD_HEIGHT;
constexpr int Y_ = BOARD_WIDTH;
constexpr int DIR_ = 1;
constexpr int TAR_ = 1;
constexpr int INPUT_SIZE = INPUT_PLANES * BOARD_HEIGHT * BOARD_WIDTH;
constexpr int POLICY_SIZE = BOARD_CELLS;


//MCTS 
constexpr float DIRICHLET_ALPHA = 0.4f;
constexpr float EXPLORATION_FRACTION = 0.25f;
constexpr size_t MAX_GAME_MOVES = 20; 


//SELFPLAY CONFIG
constexpr size_t POOL_CAPACITY = 100;