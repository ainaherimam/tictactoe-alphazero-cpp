#include "board.h"
#include <torch/torch.h>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "iterator"


Board::Board(int size)
    : board_size(4),
    empty_board(4, std::vector<Cell_state>(4, Cell_state::Empty)),
    history(4, std::vector<std::vector<Cell_state>>(4, std::vector<Cell_state>(4, Cell_state::Empty))),
    board(4, std::vector<Cell_state>(4, Cell_state::Empty)) {

        // board[2][1] = Cell_state::O;
        // board[2][3] = Cell_state::X;
        // board[2][1] = Cell_state::O;
        // board[3][3] = Cell_state::X;
        // board[0][0] = Cell_state::O;
        
}

int Board::get_board_size() const { return board_size; }

bool Board::is_within_bounds(int move_x, int move_y) const {
    return move_x >= 0 && move_x < 4 && move_y >= 0 && move_y < 4;
}

void Board::add_history() {
    if (history.size() == 4) {
        history.erase(history.begin());
    }
    history.push_back(board);
}

bool Board::is_in_vector(int x, int y, const std::vector<std::array<int, 2>>& vector) const {
    return std::find(vector.begin(), vector.end(), std::array<int, 2>{x, y}) != vector.end();
}

void Board::print_valid_moves(std::vector<std::array<int, 4>> moves) const {
    int index = 1;
    for (const auto& move : moves) {
        char column = 'A' + move[1];
        int row = move[0] + 1;
        
        std::cout << index << " - (" << row << column << ")\n";
        index++;
    }
}


std::vector<std::array<int, 4>> Board::get_valid_moves(Cell_state player) const {
    std::vector<std::array<int, 4>> valid_moves;
    

    for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 4; ++y) {
            // If the cell is empty, it's a valid move
            if (board[x][y] == Cell_state::Empty) {
                valid_moves.emplace_back(std::array<int, 4>{x, y, 0, 0});
            }
        }
    }
    
    return valid_moves;
}

void Board::make_move(int move_x, int move_y, int dir, int tar, Cell_state player) {
    add_history();
    
    if (is_within_bounds(move_x, move_y) && board[move_x][move_y] == Cell_state::Empty) {
        board[move_x][move_y] = player;
    }
}


torch::Tensor Board::to_tensor(Cell_state player) const {
    torch::Tensor stacked = torch::zeros({3, 4, 4}, torch::kFloat32);
    
    for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 4; ++y) {
            // Plane 0: Current player's pieces
            if (board[x][y] == player) {
                stacked[0][x][y] = 1.0f;
            }
            // Plane 1: Opponent's pieces
            else if (board[x][y] != Cell_state::Empty) {
                stacked[1][x][y] = 1.0f;
            }
            // Plane 2: Current player indicator (0 for X, 1 for O)
            stacked[2][x][y] = (player == Cell_state::X) ? 0.0f : 1.0f;
        }
    }
    
    return stacked;
}

Cell_state Board::check_winner() const {
    // Helper lambda to get the opponent (for Misère rules if kept)
    auto opponent = [](Cell_state player) -> Cell_state {
        if (player == Cell_state::X) return Cell_state::O;
        if (player == Cell_state::O) return Cell_state::X;
        return Cell_state::Empty;
    };

    // Check horizontal 3-in-a-row (each row has 2 possible sequences)
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) { // j can be 0 or 1 (starting columns)
            if (board[i][j] != Cell_state::Empty &&
                board[i][j] == board[i][j+1] &&
                board[i][j+1] == board[i][j+2]) {
                return opponent(board[i][j]);
            }
        }
    }
    
    // Check vertical 3-in-a-row (each column has 2 possible sequences)
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 2; ++i) { // i can be 0 or 1 (starting rows)
            if (board[i][j] != Cell_state::Empty &&
                board[i][j] == board[i+1][j] &&
                board[i+1][j] == board[i+2][j]) {
                return opponent(board[i][j]);
            }
        }
    }
    
    // Check diagonal 3-in-a-row (top-left to bottom-right direction)
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (board[i][j] != Cell_state::Empty &&
                board[i][j] == board[i+1][j+1] &&
                board[i+1][j+1] == board[i+2][j+2]) {
                return opponent(board[i][j]);
            }
        }
    }
    
    // Check anti-diagonal 3-in-a-row (top-right to bottom-left direction)
    for (int i = 0; i < 2; ++i) {
        for (int j = 2; j < 4; ++j) {
            if (board[i][j] != Cell_state::Empty &&
                board[i][j] == board[i+1][j-1] &&
                board[i+1][j-1] == board[i+2][j-2]) {
                return opponent(board[i][j]);
            }
        }
    }
    
    return Cell_state::Empty; // No winning line yet
}

// Cell_state Board::check_winner() const {

//     // Check horizontal 4-in-a-row
//     for (int i = 0; i < 4; ++i) {
//         if (board[i][0] != Cell_state::Empty &&
//             board[i][0] == board[i][1] &&
//             board[i][1] == board[i][2] &&
//             board[i][2] == board[i][3]) {
//             return board[i][0];
//         }
//     }

//     // Check vertical 4-in-a-row
//     for (int j = 0; j < 4; ++j) {
//         if (board[0][j] != Cell_state::Empty &&
//             board[0][j] == board[1][j] &&
//             board[1][j] == board[2][j] &&
//             board[2][j] == board[3][j]) {
//             return board[0][j];
//         }
//     }

//     // Check main diagonal (top-left → bottom-right)
//     if (board[0][0] != Cell_state::Empty &&
//         board[0][0] == board[1][1] &&
//         board[1][1] == board[2][2] &&
//         board[2][2] == board[3][3]) {
//         return board[0][0];
//     }

//     // Check anti-diagonal (top-right → bottom-left)
//     if (board[0][3] != Cell_state::Empty &&
//         board[0][3] == board[1][2] &&
//         board[1][2] == board[2][1] &&
//         board[2][1] == board[3][0]) {
//         return board[0][3];
//     }

//     return Cell_state::Empty; // No winner
// }



torch::Tensor Board::get_legal_mask(Cell_state player) const {
    const int X = 4;
    const int Y = 4;
    const int DIR = 1; 
    const int TAR = 1; 
    const int total_size = X * Y * DIR * TAR;

    torch::Tensor all_moves = torch::zeros({total_size}, torch::kFloat32);

    // Helper: convert (x,y) → flat index
    auto index = [&](int x, int y, int dir, int tar) -> int {
        return x * Y + y;
    };
    
    auto valid_moves = get_valid_moves(player);
    for (const auto& move : valid_moves) {
        int idx = index(move[0], move[1], move[2], move[3]);
        
        if (idx >= 0 && idx < total_size)
            all_moves[idx] = 1.0f;
    }

    return all_moves;
}

void Board::display_board(std::ostream& os) const {
    const int ROWS = 4;
    const int COLS = 4;

    os << "    ";
    for (int c = 0; c < COLS; ++c)
        os << static_cast<char>('A' + c) << "   ";
    os << "\n";

    for (int r = 0; r < ROWS; ++r) {
        os << (r + 1) << "   ";
        for (int c = 0; c < COLS; ++c) {
            os << board[r][c];
            if (c < COLS - 1)
                os << " | ";
        }
        os << "\n";

        if (r < ROWS - 1) {
            os << "   ───┼───┼───┼───\n";
        }
    }
    std::cout << "\n";
}

std::ostream& operator<<(std::ostream& os, const Board& board) {
    board.display_board(os);
    return os;
}