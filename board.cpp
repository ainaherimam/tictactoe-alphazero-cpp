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
    : board_size(3),
    empty_board(3, std::vector<Cell_state>(3, Cell_state::Empty)),
    history(4, std::vector<std::vector<Cell_state>>(3, std::vector<Cell_state>(3, Cell_state::Empty))),
    board(3, std::vector<Cell_state>(3, Cell_state::Empty)) {
}

int Board::get_board_size() const { return board_size; }

bool Board::is_within_bounds(int move_x, int move_y) const {
    return move_x >= 0 && move_x < 3 && move_y >= 0 && move_y < 3;
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
    

    for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 3; ++y) {
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
    torch::Tensor stacked = torch::zeros({11, 3, 3}, torch::kFloat32);

    // Helper lambda to fill a plane for a board
    auto fill_planes = [](torch::Tensor& tensor, int start_plane, 
                        const std::vector<std::vector<Cell_state>>& b,
                        Cell_state current_player) 
    {
        for (int x = 0; x < 3; ++x) {
            for (int y = 0; y < 3; ++y) {
                if (b[x][y] == current_player) {
                    tensor[start_plane][x][y] = 1.0f;
                } else if (b[x][y] != Cell_state::Empty) {
                    tensor[start_plane + 1][x][y] = 1.0f;
                }
                // Current player plane
                float fill_value = (current_player == Cell_state::X) ? 0.0f : 1.0f;
                tensor[10][x][y] = fill_value;
            }
        }
    };

    // Start with current board
    fill_planes(stacked, 0, board, player);

    // Fill history boards in reverse order: T-1, T-2, ...
    for (size_t i = 0; i < 4; ++i) {
        int plane_index = (i + 1) * 2;
        const auto& hist_board = history[3 - i];
        fill_planes(stacked, plane_index, hist_board, player);
    }

    return stacked; 
}


Cell_state Board::check_winner() const {
    // Check rows
    for (int i = 0; i < 3; ++i) {
        if (board[i][0] != Cell_state::Empty &&
            board[i][0] == board[i][1] &&
            board[i][1] == board[i][2]) {
            return board[i][0];
        }
    }
    
    // Check columns
    for (int j = 0; j < 3; ++j) {
        if (board[0][j] != Cell_state::Empty &&
            board[0][j] == board[1][j] &&
            board[1][j] == board[2][j]) {
            return board[0][j];
        }
    }
    
    // Check diagonal
    if (board[0][0] != Cell_state::Empty &&
        board[0][0] == board[1][1] &&
        board[1][1] == board[2][2]) {
        return board[0][0];
    }
    
    if (board[0][2] != Cell_state::Empty &&
        board[0][2] == board[1][1] &&
        board[1][1] == board[2][0]) {
        return board[0][2];
    }
    
    return Cell_state::Empty;
}

torch::Tensor Board::get_legal_mask(Cell_state player) const {
    const int X = 3;
    const int Y = 3;
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
    const int ROWS = 3;
    const int COLS = 3;

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
            os << "   ───┼───┼───\n";
        }
    }
    std::cout << "\n";
}

std::ostream& operator<<(std::ostream& os, const Board& board) {
    board.display_board(os);
    return os;
}