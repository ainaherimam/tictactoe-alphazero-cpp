#include "board.h"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <vector>
#include "constants.h"


Board::Board(int board_size)
    : board_size(board_size), 
      empty_board(board_size, std::vector<Cell_state>(board_size, Cell_state::Empty)),
    history(board_size, std::vector<std::vector<Cell_state>>(board_size, std::vector<Cell_state>(board_size, Cell_state::Empty))),
    board(board_size, std::vector<Cell_state>(board_size, Cell_state::Empty)) {

        // board[0][0] = Cell_state::X;
        // board[3][3] = Cell_state::O;
        // board[2][1] = Cell_state::O;
        // board[3][3] = Cell_state::X;
        // board[0][0] = Cell_state::O;
        
}

int Board::get_board_size() const { return board_size; }

bool Board::is_within_bounds(Move move) const {
    return move.x >= 0 && move.y < board_size && move.y >= 0 && move.x < board_size;
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

void Board::print_valid_moves(std::vector<Move> moves) const {
    int index = 1;
    for (const auto& move : moves) {
        char column = 'A' + move.y;
        int row = move.x + 1;
        
        std::cout << index << " - (" << row << column << ")\n";
        index++;
    }
}


std::vector<Move> Board::get_valid_moves(Cell_state player) const {
    
    std::vector<Move> valid_moves;
    
    for (int x = 0; x < board_size; ++x) {
        for (int y = 0; y < board_size; ++y) {
            // Empty cells means valid move
            if (board[x][y] == Cell_state::Empty) {
                valid_moves.push_back(Move{.x = x,.y = y,.dir = -1,.tar = -1});
            }
        }
    }
    
    return valid_moves;
}

void Board::make_move(Move move, Cell_state player) {
    add_history();
    
    if (is_within_bounds(move) && board[move.x][move.y] == Cell_state::Empty) {
        board[move.x][move.y] = player;
    }
}

void Board::to_float_array(Cell_state player, float* output) const {
    // Initialize to zero
    for (int i = 0; i < 3 * X_ * Y_; ++i) {
        output[i] = 0.0f;
    }
    
    for (int x = 0; x < board_size; ++x) {
        for (int y = 0; y < board_size; ++y) {
            int plane0_idx = 0 * 16 + x * 4 + y;
            int plane1_idx = 1 * 16 + x * 4 + y;
            int plane2_idx = 2 * 16 + x * 4 + y;
            
            // Plane 0: Current player's pieces
            if (board[x][y] == player) {
                output[plane0_idx] = 1.0f;
            }
            // Plane 1: Opponent's pieces
            else if (board[x][y] != Cell_state::Empty) {
                output[plane1_idx] = 1.0f;
            }
            // Plane 2: Current player indicator (0 for X, 1 for O)
            output[plane2_idx] = (player == Cell_state::X) ? 0.0f : 1.0f;
        }
    }
}


Cell_state Board::check_winner() const {
    // lambda function to get the opponent
    auto opponent = [](Cell_state player) -> Cell_state {
        if (player == Cell_state::X) return Cell_state::O;
        if (player == Cell_state::O) return Cell_state::X;
        return Cell_state::Empty;
    };

    // Check horizontal 3-in-a-rows
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (board[i][j] != Cell_state::Empty &&
                board[i][j] == board[i][j+1] &&
                board[i][j+1] == board[i][j+2]) {
                return opponent(board[i][j]);
            }
        }
    }
    
    // Check vertical 3-in-a-rows
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 2; ++i) {
            if (board[i][j] != Cell_state::Empty &&
                board[i][j] == board[i+1][j] &&
                board[i+1][j] == board[i+2][j]) {
                return opponent(board[i][j]);
            }
        }
    }
    
    // Check diagonal 3-in-a-rows
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (board[i][j] != Cell_state::Empty &&
                board[i][j] == board[i+1][j+1] &&
                board[i+1][j+1] == board[i+2][j+2]) {
                return opponent(board[i][j]);
            }
        }
    }
    
    // Check anti-diagonal 3-in-a-rows
    for (int i = 0; i < 2; ++i) {
        for (int j = 2; j < 4; ++j) {
            if (board[i][j] != Cell_state::Empty &&
                board[i][j] == board[i+1][j-1] &&
                board[i+1][j-1] == board[i+2][j-2]) {
                return opponent(board[i][j]);
            }
        }
    }
    
    return Cell_state::Empty;
}

void Board::get_legal_mask(Cell_state player, float* output) const {

    const int total_size = X_ * Y_ * DIR_ * TAR_;

    // Initialize to zero
    for (int i = 0; i < total_size; ++i) {
        output[i] = 0.0f;
    }

    // Helper: convert (x,y) → flat index
    auto index = [&](int x, int y, int dir, int tar) -> int {
        return x * Y_ + y;
    };
    
    auto valid_moves = get_valid_moves(player);
    for (const auto& move : valid_moves) {
        int idx = index(move.x, move.y, move.dir, move.tar);
        
        if (idx >= 0 && idx < total_size)
            output[idx] = 1.0f;
    }
}


void Board::display_board(std::ostream& os) const {

    os << "    ";
    for (int c = 0; c < X_; ++c)
        os << static_cast<char>('A' + c) << "   ";
    os << "\n";

    for (int r = 0; r < X_; ++r) {
        os << (r + 1) << "   ";
        for (int c = 0; c < Y_; ++c) {
            os << board[r][c];
            if (c < Y_ - 1)
                os << " | ";
        }
        os << "\n";

        if (r < Y_ - 1) {
            os << "   ───┼───┼───┼───\n";
        }
    }
    std::cout << "\n";
}

std::ostream& operator<<(std::ostream& os, const Board& board) {
    board.display_board(os);
    return os;
}