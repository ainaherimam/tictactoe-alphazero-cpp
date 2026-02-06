#ifndef BOARD_H
#define BOARD_H

#include <array>
#include <iostream>
#include <vector>
#include "cell_state.h"

/**
 * @struct Move
 * @brief Represents a move on the board with position coordinates.
 */
struct Move {
    int x;   ///< Row coordinate (0-indexed)
    int y;   ///< Column coordinate (0-indexed)
    int dir; ///< Direction indicator (currently unused, set to -1)
    int tar; ///< Target indicator (currently unused, set to -1)

    /**
     * @brief Converts the move to a string format.
     * @return String in format "(rowColumn)", e.g., "(1A)"
     */
    std::string to_string() const {
        char column = 'A' + y;
        int row = x + 1;
        return "(" + std::to_string(row) + column + ")";
    }

    /**
     * @brief Subscript operator for array-like access.
     */
    const int& operator[](size_t i) const {
        return const_cast<Move&>(*this)[i];
    }
};

/**
 * @class Board
 * @brief Represents a Tic-Tac-Toe game board (default to a 4x4 grid)
 * 
 * This class manages the game state, validates moves, and determines winners
 * according to rules (can switch between Misere and Normal TTT).
 */
class Board {
public:
    /**
     * @brief Constructs a new Board with the specified size.
     * @param board_size Size of the board (default: 4 for 4x4 grid)
     */
    Board(int board_size = 4);
    
    /**
     * @brief Gets the size of the board.
     * @return The board size
     */
    int get_board_size() const;
    
    /**
     * @brief Checks if a move is within the board boundaries.
     * @param move The move to validate
     */
    bool is_within_bounds(Move move) const;
    
    /**
     * @brief Prints valid moves to standard output.
     * @param moves Vector of moves to print
     */
    void print_valid_moves(std::vector<Move> moves) const;
    
    /**
     * @brief Gets all valid moves for the given player.
     * @param player The player whose valid moves to find
     * @return Vector of all valid moves (empty cells)
     */
    std::vector<Move> get_valid_moves(Cell_state player) const;
    
    /**
     * @brief Executes a move on the board for the given player.
     * @param move The move to make
     * @param player The player making the move
     */
    void make_move(Move move, Cell_state player);
    
    /**
     * @brief Converts the board state to a float array for data collection.
     * @param player The current player
     * @param output Pointer to pre-allocated float array of size [3 * 4 * 4]
     *               Plane 0: Current player's pieces
     *               Plane 1: Opponent's pieces
     *               Plane 2: Current player indicator (0 for X, 1 for O)
     */
    void to_float_array(Cell_state player, float* output) const;
    
    /**
     * @brief Checks for a winner based on rules.
     * @return The winning player, or Empty if no winner
     */
    Cell_state check_winner() const;
    
    /**
     * @brief Generates a legal move mask for data collection.
     * @param player The player to generate the mask for
     * @param output Pointer to pre-allocated float array of size [4 * 4 * 1 * 1]
     *               1.0 indicates a legal move, 0.0 indicates illegal
     */
    void get_legal_mask(Cell_state player, float* output) const;
    
    /**
     * @brief Displays the board to the specified output stream.
     * @param os The output stream to write to
     */
    void display_board(std::ostream& os) const;
    
    /**
     * @brief Stream insertion operator for Board.
     * @param os Output stream
     * @param board Board to display
     * @return Reference to the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const Board& board);

private:
    const int board_size;                                    ///< Size of the board (rows and columns)
    std::vector<std::vector<Cell_state>> empty_board;        ///< Template for an empty board
    std::vector<std::vector<std::vector<Cell_state>>> history; ///< Game history (4 boards states for now)
    std::vector<std::vector<Cell_state>> board;              ///< Current board state

    /**
     * @brief Adds the current board state to the game history.
     */
    void add_history();
    
    /**
     * @brief Helper to check if a position exists in a vector of coordinates.
     * @param x Row 
     * @param y Column
     * @param vector Vector of pairs (x,y) to search
     * @return true if position is found, false otherwise
     */
    bool is_in_vector(int x, int y, const std::vector<std::array<int, 2>>& vector) const;
};

#endif