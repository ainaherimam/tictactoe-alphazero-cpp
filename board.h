#ifndef BOARD_H
#define BOARD_H

#include <array>
#include <string>
#include <utility>
#include <vector>

#include "cell_state.h"
#include <torch/torch.h>

/**
 * @brief Tic Tac Toe board game implementation
 *
 *
 * The Board class provides functionality for managing and interacting with the
 * game board, including:
 *   - Initializing the board
 *   - Displaying the board in the console
 *   - Making a move on the board
 *   - Checking if there is a winner
 *
 * The board is represented internally as a 2D vector of Cell_state. The
 * Cell_state enum represents the state of a cell on the board (empty,X:occupied
 * by player 1,  or O: occupied by player 2).
 */
class Board {

public:
    /**
     * @brief Constructor for Board class
     *
     * @param size Integer to set the size of the board
     */
    Board(int size);

    /**
     * @brief Clear paths and all restricted moves from previous turn
     */
    void clear_state() {
        path.clear();
        restricted_move = {-1, -1};
    }

    /**
     * @brief Getter for the size of the board
     *
     * @return The size of the board
     */
    int get_board_size() const;

    /**
     * @brief Checks if a given cell is within the bounds of the board
     *
     * @param move_x The x-coordinate of the cell
     * @param move_y The y-coordinate of the cell
     *
     * @return True if the cell is within the bounds of the board, false otherwise
     */
    bool is_within_bounds(int move_x, int move_y) const;

    /**
     * @brief Print out a given vector of valid moves
     *
     * @param moves A vector of valid moves
     */
    void print_valid_moves(std::vector<std::array<int, 4>> moves) const;

    /**
     * @brief Check if (x,y) is in the given vector
     *
     * @param x The x-coordinate of the cell
     * @param y The y-coordinate of the cell
     * @param vector A vector to search in
     *
     * @return True if (x,y) is in the vector, false otherwise
     */
    bool is_in_vector(int x, int y, const std::vector<std::array<int, 2>>& vector) const;

    /**
     * @brief Get all valid moves available on the board for the given player
     *
     * @param player The current player
     *
     * @return A vector containing all valid moves
     */
    std::vector<std::array<int, 4>> get_valid_moves(Cell_state player) const;

    /**
     * @brief Place a piece on the board
     *
     * Updates the board state based on the specified move.
     *
     * @param move_x The x-coordinate of the cell
     * @param move_y The y-coordinate of the cell
     * @param dir The direction of the move (not used on TTT)
     * @param tar The chosen target (not used on TTT)
     * @param player The current player
     */
    void make_move(int move_x, int move_y, int dir, int tar, Cell_state player);

    /**
     * @brief Checks the winner
     *
     * @return The cell state of the winner
     */
    Cell_state check_winner() const;


    /**
     * @brief Converts the current board state to a tensor representation for data collection
     *
     * @param player The current player
     *
     * @return Tensor representation of the board state
     */
    torch::Tensor to_tensor(Cell_state player) const;

    /**
     * @brief Fill a given tensor with the tensor represenatation of the board for data collection
     *
     * @param player The current player
     */
    void fill_tensor(torch::Tensor& tensor, Cell_state player) const;

    /**
     * @brief Fill a given tensor with the mask of all legal and illegal moves of the board for data collection
     *
     * @param player The current player
     *
     * @return Tensor representation of the board state
     */
    void fill_mask(torch::Tensor& mask, Cell_state player) const;

    /**
     * @brief Adds the current board state to history
     */
    void add_history();

    /**
     * @brief Gets the legal move mask for the given player (useful for NN)
     *
     * @param player The current player
     *
     * @return Tensor representing the legal move mask (useful for NN)
     */
    torch::Tensor get_legal_mask(Cell_state player) const;

    /**
     * @brief Outputs the current state of the board to an output stream
     *
     * @param os The output stream to which the board state is to be printed
     */
    void display_board(std::ostream& os) const;

    /**
     * @brief Overloads the << operator for the Board class
     *
     * This function allows the board to be directly printed to an output stream
     * (such as std::cout) by calling the display_board() method of the Board class.
     *
     * @param os The output stream to which the board state is to be printed
     * @param board The board to be printed
     *
     * @return The output stream with the board state appended
     */
    friend std::ostream& operator<<(std::ostream& os, const Board& board);

private:
    /**
     * @brief The size of the board
     */
    int board_size;

    /**
     * @brief A 2D vector representing the game board
     *
     * Each Cell_state signifies the state of a cell in the board - it can be
     * either empty or occupied by one of the two players.
     */
    std::vector<std::vector<Cell_state>> board;

    /**
     * @brief A 2D vector representing an empty board template
     *
     * Each Cell_state signifies the state of a cell in the board - it can be
     * either empty or occupied by one of the two players.
     */
    std::vector<std::vector<Cell_state>> empty_board;

    /**
     * @brief History of board states
     */
    std::vector<std::vector<std::vector<Cell_state>>> history;

    /**
     * @brief A restricted move that cannot be performed on the current turn
     */
    std::array<int, 2> restricted_move = {-1, -1};

    /**
     * @brief A vector containing all moves previously done by the current player
     */
    std::vector<std::array<int, 2>> path;

};

#endif
