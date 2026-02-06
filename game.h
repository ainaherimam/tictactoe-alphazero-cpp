#ifndef GAME_H
#define GAME_H

#include <memory>
#include <iostream>
#include <cstring>
#include "board.h"
#include "cell_state.h"
#include "player.h"
#include "position_pool.h"

/**
 * @class Game
 * @brief Manages a game session between two players with optional training data collection.
 * 
 * Handles game flow, move execution, player switching, and position collection for
 * training. Supports both evaluation mode (no data collection) and training mode.
 */
class Game {
public:
    /**
     * @brief Create a new Game instance.
     * @param player_1 First player
     * @param player_2 Second player
     * @param pool Reference to position pool for training data collection
     * @param is_evaluation If true, disables data collection and temperature adjustment (default: false)
     */
    Game(std::unique_ptr<Player> player_1,
         std::unique_ptr<Player> player_2,
         PositionPool& pool,
         bool is_evaluation = false);
 
    /**
     * @brief Executes a complete game with data collection and temperature scheduling.
     * @return The winner of the game (or Empty if draw/timeout)
     */
    Cell_state play();

    /**
     * @brief Gets the move history of the game.
     * @return Const reference to move history string
     */
    const std::string& get_move_history() const {
        return move_history;
    }
    
    /**
     * @brief Gets the current board state.
     * @return Const reference to the game board
     */
    const Board& getBoard() const {
        return board;
    }

private:
    Board board;                            ///< Game board
    std::unique_ptr<Player> players[2];     ///< The two players
    int current_player_index;               ///< Index of current player (0 fox X or 1 for O)
    bool is_evaluation;                     ///< Evaluation mode flag (disables data collection)
    std::string move_history;               ///< String record of all moves made
    size_t current_game_moves_;             ///< Number of moves made in current game
    
    PositionPool& pool_;                    ///< Reference to position pool for data collection

    /**
     * @brief Switches to the other player.
     */
    void switch_player();
  
    /**
     * @brief Collects current position for training data.
     * @param policy Pointer to policy array from neural network
     * @param player_index Index of the player making the move
     * @param move_number Current move number in the game
     */
    void collect_position(const float* policy,
                          uint8_t player_index,
                          uint16_t move_number);
};

#endif