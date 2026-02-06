#ifndef POSITION_POOL_H
#define POSITION_POOL_H

#include <vector>
#include <array>
#include <cstdint>
#include "cell_state.h"
#include "constants.h"

/**
 * @struct Position
 * @brief Represents a single game position with board state, policy, and outcome.
 * 
 * Used for collecting training data during self-play.
 */
struct Position {
    std::array<float, BOARD_SIZE> board;      ///< board state
    std::array<float, POLICY_SIZE> policy;    ///< policy distribution from mcts
    std::array<float, POLICY_SIZE> mask;      ///< legal move mask (1.0 = legal, 0.0 = illegal)
    float z;                                  ///< Game outcome z
    uint8_t player_index;                     ///< Next player to move from the board state (0 for X, 1 for O)
    
    /**
     * @brief Default constructor.
     */
    Position() : z(0.0f), player_index(0) {
        board.fill(0.0f);
        policy.fill(0.0f);
        mask.fill(0.0f);
    }
};

/**
 * @class PositionPool
 * @brief Manages collection of positions from a single game for training purpose.
 * 
 * Collects positions during game play and assigns final z-values (outcomes)
 * once the game is complete.
 */
class PositionPool {
public:
    /**
     * @brief Create a position pool with fixed capacity.
     * @param capacity Maximum number of positions that can be stored
     */
    explicit PositionPool(size_t capacity);
    
    /**
     * @brief Acquires a new position slot for the current move.
     * @return Reference to the newly added position
     * @throws std::runtime_error if pool is at capacity
     */
    Position& acquire_position();
    
    /**
     * @brief Finalizes the game by updating all z-values based on outcome.
     * @param winner The winning player (X, O, or Empty for draw)
     */
    void finalize_game(Cell_state winner);
    
    /**
     * @brief Gets all positions collected in the current game.
     * @return Const reference to vector of positions
     */
    const std::vector<Position>& get_positions() const {
        return positions_;
    }

    /**
     * @brief Gets a specific position by index.
     */
    const Position& get_position(size_t index) const {
        return positions_[index];
    }
    
    /**
     * @brief Gets the number of positions in the current game.
     */
    size_t size() const {
        return positions_.size();
    }
    
    /**
     * @brief Clears all positions to prepare for the next game.
     */
    void reset();
    
private:
    std::vector<Position> positions_;  ///< Collection of positions from current game
    size_t capacity_;                  ///< Maximum capacity of the pool
};

#endif // POSITION_POOL_H