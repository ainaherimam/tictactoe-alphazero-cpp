#ifndef POSITION_POOL_H
#define POSITION_POOL_H

#include <vector>
#include <array>
#include <cstdint>
#include "cell_state.h"
#include "constants.h"

struct Position {
    std::array<float, BOARD_SIZE> board;
    std::array<float, POLICY_SIZE> policy;
    std::array<float, POLICY_SIZE> mask;
    float z;
    uint8_t player_index;  // 0 or 1 for X or O
    
    Position() : z(0.0f), player_index(0) {
        board.fill(0.0f);
        policy.fill(0.0f);
        mask.fill(0.0f);
    }
};

class PositionPool {
public:
    explicit PositionPool(size_t capacity);
    
    // Add a position for the current move
    Position& acquire_position();
    
    // Finalize the game and update all z-values
    void finalize_game(Cell_state winner);
    
    // Get all positions from current game
    const std::vector<Position>& get_positions() const { return positions_; }

    const Position& get_position(size_t index) const { return positions_[index]; }
    
    // Get number of moves in current game
    size_t size() const { return positions_.size(); }
    
    // Reset for next game
    void reset();
    
private:
    std::vector<Position> positions_;
    size_t capacity_;
};

#endif // POSITION_POOL_H