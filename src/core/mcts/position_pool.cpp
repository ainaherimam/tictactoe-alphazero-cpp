#include "core/mcts/position_pool.h"
#include <stdexcept>

PositionPool::PositionPool(size_t capacity)
    : capacity_(capacity) {
    // Reserve the full capacity upfront so that emplace_back() never
    // reallocates the underlying buffer. Without this, a reallocation
    // between acquire_position() and the writes in collect_position()
    // would leave a dangling Position& pointing at freed memory,
    // causing silent data corruption in board/policy/mask.
    positions_.reserve(capacity);
}

Position& PositionPool::acquire_position() {
    if (positions_.size() >= capacity_) {
        throw std::runtime_error("PositionPool is full!");
    }

    positions_.emplace_back();
    // Safe: emplace_back() will not reallocate because we reserved
    // capacity upfront in the constructor.
    return positions_.back();
}

void PositionPool::finalize_game(Cell_state winner) {
    for (Position& pos : positions_) {
        bool is_player_x = (pos.player_index == 0);


        //if X is the winner, all X turn to play position is valued 1.0, all O turn to play is valued -1.0
        //if O is the winner, all O turn to play position is valued 1.0, all X turn to play is valued -1.0
        
        if (winner == Cell_state::X) {
            pos.z = is_player_x ? 1.0f : -1.0f;
        } else if (winner == Cell_state::O) {
            pos.z = is_player_x ? -1.0f : 1.0f;
        } else {
            pos.z = 0.0f;
        }
    }
}

void PositionPool::reset() {
    positions_.clear();
}