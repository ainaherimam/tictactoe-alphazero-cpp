#ifndef CELL_STATE_H
#define CELL_STATE_H

#include <ostream>

/**
 * @enum Cell_state
 * @brief Represents the state of a cell in the game.
 * 
 * A cell in the game can be in one of three states:
 * Empty, X, or O.
 * 
 * Enumeration values:
 * @value Empty The cell has not been claimed by any player.
 * @value X The cell has been claimed by player 1.
 * @value O The cell has been claimed by player 2.
 */
enum class Cell_state {
  Empty,
  X,
  O
};

/**
 * @brief Overloaded stream insertion operator for the Cell_state.
 * 
 * @param os The output stream to write to.
 * @param state The Cell_state to be written to the stream.
 * 
 * @return A reference to the output stream with the given state character.
 */
std::ostream& operator<<(std::ostream& os, const Cell_state& state);

#endif  // CELL_STATE_H