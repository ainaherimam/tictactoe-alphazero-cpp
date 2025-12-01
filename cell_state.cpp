#include "cell_state.h"

std::ostream& operator<<(std::ostream& os, const Cell_state& state) {
  switch (state) {
    case Cell_state::Empty:
      os << '.';
      break;
    case Cell_state::X:
      os << 'x';
      break;
    case Cell_state::O:
      os << 'o';
      break;
  }
  return os;
}