#pragma once

#include <cstdint>
#include <array>
#include <unordered_map>
#include <vector>

/**
 * Perfect Negamax solver for 4x4 Misere Tic-Tac-Toe.
 *
 * Rules: A player LOSES if they complete a 3-in-a-row (horizontal, vertical,
 * or diagonal). If the board fills with no line completed, it is a draw.
 *
 * Board layout (bit indices):
 *   0  1  2  3
 *   4  5  6  7
 *   8  9 10 11
 *  12 13 14 15
 *
 * Uses two 16-bit masks (one per player) to represent the board.
 */
class MisereSolver {
public:
    struct Result {
        int value;   // +1 current player wins, 0 draw, -1 current player loses
        int best_move; // best cell index (0-15), or -1 if terminal
    };

    MisereSolver();

    /// Solve from the initial empty board. Returns the game-theoretic value
    /// for the first player (X).
    Result solve();

    /// Solve an arbitrary position. `is_x_turn` indicates whose move it is.
    Result solve(uint16_t board_x, uint16_t board_o, bool is_x_turn);

    /// Check if `bits` contains any completed 3-in-a-row line.
    static bool has_line(uint16_t bits);

    /// Return the number of transposition table entries after a solve.
    size_t table_size() const { return tt_.size(); }

    /// Retrieve the optimal move(s) for any position that was solved.
    /// Returns empty if the position was not encountered.
    std::vector<int> get_optimal_moves(uint16_t board_x, uint16_t board_o, bool is_x_turn) const;

    /// Exact game-theoretic value for a position from the current player's perspective.
    /// Uses the TT (EXACT entries) if available; otherwise runs targeted negamax without
    /// clearing the TT. Call solve() first to populate the TT for fast lookups.
    int get_position_value(uint16_t bx, uint16_t bo, bool is_x_turn);

    struct ActionValue {
        int cell;   // cell index 0-15
        int value;  // from current player's perspective: +1 win, 0 draw, -1 loss
    };

    /// Returns the exact game-theoretic value for every legal action at (bx, bo).
    /// Handles canonical-form lookups internally — results are in original cell indices.
    /// Call solve() first to populate the TT; missing entries are computed on demand.
    std::vector<ActionValue> get_action_values(uint16_t bx, uint16_t bo, bool is_x_turn);

    /// Best move for a position using the populated TT (does NOT clear the TT).
    /// Returns -1 if no move is available (terminal position).
    int get_best_move(uint16_t bx, uint16_t bo, bool is_x_turn);

    struct DrawPosition {
        uint16_t bx;    // canonical X board
        uint16_t bo;    // canonical O board
        int      depth; // number of pieces on the board
    };

    /// Return all positions in the TT that are exact draws (value == 0).
    /// Positions are in canonical form.
    std::vector<DrawPosition> get_draw_positions() const;

private:
    // ---- Symmetry -----------------------------------------------------------
    /// Map a cell index through one of 8 symmetry transforms.
    static uint16_t transform_board(uint16_t bits, int sym);

    /// Return the canonical (smallest) representation of a position
    /// under the 8 symmetries of the square.
    static void canonicalize(uint16_t bx, uint16_t bo,
                             uint16_t& canon_bx, uint16_t& canon_bo,
                             int& best_sym);

    // ---- Zobrist ------------------------------------------------------------
    uint64_t zobrist_x_[16];
    uint64_t zobrist_o_[16];

    uint64_t compute_hash(uint16_t bx, uint16_t bo) const;

    // ---- Transposition table ------------------------------------------------
    enum TTFlag : uint8_t { EXACT, LOWER_BOUND, UPPER_BOUND };

    struct TTEntry {
        uint64_t hash;
        uint16_t cbx;   // canonical board X
        uint16_t cbo;   // canonical board O
        int8_t   value;
        uint8_t  depth;
        TTFlag   flag;
    };

    std::unordered_map<uint64_t, TTEntry> tt_;

    // ---- Negamax ------------------------------------------------------------
    int negamax(uint16_t bx, uint16_t bo, bool is_x_turn,
                int alpha, int beta, int depth);

    // ---- Post-solve delay-aware move selection ------------------------------
    int find_best_move(uint16_t bx, uint16_t bo, bool is_x_turn,
                       int position_value, int depth);
    int depth_to_end(uint16_t bx, uint16_t bo, bool is_x_turn, int depth);

    // ---- Line masks (3-in-a-row on 4x4 board) ------------------------------
    static constexpr int NUM_LINES = 24;
    static constexpr uint16_t LINE_MASKS[NUM_LINES] = {
        // Rows (2 windows of 3 per row × 4 rows = 8)
        (1<<0)|(1<<1)|(1<<2),    // row 0: cells 0,1,2
        (1<<1)|(1<<2)|(1<<3),    // row 0: cells 1,2,3
        (1<<4)|(1<<5)|(1<<6),    // row 1: cells 4,5,6
        (1<<5)|(1<<6)|(1<<7),    // row 1: cells 5,6,7
        (1<<8)|(1<<9)|(1<<10),   // row 2: cells 8,9,10
        (1<<9)|(1<<10)|(1<<11),  // row 2: cells 9,10,11
        (1<<12)|(1<<13)|(1<<14), // row 3: cells 12,13,14
        (1<<13)|(1<<14)|(1<<15), // row 3: cells 13,14,15
        // Columns (2 windows of 3 per col × 4 cols = 8)
        (1<<0)|(1<<4)|(1<<8),    // col 0: cells 0,4,8
        (1<<4)|(1<<8)|(1<<12),   // col 0: cells 4,8,12
        (1<<1)|(1<<5)|(1<<9),    // col 1: cells 1,5,9
        (1<<5)|(1<<9)|(1<<13),   // col 1: cells 5,9,13
        (1<<2)|(1<<6)|(1<<10),   // col 2: cells 2,6,10
        (1<<6)|(1<<10)|(1<<14),  // col 2: cells 6,10,14
        (1<<3)|(1<<7)|(1<<11),   // col 3: cells 3,7,11
        (1<<7)|(1<<11)|(1<<15),  // col 3: cells 7,11,15
        // Diagonals (top-left to bottom-right, 4 lines of length 3)
        (1<<0)|(1<<5)|(1<<10),   // cells 0,5,10
        (1<<5)|(1<<10)|(1<<15),  // cells 5,10,15
        (1<<1)|(1<<6)|(1<<11),   // cells 1,6,11
        (1<<4)|(1<<9)|(1<<14),   // cells 4,9,14
        // Anti-diagonals (top-right to bottom-left, 4 lines of length 3)
        (1<<3)|(1<<6)|(1<<9),    // cells 3,6,9
        (1<<6)|(1<<9)|(1<<12),   // cells 6,9,12
        (1<<2)|(1<<5)|(1<<8),    // cells 2,5,8
        (1<<7)|(1<<10)|(1<<13),  // cells 7,10,13
    };

    // ---- Move ordering ------------------------------------------------------
    /// Static move ordering: center cells first, then edges, then corners.
    static constexpr int MOVE_ORDER[16] = {
        5, 6, 9, 10,           // center 4
        1, 2, 4, 7, 8, 11, 13, 14, // edges 8
        0, 3, 12, 15           // corners 4
    };

    // ---- Depth-to-end memoization -------------------------------------------
    /// Caches depth_to_end results, keyed by canonical Zobrist hash.
    std::unordered_map<uint64_t, int> depth_cache_;

    // ---- Symmetry tables (precomputed) --------------------------------------
    /// For each of the 8 D4 symmetries, the permutation of cell indices.
    static constexpr int SYM_PERM[8][16] = {
        // 0: identity
        { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
        // 1: rotate 90 CW
        {12,  8,  4,  0, 13,  9,  5,  1, 14, 10,  6,  2, 15, 11,  7,  3},
        // 2: rotate 180
        {15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0},
        // 3: rotate 270 CW
        { 3,  7, 11, 15,  2,  6, 10, 14,  1,  5,  9, 13,  0,  4,  8, 12},
        // 4: reflect horizontal (flip rows)
        {12, 13, 14, 15,  8,  9, 10, 11,  4,  5,  6,  7,  0,  1,  2,  3},
        // 5: reflect vertical (flip columns)
        { 3,  2,  1,  0,  7,  6,  5,  4, 11, 10,  9,  8, 15, 14, 13, 12},
        // 6: reflect main diagonal (transpose)
        { 0,  4,  8, 12,  1,  5,  9, 13,  2,  6, 10, 14,  3,  7, 11, 15},
        // 7: reflect anti-diagonal
        {15, 11,  7,  3, 14, 10,  6,  2, 13,  9,  5,  1, 12,  8,  4,  0},
    };
};
