// generate_initial_boards.cpp
// Generates initial_boards.h by BFS-enumerating board positions and
// classifying them into 4 groups using the perfect Misère solver.
//
// Groups:
//   0. X moves first, X wins
//   1. O moves first, X wins
//   2. X moves first, draw
//   3. O moves first, draw
//
// Positions where O wins are excluded — the perfect solver always wins
// those, making them useless for evaluation (theory_score always 0%).
//
// Positions with fewer pieces are prioritized (BFS order).
// Usage: ./GenerateInitialBoards [boards_per_group]

#include "core/solver/misere_solver.h"
#include <bit>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <queue>
#include <set>
#include <string>
#include <vector>

static constexpr int NUM_GROUPS = 4;

static const char* GROUP_NAMES[NUM_GROUPS] = {
    "X first, X wins",
    "O first, X wins",
    "X first, draw",
    "O first, draw",
};

static const char* GROUP_TAGS[NUM_GROUPS] = {
    "xfirst_xwins",
    "ofirst_xwins",
    "xfirst_draw",
    "ofirst_draw",
};

struct Position {
    uint16_t bx;
    uint16_t bo;
};

// Classify a non-terminal position into one of 4 groups.
// Returns group index 0-3, or -1 if O wins (skipped — solver always wins those).
// solver_value is from current player's perspective: +1 win, 0 draw, -1 loss.
static int classify(bool is_x_turn, int solver_value) {
    if (is_x_turn) {
        if (solver_value > 0) return 0;   // X first, X wins
        if (solver_value < 0) return -1;  // X first, O wins — skip
        return 2;                          // X first, draw
    } else {
        if (solver_value > 0) return -1;  // O first, O wins — skip
        if (solver_value < 0) return 1;   // O first, X wins
        return 3;                          // O first, draw
    }
}

int main(int argc, char* argv[]) {
    int boards_per_group = 100;
    if (argc > 1) {
        boards_per_group = std::stoi(argv[1]);
        if (boards_per_group < 1) boards_per_group = 1;
    }

    std::cout << "Generating " << boards_per_group << " boards per group ("
              << NUM_GROUPS << " groups, " << boards_per_group * NUM_GROUPS
              << " total)\n";

    // Solve the game to populate the transposition table
    MisereSolver solver;
    auto t0 = std::chrono::high_resolution_clock::now();
    solver.solve();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Solver: "
              << std::chrono::duration<double, std::milli>(t1 - t0).count()
              << " ms  (TT size: " << solver.table_size() << ")\n";

    // BFS from empty board — visits positions in order of piece count
    std::vector<std::vector<Position>> groups(NUM_GROUPS);
    int total_needed = boards_per_group * NUM_GROUPS;
    int total_found = 0;

    // Track visited canonical positions
    std::set<uint32_t> visited;
    auto canon_key = [](uint16_t bx, uint16_t bo) -> uint32_t {
        return (static_cast<uint32_t>(bx) << 16) | bo;
    };

    // Canonicalize using solver's symmetry (we replicate the logic here
    // since canonicalize is private — we use a simpler approach:
    // store all 8 symmetry variants as "visited" when we first see a position)
    static constexpr int SYM_PERM[8][16] = {
        { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
        {12,  8,  4,  0, 13,  9,  5,  1, 14, 10,  6,  2, 15, 11,  7,  3},
        {15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0},
        { 3,  7, 11, 15,  2,  6, 10, 14,  1,  5,  9, 13,  0,  4,  8, 12},
        {12, 13, 14, 15,  8,  9, 10, 11,  4,  5,  6,  7,  0,  1,  2,  3},
        { 3,  2,  1,  0,  7,  6,  5,  4, 11, 10,  9,  8, 15, 14, 13, 12},
        { 0,  4,  8, 12,  1,  5,  9, 13,  2,  6, 10, 14,  3,  7, 11, 15},
        {15, 11,  7,  3, 14, 10,  6,  2, 13,  9,  5,  1, 12,  8,  4,  0},
    };

    auto transform = [](uint16_t bits, int sym) -> uint16_t {
        uint16_t result = 0;
        uint16_t tmp = bits;
        while (tmp) {
            int cell = std::countr_zero(static_cast<unsigned>(tmp));
            result |= static_cast<uint16_t>(1u << SYM_PERM[sym][cell]);
            tmp &= tmp - 1;
        }
        return result;
    };

    auto canonicalize = [&](uint16_t bx, uint16_t bo,
                            uint16_t& cbx, uint16_t& cbo) {
        cbx = bx;
        cbo = bo;
        for (int s = 1; s < 8; ++s) {
            uint16_t tx = transform(bx, s);
            uint16_t to_val = transform(bo, s);
            if (tx < cbx || (tx == cbx && to_val < cbo)) {
                cbx = tx;
                cbo = to_val;
            }
        }
    };

    // BFS queue stores (bx, bo) — the raw (non-canonical) position that we
    // output; canonical form is only used for deduplication.
    std::queue<Position> bfs;
    {
        uint16_t cbx, cbo;
        canonicalize(0, 0, cbx, cbo);
        visited.insert(canon_key(cbx, cbo));
        bfs.push({0, 0});
    }

    int positions_examined = 0;

    while (!bfs.empty() && total_found < total_needed) {
        Position pos = bfs.front();
        bfs.pop();

        uint16_t bx = pos.bx, bo = pos.bo;
        int nx = std::popcount(static_cast<unsigned>(bx));
        int no = std::popcount(static_cast<unsigned>(bo));
        bool is_x_turn = (nx == no);
        int depth = nx + no;

        // Skip terminal positions
        if (depth > 0) {
            uint16_t last_bits = is_x_turn ? bo : bx;
            if (MisereSolver::has_line(last_bits)) continue;
        }
        if ((bx | bo) == 0xFFFF) continue;

        ++positions_examined;

        // Classify this position (group == -1 means O wins — skip)
        int value = solver.get_position_value(bx, bo, is_x_turn);
        int group = classify(is_x_turn, value);

        if (group >= 0 && static_cast<int>(groups[group].size()) < boards_per_group) {
            // Store the canonical form for the output (cleaner)
            uint16_t cbx, cbo;
            canonicalize(bx, bo, cbx, cbo);
            groups[group].push_back({cbx, cbo});
            ++total_found;
        }

        // Expand children (if we still need boards)
        if (total_found < total_needed) {
            uint16_t occupied = bx | bo;
            for (int cell = 0; cell < 16; ++cell) {
                if (occupied & (1u << cell)) continue;

                uint16_t new_bx = bx, new_bo = bo;
                if (is_x_turn) new_bx |= static_cast<uint16_t>(1u << cell);
                else           new_bo |= static_cast<uint16_t>(1u << cell);

                // Skip if the move creates a line (terminal child)
                uint16_t our_bits = is_x_turn ? new_bx : new_bo;
                if (MisereSolver::has_line(our_bits)) continue;

                // Canonicalize for deduplication
                uint16_t cbx, cbo;
                canonicalize(new_bx, new_bo, cbx, cbo);
                uint32_t key = canon_key(cbx, cbo);
                if (visited.count(key)) continue;
                visited.insert(key);
                bfs.push({cbx, cbo});
            }
        }
    }

    // Print summary
    std::cout << "\nPositions examined: " << positions_examined << "\n";
    for (int g = 0; g < NUM_GROUPS; ++g) {
        std::cout << "  " << GROUP_NAMES[g] << ": " << groups[g].size() << "\n";
    }

    // Write initial_boards.h
    const std::string output_path = "executables/initial_boards.h";
    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << output_path << "\n";
        return 1;
    }

    out << R"(#pragma once

// ============================================================
// initial_boards.h — Auto-generated by GenerateInitialBoards
//
// Board positions for Elo evaluation, classified into 4 groups
// by who moves first and the game-theoretic outcome (solved by
// the perfect Misère Negamax solver).
// O-wins positions are excluded (solver always wins → 0% theory score).
//
// Boards are stored as (bx, bo) bitboard pairs. Bit i corresponds
// to cell i in row-major order:
//   0  1  2  3
//   4  5  6  7
//   8  9 10 11
//  12 13 14 15
//
// Who moves first is determined by piece count:
//   popcount(bx) == popcount(bo)     → X to move
//   popcount(bx) == popcount(bo) + 1 → O to move
// ============================================================

#include "core/game/cell_state.h"
#include "core/game/constants.h"

#include <array>
#include <bit>
#include <cstdint>

// Number of boards per group (editable)
)";

    out << "static constexpr int BOARDS_PER_GROUP = " << boards_per_group << ";\n";
    out << "static constexpr int NUM_GROUPS = " << NUM_GROUPS << ";\n";
    out << "static constexpr int TOTAL_BOARDS = BOARDS_PER_GROUP * NUM_GROUPS;\n";
    out << "\n";

    out << R"(// Group indices
enum BoardGroup {
    X_FIRST_X_WINS = 0,
    O_FIRST_X_WINS = 1,
    X_FIRST_DRAW   = 2,
    O_FIRST_DRAW   = 3,
};

struct InitialBoard {
    uint16_t bx;   // X bitboard
    uint16_t bo;   // O bitboard
};

)";

    // Write each group
    for (int g = 0; g < NUM_GROUPS; ++g) {
        out << "// " << GROUP_NAMES[g]
            << " (" << groups[g].size() << " boards)\n";
        out << "static constexpr InitialBoard GROUP_" << GROUP_TAGS[g]
            << "[BOARDS_PER_GROUP] = {\n";

        for (size_t i = 0; i < groups[g].size(); ++i) {
            const auto& p = groups[g][i];
            int nx = std::popcount(static_cast<unsigned>(p.bx));
            int no = std::popcount(static_cast<unsigned>(p.bo));
            out << "    {0x" << std::hex << p.bx << ", 0x" << p.bo << std::dec
                << "},  // " << (nx + no) << " pieces\n";
        }
        // Pad with zeros if we didn't find enough
        for (size_t i = groups[g].size();
             i < static_cast<size_t>(boards_per_group); ++i) {
            out << "    {0x0, 0x0},  // placeholder\n";
        }
        out << "};\n\n";
    }

    // Write the unified array
    out << R"(// All groups combined into a single flat array
static constexpr const InitialBoard* ALL_GROUPS[NUM_GROUPS] = {
    GROUP_xfirst_xwins,
    GROUP_ofirst_xwins,
    GROUP_xfirst_draw,
    GROUP_ofirst_draw,
};

// Helper: determine whose turn it is from a bitboard pair
inline Cell_state first_mover_of(uint16_t bx, uint16_t bo) {
    return (std::popcount(static_cast<unsigned>(bx))
         == std::popcount(static_cast<unsigned>(bo)))
         ? Cell_state::X : Cell_state::O;
}

// Helper: convert bitboard pair to cell array for Board::load_board()
inline std::array<Cell_state, BOARD_CELLS> bitboards_to_cells(uint16_t bx, uint16_t bo) {
    std::array<Cell_state, BOARD_CELLS> cells;
    for (int i = 0; i < BOARD_CELLS; ++i) {
        if (bx & (1u << i))      cells[i] = Cell_state::X;
        else if (bo & (1u << i)) cells[i] = Cell_state::O;
        else                     cells[i] = Cell_state::Empty;
    }
    return cells;
}
)";

    out.close();
    std::cout << "\nWrote " << output_path << "\n";

    // -----------------------------------------------------------------
    // Write a human-readable text file with board grids
    // -----------------------------------------------------------------
    const std::string txt_path = "executables/initial_boards.txt";
    std::ofstream txt(txt_path);
    if (!txt.is_open()) {
        std::cerr << "Failed to open " << txt_path << "\n";
        return 1;
    }

    txt << "Initial boards for Elo evaluation\n";
    txt << "==================================\n\n";
    txt << "Board layout (cell indices):\n";
    txt << "   0  1  2  3\n";
    txt << "   4  5  6  7\n";
    txt << "   8  9 10 11\n";
    txt << "  12 13 14 15\n\n";
    txt << "Legend: X = first player, O = second player, . = empty\n";
    txt << "==================================\n";

    for (int g = 0; g < NUM_GROUPS; ++g) {
        txt << "\n--- " << GROUP_NAMES[g] << " (" << groups[g].size() << " boards) ---\n\n";
        for (size_t i = 0; i < groups[g].size(); ++i) {
            const auto& p = groups[g][i];
            int nx = std::popcount(static_cast<unsigned>(p.bx));
            int no = std::popcount(static_cast<unsigned>(p.bo));
            txt << "Board " << (i + 1) << "  (" << (nx + no) << " pieces"
                << ", bx=0x" << std::hex << p.bx << " bo=0x" << p.bo << std::dec << ")\n";
            for (int r = 0; r < 4; ++r) {
                txt << "  ";
                for (int c = 0; c < 4; ++c) {
                    int cell = r * 4 + c;
                    if      (p.bx & (1u << cell)) txt << " X";
                    else if (p.bo & (1u << cell)) txt << " O";
                    else                           txt << " .";
                }
                txt << "\n";
            }
            txt << "\n";
        }
    }

    txt.close();
    std::cout << "Wrote " << txt_path << "\n";
    return 0;
}
