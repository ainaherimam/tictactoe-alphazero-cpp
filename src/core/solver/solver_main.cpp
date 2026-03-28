#include "core/solver/misere_solver.h"
#include <iostream>
#include <chrono>
#include <string>
#include <bit>
#include <iomanip>

static void print_board(uint16_t bx, uint16_t bo) {
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            int cell = r * 4 + c;
            if (bx & (1u << cell))      std::cout << " X";
            else if (bo & (1u << cell)) std::cout << " O";
            else                         std::cout << " .";
        }
        std::cout << "\n";
    }
}

static std::string value_str(int v) {
    if (v > 0)  return "WIN  (+1)";
    if (v < 0)  return "LOSS (-1)";
    return "DRAW ( 0)";
}

static void print_move_values(MisereSolver& solver, uint16_t bx, uint16_t bo, bool is_x_turn) {
    std::string who = is_x_turn ? "X" : "O";
    std::cout << "\n  Available moves for " << who << ":\n";
    std::cout << "  Cell  (row,col)  Value\n";
    std::cout << "  ----  ---------  ---------\n";

    // Uses populated TT — no re-solve per move.
    auto avs = solver.get_action_values(bx, bo, is_x_turn);
    for (const auto& av : avs) {
        std::cout << "  " << std::setw(4) << av.cell
                  << "  (" << av.cell / 4 << "," << av.cell % 4 << ")      "
                  << value_str(av.value);
        if (av.value == -1) std::cout << "  [completes 3-in-a-row]";
        std::cout << "\n";
    }
    std::cout << "\n";
}

static void print_position_stats(MisereSolver& solver, uint16_t bx, uint16_t bo, bool is_x_turn) {
    int x_pieces = std::popcount(static_cast<unsigned>(bx));
    int o_pieces = std::popcount(static_cast<unsigned>(bo));
    int total = x_pieces + o_pieces;
    int empty = 16 - total;

    // Uses populated TT — no re-solve.
    int val = solver.get_position_value(bx, bo, is_x_turn);

    std::cout << "  --- Position stats ---\n";
    std::cout << "  Pieces: X=" << x_pieces << " O=" << o_pieces
              << "  Empty=" << empty
              << "  Turn=" << (is_x_turn ? "X" : "O")
              << "  Eval=" << value_str(val) << "\n";
}

int main() {
    MisereSolver solver;

    std::cout << "=== 4x4 Misere Tic-Tac-Toe Perfect Solver (3-in-a-row loses) ===\n\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = solver.solve();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Result for first player (X): " << value_str(result.value) << "\n";
    std::cout << "Best opening move: cell " << result.best_move
              << " (row " << result.best_move / 4
              << ", col " << result.best_move % 4 << ")\n";
    std::cout << "Solve time: " << ms << " ms\n";
    std::cout << "Transposition table entries: " << solver.table_size() << "\n\n";

    // Collect draw positions
    auto draws = solver.get_draw_positions();
    size_t draw_limit = std::min(draws.size(), size_t(100));

    std::cout << "=== Draw positions under perfect play: " << draws.size()
              << " (showing first " << draw_limit << ") ===\n\n";
    for (size_t i = 0; i < draw_limit; ++i) {
        const auto& dp = draws[i];
        bool x_turn = (dp.depth % 2 == 0);
        std::cout << "[" << (i + 1) << "] depth " << dp.depth
                  << ", " << (x_turn ? "X" : "O") << " to move:\n";
        print_board(dp.bx, dp.bo);
        std::cout << "\n";
    }

    // Interactive mode
    std::cout << "Play against the solver? (y/n): ";
    std::string answer;
    std::getline(std::cin, answer);
    if (answer.empty() || (answer[0] != 'y' && answer[0] != 'Y'))
        return 0;

    // Select starting position
    uint16_t bx = 0, bo = 0;
    bool x_turn = true;

    if (draw_limit > 0) {
        std::cout << "\nStart from a draw position? Enter number [1-" << draw_limit
                  << "] or 0 for empty board: ";
        std::string choice_str;
        std::getline(std::cin, choice_str);
        int choice = 0;
        try { choice = std::stoi(choice_str); } catch (...) {}

        if (choice >= 1 && choice <= (int)draw_limit) {
            const auto& dp = draws[choice - 1];
            bx = dp.bx;
            bo = dp.bo;
            x_turn = (dp.depth % 2 == 0);
            std::cout << "\nStarting from draw position #" << choice
                      << " (depth " << dp.depth << "):\n";
            print_board(bx, bo);
            std::cout << "\n";
        } else {
            std::cout << "\nStarting from empty board.\n\n";
        }
    }

    // Choose solver role
    std::string next_player = x_turn ? "X" : "O";
    std::string other_player = x_turn ? "O" : "X";
    std::cout << "Next to move: " << next_player << "\n";
    std::cout << "Solver plays as (1 = " << next_player << "/next, 2 = "
              << other_player << "/after): ";
    std::string role_str;
    std::getline(std::cin, role_str);
    bool solver_goes_next = (role_str.empty() || role_str[0] == '1');
    bool solver_is_x = solver_goes_next ? x_turn : !x_turn;

    if (solver_is_x) {
        std::cout << "\nSolver is X. You are O.\n";
    } else {
        std::cout << "\nYou are X. Solver is O.\n";
    }
    std::cout << "Enter moves as cell number (0-15):\n";
    std::cout << " 0  1  2  3\n 4  5  6  7\n 8  9 10 11\n12 13 14 15\n\n";

    // Game loop
    while (true) {
        print_board(bx, bo);
        uint16_t occupied = bx | bo;

        // Check terminal
        if (MisereSolver::has_line(bx)) {
            std::cout << "X completed 3-in-a-row -- X LOSES (misere)!\n";
            break;
        }
        if (MisereSolver::has_line(bo)) {
            std::cout << "O completed 3-in-a-row -- O LOSES (misere)!\n";
            break;
        }
        if (occupied == 0xFFFF) {
            std::cout << "Board full -- DRAW!\n";
            break;
        }

        // Print stats and move values
        print_position_stats(solver, bx, bo, x_turn);
        print_move_values(solver, bx, bo, x_turn);

        bool solver_turn = (x_turn == solver_is_x);

        if (solver_turn) {
            int best = solver.get_best_move(bx, bo, x_turn);
            int eval = solver.get_position_value(bx, bo, x_turn);
            std::cout << "Solver plays: cell " << best
                      << " (row " << best / 4
                      << ", col " << best % 4 << ")"
                      << "  [eval: " << value_str(eval) << "]\n\n";
            if (x_turn) bx |= (1u << best);
            else        bo |= (1u << best);
        } else {
            int cell = -1;
            while (true) {
                std::cout << "Your move (0-15): ";
                std::cin >> cell;
                if (cell >= 0 && cell < 16 && !(occupied & (1u << cell)))
                    break;
                std::cout << "Invalid move. Try again.\n";
            }
            if (x_turn) bx |= (1u << cell);
            else        bo |= (1u << cell);
        }
        x_turn = !x_turn;
    }

    return 0;
}
