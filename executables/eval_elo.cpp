// eval_elo.cpp
// AlphaZero-style Elo evaluation: all checkpoints in checkpoints/ vs the
// perfect Misère Negamax solver.
//
// Checkpoint always plays as X; solver always plays as O.
// Boards are pre-classified into 6 groups by who moves first and the
// game-theoretic outcome.  Score is awarded only when the checkpoint (X)
// achieves the theoretical outcome for that position:
//
//   Theory X wins + actual X wins  → W  (full point)
//   Theory draw   + actual draw    → D  (half point)
//   Everything else                → L  (no point)
//
//   score  S  = (W + 0.5·D) / N
//   Elo       = 400 · log10( S / (1 − S) )
//
// Output: elo_results.csv (one row per checkpoint, sorted by checkpoint number)

#include "core/game/player.h"
#include "core/game/cell_state.h"
#include "core/game/board.h"
#include "core/game/constants.h"
#include "core/mcts/mcts_config.h"
#include "core/solver/misere_solver.h"
#include "inference/shared_memory/inference_queue_shm.h"
#include "initial_boards.h"

#include <iostream>
#include <fstream>
#include <memory>
#include <thread>
#include <vector>
#include <atomic>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <string>
#include <cmath>

namespace fs = std::filesystem;


// ============================================================
// SolverPlayer — wraps MisereSolver as a Player
// check_winner() returns the WINNER (opponent of who formed 3-in-a-row),
// so winner == X means X won; winner == O means O won.
// ============================================================
class SolverPlayer : public Player {
public:
    explicit SolverPlayer(std::shared_ptr<MisereSolver> solver)
        : solver_(std::move(solver)) {}

    std::pair<Move, std::vector<float>> choose_move(
        const Board& board, Cell_state player) override
    {
        float planes[INPUT_SIZE];
        board.to_float_array(Cell_state::X, planes);

        uint16_t bx = 0, bo = 0;
        for (int i = 0; i < BOARD_CELLS; ++i) {
            if (planes[i]             > 0.5f) bx |= static_cast<uint16_t>(1u << i);
            if (planes[BOARD_CELLS+i] > 0.5f) bo |= static_cast<uint16_t>(1u << i);
        }

        bool is_x_turn = (player == Cell_state::X);
        int cell = solver_->get_best_move(bx, bo, is_x_turn);

        Move move;
        move.x   = cell / BOARD_WIDTH;
        move.y   = cell % BOARD_WIDTH;
        move.dir = -1;
        move.tar = -1;

        std::vector<float> policy(BOARD_CELLS, 0.0f);
        if (cell >= 0) policy[cell] = 1.0f;
        return {move, policy};
    }

private:
    std::shared_ptr<MisereSolver> solver_;
};


// ============================================================
// Elo formula — AlphaZero paper variant.
// Solver is the reference player, anchored at Elo 0.
//
//   S   = (W + 0.5·D) / N          (expected score in [0,1])
//   Elo = 400 · log10( S / (1−S) )
//
// S is clamped away from 0/1 to keep Elo finite.
// ============================================================
static double compute_elo(int wins, int draws, int losses) {
    int n = wins + draws + losses;
    if (n == 0) return 0.0;
    double s = (wins + 0.5 * draws) / static_cast<double>(n);
    s = std::max(1e-6, std::min(1.0 - 1e-6, s));
    return 400.0 * std::log10(s / (1.0 - s));
}


// ============================================================
// Play one game from a fixed initial board position.
//
// Checkpoint always plays as X; solver always plays as O.
//
// check_winner() semantics (Misère, implemented in board.cpp):
//   returns the OPPONENT of the player who formed 3-in-a-row,
//   i.e. the WINNER of the game.  Empty = draw (board full).
//
// Returns: winner (X, O, or Empty=draw).
// ============================================================
static Cell_state play_from_board(
    const InitialBoard& init,
    std::shared_ptr<SharedMemoryInferenceQueue> queue,
    std::shared_ptr<MisereSolver> solver)
{
    Board board;
    auto cells = bitboards_to_cells(init.bx, init.bo);
    board.load_board(cells);

    // Guard: already terminal before we start
    Cell_state early = board.check_winner();
    if (early != Cell_state::Empty) return early;

    Cell_state current = first_mover_of(init.bx, init.bo);

    // Eval config: no Dirichlet noise, greedy (temperature = 0)
    Mcts_config cfg(
        /*c_puct         =*/ 1.4,
        /*iterations     =*/ 100,
        LogLevel::NONE,
        /*temperature    =*/ 0.0f,
        /*dirichlet_alpha=*/ 0.3f,
        /*dirichlet_eps  =*/ 0.0f,
        queue,
        /*max_depth      =*/ 5,
        /*tree_reuse     =*/ false,
        /*model_id       =*/ 0);

    Mcts_player_selfplay mcts(cfg);
    SolverPlayer         sol(solver);

    for (;;) {
        Cell_state w = board.check_winner();
        if (w != Cell_state::Empty) return w;

        auto valid = board.get_valid_moves(current);
        if (valid.empty()) return Cell_state::Empty;   // draw

        Move chosen;
        if (current == Cell_state::X)
            chosen = mcts.choose_move(board, current).first;  // checkpoint plays X
        else
            chosen = sol.choose_move(board, current).first;   // solver plays O

        board.make_move(chosen, current);
        current = (current == Cell_state::X) ? Cell_state::O : Cell_state::X;
    }
}


// ============================================================
// Atomic result counters (from X's perspective) — one per board.
// ============================================================
struct AtomicCounts {
    std::atomic<int> x_wins{0};   // X (checkpoint) won
    std::atomic<int> draws{0};    // draw
    std::atomic<int> o_wins{0};   // O (solver) won
};


// ============================================================
// Game task (one game)
// ============================================================
struct GameTask {
    int board_idx;
};


// ============================================================
// Worker thread
// ============================================================
static void worker_fn(
    std::shared_ptr<SharedMemoryInferenceQueue>  queue,
    std::shared_ptr<MisereSolver>                solver,
    const std::vector<InitialBoard>&             boards,
    const std::vector<GameTask>&                 tasks,
    std::atomic<int>&                            task_counter,
    std::vector<AtomicCounts>&                   board_results)
{
    while (true) {
        int i = task_counter.fetch_add(1, std::memory_order_relaxed);
        if (i >= static_cast<int>(tasks.size())) break;

        const GameTask&     t    = tasks[i];
        const InitialBoard& init = boards[t.board_idx];

        // Checkpoint always X, solver always O
        Cell_state winner = play_from_board(init, queue, solver);

        AtomicCounts& cnt = board_results[t.board_idx];
        if (winner == Cell_state::X)
            cnt.x_wins.fetch_add(1, std::memory_order_relaxed);
        else if (winner == Cell_state::Empty)
            cnt.draws.fetch_add(1, std::memory_order_relaxed);
        else
            cnt.o_wins.fetch_add(1, std::memory_order_relaxed);
    }
}


// ============================================================
// Checkpoint discovery — scans dir for checkpoint_N folders
// ============================================================
static std::vector<int> discover_checkpoints(const std::string& dir) {
    std::vector<int> result;
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        std::cerr << "[WARNING] Checkpoints directory not found: " << dir << "\n";
        return result;
    }
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_directory()) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind("checkpoint_", 0) != 0) continue;
        try {
            result.push_back(std::stoi(name.substr(11)));
        } catch (...) {}
    }
    std::sort(result.begin(), result.end());
    return result;
}


// ============================================================
// Checkpoint reload trigger
// ============================================================
static bool reload_checkpoint(const std::string& checkpoints_dir,
                               const std::string& queue_name,
                               int gen,
                               std::shared_ptr<SharedMemoryInferenceQueue> queue,
                               int timeout_ms = 60'000)
{
    const std::string trigger_path = checkpoints_dir +
        "reload." + queue_name + "." + std::to_string(gen) + ".trigger";

    {
        std::ofstream f(trigger_path);
        if (!f.is_open()) {
            std::cerr << "  Failed to create trigger\n";
            return false;
        }
        f << gen << "\n";
    }

    if (!queue->wait_for_server(timeout_ms)) {
        std::cerr << "  ERROR: server not ready\n";
        return false;
    }
    return true;
}


// ============================================================
// Group names (parallel to BoardGroup enum in initial_boards.h)
// ============================================================
static constexpr const char* GROUP_NAMES[NUM_GROUPS] = {
    "X_first_X_wins",
    "O_first_X_wins",
    "X_first_Draw  ",
    "O_first_Draw  ",
};


// ============================================================
// Theoretical outcome per group (from X's / checkpoint's perspective)
// ============================================================
enum class TheoreticalOutcome { X_WINS, DRAW };

static TheoreticalOutcome group_theory(int g) {
    switch (g) {
        case X_FIRST_X_WINS: case O_FIRST_X_WINS: return TheoreticalOutcome::X_WINS;
        default:                                   return TheoreticalOutcome::DRAW;
    }
}


// ============================================================
// Result for one checkpoint
// ============================================================
struct GroupStats {
    int wins   = 0;   // theory-matched wins (X won when theory says X wins)
    int draws  = 0;   // theory-matched draws
    int losses = 0;   // everything else
    // raw counts (from X's perspective)
    int raw_x_wins = 0;
    int raw_draws  = 0;
    int raw_o_wins = 0;
};

struct CheckpointElo {
    int    checkpoint    = 0;
    int    total_wins    = 0;
    int    total_draws   = 0;
    int    total_losses  = 0;
    double score         = 0.0;
    double elo           = 0.0;
    std::array<GroupStats, NUM_GROUPS> by_group{};
};


// ============================================================
// Evaluate one checkpoint against the solver
// ============================================================
static CheckpointElo evaluate_checkpoint(
    const std::string& checkpoints_dir,
    int gen,
    const std::string& queue_name,
    std::shared_ptr<SharedMemoryInferenceQueue> queue,
    std::shared_ptr<MisereSolver> solver,
    const std::vector<InitialBoard>& boards,
    int games_per_position_per_side,
    int num_workers)
{
    CheckpointElo result;
    result.checkpoint = gen;

    if (!reload_checkpoint(checkpoints_dir, queue_name, gen, queue))
        return result;

    const int num_boards = static_cast<int>(boards.size());

    // Build flat task list: one game per board per repetition
    // (checkpoint always X, solver always O — no role switching)
    std::vector<GameTask> tasks;
    tasks.reserve(static_cast<size_t>(num_boards) * games_per_position_per_side);
    for (int b = 0; b < num_boards; ++b)
        for (int g = 0; g < games_per_position_per_side; ++g)
            tasks.push_back({b});

    std::vector<AtomicCounts> board_results(num_boards);

    std::atomic<int> task_counter{0};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(num_workers));
    for (int i = 0; i < num_workers; ++i)
        workers.emplace_back(worker_fn, queue, solver,
                             std::cref(boards), std::cref(tasks),
                             std::ref(task_counter),
                             std::ref(board_results));
    for (auto& w : workers) w.join();

    // ---- Aggregate results ----------------------------------------
    // Score rule: checkpoint (X) earns a point only when the actual
    // outcome matches the theoretical value for the position.
    //   Theory X_WINS + actual X wins → W (full point)
    //   Theory DRAW   + actual draw   → D (half point)
    //   Everything else               → L (no point)
    int total_W = 0, total_D = 0, total_L = 0;
    for (int b = 0; b < num_boards; ++b) {
        const auto& r = board_results[b];
        int xw = r.x_wins.load();
        int dr = r.draws.load();
        int ow = r.o_wins.load();

        int g = b / BOARDS_PER_GROUP;
        TheoreticalOutcome theory = group_theory(g);

        int gw = 0, gd = 0, gl = 0;
        if (theory == TheoreticalOutcome::X_WINS) {
            // Checkpoint should win — X wins score, rest are losses
            gw = xw;
            gd = 0;
            gl = dr + ow;
        } else {
            // Checkpoint should draw — draws score, rest are losses
            gw = 0;
            gd = dr;
            gl = xw + ow;
        }

        total_W += gw;
        total_D += gd;
        total_L += gl;

        result.by_group[g].wins       += gw;
        result.by_group[g].draws      += gd;
        result.by_group[g].losses     += gl;
        result.by_group[g].raw_x_wins += xw;
        result.by_group[g].raw_draws  += dr;
        result.by_group[g].raw_o_wins += ow;
    }

    const int    total_N = total_W + total_D + total_L;
    const double s       = total_N > 0
                           ? (total_W + 0.5 * total_D) / static_cast<double>(total_N)
                           : 0.0;
    const double elo     = compute_elo(total_W, total_D, total_L);

    std::cout << "  W=" << total_W << "  D=" << total_D << "  L=" << total_L
              << "  Elo=" << std::fixed << std::setprecision(1) << elo << "\n";

    // Per-category breakdown
    std::cout << "  Per category (checkpoint=X, solver=O):\n";
    for (int g = 0; g < NUM_GROUPS; ++g) {
        const auto& gs = result.by_group[g];
        int n_score = gs.wins + gs.draws + gs.losses;
        std::cout << "    [" << GROUP_NAMES[g] << "]"
                  << "  raw(X=" << std::setw(3) << gs.raw_x_wins
                  << " D=" << std::setw(3) << gs.raw_draws
                  << " O=" << std::setw(3) << gs.raw_o_wins << ")";
        if (n_score > 0) {
            double cat_score = (gs.wins + 0.5 * gs.draws) / static_cast<double>(n_score) * 100.0;
            std::cout << "  theory_score=" << std::fixed << std::setprecision(1) << cat_score << "%";
        }
        std::cout << "\n";
    }

    result.total_wins = total_W;  result.total_draws = total_D;
    result.total_losses = total_L;  result.score = s;  result.elo = elo;
    return result;
}


// ============================================================
// main
// ============================================================
int main() {
    const std::string CHECKPOINTS_DIR    = "checkpoints/";
    const std::string QUEUE_NAME         = "mcts_jax_inference";
    const int         NUM_WORKERS        = 10;
    const int         GAMES_PER_POS_SIDE = 1;   // per (position, side) pair

    // ---- Pre-solve ------------------------------------------------
    auto solver = std::make_shared<MisereSolver>();
    auto t0 = std::chrono::high_resolution_clock::now();
    auto solve_result = solver->solve();
    solver->populate_all_positions();   // make TT thread-safe (read-only)
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Solver: " << std::fixed << std::setprecision(0)
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    // ---- Discover checkpoints -------------------------------------
    std::vector<int> checkpoints = discover_checkpoints(CHECKPOINTS_DIR);
    if (checkpoints.empty()) {
        std::cerr << "No checkpoints found\n";
        return 1;
    }

    // ---- Build flat board list from all groups ----------------------
    std::vector<InitialBoard> boards;
    boards.reserve(TOTAL_BOARDS);
    for (int g = 0; g < NUM_GROUPS; ++g)
        for (int b = 0; b < BOARDS_PER_GROUP; ++b)
            boards.push_back(ALL_GROUPS[g][b]);

    std::cout << "Checkpoints: " << checkpoints.size()
              << "  Positions: " << boards.size()
              << "  Workers: " << NUM_WORKERS << "\n";

    // ---- Connect to inference queue --------------------------------
    auto queue = std::make_shared<SharedMemoryInferenceQueue>("/" + QUEUE_NAME);
    if (!queue->wait_for_server(30'000)) {
        std::cerr << "Inference server not ready\n";
        return 1;
    }
    std::cout << "\n";

    // ---- CSV output -----------------------------------------------
    std::ofstream csv("elo_results.csv");
    if (!csv.is_open()) {
        std::cerr << "Failed to open elo_results.csv.\n";
        return 1;
    }
    // W/D/L are theory-matched counts; raw_X/raw_D/raw_O are actual game outcomes
    csv << "checkpoint,theory_wins,theory_draws,theory_losses,total_games,score,elo,"
           "xfirst_xwins_W,xfirst_xwins_D,xfirst_xwins_L,xfirst_xwins_rawX,xfirst_xwins_rawD,xfirst_xwins_rawO,"
           "ofirst_xwins_W,ofirst_xwins_D,ofirst_xwins_L,ofirst_xwins_rawX,ofirst_xwins_rawD,ofirst_xwins_rawO,"
           "xfirst_draw_W,xfirst_draw_D,xfirst_draw_L,xfirst_draw_rawX,xfirst_draw_rawD,xfirst_draw_rawO,"
           "ofirst_draw_W,ofirst_draw_D,ofirst_draw_L,ofirst_draw_rawX,ofirst_draw_rawD,ofirst_draw_rawO\n";

    // ---- Evaluate -------------------------------------------------
    std::vector<CheckpointElo> all_results;
    all_results.reserve(checkpoints.size());

    for (size_t idx = 0; idx < checkpoints.size(); ++idx) {
        int gen = checkpoints[idx];
        std::cout << "[" << (idx + 1) << "/" << checkpoints.size() << "] ";

        CheckpointElo r = evaluate_checkpoint(
            CHECKPOINTS_DIR, gen, QUEUE_NAME, queue, solver,
            boards, GAMES_PER_POS_SIDE, NUM_WORKERS);

        all_results.push_back(r);

        const int n = r.total_wins + r.total_draws + r.total_losses;
        csv << r.checkpoint << ","
            << r.total_wins << "," << r.total_draws << "," << r.total_losses << ","
            << n << ","
            << std::fixed << std::setprecision(6) << r.score << ","
            << std::setprecision(2) << r.elo;
        for (int g = 0; g < NUM_GROUPS; ++g)
            csv << "," << r.by_group[g].wins
                << "," << r.by_group[g].draws
                << "," << r.by_group[g].losses
                << "," << r.by_group[g].raw_x_wins
                << "," << r.by_group[g].raw_draws
                << "," << r.by_group[g].raw_o_wins;
        csv << "\n";
        csv.flush();
    }
    csv.close();

    // ---- Summary --------------------------------------------------
    std::cout << "\nCheckpoint  W    D    L    Elo\n";
    for (const auto& r : all_results) {
        std::cout << std::setw(10) << r.checkpoint
                  << std::setw(5) << r.total_wins
                  << std::setw(5) << r.total_draws
                  << std::setw(5) << r.total_losses
                  << std::setw(8) << std::fixed << std::setprecision(1) << r.elo << "\n";
    }
    std::cout << "\nResults: elo_results.csv\n";
    return 0;
}
