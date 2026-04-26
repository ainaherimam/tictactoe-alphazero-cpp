// eval_against_solver.cpp
// Detailed evaluation: every checkpoint in checkpoints/ vs the perfect
// Misère Negamax solver.
//
// Same structure as eval_elo.cpp (checkpoint=X, solver=O, 4 groups of
// 100 pre-classified boards), but with two extras:
//
//   1. Every AZ move is analyzed by the solver in parallel to play:
//      solver->get_action_values() returns the exact game-theoretic
//      value of every legal action. If the AZ move's value is below
//      the optimum, it is recorded as a BLUNDER (with severity).
//
//   2. Every game is written to disk as an annotated PGN-like file:
//        eval_games/checkpoint_<N>/P1_win_theoretical_board_start/*.pgn
//        eval_games/checkpoint_<N>/draw_theoretical_board_start/*.pgn
//      Blunder moves are tagged (??) and a "=== AZ Blunders ===" section
//      lists each blunder's position, chosen move, optimal move(s), and
//      the value drop it caused.
//
// Aggregate output: solver_eval_results.csv — per-checkpoint Elo plus
// blunder statistics.

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
#include <sstream>
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
#include <climits>

namespace fs = std::filesystem;


// ============================================================
// SolverPlayer — wraps MisereSolver as a Player
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
// Elo formula (same as eval_elo.cpp)
// ============================================================
static double compute_elo(int wins, int draws, int losses) {
    int n = wins + draws + losses;
    if (n == 0) return 0.0;
    double s = (wins + 0.5 * draws) / static_cast<double>(n);
    s = std::max(1e-6, std::min(1.0 - 1e-6, s));
    return 400.0 * std::log10(s / (1.0 - s));
}


// ============================================================
// Group names & theoretical outcome per group
// ============================================================
static constexpr const char* GROUP_NAMES[NUM_GROUPS] = {
    "X_first_X_wins",
    "O_first_X_wins",
    "X_first_Draw",
    "O_first_Draw",
};

enum class TheoreticalOutcome { X_WINS, DRAW };

static TheoreticalOutcome group_theory(int g) {
    switch (g) {
        case X_FIRST_X_WINS: case O_FIRST_X_WINS: return TheoreticalOutcome::X_WINS;
        default:                                   return TheoreticalOutcome::DRAW;
    }
}

static const char* theory_folder(int g) {
    return group_theory(g) == TheoreticalOutcome::X_WINS
        ? "P1_win_theoretical_board_start"
        : "draw_theoretical_board_start";
}


// ============================================================
// Blunder record (one per AZ move that was sub-optimal)
// ============================================================
struct BlunderInfo {
    int      move_number;     // 1-indexed (game move count)
    uint16_t bx_before;       // position right before the blunder
    uint16_t bo_before;
    int      chosen_cell;     // 0..15 — cell AZ actually played
    int      chosen_value;    // game-theoretic value of the chosen move (X's POV)
    int      best_value;      // best achievable value from this position
    std::vector<int> best_cells; // all cells that achieve best_value
};


// ============================================================
// Per-game record — produced by play_and_analyze, consumed by
// write_game_pgn and the aggregate statistics block.
// ============================================================
struct GameRecord {
    int         board_idx     = 0;
    int         group         = 0;
    uint16_t    init_bx       = 0;
    uint16_t    init_bo       = 0;
    Cell_state  winner        = Cell_state::Empty;
    std::string moves;        // "1.4a 2.3b??  3.2c ..."   (?? marks AZ blunders)
    std::vector<BlunderInfo> az_blunders;
    int         az_moves_analyzed = 0;
};


// ============================================================
// Cell -> "RC" coord string ("4a", "3b", ...), matches game.cpp
// Row index is BOARD_SIZE - x (bottom-up), column is 'a' + y.
// ============================================================
static std::string cell_to_coord(int cell) {
    int x = cell / BOARD_WIDTH;
    int y = cell % BOARD_WIDTH;
    std::string s;
    s += char('0' + (BOARD_SIZE - x));
    s += char('a' + y);
    return s;
}


// ============================================================
// Draw a 4x4 board from a (bx, bo) bitboard pair to a stream.
// ============================================================
static void draw_bitboards(std::ostream& os, uint16_t bx, uint16_t bo) {
    os << "    A   B   C   D\n";
    for (int r = 0; r < BOARD_SIZE; ++r) {
        os << (BOARD_SIZE - r) << "   ";
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int bit = r * BOARD_WIDTH + c;
            char ch = '.';
            if (bx & (1u << bit)) ch = 'X';
            else if (bo & (1u << bit)) ch = 'O';
            os << ch;
            if (c < BOARD_SIZE - 1) os << " | ";
        }
        os << "\n";
        if (r < BOARD_SIZE - 1) {
            os << "    ---+---+---+---\n";
        }
    }
}


// ============================================================
// Play one game + analyze every AZ (X) move against the solver.
//
// For every AZ move: call solver->get_action_values() on the
// position right before the move. Compare the value of the chosen
// cell against the best achievable value — if lower, record a
// BlunderInfo (with severity = best_value - chosen_value).
//
// Mirrors eval_elo.cpp's play_from_board but also records the full
// move history and blunder data into the returned GameRecord.
// ============================================================
static GameRecord play_and_analyze(
    const InitialBoard& init,
    int board_idx,
    int group,
    std::shared_ptr<SharedMemoryInferenceQueue> queue,
    std::shared_ptr<MisereSolver> solver)
{
    GameRecord rec;
    rec.board_idx = board_idx;
    rec.group     = group;
    rec.init_bx   = init.bx;
    rec.init_bo   = init.bo;

    Board board;
    auto cells = bitboards_to_cells(init.bx, init.bo);
    board.load_board(cells);

    // Guard: already terminal before we start
    Cell_state early = board.check_winner();
    if (early != Cell_state::Empty) { rec.winner = early; return rec; }

    Cell_state current = first_mover_of(init.bx, init.bo);

    // Eval config: no Dirichlet noise, greedy (temperature = 0)
    Mcts_config cfg(
        /*c_puct         =*/ 2.5,
        /*iterations     =*/ 100,
        LogLevel::NONE,
        /*temperature    =*/ 0.0f,
        /*dirichlet_alpha=*/ 0.3f,
        /*dirichlet_eps  =*/ 0.0f,
        queue,
        /*max_depth      =*/ 12,
        /*tree_reuse     =*/ false,
        /*model_id       =*/ 0);

    Mcts_player_selfplay mcts(cfg);
    SolverPlayer         sol(solver);

    int move_num = 0;

    for (;;) {
        Cell_state w = board.check_winner();
        if (w != Cell_state::Empty) { rec.winner = w; break; }

        auto valid = board.get_valid_moves(current);
        if (valid.empty()) { rec.winner = Cell_state::Empty; break; }

        // Extract bitboards BEFORE this move (needed for solver analysis)
        uint16_t bx_before = 0, bo_before = 0;
        {
            float planes[INPUT_SIZE];
            board.to_float_array(Cell_state::X, planes);
            for (int i = 0; i < BOARD_CELLS; ++i) {
                if (planes[i]             > 0.5f) bx_before |= static_cast<uint16_t>(1u << i);
                if (planes[BOARD_CELLS+i] > 0.5f) bo_before |= static_cast<uint16_t>(1u << i);
            }
        }

        bool is_az_move = (current == Cell_state::X);

        Move chosen;
        if (is_az_move)
            chosen = mcts.choose_move(board, current).first;
        else
            chosen = sol.choose_move(board, current).first;

        int chosen_cell = chosen.x * BOARD_WIDTH + chosen.y;

        // ---- Solver co-analysis: only on AZ's turn -----------------
        bool was_blunder = false;
        if (is_az_move) {
            auto action_vals = solver->get_action_values(bx_before, bo_before, /*is_x_turn=*/true);

            if (!action_vals.empty()) {
                int best_val    = INT_MIN;
                int chosen_val  = INT_MIN;
                for (const auto& av : action_vals) {
                    if (av.value > best_val) best_val = av.value;
                    if (av.cell == chosen_cell) chosen_val = av.value;
                }

                std::vector<int> best_cells;
                for (const auto& av : action_vals)
                    if (av.value == best_val) best_cells.push_back(av.cell);

                rec.az_moves_analyzed++;

                if (chosen_val < best_val) {
                    was_blunder = true;
                    BlunderInfo bi;
                    bi.move_number   = move_num + 1;
                    bi.bx_before     = bx_before;
                    bi.bo_before     = bo_before;
                    bi.chosen_cell   = chosen_cell;
                    bi.chosen_value  = chosen_val;
                    bi.best_value    = best_val;
                    bi.best_cells    = std::move(best_cells);
                    rec.az_blunders.push_back(std::move(bi));
                }
            }
        }

        // ---- Append to move history, matching game.cpp's format ----
        rec.moves += std::to_string(move_num + 1) + "."
                   + cell_to_coord(chosen_cell);
        if (was_blunder) rec.moves += "??";
        rec.moves += " ";

        board.make_move(chosen, current);
        current = (current == Cell_state::X) ? Cell_state::O : Cell_state::X;
        ++move_num;
    }

    return rec;
}


// ============================================================
// Write one annotated PGN file (mirrors main.cpp/GameLogger's
// layout with extra headers + a per-blunder analysis section).
// ============================================================
static void write_game_pgn(const std::string& path, const GameRecord& rec) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[WARNING] cannot open " << path << " for writing\n";
        return;
    }

    const std::string result =
        (rec.winner == Cell_state::X) ? "1-0" :
        (rec.winner == Cell_state::O) ? "0-1" : "1/2-1/2";

    const std::string actual =
        (rec.winner == Cell_state::X) ? "X_wins" :
        (rec.winner == Cell_state::O) ? "O_wins" : "draw";

    const std::string theory =
        group_theory(rec.group) == TheoreticalOutcome::X_WINS ? "X_wins" : "draw";

    const std::string first =
        first_mover_of(rec.init_bx, rec.init_bo) == Cell_state::X ? "X" : "O";

    f << "[Event \"AZ vs Solver (detailed)\"]\n"
      << "[X \"Checkpoint (AlphaZero)\"]\n"
      << "[O \"Perfect Solver\"]\n"
      << "[Group \"" << GROUP_NAMES[rec.group] << "\"]\n"
      << "[FirstMover \"" << first << "\"]\n"
      << "[Theory \"" << theory << "\"]\n"
      << "[Actual \"" << actual << "\"]\n"
      << "[InitialBx \"0x" << std::hex << rec.init_bx << std::dec << "\"]\n"
      << "[InitialBo \"0x" << std::hex << rec.init_bo << std::dec << "\"]\n"
      << "[AzMovesAnalyzed \"" << rec.az_moves_analyzed << "\"]\n"
      << "[AzBlunders \"" << rec.az_blunders.size() << "\"]\n"
      << result << "\n\n";

    f << "Initial board:\n";
    draw_bitboards(f, rec.init_bx, rec.init_bo);
    f << "\n";

    f << rec.moves << "\n";

    if (!rec.az_blunders.empty()) {
        f << "\n=== AZ Blunders ===\n"
          << "(Values are from X's perspective: +1 win, 0 draw, -1 loss)\n\n";
        for (const auto& b : rec.az_blunders) {
            std::ostringstream best;
            for (size_t k = 0; k < b.best_cells.size(); ++k) {
                if (k) best << ",";
                best << cell_to_coord(b.best_cells[k]);
            }

            // Severity label
            const char* label = "blunder";
            if      (b.best_value == +1 && b.chosen_value <= -1) label = "WIN -> LOSS";
            else if (b.best_value == +1 && b.chosen_value ==  0) label = "WIN -> DRAW";
            else if (b.best_value ==  0 && b.chosen_value <= -1) label = "DRAW -> LOSS";

            f << "Move " << b.move_number
              << "  (" << label << ")"
              << ": played " << cell_to_coord(b.chosen_cell)
              << " value=" << std::showpos << b.chosen_value << std::noshowpos
              << "  best=" << std::showpos << b.best_value << std::noshowpos
              << " via [" << best.str() << "]\n"
              << "Position before move:\n";
            draw_bitboards(f, b.bx_before, b.bo_before);
            f << "\n";
        }
    }

    f.close();
}


// ============================================================
// Atomic aggregates — per-board outcome counts + per-checkpoint
// blunder tallies (one instance per worker-shared bag).
// ============================================================
struct AtomicCounts {
    std::atomic<int> x_wins{0};
    std::atomic<int> draws{0};
    std::atomic<int> o_wins{0};
};

struct AtomicBlunderStats {
    std::atomic<int> az_moves_analyzed{0};
    std::atomic<int> total_blunders{0};
    std::atomic<int> games_with_blunder{0};
    std::atomic<int> win_to_draw{0};   // best=+1, chosen=0
    std::atomic<int> win_to_loss{0};   // best=+1, chosen=-1
    std::atomic<int> draw_to_loss{0};  // best= 0, chosen=-1
};


// ============================================================
// One game task. (board_idx tells the worker which board to use.)
// ============================================================
struct GameTask {
    int board_idx;
};


// ============================================================
// Worker thread — same task-queue pattern as eval_elo, extended
// to write per-game PGNs and update blunder stats.
// ============================================================
static void worker_fn(
    std::shared_ptr<SharedMemoryInferenceQueue>  queue,
    std::shared_ptr<MisereSolver>                solver,
    const std::vector<InitialBoard>&             boards,
    const std::vector<GameTask>&                 tasks,
    std::atomic<int>&                            task_counter,
    std::vector<AtomicCounts>&                   board_results,
    AtomicBlunderStats&                          blunder_stats,
    const std::string&                           ckpt_games_dir,
    std::atomic<uint64_t>&                       game_id_counter)
{
    while (true) {
        int i = task_counter.fetch_add(1, std::memory_order_relaxed);
        if (i >= static_cast<int>(tasks.size())) break;

        const GameTask&     t    = tasks[i];
        const InitialBoard& init = boards[t.board_idx];
        int group = t.board_idx / BOARDS_PER_GROUP;

        GameRecord rec = play_and_analyze(init, t.board_idx, group, queue, solver);

        // ---- Update win/draw/loss counts for this board -----------
        AtomicCounts& cnt = board_results[t.board_idx];
        if      (rec.winner == Cell_state::X)     cnt.x_wins.fetch_add(1, std::memory_order_relaxed);
        else if (rec.winner == Cell_state::Empty) cnt.draws .fetch_add(1, std::memory_order_relaxed);
        else                                      cnt.o_wins.fetch_add(1, std::memory_order_relaxed);

        // ---- Update blunder aggregates ----------------------------
        blunder_stats.az_moves_analyzed.fetch_add(rec.az_moves_analyzed, std::memory_order_relaxed);
        blunder_stats.total_blunders   .fetch_add(static_cast<int>(rec.az_blunders.size()),
                                                  std::memory_order_relaxed);
        if (!rec.az_blunders.empty())
            blunder_stats.games_with_blunder.fetch_add(1, std::memory_order_relaxed);
        for (const auto& b : rec.az_blunders) {
            if      (b.best_value == +1 && b.chosen_value ==  0) blunder_stats.win_to_draw .fetch_add(1, std::memory_order_relaxed);
            else if (b.best_value == +1 && b.chosen_value <= -1) blunder_stats.win_to_loss .fetch_add(1, std::memory_order_relaxed);
            else if (b.best_value ==  0 && b.chosen_value <= -1) blunder_stats.draw_to_loss.fetch_add(1, std::memory_order_relaxed);
        }

        // ---- Write the game file ----------------------------------
        uint64_t gid = game_id_counter.fetch_add(1, std::memory_order_relaxed);
        std::ostringstream fname;
        fname << ckpt_games_dir << "/" << theory_folder(group) << "/game_"
              << std::setfill('0') << std::setw(6) << gid
              << "_" << GROUP_NAMES[group]
              << "_board" << std::setw(3) << std::setfill('0') << t.board_idx
              << "_" << (rec.winner == Cell_state::X ? "Xwon"  :
                         rec.winner == Cell_state::O ? "Owon"  : "draw")
              << ".pgn";
        write_game_pgn(fname.str(), rec);
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
// Checkpoint reload trigger (same as eval_elo)
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
// Per-checkpoint aggregated result (reused for stdout + CSV)
// ============================================================
struct GroupStats {
    int wins   = 0;
    int draws  = 0;
    int losses = 0;
    int raw_x_wins = 0;
    int raw_draws  = 0;
    int raw_o_wins = 0;
};

struct CheckpointResult {
    int    checkpoint    = 0;
    int    total_wins    = 0;
    int    total_draws   = 0;
    int    total_losses  = 0;
    double score         = 0.0;
    double elo           = 0.0;
    std::array<GroupStats, NUM_GROUPS> by_group{};

    // Blunder stats (checkpoint-wide)
    int az_moves_analyzed = 0;
    int total_blunders    = 0;
    int games_with_blunder = 0;
    int win_to_draw       = 0;
    int win_to_loss       = 0;
    int draw_to_loss      = 0;
};


// ============================================================
// Evaluate one checkpoint — full per-game logging + blunder stats.
// ============================================================
static CheckpointResult evaluate_checkpoint(
    const std::string& checkpoints_dir,
    int gen,
    const std::string& queue_name,
    std::shared_ptr<SharedMemoryInferenceQueue> queue,
    std::shared_ptr<MisereSolver> solver,
    const std::vector<InitialBoard>& boards,
    int games_per_position,
    int num_workers,
    const std::string& eval_games_root)
{
    CheckpointResult result;
    result.checkpoint = gen;

    if (!reload_checkpoint(checkpoints_dir, queue_name, gen, queue))
        return result;

    const int num_boards = static_cast<int>(boards.size());

    // ---- Create per-checkpoint output directory tree ------------
    const std::string ckpt_name = "checkpoint_" + std::to_string(gen);
    const std::string ckpt_dir  = eval_games_root + "/" + ckpt_name;
    fs::create_directories(ckpt_dir + "/P1_win_theoretical_board_start");
    fs::create_directories(ckpt_dir + "/draw_theoretical_board_start");

    // ---- Build flat task list -----------------------------------
    std::vector<GameTask> tasks;
    tasks.reserve(static_cast<size_t>(num_boards) * games_per_position);
    for (int b = 0; b < num_boards; ++b)
        for (int g = 0; g < games_per_position; ++g)
            tasks.push_back({b});

    std::vector<AtomicCounts>   board_results(num_boards);
    AtomicBlunderStats          blunder_stats;
    std::atomic<int>            task_counter{0};
    std::atomic<uint64_t>       game_id_counter{1};

    // ---- Launch workers -----------------------------------------
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(num_workers));
    for (int i = 0; i < num_workers; ++i)
        workers.emplace_back(worker_fn, queue, solver,
                             std::cref(boards), std::cref(tasks),
                             std::ref(task_counter),
                             std::ref(board_results),
                             std::ref(blunder_stats),
                             std::cref(ckpt_dir),
                             std::ref(game_id_counter));
    for (auto& w : workers) w.join();

    // ---- Aggregate W/D/L (theory-matched) ------------------------
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
            gw = xw;
            gd = 0;
            gl = dr + ow;
        } else {
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

    result.total_wins    = total_W;
    result.total_draws   = total_D;
    result.total_losses  = total_L;
    result.score         = s;
    result.elo           = elo;

    result.az_moves_analyzed  = blunder_stats.az_moves_analyzed.load();
    result.total_blunders     = blunder_stats.total_blunders.load();
    result.games_with_blunder = blunder_stats.games_with_blunder.load();
    result.win_to_draw        = blunder_stats.win_to_draw.load();
    result.win_to_loss        = blunder_stats.win_to_loss.load();
    result.draw_to_loss       = blunder_stats.draw_to_loss.load();

    // ---- Stdout summary -----------------------------------------
    std::cout << "  W=" << total_W << "  D=" << total_D << "  L=" << total_L
              << "  Elo=" << std::fixed << std::setprecision(1) << elo << "\n";

    std::cout << "  Per category (checkpoint=X, solver=O):\n";
    for (int g = 0; g < NUM_GROUPS; ++g) {
        const auto& gs = result.by_group[g];
        int n_score = gs.wins + gs.draws + gs.losses;
        std::cout << "    [" << std::left << std::setw(15) << GROUP_NAMES[g] << std::right << "]"
                  << "  raw(X=" << std::setw(3) << gs.raw_x_wins
                  << " D="      << std::setw(3) << gs.raw_draws
                  << " O="      << std::setw(3) << gs.raw_o_wins << ")";
        if (n_score > 0) {
            double cat_score = (gs.wins + 0.5 * gs.draws) / static_cast<double>(n_score) * 100.0;
            std::cout << "  theory_score=" << std::fixed << std::setprecision(1) << cat_score << "%";
        }
        std::cout << "\n";
    }

    double blunder_rate = result.az_moves_analyzed > 0
        ? 100.0 * result.total_blunders / result.az_moves_analyzed
        : 0.0;
    std::cout << "  AZ moves analyzed: " << result.az_moves_analyzed
              << "  blunders: "            << result.total_blunders
              << " (" << std::fixed << std::setprecision(2) << blunder_rate << "%)"
              << "  games-with-blunder: " << result.games_with_blunder << "\n"
              << "  severity breakdown:"
              << "  WIN->DRAW=" << result.win_to_draw
              << "  WIN->LOSS=" << result.win_to_loss
              << "  DRAW->LOSS=" << result.draw_to_loss << "\n"
              << "  Games written to: " << ckpt_dir << "/\n";

    return result;
}


// ============================================================
// main
// ============================================================
int main() {
    const std::string CHECKPOINTS_DIR = "checkpoints/";
    const std::string QUEUE_NAME      = "mcts_jax_inference";
    const std::string EVAL_GAMES_DIR  = "eval_games";
    const int         NUM_WORKERS     = 10;
    const int         GAMES_PER_POS   = 1;     // one game per initial board

    // ---- Pre-solve ----------------------------------------------
    auto solver = std::make_shared<MisereSolver>();
    auto t0 = std::chrono::high_resolution_clock::now();
    solver->solve();
    solver->populate_all_positions();    // make TT read-only / thread-safe
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Solver: " << std::fixed << std::setprecision(0)
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    // ---- Discover checkpoints -----------------------------------
    std::vector<int> checkpoints = discover_checkpoints(CHECKPOINTS_DIR);
    if (checkpoints.empty()) {
        std::cerr << "No checkpoints found\n";
        return 1;
    }

    // ---- Build flat board list from all 4 groups ----------------
    std::vector<InitialBoard> boards;
    boards.reserve(TOTAL_BOARDS);
    for (int g = 0; g < NUM_GROUPS; ++g)
        for (int b = 0; b < BOARDS_PER_GROUP; ++b)
            boards.push_back(ALL_GROUPS[g][b]);

    std::cout << "Checkpoints: " << checkpoints.size()
              << "  Positions: " << boards.size()
              << "  Workers: "   << NUM_WORKERS << "\n";

    // ---- Connect to inference queue -----------------------------
    auto queue = std::make_shared<SharedMemoryInferenceQueue>("/" + QUEUE_NAME);
    if (!queue->wait_for_server(30'000)) {
        std::cerr << "Inference server not ready\n";
        return 1;
    }

    // ---- Prepare eval_games/ root -------------------------------
    fs::create_directories(EVAL_GAMES_DIR);
    std::cout << "Games output root: " << EVAL_GAMES_DIR << "/\n\n";

    // ---- CSV output ---------------------------------------------
    std::ofstream csv("solver_eval_results.csv");
    if (!csv.is_open()) {
        std::cerr << "Failed to open solver_eval_results.csv.\n";
        return 1;
    }
    csv << "checkpoint,theory_wins,theory_draws,theory_losses,total_games,score,elo,"
           "az_moves_analyzed,total_blunders,blunder_rate_pct,games_with_blunder,"
           "win_to_draw,win_to_loss,draw_to_loss,"
           "xfirst_xwins_W,xfirst_xwins_D,xfirst_xwins_L,xfirst_xwins_rawX,xfirst_xwins_rawD,xfirst_xwins_rawO,"
           "ofirst_xwins_W,ofirst_xwins_D,ofirst_xwins_L,ofirst_xwins_rawX,ofirst_xwins_rawD,ofirst_xwins_rawO,"
           "xfirst_draw_W,xfirst_draw_D,xfirst_draw_L,xfirst_draw_rawX,xfirst_draw_rawD,xfirst_draw_rawO,"
           "ofirst_draw_W,ofirst_draw_D,ofirst_draw_L,ofirst_draw_rawX,ofirst_draw_rawD,ofirst_draw_rawO\n";

    // ---- Evaluate each checkpoint -------------------------------
    std::vector<CheckpointResult> all_results;
    all_results.reserve(checkpoints.size());

    for (size_t idx = 0; idx < checkpoints.size(); ++idx) {
        int gen = checkpoints[idx];
        std::cout << "[" << (idx + 1) << "/" << checkpoints.size() << "] checkpoint "
                  << gen << "\n";

        CheckpointResult r = evaluate_checkpoint(
            CHECKPOINTS_DIR, gen, QUEUE_NAME, queue, solver,
            boards, GAMES_PER_POS, NUM_WORKERS, EVAL_GAMES_DIR);

        all_results.push_back(r);

        const int n = r.total_wins + r.total_draws + r.total_losses;
        double blunder_rate = r.az_moves_analyzed > 0
            ? 100.0 * r.total_blunders / r.az_moves_analyzed : 0.0;

        csv << r.checkpoint << ","
            << r.total_wins << "," << r.total_draws << "," << r.total_losses << ","
            << n << ","
            << std::fixed << std::setprecision(6) << r.score << ","
            << std::setprecision(2) << r.elo << ","
            << r.az_moves_analyzed << ","
            << r.total_blunders   << ","
            << std::setprecision(4) << blunder_rate << ","
            << r.games_with_blunder << ","
            << r.win_to_draw << "," << r.win_to_loss << "," << r.draw_to_loss;
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

    // ---- Summary ------------------------------------------------
    std::cout << "\nCheckpoint    W    D    L     Elo  Blunders(rate)\n";
    for (const auto& r : all_results) {
        double rate = r.az_moves_analyzed > 0
            ? 100.0 * r.total_blunders / r.az_moves_analyzed : 0.0;
        std::cout << std::setw(10) << r.checkpoint
                  << std::setw(5)  << r.total_wins
                  << std::setw(5)  << r.total_draws
                  << std::setw(5)  << r.total_losses
                  << std::setw(8)  << std::fixed << std::setprecision(1) << r.elo
                  << std::setw(8)  << r.total_blunders
                  << " ("          << std::fixed << std::setprecision(2) << rate << "%)\n";
    }
    std::cout << "\nResults: solver_eval_results.csv\n";
    std::cout << "Game files: " << EVAL_GAMES_DIR << "/checkpoint_<N>/{P1_win,draw}_theoretical_board_start/\n";
    return 0;
}
