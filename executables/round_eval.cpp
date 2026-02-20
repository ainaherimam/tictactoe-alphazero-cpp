#include "core/game/game.h"
#include "core/mcts/position_pool.h"
#include "core/game/player.h"
#include <sys/stat.h>
#include "core/utils/game_logger.h"
#include "core/game/cell_state.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <thread>
#include <vector>
#include "core/game/constants.h"
#include <chrono>
#include <atomic>
#include <mutex>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <sstream>
#include <ctime>
#include <string>
#include <stdexcept>

namespace fs = std::filesystem;


// ---------------------------------------------------------------------------
// EvaluationResults
// ---------------------------------------------------------------------------
struct EvaluationResults {
    std::atomic<int> best_as_p1_wins{0};
    std::atomic<int> best_as_p1_losses{0};
    std::atomic<int> best_as_p1_draws{0};

    std::atomic<int> best_as_p2_wins{0};
    std::atomic<int> best_as_p2_losses{0};
    std::atomic<int> best_as_p2_draws{0};

    std::mutex cout_mutex;
};

// Winrate result for a single matchup
struct MatchupResult {
    int   checkpoint_a;
    int   checkpoint_b;

    int    a_wins    = 0;
    int    b_wins    = 0;
    int    draws     = 0;
    double b_winrate = 0.0;

    int   as_p1_a_wins = 0;
    int   as_p1_b_wins = 0;
    int   as_p1_draws  = 0;

    int   as_p2_a_wins = 0;
    int   as_p2_b_wins = 0;
    int   as_p2_draws  = 0;
};


// ---------------------------------------------------------------------------
// EvalMode — selects which evaluation strategy to run
// ---------------------------------------------------------------------------
enum class EvalMode {
    FULL_ROUND_ROBIN,   // every pair (i, j) with i < j
    CONSECUTIVE,        // only (i, i+1) pairs
    PROGRESSIVE         // best=ckpt[0]; each next ckpt is candidate;
                        // if candidate >55% winrate it becomes the new best
};

static std::string eval_mode_name(EvalMode mode) {
    switch (mode) {
        case EvalMode::FULL_ROUND_ROBIN: return "Full Round-Robin";
        case EvalMode::CONSECUTIVE:      return "Consecutive";
        case EvalMode::PROGRESSIVE:      return "Progressive";
    }
    return "Unknown";
}


// ---------------------------------------------------------------------------
// RoundRobinLogger
//   One line per matchup:
//   ckptA_vs_ckptB_winsA_winsB_winrateA%_winrateB%
// ---------------------------------------------------------------------------
class RoundRobinLogger {
public:
    explicit RoundRobinLogger(const std::string& log_path) : log_path_(log_path) {
        file_.open(log_path, std::ios::out | std::ios::trunc);
        if (!file_.is_open())
            std::cerr << "[Logger] WARNING: could not open log file: " << log_path << "\n";
    }

    ~RoundRobinLogger() { if (file_.is_open()) file_.close(); }

    void write_header(const std::vector<int>&, int, int) {}

    void write_matchup(const MatchupResult& r, int, int, int, int, int, int, int, int) {
        int decisive = r.a_wins + r.b_wins;
        double wr_a  = decisive > 0 ? (static_cast<double>(r.a_wins) / decisive) * 100.0 : 0.0;
        double wr_b  = decisive > 0 ? (static_cast<double>(r.b_wins) / decisive) * 100.0 : 0.0;

        std::ostringstream line;
        line << r.checkpoint_a << "_vs_" << r.checkpoint_b
             << "_" << r.a_wins
             << "_" << r.b_wins
             << "_" << std::fixed << std::setprecision(1) << wr_a << "%"
             << "_" << std::fixed << std::setprecision(1) << wr_b << "%";

        std::cout << line.str() << "\n";
        if (file_.is_open()) { file_ << line.str() << "\n"; file_.flush(); }
    }

    void write_summary(const std::vector<int>&,
                       const std::vector<std::vector<double>>&,
                       const std::vector<std::pair<double,int>>&) {}

private:
    std::string   log_path_;
    std::ofstream file_;
};


// ---------------------------------------------------------------------------
// Worker threads
// ---------------------------------------------------------------------------
void eval_worker_best_as_p1(int worker_id,
                             std::shared_ptr<SharedMemoryInferenceQueue> queue_best,
                             std::shared_ptr<SharedMemoryInferenceQueue> queue_candidate,
                             std::shared_ptr<EvaluationResults> results,
                             int games_per_worker)
{
    PositionPool dummy_pool(100);

    for (int game_num = 0; game_num < games_per_worker; ++game_num) {
        Mcts_config config_best     (1.4, 100, LogLevel::NONE, 0.0, 0.3, 0.0, queue_best,      -1, false, 0);
        Mcts_config config_candidate(1.4, 100, LogLevel::NONE, 0.0, 0.3, 0.0, queue_candidate, -1, false, 0);

        auto player1 = std::make_unique<Mcts_player_selfplay>(config_best);
        auto player2 = std::make_unique<Mcts_player_selfplay>(config_candidate);

        Game game(std::move(player1), std::move(player2), dummy_pool, false);
        Cell_state winner = game.play();

        if      (winner == Cell_state::X) results->best_as_p1_wins++;
        else if (winner == Cell_state::O) results->best_as_p1_losses++;
        else                              results->best_as_p1_draws++;

        dummy_pool.reset();
    }
}

void eval_worker_best_as_p2(int worker_id,
                             std::shared_ptr<SharedMemoryInferenceQueue> queue_best,
                             std::shared_ptr<SharedMemoryInferenceQueue> queue_candidate,
                             std::shared_ptr<EvaluationResults> results,
                             int games_per_worker)
{
    PositionPool dummy_pool(100);

    for (int game_num = 0; game_num < games_per_worker; ++game_num) {
        Mcts_config config_candidate(1.4, 100, LogLevel::NONE, 0.0, 0.3, 0.0, queue_candidate, -1, false, 0);
        Mcts_config config_best     (1.4, 100, LogLevel::NONE, 0.0, 0.3, 0.0, queue_best,      -1, false, 0);

        auto player1 = std::make_unique<Mcts_player_selfplay>(config_candidate);
        auto player2 = std::make_unique<Mcts_player_selfplay>(config_best);

        Game game(std::move(player1), std::move(player2), dummy_pool, false);
        Cell_state winner = game.play();

        if      (winner == Cell_state::O) results->best_as_p2_wins++;
        else if (winner == Cell_state::X) results->best_as_p2_losses++;
        else                              results->best_as_p2_draws++;

        dummy_pool.reset();
    }
}


// ---------------------------------------------------------------------------
// write_trigger
// ---------------------------------------------------------------------------
static void write_trigger(const std::string& path, int gen) {
    std::ofstream f(path);
    if (f.is_open()) {
        f << "Trigger for generation " << gen << "\n";
        f.close();
        std::cout << "  Created trigger file: " << path << "\n";
    } else {
        std::cerr << "  Failed to create trigger file: " << path << "\n";
    }
}


// ---------------------------------------------------------------------------
// reload_checkpoint
// ---------------------------------------------------------------------------
static bool reload_checkpoint(const std::string& checkpoints_dir,
                               const std::string& queue_name,
                               int checkpoint_gen,
                               std::shared_ptr<SharedMemoryInferenceQueue> queue,
                               int timeout_ms = 60000)
{
    std::string trigger_path = checkpoints_dir +
        "reload." + queue_name + "." + std::to_string(checkpoint_gen) + ".trigger";

    write_trigger(trigger_path, checkpoint_gen);

    std::cout << "  Waiting for " << queue_name
              << " to reload checkpoint " << checkpoint_gen << "...\n";

    if (!queue->wait_for_server(timeout_ms)) {
        std::cerr << "  ERROR: " << queue_name
                  << " did not become ready after reload trigger for checkpoint "
                  << checkpoint_gen << ".\n";
        return false;
    }

    std::cout << "  " << queue_name << " ready with checkpoint " << checkpoint_gen << ".\n";
    return true;
}


// ---------------------------------------------------------------------------
// evaluate_pair
// ---------------------------------------------------------------------------
MatchupResult evaluate_pair(const std::string& checkpoints_dir,
                             int checkpoint_a,
                             int checkpoint_b,
                             std::shared_ptr<SharedMemoryInferenceQueue> queue_best,
                             std::shared_ptr<SharedMemoryInferenceQueue> queue_candidate,
                             int games_per_side = 50,
                             int num_workers    = 4)
{
    MatchupResult result;
    result.checkpoint_a = checkpoint_a;
    result.checkpoint_b = checkpoint_b;

    std::cout << "\n--- Matchup: checkpoint " << checkpoint_a
              << " vs checkpoint " << checkpoint_b << " ---\n";

    if (!reload_checkpoint(checkpoints_dir, "mcts_best_model",      checkpoint_a, queue_best))      return result;
    if (!reload_checkpoint(checkpoints_dir, "mcts_candidate_model", checkpoint_b, queue_candidate)) return result;

    auto eval_results = std::make_shared<EvaluationResults>();

    int games_per_worker = std::max(1, games_per_side / num_workers);

    std::vector<std::thread> workers;

    for (int i = 0; i < num_workers; ++i)
        workers.emplace_back(eval_worker_best_as_p1,
                             i, queue_best, queue_candidate,
                             eval_results, games_per_worker);

    for (int i = 0; i < num_workers; ++i)
        workers.emplace_back(eval_worker_best_as_p2,
                             i + num_workers, queue_best, queue_candidate,
                             eval_results, games_per_worker);

    for (auto& w : workers) w.join();

    result.as_p1_a_wins = eval_results->best_as_p1_wins;
    result.as_p1_b_wins = eval_results->best_as_p1_losses;
    result.as_p1_draws  = eval_results->best_as_p1_draws;

    result.as_p2_a_wins = eval_results->best_as_p2_wins;
    result.as_p2_b_wins = eval_results->best_as_p2_losses;
    result.as_p2_draws  = eval_results->best_as_p2_draws;

    result.a_wins = result.as_p1_a_wins + result.as_p2_a_wins;
    result.b_wins = result.as_p1_b_wins + result.as_p2_b_wins;
    result.draws  = result.as_p1_draws  + result.as_p2_draws;

    int decisive = result.a_wins + result.b_wins;
    if (decisive > 0)
        result.b_winrate = (static_cast<double>(result.b_wins) / decisive) * 100.0;

    std::cout << "  Result  ->  A wins: " << result.a_wins
              << "  B wins: " << result.b_wins
              << "  Draws: "  << result.draws
              << "  B winrate: " << std::fixed << std::setprecision(1)
              << result.b_winrate << "%\n";

    return result;
}


// ---------------------------------------------------------------------------
// validate_checkpoints
// ---------------------------------------------------------------------------
std::vector<int> validate_checkpoints(const std::string& checkpoints_dir,
                                       const std::vector<int>& requested)
{
    std::vector<int> valid;
    for (int gen : requested) {
        fs::path p = fs::path(checkpoints_dir) / ("checkpoint_" + std::to_string(gen));
        if (fs::exists(p) && fs::is_directory(p)) {
            valid.push_back(gen);
        } else {
            std::cerr << "[WARNING] Checkpoint not found on disk, skipping: " << p << "\n";
        }
    }
    std::sort(valid.begin(), valid.end());
    return valid;
}


// ---------------------------------------------------------------------------
// shared helpers used by both evaluation runners
// ---------------------------------------------------------------------------
static std::ofstream open_csv(const std::string& path) {
    std::ofstream csv(path);
    csv << "checkpoint_a,checkpoint_b,"
           "as_p1_a_wins,as_p1_b_wins,as_p1_draws,"
           "as_p2_a_wins,as_p2_b_wins,as_p2_draws,"
           "total_a_wins,total_b_wins,total_draws,b_winrate\n";
    return csv;
}

static void write_csv_row(std::ofstream& csv, const MatchupResult& r) {
    csv << r.checkpoint_a << "," << r.checkpoint_b << ","
        << r.as_p1_a_wins << "," << r.as_p1_b_wins << "," << r.as_p1_draws << ","
        << r.as_p2_a_wins << "," << r.as_p2_b_wins << "," << r.as_p2_draws << ","
        << r.a_wins << "," << r.b_wins << "," << r.draws << ","
        << std::fixed << std::setprecision(2) << r.b_winrate << "\n";
    csv.flush();
}

static void run_matchup_and_log(const std::string&                          checkpoints_dir,
                                 int                                          ckpt_a,
                                 int                                          ckpt_b,
                                 std::shared_ptr<SharedMemoryInferenceQueue>  queue_best,
                                 std::shared_ptr<SharedMemoryInferenceQueue>  queue_candidate,
                                 int                                          games_per_side,
                                 int                                          num_workers,
                                 int                                          matchup_idx,
                                 int                                          total_matchups,
                                 RoundRobinLogger&                            logger,
                                 std::ofstream&                               csv)
{
    std::cout << "\n[Matchup " << matchup_idx << "/" << total_matchups << "] "
              << "checkpoint " << ckpt_a << " (A) vs checkpoint " << ckpt_b << " (B)\n";

    MatchupResult r = evaluate_pair(checkpoints_dir, ckpt_a, ckpt_b,
                                    queue_best, queue_candidate,
                                    games_per_side, num_workers);

    logger.write_matchup(r, matchup_idx, total_matchups,
                         r.as_p1_a_wins, r.as_p1_b_wins, r.as_p1_draws,
                         r.as_p2_a_wins, r.as_p2_b_wins, r.as_p2_draws);

    write_csv_row(csv, r);
}


// ---------------------------------------------------------------------------
// run_full_round_robin
//   Evaluates every pair (i, j) with i < j.
// ---------------------------------------------------------------------------
void run_full_round_robin(const std::string& checkpoints_dir,
                           const std::vector<int>& checkpoints,
                           std::shared_ptr<SharedMemoryInferenceQueue> queue_best,
                           std::shared_ptr<SharedMemoryInferenceQueue> queue_candidate,
                           int games_per_side = 50,
                           int num_workers    = 4)
{
    if (checkpoints.size() < 2) {
        std::cout << "Need at least 2 checkpoints for a round-robin. Found "
                  << checkpoints.size() << ".\n";
        return;
    }

    int n              = static_cast<int>(checkpoints.size());
    int total_matchups = n * (n - 1) / 2;

    RoundRobinLogger logger("round_robin_log.txt");
    logger.write_header(checkpoints, games_per_side, num_workers);

    std::cout << "\n========== FULL ROUND-ROBIN EVALUATION ==========\n";
    std::cout << "Checkpoints (" << n << "): ";
    for (int g : checkpoints) std::cout << g << " ";
    std::cout << "\nTotal matchups : " << total_matchups << "\n";
    std::cout << "Games/matchup  : " << games_per_side * 2
              << " (" << games_per_side << " each side)\n";
    std::cout << "==================================================\n";

    auto csv = open_csv("round_robin_results.csv");

    // For the summary ranking matrix
    std::vector<std::vector<double>> winrate(n, std::vector<double>(n, 0.0));

    int matchup_idx = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            ++matchup_idx;

            MatchupResult r = evaluate_pair(checkpoints_dir,
                                            checkpoints[i], checkpoints[j],
                                            queue_best, queue_candidate,
                                            games_per_side, num_workers);

            winrate[i][j] = r.b_winrate;
            winrate[j][i] = 100.0 - r.b_winrate;

            logger.write_matchup(r, matchup_idx, total_matchups,
                                 r.as_p1_a_wins, r.as_p1_b_wins, r.as_p1_draws,
                                 r.as_p2_a_wins, r.as_p2_b_wins, r.as_p2_draws);

            write_csv_row(csv, r);
        }
    }

    // Build ranking by average winrate
    std::vector<std::pair<double, int>> ranking;
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j)
            if (i != j) sum += (100.0 - winrate[i][j]);
        ranking.push_back({sum / (n - 1), i});
    }
    std::sort(ranking.begin(), ranking.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });

    logger.write_summary(checkpoints, winrate, ranking);

    std::cout << "\nDetailed log : round_robin_log.txt\n";
    std::cout << "Machine CSV  : round_robin_results.csv\n";
    std::cout << "==================================================\n";
}


// ---------------------------------------------------------------------------
// run_consecutive_evaluation
//   Evaluates consecutive pairs only: [0 vs 1], [1 vs 2], [2 vs 3], ...
// ---------------------------------------------------------------------------
void run_consecutive_evaluation(const std::string& checkpoints_dir,
                                 const std::vector<int>& checkpoints,
                                 std::shared_ptr<SharedMemoryInferenceQueue> queue_best,
                                 std::shared_ptr<SharedMemoryInferenceQueue> queue_candidate,
                                 int games_per_side = 50,
                                 int num_workers    = 4)
{
    if (checkpoints.size() < 2) {
        std::cout << "Need at least 2 checkpoints. Found " << checkpoints.size() << ".\n";
        return;
    }

    int total_matchups = static_cast<int>(checkpoints.size()) - 1;

    RoundRobinLogger logger("consecutive_log.txt");

    std::cout << "\n========== CONSECUTIVE CHECKPOINT EVALUATION ==========\n";
    std::cout << "Checkpoints (" << checkpoints.size() << "): ";
    for (int g : checkpoints) std::cout << g << " ";
    std::cout << "\nTotal matchups : " << total_matchups << "\n";
    std::cout << "Games/matchup  : " << games_per_side * 2
              << " (" << games_per_side << " each side)\n";
    std::cout << "=======================================================\n";

    auto csv = open_csv("consecutive_results.csv");

    for (int i = 0; i < total_matchups; ++i) {
        run_matchup_and_log(checkpoints_dir,
                            checkpoints[i], checkpoints[i + 1],
                            queue_best, queue_candidate,
                            games_per_side, num_workers,
                            i + 1, total_matchups,
                            logger, csv);
    }

    std::cout << "\nDetailed log : consecutive_log.txt\n";
    std::cout << "Machine CSV  : consecutive_results.csv\n";
    std::cout << "=======================================================\n";
}


// ---------------------------------------------------------------------------
// run_progressive_evaluation
//   best  = checkpoints[0]  (initial seed)
//   For every subsequent checkpoint as candidate:
//     - Play best vs candidate
//     - If candidate winrate > PROMOTION_THRESHOLD (55%) →
//         candidate is promoted to best, checkpoint number is logged
//     - Otherwise best stays, move on to the next candidate
// ---------------------------------------------------------------------------
static constexpr double PROMOTION_THRESHOLD = 55.0;

void run_progressive_evaluation(const std::string& checkpoints_dir,
                                 const std::vector<int>& checkpoints,
                                 std::shared_ptr<SharedMemoryInferenceQueue> queue_best,
                                 std::shared_ptr<SharedMemoryInferenceQueue> queue_candidate,
                                 int games_per_side = 50,
                                 int num_workers    = 4)
{
    if (checkpoints.size() < 2) {
        std::cout << "Need at least 2 checkpoints for progressive eval. Found "
                  << checkpoints.size() << ".\n";
        return;
    }

    std::cout << "\n========== PROGRESSIVE CHECKPOINT EVALUATION ==========\n";
    std::cout << "Checkpoints (" << checkpoints.size() << "): ";
    for (int g : checkpoints) std::cout << g << " ";
    std::cout << "\n";
    std::cout << "Promotion threshold : " << PROMOTION_THRESHOLD << "%\n";
    std::cout << "Games/matchup       : " << games_per_side * 2
              << " (" << games_per_side << " each side)\n";
    std::cout << "=======================================================\n";

    // Human-readable promotion log
    std::ofstream promo_log("progressive_promotions.txt");
    if (!promo_log.is_open())
        std::cerr << "[Logger] WARNING: could not open progressive_promotions.txt\n";

    auto csv = open_csv("progressive_results.csv");

    int current_best     = checkpoints[0];
    int total_candidates = static_cast<int>(checkpoints.size()) - 1;

    std::cout << "\nInitial best model: checkpoint " << current_best << "\n";
    if (promo_log.is_open())
        promo_log << "Initial best: " << current_best << "\n\n";

    for (int i = 1; i <= total_candidates; ++i) {
        int candidate = checkpoints[i];

        std::cout << "\n[" << i << "/" << total_candidates << "] "
                  << "Best=" << current_best
                  << "  vs  Candidate=" << candidate << "\n";

        MatchupResult r = evaluate_pair(checkpoints_dir,
                                        current_best, candidate,
                                        queue_best, queue_candidate,
                                        games_per_side, num_workers);

        write_csv_row(csv, r);

        if (r.b_winrate > PROMOTION_THRESHOLD) {
            // Format winrate string for the log line
            std::ostringstream wr_str;
            wr_str << std::fixed << std::setprecision(1) << r.b_winrate;

            std::string msg = "PROMOTED: checkpoint " + std::to_string(candidate)
                            + "  (beat " + std::to_string(current_best)
                            + " with " + wr_str.str() + "% winrate)";

            std::cout << "  >>> " << msg << "\n";
            if (promo_log.is_open()) { promo_log << msg << "\n"; promo_log.flush(); }

            current_best = candidate;
        } else {
            std::cout << "  --- No promotion (candidate winrate "
                      << std::fixed << std::setprecision(1) << r.b_winrate
                      << "% <= " << PROMOTION_THRESHOLD << "%)  "
                      << "Best stays: checkpoint " << current_best << "\n";
        }
    }

    std::cout << "\nFinal best model: checkpoint " << current_best << "\n";

    if (promo_log.is_open()) {
        promo_log << "\nFinal best: " << current_best << "\n";
        promo_log.close();
    }

    std::cout << "\nPromotion log : progressive_promotions.txt\n";
    std::cout << "Machine CSV   : progressive_results.csv\n";
    std::cout << "=======================================================\n";
}


// ---------------------------------------------------------------------------
// run_evaluation  —  unified entry point
// ---------------------------------------------------------------------------
void run_evaluation(EvalMode                                    mode,
                    const std::string&                          checkpoints_dir,
                    const std::vector<int>&                     checkpoints,
                    std::shared_ptr<SharedMemoryInferenceQueue> queue_best,
                    std::shared_ptr<SharedMemoryInferenceQueue> queue_candidate,
                    int                                         games_per_side = 50,
                    int                                         num_workers    = 4)
{
    switch (mode) {
        case EvalMode::FULL_ROUND_ROBIN:
            run_full_round_robin(checkpoints_dir, checkpoints,
                                 queue_best, queue_candidate,
                                 games_per_side, num_workers);
            break;
        case EvalMode::CONSECUTIVE:
            run_consecutive_evaluation(checkpoints_dir, checkpoints,
                                       queue_best, queue_candidate,
                                       games_per_side, num_workers);
            break;
        case EvalMode::PROGRESSIVE:
            run_progressive_evaluation(checkpoints_dir, checkpoints,
                                       queue_best, queue_candidate,
                                       games_per_side, num_workers);
            break;
    }
}


// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------
    const std::string    CHECKPOINTS_DIR  = "checkpoints/";
    const int            NUM_WORKERS      = 8;
    const int            GAMES_PER_SIDE   = 200;
    const std::vector<int> CHECKPOINT_LIST = {0, 300, 600, 1500, 1800, 3000, 5400, 6300, 7500, 8700};

    // -----------------------------------------------------------------------
    // Mode selection  (default: PROGRESSIVE)
    //   --progressive  progressive best-model promotion  [default]
    //   --consec       consecutive pairs (i vs i+1)
    //   --full         full round-robin (every pair)
    // -----------------------------------------------------------------------
    EvalMode mode = EvalMode::PROGRESSIVE;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if      (arg == "--full")        mode = EvalMode::FULL_ROUND_ROBIN;
        else if (arg == "--consec")      mode = EvalMode::CONSECUTIVE;
        else if (arg == "--progressive") mode = EvalMode::PROGRESSIVE;
        else {
            std::cerr << "Unknown argument: " << arg << "\n"
                      << "Usage: " << argv[0]
                      << " [--progressive | --consec | --full]\n";
            return 1;
        }
    }

    std::cout << "=== Checkpoint Evaluation  [mode: " << eval_mode_name(mode) << "] ===\n";
    std::cout << "Checkpoints dir : " << CHECKPOINTS_DIR << "\n";
    std::cout << "Checkpoint list : ";
    for (int g : CHECKPOINT_LIST) std::cout << g << " ";
    std::cout << "\n";
    std::cout << "Games per side  : " << GAMES_PER_SIDE << "\n";
    std::cout << "Workers         : " << NUM_WORKERS    << "\n\n";

    // -----------------------------------------------------------------------
    // Connect to inference queues
    // -----------------------------------------------------------------------
    auto queue_best      = std::make_shared<SharedMemoryInferenceQueue>("/mcts_best_model");
    auto queue_candidate = std::make_shared<SharedMemoryInferenceQueue>("/mcts_candidate_model");

    std::cout << "Waiting for best model server...\n";
    if (!queue_best->wait_for_server(30000)) {
        std::cerr << "Fatal: Best model inference server not ready. Exiting.\n";
        return 1;
    }
    std::cout << "Best model server ready!\n";

    std::cout << "Waiting for candidate model server...\n";
    if (!queue_candidate->wait_for_server(30000)) {
        std::cerr << "Fatal: Candidate model inference server not ready. Exiting.\n";
        return 1;
    }
    std::cout << "Candidate model server ready!\n\n";

    // -----------------------------------------------------------------------
    // Validate checkpoints
    // -----------------------------------------------------------------------
    std::vector<int> checkpoints = validate_checkpoints(CHECKPOINTS_DIR, CHECKPOINT_LIST);

    if (checkpoints.empty()) {
        std::cerr << "No valid checkpoints found in " << CHECKPOINTS_DIR << "\n";
        return 1;
    }

    // -----------------------------------------------------------------------
    // Run
    // -----------------------------------------------------------------------
    run_evaluation(mode,
                   CHECKPOINTS_DIR,
                   checkpoints,
                   queue_best,
                   queue_candidate,
                   GAMES_PER_SIDE,
                   NUM_WORKERS);

    return 0;
}