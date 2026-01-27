#include "game_dataset.h"
#include <random>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <iomanip>
#include <algorithm>

GameDataset::GameDataset(size_t max_size_) : max_size(max_size_) {
    boards.resize(max_size);
    pi_targets.resize(max_size);
    z_targets.resize(max_size);
    legal_mask.resize(max_size);
}

void GameDataset::add_position(torch::Tensor board, torch::Tensor pi, 
                               torch::Tensor z, torch::Tensor mask) {
    std::lock_guard<std::mutex> lock(mutex_);
    

    boards[next_index] = board;
    pi_targets[next_index] = pi;
    z_targets[next_index] = z;
    legal_mask[next_index] = mask;
    
    // Circular buffer advancement
    next_index = (next_index + 1) % max_size;
    if (current_size < max_size) {
        current_size++;
    }
}

torch::data::Example<> GameDataset::get(size_t) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::uniform_int_distribution<size_t> dist(0, current_size - 1);
    static std::mt19937 rng(std::random_device{}());
    size_t idx = dist(rng);
    
    return {boards[idx], torch::cat({pi_targets[idx], z_targets[idx].unsqueeze(0), legal_mask[idx]})};
}

torch::optional<size_t> GameDataset::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_size;
}

void GameDataset::update_z(const std::vector<torch::Tensor>& new_z_values, Cell_state winner) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t count = new_z_values.size();
    
    // Compute z values for X and O based on winner
    float z_val_x = (winner == Cell_state::X) ? 1.0f : (winner == Cell_state::O ? -1.0f : 0.0f);
    float z_val_o = (winner == Cell_state::O) ? 1.0f : (winner == Cell_state::X ? -1.0f : 0.0f);
    
    for (size_t i = 0; i < count; ++i) {
        size_t idx = (next_index + max_size - count + i) % max_size;
        float old_val = z_targets[idx].item<float>();
        float updated_val = (old_val == 0.0f) ? z_val_x : z_val_o;
        z_targets[idx] = torch::tensor(updated_val, torch::dtype(torch::kFloat32));
    }
}

void GameDataset::save(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    

    torch::save(boards, path + "_boards.pt");
    torch::save(pi_targets, path + "_pi.pt");
    torch::save(z_targets, path + "_z.pt");
    torch::save(legal_mask, path + "_mask.pt");
    
    auto metadata = torch::tensor({static_cast<int64_t>(current_size), 
                                   static_cast<int64_t>(next_index)}, 
                                  torch::kInt64);
    torch::save(metadata, path + "_metadata.pt");
}

void GameDataset::load(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        torch::load(boards, path + "_boards.pt");
        torch::load(pi_targets, path + "_pi.pt");
        torch::load(z_targets, path + "_z.pt");
        torch::load(legal_mask, path + "_mask.pt");
        
        torch::Tensor metadata;
        torch::load(metadata, path + "_metadata.pt");
        
        auto metadata_acc = metadata.accessor<int64_t, 1>();
        current_size = static_cast<size_t>(metadata_acc[0]);
        next_index = static_cast<size_t>(metadata_acc[1]);
        

    } catch (const std::exception& e) {
        std::cerr << "Error loading dataset: " << e.what() << std::endl;
        throw;
    }
}

//Thread-safe merge for parallel self-play
void GameDataset::merge(const GameDataset& other) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (size_t i = 0; i < other.current_size; ++i) {

        boards[next_index] = other.boards[i].clone();
        pi_targets[next_index] = other.pi_targets[i].clone();
        z_targets[next_index] = other.z_targets[i].clone();
        legal_mask[next_index] = other.legal_mask[i].clone();
        

        next_index = (next_index + 1) % max_size;
        if (current_size < max_size) {
            current_size++;
        }
    }
}

// Clear the dataset
void GameDataset::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    current_size = 0;
    next_index = 0;

    // boards.clear();
    // pi_targets.clear();
    // z_targets.clear();
    // legal_mask.clear();
    // boards.resize(max_size);
    // pi_targets.resize(max_size);
    // z_targets.resize(max_size);
    // legal_mask.resize(max_size);
}



int GameDataset::count_pieces(const torch::Tensor& board) const {
    auto plane0 = board[0].sum().item<float>();
    auto plane1 = board[1].sum().item<float>();
    return static_cast<int>(plane0 + plane1);
}

bool GameDataset::is_empty_board(const torch::Tensor& board) const {
    return count_pieces(board) == 0;
}


double GameDataset::compute_entropy(const torch::Tensor& policy) const {
    auto policy_safe = policy.clamp(1e-10, 1.0);
    auto log_policy = torch::log(policy_safe);
    auto entropy = -(policy_safe * log_policy).sum().item<double>();
    return entropy;
}

std::string GameDataset::board_to_hash(const torch::Tensor& board) const {
    std::stringstream ss;
    auto accessor = board.accessor<float, 3>();
    
    for (int i = 0; i < 2; ++i) {
        for (int x = 0; x < 4; ++x) {
            for (int y = 0; y < 4; ++y) {
                ss << static_cast<int>(accessor[i][x][y]);
            }
        }
    }
    return ss.str();
}

GameDatasetAnalysisResults GameDataset::analyze() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    GameDatasetAnalysisResults results;
    
    if (current_size == 0) {
        return results;
    }
    
    results.total_positions = current_size;
    
    std::unordered_set<std::string> unique_board_hashes;
    
    // Track game boundaries
    std::vector<size_t> game_lengths;
    size_t current_game_length = 0;
    bool first_position = true;
    

    double total_entropy = 0.0;
    double total_legal_moves = 0.0;
    double total_policy_max = 0.0;
    double total_value = 0.0;
    double total_abs_value = 0.0;
    std::vector<double> all_values;
    std::vector<int> piece_counts;
    
    // Calculate the starting index for iteration (oldest position in circular buffer)
    size_t start_idx = (current_size < max_size) ? 0 : next_index;
    
    for (size_t i = 0; i < current_size; ++i) {
        // Handle circular buffer indexing correctly
        size_t idx = (start_idx + i) % max_size;
        
        const auto& board = boards[idx];
        const auto& pi = pi_targets[idx];
        const auto& z = z_targets[idx];
        const auto& mask = legal_mask[idx];
        
        // Unique boards
        std::string board_hash = board_to_hash(board);
        unique_board_hashes.insert(board_hash);
        
        // Game phase classification
        int num_pieces = count_pieces(board);
        piece_counts.push_back(num_pieces);
        
        if (num_pieces <= 3) {
            results.early_game_positions++;
        } else if (num_pieces <= 9) {
            results.mid_game_positions++;
        } else {
            results.late_game_positions++;
        }
        // Game length tracking (detect game boundaries via empty board)
        // Skip the first position check to avoid treating the very first board as a boundary
        if (!first_position && is_empty_board(board) && current_game_length > 0) {
            game_lengths.push_back(current_game_length);
            current_game_length = 0;
        }
        first_position = false;
        current_game_length++;
        
        // Value statistics
        float z_value = z.item<float>();
        total_value += z_value;
        total_abs_value += std::abs(z_value);
        all_values.push_back(z_value);
        
        if (z_value > 0.5f) {
            results.x_wins++;
        } else if (z_value < -0.5f) {
            results.o_wins++;
        } else {
            results.draws++;
        }
        
        // Policy statistics
        auto legal_pi = pi * mask;
        auto num_legal = mask.sum().item<float>();
        total_legal_moves += num_legal;
        
        if (num_legal > 0) {
            // Normalize policy over legal moves
            auto normalized_pi = legal_pi / (legal_pi.sum() + 1e-10);
            
            // Entropy
            double entropy = compute_entropy(normalized_pi);
            total_entropy += entropy;
            
            // Max probability (confidence)
            auto max_prob = normalized_pi.max().item<float>();
            total_policy_max += max_prob;
        }
    }
    
    // Finalize last game if exists
    if (current_game_length > 0) {
        game_lengths.push_back(current_game_length);
    }
    

    results.unique_boards = unique_board_hashes.size();
    results.uniqueness_ratio = static_cast<double>(results.unique_boards) / results.total_positions;

    results.avg_policy_entropy = total_entropy / current_size;
    results.avg_legal_moves = total_legal_moves / current_size;
    results.avg_policy_confidence = total_policy_max / current_size;
    results.max_policy_confidence = total_policy_max / current_size;
    
    results.avg_value = total_value / current_size;
    results.avg_abs_value = total_abs_value / current_size;
    
    double sum_sq_diff = 0.0;
    for (double val : all_values) {
        double diff = val - results.avg_value;
        sum_sq_diff += diff * diff;
    }
    results.value_std = std::sqrt(sum_sq_diff / current_size);
    
    // Game length statistics
    results.num_games = game_lengths.size();
    if (!game_lengths.empty()) {
        double total_length = 0.0;
        results.min_game_length = game_lengths[0];
        results.max_game_length = game_lengths[0];
        
        for (size_t len : game_lengths) {
            total_length += len;
            if (len < results.min_game_length) results.min_game_length = len;
            if (len > results.max_game_length) results.max_game_length = len;
        }
        results.avg_game_length = total_length / game_lengths.size();
    }
    
    // Position diversity (variance in piece counts)
    if (!piece_counts.empty()) {
        double avg_pieces = 0.0;
        for (int pc : piece_counts) {
            avg_pieces += pc;
        }
        avg_pieces /= piece_counts.size();
        
        double variance = 0.0;
        for (int pc : piece_counts) {
            variance += (pc - avg_pieces) * (pc - avg_pieces);
        }
        variance /= piece_counts.size();
        results.position_diversity = std::sqrt(variance);
    }
    
    // Policy sharpness (inverse of average entropy, normalized)
    double max_entropy = std::log(16.0); // log of max possible moves (4x4 board)
    results.policy_sharpness = 1.0 - (results.avg_policy_entropy / max_entropy);
    
    return results;
}

void GameDataset::print_analysis() const {
    auto results = analyze();
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  GAME DATASET ANALYSIS REPORT\n";
    std::cout << std::string(70, '=') << "\n\n";
    

    std::cout << "DATASET SIZE:\n";
    std::cout << "  Total positions:       " << results.total_positions << "\n";
    std::cout << "  Unique boards:         " << results.unique_boards 
              << " (" << std::fixed << std::setprecision(1) 
              << results.uniqueness_ratio * 100 << "%)\n";
    std::cout << "  Total games:           " << results.num_games << "\n\n";
    

    std::cout << "GAME PHASE DISTRIBUTION:\n";
    std::cout << "  Early game (≤3 pcs):   " << results.early_game_positions 
              << " (" << std::setprecision(1) 
              << 100.0 * results.early_game_positions / results.total_positions << "%)\n";
    std::cout << "  Mid game (4-9 pcs):    " << results.mid_game_positions 
              << " (" << std::setprecision(1) 
              << 100.0 * results.mid_game_positions / results.total_positions << "%)\n";
    std::cout << "  Late game (≥10 pcs):   " << results.late_game_positions 
              << " (" << std::setprecision(1) 
              << 100.0 * results.late_game_positions / results.total_positions << "%)\n\n";
    

    std::cout << "GAME OUTCOMES:\n";
    std::cout << "  X wins:                " << results.x_wins 
              << " (" << std::setprecision(1) 
              << 100.0 * results.x_wins / results.total_positions << "%)\n";
    std::cout << "  O wins:                " << results.o_wins 
              << " (" << std::setprecision(1) 
              << 100.0 * results.o_wins / results.total_positions << "%)\n";
    std::cout << "  Draws:                 " << results.draws 
              << " (" << std::setprecision(1) 
              << 100.0 * results.draws / results.total_positions << "%)\n\n";
    

    std::cout << "POLICY STATISTICS:\n";
    std::cout << "  Avg entropy:           " << std::setprecision(3) 
              << results.avg_policy_entropy << "\n";
    std::cout << "  Avg legal moves:       " << std::setprecision(2) 
              << results.avg_legal_moves << "\n";
    std::cout << "  Avg confidence:        " << std::setprecision(3) 
              << results.avg_policy_confidence << "\n";
    std::cout << "  Policy sharpness:      " << std::setprecision(3) 
              << results.policy_sharpness 
              << " (higher = more decisive)\n\n";
    

    std::cout << "VALUE STATISTICS:\n";
    std::cout << "  Avg value:             " << std::setprecision(3) 
              << results.avg_value << "\n";
    std::cout << "  Value std dev:         " << std::setprecision(3) 
              << results.value_std << "\n";
    std::cout << "  Avg |value|:           " << std::setprecision(3) 
              << results.avg_abs_value << "\n\n";
    

    if (results.num_games > 0) {
        std::cout << "GAME LENGTH STATISTICS:\n";
        std::cout << "  Avg game length:       " << std::setprecision(1) 
                  << results.avg_game_length << " moves\n";
        std::cout << "  Min game length:       " << results.min_game_length << " moves\n";
        std::cout << "  Max game length:       " << results.max_game_length << " moves\n\n";
    }
    

    std::cout << "DATA QUALITY METRICS:\n";
    std::cout << "  Position diversity:    " << std::setprecision(2) 
              << results.position_diversity 
              << " (piece count std dev)\n";
    std::cout << "  Uniqueness ratio:      " << std::setprecision(1) 
              << results.uniqueness_ratio * 100 << "%\n";
    
    std::cout << "\n" << std::string(70, '=') << "\n\n";
}



void GameDataset::log_metrics(MetricsLogger& logger, int iteration, const std::string& prefix) const {
    auto results = analyze();
    
    // Dataset size metrics
    // logger.add_scalar(prefix + "/total_positions", results.total_positions);
    logger.add_scalar("iteration_number", iteration);
    logger.add_scalar(prefix + "/unique_boards", results.unique_boards);
    logger.add_scalar(prefix + "/uniqueness_ratio", results.uniqueness_ratio);
    
    // Game phase distribution (counts)
    // logger.add_scalar(prefix + "/early_game_positions", results.early_game_positions);
    // logger.add_scalar(prefix + "/mid_game_positions", results.mid_game_positions);
    // logger.add_scalar(prefix + "/late_game_positions", results.late_game_positions);
    
    // Game phase distribution (percentages)
    if (results.total_positions > 0) {
        logger.add_scalar(prefix + "/early_game_pct", 
            100.0 * results.early_game_positions / results.total_positions);
        logger.add_scalar(prefix + "/mid_game_pct", 
            100.0 * results.mid_game_positions / results.total_positions);
        logger.add_scalar(prefix + "/late_game_pct", 
            100.0 * results.late_game_positions / results.total_positions);
    }
    
    // Game outcomes (counts)
    // logger.add_scalar(prefix + "/x_wins", results.x_wins);
    // logger.add_scalar(prefix + "/o_wins", results.o_wins);
    // logger.add_scalar(prefix + "/draws", results.draws);
    
    // Game outcomes (percentages)
    if (results.total_positions > 0) {
        logger.add_scalar(prefix + "/x_win_pct", 
            100.0 * results.x_wins / results.total_positions);
        logger.add_scalar(prefix + "/o_win_pct", 
            100.0 * results.o_wins / results.total_positions);
        logger.add_scalar(prefix + "/draw_pct", 
            100.0 * results.draws / results.total_positions);
        
        // Win rate balance (absolute difference between X and O)
        double win_imbalance = std::abs(
            static_cast<double>(results.x_wins - results.o_wins) / results.total_positions
        );
        logger.add_scalar(prefix + "/win_imbalance", win_imbalance);
    }
    
    // Policy statistics
    logger.add_scalar(prefix + "/avg_policy_entropy", results.avg_policy_entropy);
    // logger.add_scalar(prefix + "/avg_legal_moves", results.avg_legal_moves);
    logger.add_scalar(prefix + "/avg_policy_confidence", results.avg_policy_confidence);
    logger.add_scalar(prefix + "/policy_sharpness", results.policy_sharpness);
    
    // // Value statistics
    // logger.add_scalar(prefix + "/avg_value", results.avg_value);
    // logger.add_scalar(prefix + "/value_std", results.value_std);
    // logger.add_scalar(prefix + "/avg_abs_value", results.avg_abs_value);
    
    // // Game length statistics
    if (results.num_games > 0) {
        logger.add_scalar(prefix + "/avg_game_length", results.avg_game_length);
    //     logger.add_scalar(prefix + "/min_game_length", results.min_game_length);
    //     logger.add_scalar(prefix + "/max_game_length", results.max_game_length);
    }
    
    // Data quality metrics
    logger.add_scalar(prefix + "/position_diversity", results.position_diversity);
    logger.flush_metrics();
}