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

std::string GameDataset::hash_board(const torch::Tensor& board) const {
    auto accessor = board.accessor<float, 3>();
    std::ostringstream oss;
    
    for (int z = 0; z < 2; ++z) {
        for (int i = 0; i < board.size(1); ++i) {
            for (int j = 0; j < board.size(2); ++j) {
                if (i > 0 || j > 0) oss << ",";
                oss << accessor[z][i][j];
            }
        }
    }
    return oss.str();
}

void GameDataset::add_position(torch::Tensor board, torch::Tensor pi, 
                                torch::Tensor z, torch::Tensor mask) {
    // Simply add to circular buffer - NO deduplication
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
    std::uniform_int_distribution<size_t> dist(0, current_size - 1);
    static std::mt19937 rng(std::random_device{}());
    size_t idx = dist(rng);
    return {boards[idx], torch::cat({pi_targets[idx], z_targets[idx].unsqueeze(0), legal_mask[idx]})};
}

torch::optional<size_t> GameDataset::size() const {
    return max_size;
}

void GameDataset::update_last_z(const std::vector<torch::Tensor>& new_z_values, Cell_state winner) {
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
    torch::save(boards, path + "_boards.pt");
    torch::save(pi_targets, path + "_pi.pt");
    torch::save(z_targets, path + "_z.pt");
    torch::save(legal_mask, path + "_mask.pt");
}


DatasetAnalysis GameDataset::analyze_dataset() const {
    DatasetAnalysis analysis;
    analysis.total_states = current_size;
    
    std::unordered_map<std::string, int> state_freq;
    std::vector<double> entropies;
    
    // Track game boundaries by detecting empty boards (new game start)
    std::vector<int> game_lengths;
    int current_game_length = 0;
    
    std::cout << "=== DATASET ANALYSIS ===\n\n";
    std::cout << "Analyzing " << current_size << " positions...\n\n";
    
    // 1. Analyze each position
    for (size_t i = 0; i < current_size; ++i) {
        // Hash board state for uniqueness tracking
        std::string board_hash = hash_board(boards[i]);
        state_freq[board_hash]++;
        
        // Calculate policy entropy
        auto pi = pi_targets[i];
        auto mask = legal_mask[i];
        
        double entropy = 0.0;
        int legal_moves = 0;
        
        auto pi_accessor = pi.accessor<float, 1>();
        auto mask_accessor = mask.accessor<float, 1>();
        
        for (int j = 0; j < pi.size(0); ++j) {
            if (mask_accessor[j] > 0.5f) {
                legal_moves++;
                float p = pi_accessor[j];
                if (p > 1e-8) {
                    entropy -= p * std::log(p);
                }
            }
        }
        
        entropies.push_back(entropy);
        
        // Track game lengths by checking if board is empty (new game)
        // Check if current board is empty by looking at first two planes (current position)
        auto board_accessor = boards[i].accessor<float, 3>();
        bool is_empty_board = true;
        
        // Check planes 0 and 1 (current player and opponent pieces)
        for (int x = 0; x < 4; ++x) {
            for (int y = 0; y < 4; ++y) {
                if (board_accessor[0][x][y] > 0.5f || board_accessor[1][x][y] > 0.5f) {
                    is_empty_board = false;
                    break;
                }
            }
            if (!is_empty_board) break;
        }
        
        if (is_empty_board && i > 0) {
            // New game detected - save previous game length
            if (current_game_length > 0) {
                game_lengths.push_back(current_game_length);
            }
            current_game_length = 1;
        } else {
            current_game_length++;
        }
    }
    
    // Add last game
    if (current_game_length > 0) {
        game_lengths.push_back(current_game_length);
    }
    
    // 2. Calculate statistics
    analysis.unique_states = state_freq.size();
    analysis.uniqueness_ratio = static_cast<double>(analysis.unique_states) / analysis.total_states;
    analysis.state_frequency = state_freq;
    
    // Repetition frequency
    for (const auto& [hash, count] : state_freq) {
        analysis.repetition_counts.push_back(count);
    }
    std::sort(analysis.repetition_counts.begin(), analysis.repetition_counts.end(), std::greater<int>());
    
    // Entropy statistics
    if (!entropies.empty()) {
        double sum = 0.0;
        for (double e : entropies) sum += e;
        analysis.mean_entropy = sum / entropies.size();
        
        double var_sum = 0.0;
        for (double e : entropies) {
            var_sum += (e - analysis.mean_entropy) * (e - analysis.mean_entropy);
        }
        analysis.std_entropy = std::sqrt(var_sum / entropies.size());
    }
    
    // Game length statistics
    analysis.game_lengths = game_lengths;
    if (!game_lengths.empty()) {
        int sum = 0;
        analysis.min_game_length = game_lengths[0];
        analysis.max_game_length = game_lengths[0];
        
        for (int len : game_lengths) {
            sum += len;
            analysis.min_game_length = std::min(analysis.min_game_length, len);
            analysis.max_game_length = std::max(analysis.max_game_length, len);
        }
        
        analysis.mean_game_length = static_cast<double>(sum) / game_lengths.size();
        
        double var_sum = 0.0;
        for (int len : game_lengths) {
            var_sum += (len - analysis.mean_game_length) * (len - analysis.mean_game_length);
        }
        analysis.std_game_length = std::sqrt(var_sum / game_lengths.size());
    }
    
    // 3. Check terminal state consistency
    analysis.mismatched_terminal_states = 0;
    for (size_t i = 0; i < current_size; ++i) {
        float z = z_targets[i].item<float>();
        
        // Terminal states should have |z| = 1.0 (win/loss) or 0.0 (draw)
        // Check if legal moves exist for this position
        auto mask = legal_mask[i];
        auto mask_accessor = mask.accessor<float, 1>();
        bool has_legal_moves = false;
        for (int j = 0; j < mask.size(0); ++j) {
            if (mask_accessor[j] > 0.5f) {
                has_legal_moves = true;
                break;
            }
        }
        
        // If no legal moves but z suggests ongoing game, that's suspicious
        if (!has_legal_moves && std::abs(z) < 0.99f && std::abs(z) > 0.01f) {
            analysis.mismatched_terminal_states++;
        }
    }
    
    return analysis;
}

void GameDataset::print_analysis() const {
    auto analysis = analyze_dataset();
    
    std::cout << std::fixed << std::setprecision(4);
    
    // 1. State Uniqueness
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "📊 STATE UNIQUENESS\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "Total States:        " << analysis.total_states << "\n";
    std::cout << "Unique States:       " << analysis.unique_states << "\n";
    std::cout << "Uniqueness Ratio:    " << (analysis.uniqueness_ratio * 100) << "%\n";
    
    if (analysis.uniqueness_ratio < 0.5) {
        std::cout << "⚠️  WARNING: Low uniqueness ratio - high state repetition!\n";
    } else if (analysis.uniqueness_ratio > 0.95) {
        std::cout << "✓ GOOD: High state diversity\n";
    }
    std::cout << "\n";
    
    // 2. State Repetition
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "🔁 STATE REPETITION FREQUENCY\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    
    if (!analysis.repetition_counts.empty()) {
        std::cout << "Most repeated state:  " << analysis.repetition_counts[0] << " times\n";
        
        if (analysis.repetition_counts.size() >= 5) {
            std::cout << "Top 5 repetitions:    ";
            for (int i = 0; i < 5; ++i) {
                std::cout << analysis.repetition_counts[i];
                if (i < 4) std::cout << ", ";
            }
            std::cout << "\n";
        }
        
        // Count states seen only once
        int unique_once = 0;
        for (int count : analysis.repetition_counts) {
            if (count == 1) unique_once++;
        }
        std::cout << "States seen once:     " << unique_once << " (" 
                  << (unique_once * 100.0 / analysis.unique_states) << "%)\n";
        
        if (analysis.repetition_counts[0] > 100) {
            std::cout << "⚠️  WARNING: Some states heavily repeated!\n";
        }
    }
    std::cout << "\n";
    
    // 3. Game Length Distribution
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << "🎮 GAME LENGTH DISTRIBUTION\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        
        if (!analysis.game_lengths.empty()) {
            std::cout << "Number of games:     " << analysis.game_lengths.size() << "\n";
            std::cout << "Mean game length:    " << analysis.mean_game_length << " moves\n";
            std::cout << "Std game length:     " << analysis.std_game_length << " moves\n";
            std::cout << "Min game length:     " << analysis.min_game_length << " moves\n";
            std::cout << "Max game length:     " << analysis.max_game_length << " moves\n";
            
            // Histogram
            std::map<int, int> length_histogram;
            for (int len : analysis.game_lengths) {
                int bucket = (len / 5) * 5; // Group by 5s
                length_histogram[bucket]++;
            }
            
            std::cout << "\nLength histogram:\n";
            for (const auto& [bucket, count] : length_histogram) {
                std::cout << "  " << std::setw(3) << bucket << "-" << std::setw(3) << (bucket+4) 
                        << " moves: " << std::string(count / 2, '#') << " (" << count << ")\n";
            }
            
            if (analysis.mean_game_length < 5) {
                std::cout << "⚠️  WARNING: Very short games - may indicate issues!\n";
            }
        } else {
            std::cout << "Unable to detect game boundaries\n";
        }
        std::cout << "\n";
    
    // 4. Policy Entropy
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "📈 POLICY ENTROPY\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "Mean entropy:        " << analysis.mean_entropy << "\n";
    std::cout << "Std entropy:         " << analysis.std_entropy << "\n";
    
    if (analysis.mean_entropy < 0.5) {
        std::cout << "⚠️  WARNING: Low entropy - policies too deterministic!\n";
        std::cout << "   (Early training data should be more exploratory)\n";
    } else if (analysis.mean_entropy > 2.0) {
        std::cout << "✓ GOOD: Exploratory policies\n";
    }
    std::cout << "\n";
    
    // 5. Terminal State Validation
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "🎯 TERMINAL STATE VALIDATION\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "Mismatched states:   " << analysis.mismatched_terminal_states << "\n";
    
    if (analysis.mismatched_terminal_states == 0) {
        std::cout << "✓ GOOD: All terminal states consistent\n";
    } else {
        std::cout << "⚠️  WARNING: Some terminal states don't match winner!\n";
    }
    std::cout << "\n";
    
    // 6. Overall Assessment
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "🏁 OVERALL ASSESSMENT\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    
    int issues = 0;
    if (analysis.uniqueness_ratio < 0.5) issues++;
    if (analysis.mean_entropy < 0.5) issues++;
    if (analysis.mismatched_terminal_states > 0) issues++;
    if (analysis.mean_game_length < 5) issues++;
    
    if (issues == 0) {
        std::cout << "✓ Dataset looks READY for training!\n";
    } else if (issues <= 2) {
        std::cout << "⚠️  Dataset has " << issues << " potential issue(s) - review warnings above\n";
    } else {
        std::cout << "❌ Dataset has " << issues << " issues - NOT recommended for training yet\n";
    }
    
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";
}