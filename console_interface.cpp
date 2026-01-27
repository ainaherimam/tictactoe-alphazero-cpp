#include "console_interface.h"
#include <chrono>
#include <string>
#include <thread>
#include <climits>
#include <filesystem>
#include "selfplay_manager.h"
#include "board.h"
#include "game_dataset.h"
#include "alphaz_model.h"
#include "logger.h"
#include <torch/torch.h>
#include "metrics_logger.h"  // Custom metrics logger for W&B


namespace fs = std::filesystem;


bool is_integer(const std::string& s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it)) ++it;
  return !s.empty() && it == s.end();
}

char get_yes_or_no_response(const std::string& prompt) {
  char response;
  while (true) {
    std::cout << prompt;
    std::cin >> response;

    if (std::cin.fail()) {
      std::cin.clear();  
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(),
                      '\n');  
      std::cout << "Invalid input. Please enter 'y' or 'n'.\n";
    } else if (response != 'y' && response != 'n' && response != 'Y' &&
               response != 'N') {
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(),
                      '\n');
      std::cout << "Invalid response. Please enter 'y' or 'n'.\n";
    } else {
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      return std::tolower(response);
    }
  }
}

template <>
int get_parameter_within_bounds<int>(const std::string& prompt, int lower_bound,
                                     int upper_bound) {
  std::string input;
  int value;

  while (true) {
    std::cout << prompt;
    std::cin >> input;

    // Check if input is a valid integer
    if (!is_integer(input)) {
      std::cout << "Invalid input. Please enter a valid integer.\n";
      continue;
    }

    // Convert string to int
    value = std::stoi(input);

    // Check if value is within bounds
    if (!is_in_bounds(value, lower_bound, upper_bound)) {
      std::cout << "Invalid value. Please try again.\n";
    } else {
      break;
    }
  }

  return value;
}

template <>
double get_parameter_within_bounds<double>(const std::string& prompt,
                                           double lower_bound,
                                           double upper_bound) {
  std::string input;
  double value;

  while (true) {
    std::cout << prompt;
    std::cin >> input;

    // Check if input is a valid double
    try {
      value = std::stod(input);
    } catch (std::invalid_argument&) {
      std::cout << "Invalid input. Please enter a valid number.\n";
      continue;
    }

    // Check if value is within bounds
    if (!is_in_bounds(value, lower_bound, upper_bound)) {
      std::cout << "Invalid value. Please try again.\n";
    } else {
      break;
    }
  }

  return value;
}

std::unique_ptr<Mcts_player> create_mcts_agent(
    const std::string& agent_prompt) {

    torch::Device device(torch::kCUDA);
    if (!torch::cuda::is_available()) device = torch::kCPU;
    auto best_model  = AlphaZModel::load_model("checkpoint/best.pt");
    best_model->to(device);
    best_model->eval();

    std::cout << "\nInitializing " << agent_prompt << ":\n";

    int max_iteration = get_parameter_within_bounds(
        "Max iteration number (at least 10) : ", 10, INT_MAX);

    double exploration_constant = 1.41;

    exploration_constant = get_parameter_within_bounds(
        "Enter exploration constant (between 0.1 and 2): ", 0.1, 2.0);

    LogLevel log_level = LogLevel::NONE;
    log_level = static_cast<LogLevel>(get_parameter_within_bounds("Log Level (0:None  -- 5:Full)  : ", 0, 5));
    return std::make_unique<Mcts_player>(exploration_constant, max_iteration,log_level, 0.0, 0.2, 0.15, best_model, -1, false);
    }

std::unique_ptr<Mcts_player_parallel> create_mcts_agent_parallel(
    const std::string& agent_prompt) {

    torch::Device device(torch::kCUDA);
    if (!torch::cuda::is_available()) device = torch::kCPU;
    auto best_model  = AlphaZModel::load_model("checkpoint/best.pt");
    best_model->to(device);
    best_model->eval();

    std::cout << "\nInitializing " << agent_prompt << ":\n";

    int max_iteration = get_parameter_within_bounds(
        "Max iteration number (at least 10) : ", 10, INT_MAX);

    double exploration_constant = 1.41;

    exploration_constant = get_parameter_within_bounds(
        "Enter exploration constant (between 0.1 and 2): ", 0.1, 2.0);

    LogLevel log_level = LogLevel::NONE;
    log_level = static_cast<LogLevel>(get_parameter_within_bounds("Log Level (0:None  -- 5:Full)  : ", 0, 5));
    return std::make_unique<Mcts_player_parallel>(exploration_constant, max_iteration, log_level, 0.0, 0.2, 0.15, best_model, -1, false, 1.0f, 2, 2);
    }
    

void countdown(int seconds) {
    while (seconds > 0) {
        std::cout << "The agent will start thinking loudly in " << seconds
            << " ...\n";
        std::this_thread::sleep_for(std::chrono::seconds(1));
        --seconds;
    }
}

void start_match_against_robot() {
    GameDataset dataset(1000);
    int human_player_number = get_parameter_within_bounds(
        "Enter '1' if you want to be Player 1 (X) or '2' if you "
        "want to be "
        "Player 2 (O): ",
        1, 2);

    int board_size = 4;

    auto mcts_agent = create_mcts_agent("agent");
    auto human_player = std::make_unique<Human_player>();

    if (human_player_number == 1) {
        Game game(board_size, std::move(human_player), std::move(mcts_agent), dataset, true);
        game.simple_play();
    }
    else {
        Game game(board_size, std::move(mcts_agent), std::move(human_player), dataset, true);
        game.simple_play();
    }
}

void start_robot_arena() {
    GameDataset dataset(1000);
    int board_size = 4;

    auto mcts_agent_1 = create_mcts_agent("first agent");
    auto mcts_agent_2 = create_mcts_agent("second agent");

    Game game(board_size, std::move(mcts_agent_1), std::move(mcts_agent_2), dataset);
    game.simple_play();
}


void selfplay() {
    torch::Device device(torch::kCUDA);
    if (!torch::cuda::is_available()) device = torch::kCPU;
    
    // ========================================
    // CONFIGURATION PARAMETERS
    // ========================================
    
    // Board configuration
    const int BOARD_SIZE = 4;
    const int INPUT_CHANNELS = 3;
    const int NUM_MOVES = 20;
    
    // Network architecture
    const int CONV_CHANNELS = 64;
    const int NUM_RES_BLOCKS = 3;
    
    // Dataset configuration
    const int REPLAY_BUFFER_SIZE = 10000;
    GameDataset dataset(REPLAY_BUFFER_SIZE);
    
    // Self-play configuration
    const int NUM_PARALLEL_GAMES = 12;
    const int GAMES_PER_WORKER = 12;
    const int BATCH_SIZE = 12;
    const int MCTS_SIMULATIONS = 100;
    
    // MCTS parameters
    const double EXPLORATION_FACTOR = 1.0;
    const float DIRICHLET_ALPHA = 0.2f;
    const float DIRICHLET_EPSILON = 0.15f;
    
    // Training configuration
    const int NUM_ITERATIONS = 50;
    const int TRAINING_STEPS = 50;
    const int TRAINING_BATCH_SIZE = 128;
    const float LEARNING_RATE = 1e-3f;
    
    // Checkpointing
    const int CHECKPOINT_INTERVAL = 5;
    const int EVALUATION_INTERVAL = 1;
    const std::string CHECKPOINT_DIR = "checkpoint";
    const std::string BEST_MODEL_PATH = "checkpoint/best.pt";
    
    // ========================================
    // INITIALIZE METRICS LOGGER
    // ========================================
    
    std::string log_dir = "wandb_logs/run_" + std::to_string(std::time(nullptr));
    MetricsLogger logger(log_dir);
    
    // Log hyperparameters as JSON
    std::map<std::string, std::string> config = {
        {"board_size", std::to_string(BOARD_SIZE)},
        {"input_channels", std::to_string(INPUT_CHANNELS)},
        {"num_moves", std::to_string(NUM_MOVES)},
        {"conv_channels", std::to_string(CONV_CHANNELS)},
        {"num_res_blocks", std::to_string(NUM_RES_BLOCKS)},
        {"replay_buffer_size", std::to_string(REPLAY_BUFFER_SIZE)},
        {"num_parallel_games", std::to_string(NUM_PARALLEL_GAMES)},
        {"games_per_worker", std::to_string(GAMES_PER_WORKER)},
        {"total_games_per_iteration", std::to_string(NUM_PARALLEL_GAMES * GAMES_PER_WORKER)},
        {"batch_size", std::to_string(BATCH_SIZE)},
        {"mcts_simulations", std::to_string(MCTS_SIMULATIONS)},
        {"exploration_factor", std::to_string(EXPLORATION_FACTOR)},
        {"dirichlet_alpha", std::to_string(DIRICHLET_ALPHA)},
        {"dirichlet_epsilon", std::to_string(DIRICHLET_EPSILON)},
        {"num_iterations", std::to_string(NUM_ITERATIONS)},
        {"training_steps", std::to_string(TRAINING_STEPS)},
        {"training_batch_size", std::to_string(TRAINING_BATCH_SIZE)},
        {"learning_rate", std::to_string(LEARNING_RATE)},
        {"weight_decay", "\"1e-4\""},
        {"gradient_clip_norm", "1.0"},
        {"min_learning_rate", "\"1e-4\""},
        {"checkpoint_interval", std::to_string(CHECKPOINT_INTERVAL)},
        {"evaluation_interval", std::to_string(EVALUATION_INTERVAL)}
    };
    
    logger.log_config(config);
    
    std::cout << "🌱 Starting AlphaZero self-play training...\n";
    std::cout << "Parallel games: " << NUM_PARALLEL_GAMES << " - Games per worker: " << GAMES_PER_WORKER << std::endl;
    std::cout << "Total games per iteration: " << (NUM_PARALLEL_GAMES * GAMES_PER_WORKER) << std::endl;
    std::cout << "Training iterations: " << NUM_ITERATIONS << std::endl;
    
    auto best_model = AlphaZModel::load_model(BEST_MODEL_PATH);
    best_model->to(device);
    best_model->eval();
    
    SelfPlayManager selfplay_manager;
    
    // MAIN TRAINING LOOP
    int total_games = 0;
    int global_step = 0;  // For consistent step tracking
    
    for (int iteration = 1; iteration <= NUM_ITERATIONS; ++iteration) {
        std::cout << "\n========================================\n";
        std::cout << "🔁 ITERATION " << iteration << "/" << NUM_ITERATIONS << "\n";
        std::cout << "========================================\n\n";
        
        // PHASE 1: SELF-PLAY DATA GENERATION
        std::cout << "🎮 Generating training data through parallel self-play...\n";
        
        size_t dataset_size_before = dataset.actual_size();
        auto selfplay_start = std::chrono::high_resolution_clock::now();
        
        selfplay_manager.generate_training_data(
            NUM_PARALLEL_GAMES,
            GAMES_PER_WORKER,
            best_model,
            dataset,
            BATCH_SIZE,
            MCTS_SIMULATIONS,
            BOARD_SIZE,
            EXPLORATION_FACTOR,
            DIRICHLET_ALPHA,
            DIRICHLET_EPSILON
        );
        
        auto selfplay_end = std::chrono::high_resolution_clock::now();
        auto selfplay_duration = std::chrono::duration_cast<std::chrono::seconds>(
            selfplay_end - selfplay_start).count();
        
        int games_played = NUM_PARALLEL_GAMES * GAMES_PER_WORKER;
        total_games += games_played;
        
        std::cout << "✅ Total games played: " << total_games << "\n";
        std::cout << "📊 Replay buffer size: " << dataset.actual_size() 
                  << "/" << REPLAY_BUFFER_SIZE << "\n\n";
        
        // Log self-play metrics
        logger.add_scalar("iteration", iteration);
        logger.add_scalar("selfplay/games_played_this_iter", games_played);
        logger.add_scalar("selfplay/total_games", total_games);
        logger.add_scalar("selfplay/duration_sec", selfplay_duration);
        logger.add_scalar("selfplay/games_per_sec", 
                         games_played / static_cast<double>(std::max(selfplay_duration, 1L)));
        logger.add_scalar("dataset/size", dataset.actual_size());
        logger.add_scalar("dataset/utilization", 
                         dataset.actual_size() / static_cast<double>(REPLAY_BUFFER_SIZE));
        logger.flush_metrics();
        
        // Skip training if not enough data
        if (dataset.actual_size() < TRAINING_BATCH_SIZE) {
            std::cout << "⚠️  Not enough data for training yet (need at least " 
                      << TRAINING_BATCH_SIZE << " positions)\n";
            continue;
        }
        
        // PHASE 2: NEURAL NETWORK TRAINING
        std::cout << "🎯 Training network for " << TRAINING_STEPS << " steps...\n";
        
        auto candidate_model = AlphaZModel::load_model(BEST_MODEL_PATH);
        candidate_model->to(device);
        
        train(
            candidate_model,
            dataset,
            TRAINING_BATCH_SIZE,
            TRAINING_STEPS,
            LEARNING_RATE,
            device,
            5,
            true,
            iteration,
            global_step,
            logger  // Pass metrics logger
        );
        
        global_step += TRAINING_STEPS;
        
        // Set back to evaluation mode
        candidate_model->eval();
        
        // PHASE 3: CHECKPOINTING
        std::string checkpoint_path = CHECKPOINT_DIR + "/iter_" + std::to_string(iteration) + ".pt";
        candidate_model->save_model(checkpoint_path);
        std::cout << "💾 Saved checkpoint: " << checkpoint_path << "\n";
        
        // PHASE 4: EVALUATION
        if (iteration % EVALUATION_INTERVAL == 0) {
            std::cout << "\n⚔️  Evaluating new model vs current best...\n";
            
            // If you have evaluation metrics
            auto [phase1_results, phase2_results] = evaluate(BEST_MODEL_PATH, checkpoint_path);

            // Log Phase 1 results (Best model plays first)
            logger.add_scalar("iteration", iteration);
            logger.add_scalar("evaluation/phase1_candidate_winrate", phase1_results.model2_winrate);
            logger.add_scalar("evaluation/phase1_candidate_top1_accuracy", 
                  static_cast<float>(phase1_results.optimal_moves) / 
                  static_cast<float>(phase1_results.eval_moves));

            // Log Phase 2 results (Candidate model plays first)
            logger.add_scalar("evaluation/phase2_best_winrate", phase2_results.model2_winrate);
            logger.add_scalar("evaluation/phase2_candidate_top1_accuracy", 
                  static_cast<float>(phase2_results.optimal_moves) / 
                  static_cast<float>(phase2_results.eval_moves));

            // Log Combined results
            int total_candidate_wins = phase1_results.model2_wins + phase2_results.model1_wins;
            int total_best_wins = phase1_results.model1_wins + phase2_results.model2_wins;
            int total_draws = phase1_results.draws + phase2_results.draws;
            int total_games = phase1_results.total_games + phase2_results.total_games;

            float overall_candidate_winrate = static_cast<float>(total_candidate_wins) / 
                                              static_cast<float>(total_games);

            logger.add_scalar("evaluation/overall_candidate_winrate", overall_candidate_winrate);


            logger.flush_metrics();
        }
        dataset.print_analysis();
        dataset.log_metrics(logger,iteration);
        if (iteration % CHECKPOINT_INTERVAL == 0) {
            std::string dataset_path = "dataset" + std::to_string(iteration);
            dataset.save(dataset_path);
            // dataset.print_analysis();
            std::cout << "💾 Saved dataset: " << dataset_path << "_*.pt\n";
        }
        
        std::cout << "\n✅ Iteration " << iteration << " complete!\n";
    }
    
    // SAVE FINAL DATASET
    dataset.save(CHECKPOINT_DIR + "/dataset");
    std::cout << "\n🎉 Training complete! Total games played: " << total_games << "\n";
    std::cout << "📊 Upload to W&B:\n";
    std::cout << "   1. Install: pip install wandb\n";
    std::cout << "   2. Login: wandb login\n";
    std::cout << "   3. Create new run and upload " << log_dir << "/metrics.csv\n";
}


void init() {
    torch::manual_seed(0);
    torch::Device device(torch::kCUDA);
    if (!torch::cuda::is_available()) device = torch::kCPU;
    
    auto model = std::make_shared<AlphaZModel>();
    model->save_model("checkpoint/best.pt");
}

int clean() {
    std::string folder = "checkpoint";

    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.is_regular_file()) {
            fs::path p = entry.path();

            // Check if extension is ".pt"
            if (p.extension() == ".pt") {
                // Check if the filename is numeric.pt
                std::string name = p.stem().string(); // before .pt

                bool numeric = true;
                for (char c : name) {
                    if (!isdigit(c)) {
                        numeric = false;
                        break;
                    }
                }

                // If numeric, remove
                if (numeric) {
                    std::cout << "Removing: " << p << "\n";
                    fs::remove(p);
                }
            }
        }
    }
    return 0;
}

std::pair<EvaluationResults, EvaluationResults> evaluate(std::string best_model_path, std::string candidate_model_path) {
    

    // CONFIGURATION PARAMETERS
    
    const int BOARD_SIZE = 4;
    
    // Self-play configuration
    const int NUM_PARALLEL_GAMES = 12;
    const int GAMES_PER_PHASE = 100;
    const int BATCH_SIZE = 12;
    const int MCTS_SIMULATIONS = 100;
    
    // MCTS parameters (No exploration noise for evaluation)
    const double EXPLORATION_FACTOR = 1.41;
    const float DIRICHLET_ALPHA = 0.3f;
    const float DIRICHLET_EPSILON = 0.0f;
    
    // Evaluation configuration
    const float REPLACEMENT_THRESHOLD = 55.0f;
    
    // Calculate games per worker
    int games_per_worker = (GAMES_PER_PHASE + NUM_PARALLEL_GAMES - 1) / NUM_PARALLEL_GAMES;
    

    // DEVICE 
    torch::manual_seed(0);
    torch::Device device(torch::kCUDA);
    if (!torch::cuda::is_available()) device = torch::kCPU;
    

    // LOAD MODELS
    auto best_model = AlphaZModel::load_model(best_model_path);
    best_model->to(device);
    best_model->eval();
    
    auto candidate_model = AlphaZModel::load_model(candidate_model_path);
    candidate_model->to(device);
    candidate_model->eval();
    

    // INITIALIZE MANAGER
    SelfPlayManager selfplay_manager;
    
    std::cout << "\n====================\n";
    std::cout << "🔁    EVALUATION \n";
    std::cout << "======================\n\n";
    
    std::cout << "▶ Phase 1: Playing " << GAMES_PER_PHASE << " games - Best as X, Candidate as O\n\n";
    
    EvaluationResults phase1_results = selfplay_manager.generate_evaluation_games(
        NUM_PARALLEL_GAMES,
        games_per_worker,
        best_model,
        candidate_model,
        BATCH_SIZE,
        MCTS_SIMULATIONS,
        BOARD_SIZE,
        EXPLORATION_FACTOR,
        DIRICHLET_ALPHA,
        DIRICHLET_EPSILON,
        Cell_state::O
    );
    
    
    int best_wins = phase1_results.model1_wins;
    int candidate_wins = phase1_results.model2_wins;
    int draws = phase1_results.draws;
    
    std::cout << "Phase 1 Results: Best: " << best_wins 
              << " Candidate: " << candidate_wins 
              << " Draws: " << draws << "\n\n";
  
    
    std::cout << "▶ Phase 2: Playing " << GAMES_PER_PHASE << " games - Candidate as X, Best as O\n\n";
    
    EvaluationResults phase2_results = selfplay_manager.generate_evaluation_games(
        NUM_PARALLEL_GAMES,
        games_per_worker,
        candidate_model,
        best_model,
        BATCH_SIZE,
        MCTS_SIMULATIONS,
        BOARD_SIZE,
        EXPLORATION_FACTOR,
        DIRICHLET_ALPHA,
        DIRICHLET_EPSILON,
        Cell_state::X
    );
    
    // Accumulate results (candidate is player1 in phase 2, so wins are swapped)
    candidate_wins += phase2_results.model1_wins;
    best_wins += phase2_results.model2_wins;
    draws += phase2_results.draws;
    
    int total_games = phase1_results.total_games + phase2_results.total_games;
    
    std::cout << "Phase 2 Results: Best: " << phase2_results.model2_wins 
              << " Candidate: " << phase2_results.model1_wins 
              << " Draws: " << phase2_results.draws << "\n\n";
    
    // Scoring: win=1, draw=0.5, loss=0
    float candidate_score = candidate_wins + (draws * 0.5f);
    float best_score = best_wins + (draws * 0.5f);
    
    float candidate_winrate = (candidate_score / total_games) * 100.0f;
    float best_winrate = (best_score / total_games) * 100.0f;
    
    std::cout << "\n=============================\n";
    std::cout << "🏁 EVALUATION COMPLETE\n";
    std::cout << "=============================\n";
    std::cout << "Results:\n";
    std::cout << "  Best model wins     : " << best_wins << "/" << total_games << "\n";
    std::cout << "  Candidate wins      : " << candidate_wins << "/" << total_games << "\n";
    std::cout << "  Draws               : " << draws << "/" << total_games << "\n";
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Win Rates:\n";
    std::cout << "  Best model          : " << best_winrate << "%\n";
    std::cout << "  Candidate           : " << candidate_winrate << "%\n\n";
    
    
    bool should_replace = (candidate_winrate >= REPLACEMENT_THRESHOLD);
    
    if (should_replace) {
        std::cout << "✅ Candidate achieves >= " << REPLACEMENT_THRESHOLD 
                  << "% win rate. Promoting to best model...\n\n";
        
        // Archive old best model
        fs::path oldFile = best_model_path;
        std::string baseName = "best-old-";
        std::string extension = ".pt";
        int counter = 1;
        fs::path newFile;
        
        do {
            newFile = oldFile.parent_path() / (baseName + std::to_string(counter) + extension);
            counter++;
        } while (fs::exists(newFile));
        
        try {
            fs::rename(oldFile, newFile);
            fs::copy_file(candidate_model_path, best_model_path);
            std::cout << "💾 Best model updated: " << best_model_path << "\n";
        } catch (const std::exception &e) {
            std::cerr << "❌ Error updating best model: " << e.what() << '\n';
        }
    } else {
        std::cout << "❌ Candidate achieves < " << REPLACEMENT_THRESHOLD 
                  << "%. Current best model kept.\n";
    }
  return {phase1_results, phase2_results};
}

void evaluate_vs_minimax(){

  GameDataset dataset(1000);
  torch::manual_seed(0);
  torch::Device device(torch::kCUDA);
  if (!torch::cuda::is_available()) device = torch::kCPU;

  auto last_model  = AlphaZModel::load_model("checkpoint/best.pt");
  last_model->to(device);
  last_model->eval();

  // HYPERPARAMETERS
  double exploration_factor = 1.41;
  int number_iteration = 500;
  float temperature = 0.0;
  float dirichlet_alpha = 0.3;
  float dirichlet_epsilon = 0.25;
  int max_depth = -1;
  bool tree_reuse = false;

  int game_counter = 0;
  int minimax_win = 0;
  int best_model_win = 0;
  int draw = 0;
  int game_number = 2;

  std::cout << "\n=============================\n";
  std::cout << "🔁 EVALUATION VS MINIMAX\n";
  std::cout << "=============================\n\n";

  /* std::cout << "▶ Phase 1: Playing " << game_number / 2 << " games - Minimax as X, Candidate as O\n\n";

  while (game_counter < game_number / 2)
  {
      std::cout << "Game " << game_counter + 1 << " | Minimax = X | Best = O ... ";

      Game game(9,
                std::make_unique<Minimax_player>(16, true, LogLevel::EVERYTHING),
                std::make_unique<Mcts_player>(exploration_factor, number_iteration, LogLevel::NONE,
                                              temperature, dirichlet_alpha, dirichlet_epsilon,
                                              last_model, max_depth, tree_reuse),
                dataset,
                true);

      Cell_state winner = game.play();

      if (winner == Cell_state::X){
        minimax_win++;
        std::cout << "Winner: Minimax";
      }
      else if (winner == Cell_state::O){
        best_model_win++;
        std::cout << "Winner: Best model";
      }
      else {
        draw++;
        std::cout << "Draw";
      }

      std::cout << " | Score → Minimax: " << minimax_win
                << " Best: " << best_model_win << "\n";

      game_counter++;
  } */
 
  std::cout << "▶ Phase 1: Playing " << game_number / 2 << " games - Best as X, Minimax as O\n\n";

  while (game_counter < game_number)
  {
      std::cout << "Game " << game_counter + 1 << " | Best = X | Minimax = O ... ";

      Game game(9,
                std::make_unique<Mcts_player>(exploration_factor, number_iteration, LogLevel::NONE,
                                              temperature, dirichlet_alpha, dirichlet_epsilon,
                                              last_model, max_depth, tree_reuse),
                std::make_unique<Minimax_player>(16, true, LogLevel::EVERYTHING),
                dataset,
                true);

      Cell_state winner = game.play();

      if (winner == Cell_state::X){
        best_model_win++;
        std::cout << "Winner: Best model";
      }
      else if (winner == Cell_state::O){
        minimax_win++;
        std::cout << "Winner: Minimax";
      }
      else {
        draw++;
        std::cout << "Draw";
      }

      std::cout << " | Score → Minimax: " << minimax_win
                << " Best: " << best_model_win << "\n";

      game_counter++;
  } 

  std::cout << "\n=============================\n";
  std::cout << "🏁 EVALUATION COMPLETE\n";
  std::cout << "=============================\n";

  std::cout << "✅ Minimax wins      : " << minimax_win << "\n";
  std::cout << "✅ Best model wins   : " << best_model_win << "\n";
  std::cout << "🤝 Draws             : " << draw << "\n";

  if (minimax_win + best_model_win > 0){
    float minimax_winrate =
        minimax_win / static_cast<float>(minimax_win + best_model_win) * 100.0f;

    float best_model_winrate =
        best_model_win / static_cast<float>(minimax_win + best_model_win) * 100.0f;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "📊 Minimax winrate   : " << minimax_winrate << "%\n";
    std::cout << "📊 Best model winrate: " << best_model_winrate << "%\n\n";
  }
  else{
    std::cout << "⚠️  All games were draws\n\n";
  }
}

void start_human_arena() {
  GameDataset dataset(1000);
  int board_size = 4;
  auto human_player_1 = std::make_unique<Human_player>();
  auto human_player_2 = std::make_unique<Human_player>();
  Game game(board_size, std::move(human_player_1), std::move(human_player_2), dataset);
  game.simple_play();
}

void vs_minimax() {
  GameDataset dataset(1000);
  int board_size = 9;
  auto human_player_2 = std::make_unique<Human_player>();
  Game game(board_size, std::make_unique<Minimax_player>(16, true, LogLevel::EVERYTHING),std::move(human_player_2), dataset);
  std::cout << "Here";
  game.simple_play();
}

void run_console_interface() {
  print_welcome_ascii_art();
  std::cout << "Hi ;).\n";
  bool is_running = true;
  while (is_running) {
    try {
      int option = 0;
      std::cout << "\nMENU:\n"
                << "[1] Human player vs Human player\n"
                << "[2] AI player vs AI player\n"
                << "[3] Human player vs AI player\n"
                << "[4] Initialize NN to random weights\n"
                << "[5] Launch Self-Play\n"
                << "[6] Player against Minimax\n"
                << "[7] Evaluate best model vs Minimax\n"
                << "[8] (H)Exit\n";

      option = get_parameter_within_bounds("Option: ", 1, 10);
      std::cout << "\n";

      switch (option) {
        case 1:
           start_human_arena();
           break;
        case 2:
            start_robot_arena();
            break;
        case 3:
            start_match_against_robot();
          break;
        case 4:
            init();
          break;
        case 5:
          selfplay();
          break;
        case 6:
          vs_minimax();
          break;
        case 7:
          evaluate_vs_minimax();
          break;
        case 8:
          is_running = false;
          break;
        case 9:
          evaluate("checkpoint/best-old-4.pt", "checkpoint/1k.pt");
          break;
        default:
          evaluate_vs_minimax();
          break;
      }
    } catch (const std::invalid_argument& e) {
      std::cout << "Error: " << e.what() << "\n";
    } catch (const std::logic_error& e) {
      std::cout << "Error: " << e.what() << "\n";
    } catch (const std::runtime_error& e) {
      std::cout << "Error: " << e.what() << "\n";
    }
  }
  print_exit_ascii_art();
}

void print_welcome_ascii_art() {
  std::cout << R"(

(_   _)_      (_   _)          (_   _)            
  | | (_)   ___ | |   _ _    ___ | |   _      __  
  | | | | /'___)| | /'_` ) /'___)| | /'_`\  /'__`\
  | | | |( (___ | |( (_| |( (___ | |( (_) )(  ___/
  (_) (_)`\____)(_)`\__,_)`\____)(_)`\___/'`\____)
                                        by AinaHerimam          

)" << '\n';
}

void print_board_and_winner(Board& board) {
  board.display_board(std::cout);
  Cell_state winner = board.check_winner();
  std::cout << "Winner: " << winner << std::endl;
  std::cout << "------------------" << std::endl;
}

void print_exit_ascii_art() {
  std::cout << R"(
       
Thank you!!!
              ┛         
)" << '\n';
}

