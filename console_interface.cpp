#include "console_interface.h"
#include <chrono>
#include <thread>
#include <climits>
#include <random>

#include "board.h"
#include "mcts_agent.h"
#include "nn_model.h"
#include "logger.h"


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
    auto eval_model = AlphaZeroNetWithMaskImpl::load_model("checkpoint/1.pt");
    eval_model->to(device);
    eval_model->eval();

    std::cout << "\nInitializing " << agent_prompt << ":\n";

    int max_iteration = get_parameter_within_bounds(
        "Max iteration number (at least 10) : ", 10, INT_MAX);

    double exploration_constant = 1.41;

    exploration_constant = get_parameter_within_bounds(
        "Enter exploration constant (between 0.1 and 2): ", 0.1, 2.0);

    LogLevel log_level = LogLevel::NONE;
    log_level = static_cast<LogLevel>(get_parameter_within_bounds("Log Level (0:None  -- 5:Full)  : ", 0, 5));
    return std::make_unique<Mcts_player>(exploration_constant, max_iteration,log_level, 0.0, 0.3, 0.25, eval_model, -1, false);
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

    int board_size = 9;

    auto mcts_agent = create_mcts_agent("agent");
    auto human_player = std::make_unique<Human_player>();

    if (human_player_number == 1) {
        Game game(board_size, std::move(human_player), std::move(mcts_agent), dataset);
        game.simple_play();
    }
    else {
        Game game(board_size, std::move(mcts_agent), std::move(human_player), dataset);
        game.simple_play();
    }
}

void start_robot_arena() {
    GameDataset dataset(1000);
    int board_size = 9;

    auto mcts_agent_1 = create_mcts_agent("first agent");
    auto mcts_agent_2 = create_mcts_agent("second agent");

    Game game(board_size, std::move(mcts_agent_1), std::move(mcts_agent_2), dataset);
    game.simple_play();
}


void selfplay() {
  torch::manual_seed(0);
  torch::Device device(torch::kCUDA);
  if (!torch::cuda::is_available()) device = torch::kCPU;

  //DATASET
  int data_number = 2000;
  int train_every = 500;
  int last_train_index = 0;
  GameDataset dataset(data_number);

  //THE MODEL TO TRAIN
  AlphaZeroNetWithMask model;

  //HYPERPARAMETERS
  double exploration_factor = 1.41;
  int number_iteration = 100;
  float temperature = 0.0;
  float dirichlet_alpha = 0.3;
  float dirichlet_epsilon = 0.25;
  int max_depth = -1;
  bool tree_reuse = false;

  //TRAINING HYPERPARAMETERS
  int batch_size = 64;
  int epoch = 15;
  double learning_rate = 1e-3;

  auto eval_model = AlphaZeroNetWithMaskImpl::load_model("checkpoint/1.pt");
  eval_model->to(device);
  eval_model->eval();  // Set to evaluation mode

  std::cout << "🌱 Starting self-play...\n";

  int cycles = ceil(data_number / train_every);
  int game_counter = 0;

  for (int iter = 1; iter <= cycles; ++iter)
  {
      std::cout << "\n=============================\n";
      std::cout << "🔁 TRAINING CYCLE " << iter << "\n";
      std::cout << "=============================\n\n";
      std::cout << "🌱 Collecting Data...\n";

      // SELF-PLAY PHASE
      int moves_since_last_train;

      if (dataset.next_index >= last_train_index)
          moves_since_last_train = dataset.next_index - last_train_index;
      else
          moves_since_last_train = (data_number - last_train_index) + dataset.next_index;

      // Play games until we reach the training threshold
      while (moves_since_last_train < train_every)
      {
          Game game(9,
                    std::make_unique<Mcts_player>(exploration_factor, number_iteration, LogLevel::NONE, temperature, dirichlet_alpha, dirichlet_epsilon, eval_model, max_depth, tree_reuse),
                    std::make_unique<Mcts_player>(exploration_factor, number_iteration, LogLevel::NONE, temperature, dirichlet_alpha, dirichlet_epsilon, eval_model, max_depth, tree_reuse),
                    dataset,
                    true);

          Cell_state winner = game.play();
          game_counter++;

          if (dataset.next_index >= last_train_index)
              moves_since_last_train = dataset.next_index - last_train_index;
          else
              moves_since_last_train = (data_number - last_train_index) + dataset.next_index;

          std::cout << game_counter << " Games completed - Stored positions: "
                    << dataset.current_size << "/" << data_number
                    << " - Player " << winner << " won\n";
      }
      
      std::cout << "✅ Replay buffer ready (" << dataset.current_size << " samples)\n";

      // TRAINING PHASE
      std::cout << "🎯 Training model...\n";
      train(model, dataset, batch_size, epoch, learning_rate, device);
      torch::save(model, "checkpoint/" + std::to_string(iter + 2) + ".pt");

      // Update last_train_index circularly
      last_train_index = (last_train_index + train_every) % data_number;
  }
  // Save the actual dataset to disk
  dataset.save("my_dataset");
}

void init() {
    torch::manual_seed(0);
    torch::Device device(torch::kCUDA);
    if (!torch::cuda::is_available()) device = torch::kCPU;

    AlphaZeroNetWithMask model;
    torch::save(model, "checkpoint/1.pt");
    std::cout << "✅ Model saved successfully!\n";
}


void start_human_arena() {
  GameDataset dataset(1000);
  int board_size = 9;
  auto human_player_1 = std::make_unique<Human_player>();
  auto human_player_2 = std::make_unique<Human_player>();
  Game game(board_size, std::move(human_player_1), std::move(human_player_2), dataset);
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
                << "[4] Initialize NN\n"
                << "[5] Launch Self-Play\n"
                << "[6] (H)Exit\n";

      option = get_parameter_within_bounds("Option: ", 1, 6);
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
          is_running = false;
          break;
        default:
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
