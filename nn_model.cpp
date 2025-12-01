#include "nn_model.h"


GameDataset::GameDataset(size_t max_size_) : max_size(max_size_) {
    boards.resize(max_size);
    pi_targets.resize(max_size);
    z_targets.resize(max_size);
    legal_mask.resize(max_size);
}

std::string GameDataset::hash_board(const torch::Tensor& board) {
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
        std::string hash = hash_board(board);
        
        auto it = board_hash_map.find(hash);
        if (it != board_hash_map.end()) {
            
            size_t existing_idx = it->second;

            // Average policy targets element-wise
            pi_targets[existing_idx] = (pi_targets[existing_idx] + pi) / 2.0f;
            
            // Average existing values
            z_targets[existing_idx] = (z_targets[existing_idx] + z) / 2.0f;
            legal_mask[existing_idx] = mask;
        } else {
            // New unique position
            boards[next_index] = board;
            pi_targets[next_index] = pi;
            z_targets[next_index] = z;
            legal_mask[next_index] = mask;
            
            board_hash_map[hash] = next_index;
            
            next_index = (next_index + 1) % max_size;
            if (current_size < max_size)
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


AlphaZeroNetWithMaskImpl::AlphaZeroNetWithMaskImpl(int C, int H_, int W_, 
                                                   int num_moves, int channels, int n_blocks) 
    : H(H_), W(W_) {

    conv_in = register_module("conv_in", 
        torch::nn::Conv2d(torch::nn::Conv2dOptions(C, channels, 3).padding(1)));

    res_blocks = register_module("res_blocks", torch::nn::ModuleList());
    
    for (int i = 0; i < n_blocks; ++i) {
        auto block = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)),
            torch::nn::BatchNorm2d(channels),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)),
            torch::nn::BatchNorm2d(channels)
        );
        res_blocks->push_back(block);
    }

    policy_head_conv = register_module("policy_head_conv", 
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, 2, 1)));
    policy_fc = register_module("policy_fc", 
        torch::nn::Linear(2 * H * W, num_moves));

    value_head_conv = register_module("value_head_conv", 
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, 1, 1)));
    value_fc1 = register_module("value_fc1", 
        torch::nn::Linear(H * W, 64));
    value_fc2 = register_module("value_fc2", 
        torch::nn::Linear(64, 1));
}

std::pair<torch::Tensor, torch::Tensor> AlphaZeroNetWithMaskImpl::forward(torch::Tensor x, 
                                                                           torch::Tensor legal_mask) {
    x = torch::relu(conv_in->forward(x));
    
    for (size_t i = 0; i < res_blocks->size(); ++i) {
        auto residual = x.clone();
        x = res_blocks[i]->as<torch::nn::Sequential>()->forward(x);
        x = torch::relu(x + residual);
    }

    // --- Policy Head ---
    auto p = policy_head_conv->forward(x).view({x.size(0), -1});
    p = policy_fc->forward(p);
    if (legal_mask.defined()) {
        p = p.masked_fill(legal_mask == 0, -1e9);
    }
    p = torch::log_softmax(p, 1);

    // --- Value Head ---
    auto v = value_head_conv->forward(x).view({x.size(0), -1});
    v = torch::relu(value_fc1->forward(v));
    v = torch::tanh(value_fc2->forward(v)).squeeze(-1);

    return {p, v};
}

std::pair<torch::Tensor, torch::Tensor> AlphaZeroNetWithMaskImpl::predict(torch::Tensor x, 
                                                                           torch::Tensor legal_mask) {
    torch::NoGradGuard no_grad;
    return this->forward(x, legal_mask);
}

void AlphaZeroNetWithMaskImpl::save_model(const std::string& path) {
    try {
        torch::save(shared_from_this(), path);
        std::cout << "✅ Model saved to: " << path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "❌ Error saving model: " << e.msg() << std::endl;
    }
}

torch::nn::ModuleHolder<AlphaZeroNetWithMaskImpl> AlphaZeroNetWithMaskImpl::load_model(const std::string& path) {
    try {
        auto model = std::make_shared<AlphaZeroNetWithMaskImpl>(
            11, 3, 3, 9, 64, 6
        );
        torch::load(model, path);
        return model;
    } catch (const c10::Error& e) {
        std::cerr << "❌ Error loading model: " << e.msg() << std::endl;
        throw;
    }
}

torch::Tensor alphazero_loss(torch::Tensor policy_pred, torch::Tensor value_pred,
                             torch::Tensor pi_target, torch::Tensor z_target) {

    auto policy_loss = -(pi_target * policy_pred).sum(1).mean();
    auto value_loss = torch::mse_loss(value_pred, z_target);

    //MAYBE L2 REGULARIZATION HERE!!!
    return policy_loss + value_loss;
}


void train(
    AlphaZeroNetWithMask& model, 
    GameDataset& dataset, 
    int batch_size, 
    int epochs, 
    double lr, 
    torch::Device device
) {

    model->to(device);

    auto dataloader = torch::data::make_data_loader(
        dataset.map(torch::data::transforms::Stack<>()),
        batch_size
    );

    torch::optim::Adam optimizer(model->parameters(), lr);

    model->train();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double total_loss = 0.0;
        size_t batch_idx = 0;

        for (auto& batch : *dataloader) {

            // Move batch data + target to device
            auto b = batch.data.to(device);
            auto t = batch.target.to(device);

            // Split the targets
            auto pi_target = t.slice(1, 0, 9);
            auto z_target = t.slice(1, 9, 10).squeeze(1);
            auto mask = t.slice(1, 10, t.size(1));

            optimizer.zero_grad();

            auto [p, v] = model->forward(b, mask);

            auto loss = alphazero_loss(p, v, pi_target, z_target);

            loss.backward();
 
            optimizer.step();

            total_loss += loss.item<double>();
            batch_idx++;
        }

        std::cout << "Epoch " << epoch 
                  << "/" << epochs 
                  << " - Avg Loss: " << total_loss / batch_idx 
                  << std::endl;
    }
}
