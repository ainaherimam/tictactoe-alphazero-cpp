#include "inference/triton/triton_inference_client.h"
#include <grpc_client.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <cstring>
#include "core/game/constants.h"

using namespace triton::client;

InferenceClient::InferenceClient(
    const std::string& server_url,
    const std::string& model_name,
    int timeout_ms,
    int max_retries
)
    : server_url_(server_url)
    , model_name_(model_name)
    , timeout_ms_(timeout_ms)
    , max_retries_(max_retries)
{
    connect();
}

InferenceClient::~InferenceClient() {}

void InferenceClient::warmup() {
    std::cout << "[InferenceClient] Warming up..." << std::endl;

    float dummy_input[48] = {0};
    float dummy_mask[16] = {1.0f};
    float dummy_policy[16];
    float dummy_value;

    try {
        infer(dummy_input, dummy_mask, dummy_policy, &dummy_value);
        std::cout << "[InferenceClient] ✓ Warmup successful" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[InferenceClient] ⚠ Warmup failed: " << e.what() << std::endl;
    }
}

void InferenceClient::connect() {
    std::lock_guard<std::mutex> lock(connection_mutex_);

    std::cout << "[InferenceClient] Connecting to " << server_url_ << "..." << std::endl;

    std::unique_ptr<InferenceServerGrpcClient> temp_client;
    Error err = InferenceServerGrpcClient::Create(&temp_client, server_url_);

    if (!err.IsOk()) {
        throw std::runtime_error("Failed to create gRPC client: " + err.Message());
    }

    client_ = std::move(temp_client);

    // Verify server
    bool is_live = false;
    err = client_->IsServerLive(&is_live);
    if (!err.IsOk() || !is_live) {
        throw std::runtime_error("Server not reachable");
    }

    // Verify model
    bool is_ready = false;
    err = client_->IsModelReady(&is_ready, model_name_);
    if (!err.IsOk() || !is_ready) {
        throw std::runtime_error("Model not ready");
    }

    std::cout << "[InferenceClient] ✓ Connected. Model ready." << std::endl;
}

void InferenceClient::infer(
    const float* input,
    const float* mask,
    float* policy,
    float* value
) {
    total_requests_++;

    int attempt = 0;
    std::exception_ptr last_exception;

    while (attempt <= max_retries_) {
        try {
            if (attempt > 0) {
                total_retries_++;
                int backoff_ms = 10 * (1 << (attempt - 1));
                std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
                std::cout << "[InferenceClient] Retry " << attempt << "/" << max_retries_
                          << " (backoff: " << backoff_ms << "ms)" << std::endl;
            }

            infer_internal(input, mask, policy, value);
            return; // Success

        } catch (const std::exception& e) {
            last_exception = std::current_exception();
            std::cout << "[InferenceClient] Attempt " << (attempt + 1)
                      << " failed: " << e.what() << std::endl;
            attempt++;
        }
    }

    failed_requests_++;

    if (last_exception) {
        std::rethrow_exception(last_exception);
    }

    throw std::runtime_error("Inference failed after retries");
}

void InferenceClient::infer_internal(
    const float* input,
    const float* mask,
    float* policy,
    float* value
) {
    std::vector<int64_t> board_shape = {1, 48};
    std::vector<int64_t> mask_shape = {1, 16};

    InferInput* board_input_ptr = nullptr;
    InferInput* mask_input_ptr = nullptr;

    Error err = InferInput::Create(&board_input_ptr, "boards", board_shape, "FP32");
    if (!err.IsOk()) {
        throw std::runtime_error("Failed to create 'boards' input");
    }
    std::shared_ptr<InferInput> board_input(board_input_ptr);

    err = InferInput::Create(&mask_input_ptr, "mask", mask_shape, "FP32");
    if (!err.IsOk()) {
        throw std::runtime_error("Failed to create 'mask' input");
    }
    std::shared_ptr<InferInput> mask_input(mask_input_ptr);

    err = board_input->AppendRaw(
        reinterpret_cast<const uint8_t*>(input),
        48 * sizeof(float)
    );
    if (!err.IsOk()) {
        throw std::runtime_error("Failed to append board data");
    }

    err = mask_input->AppendRaw(
        reinterpret_cast<const uint8_t*>(mask),
        16 * sizeof(float)
    );
    if (!err.IsOk()) {
        throw std::runtime_error("Failed to append mask data");
    }

    InferRequestedOutput* policy_output_ptr = nullptr;
    InferRequestedOutput* value_output_ptr = nullptr;

    err = InferRequestedOutput::Create(&policy_output_ptr, "policy");
    if (!err.IsOk()) {
        throw std::runtime_error("Failed to create 'policy' output");
    }
    std::shared_ptr<InferRequestedOutput> policy_output(policy_output_ptr);

    err = InferRequestedOutput::Create(&value_output_ptr, "value");
    if (!err.IsOk()) {
        throw std::runtime_error("Failed to create 'value' output");
    }
    std::shared_ptr<InferRequestedOutput> value_output(value_output_ptr);

    std::vector<InferInput*> inputs = {board_input.get(), mask_input.get()};
    std::vector<const InferRequestedOutput*> outputs = {
        policy_output.get(),
        value_output.get()
    };

    InferOptions options(model_name_);
    options.client_timeout_ = static_cast<uint64_t>(timeout_ms_) * 1000;

    InferResult* result_ptr = nullptr;
    err = client_->Infer(&result_ptr, options, inputs, outputs);

    if (!err.IsOk()) {
        throw std::runtime_error("Inference failed: " + err.Message());
    }

    std::unique_ptr<InferResult> result(result_ptr);

    // Extract policy [1, 16]
    const uint8_t* policy_raw = nullptr;
    size_t policy_size = 0;
    err = result->RawData("policy", &policy_raw, &policy_size);
    if (!err.IsOk() || policy_size != 16 * sizeof(float)) {
        throw std::runtime_error("Policy output error");
    }
    std::memcpy(policy, policy_raw, policy_size);

    // Extract value [1, 1]
    const uint8_t* value_raw = nullptr;
    size_t value_size = 0;
    err = result->RawData("value", &value_raw, &value_size);
    if (!err.IsOk() || value_size != sizeof(float)) {
        throw std::runtime_error("Value output error");
    }
    std::memcpy(value, value_raw, value_size);
}

bool InferenceClient::is_server_alive() const {
    std::lock_guard<std::mutex> lock(connection_mutex_);

    if (!client_) {
        return false;
    }

    bool is_live = false;
    Error err = client_->IsServerLive(&is_live);

    return err.IsOk() && is_live;
}

InferenceClient::Stats InferenceClient::get_stats() const {
    uint64_t total = total_requests_.load();
    uint64_t failed = failed_requests_.load();
    uint64_t retries = total_retries_.load();

    Stats stats;
    stats.total_requests = total;
    stats.failed_requests = failed;
    stats.total_retries = retries;

    return stats;
}

void InferenceClient::reset_stats() {
    total_requests_ = 0;
    failed_requests_ = 0;
    total_retries_ = 0;
}