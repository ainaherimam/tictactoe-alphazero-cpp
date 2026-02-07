#include "triton_inference_client.h"
#include <grpc_client.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <cstring>
#include "constants.h"

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

InferenceClient::~InferenceClient() {
    // Unique_ptr handles cleanup automatically
}

void InferenceClient::connect() {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    std::cout << "[InferenceClient] Connecting to Triton server at " 
              << server_url_ << "..." << std::endl;
    
    // Create gRPC client
    std::unique_ptr<InferenceServerGrpcClient> temp_client;
    Error err = InferenceServerGrpcClient::Create(&temp_client, server_url_);
    
    if (!err.IsOk()) {
        throw std::runtime_error(
            "Failed to create Triton gRPC client: " + err.Message()
        );
    }
    
    client_ = std::move(temp_client);
    
    // Verify server is alive
    bool is_live = false;
    err = client_->IsServerLive(&is_live);
    
    if (!err.IsOk() || !is_live) {
        throw std::runtime_error(
            "Triton server not reachable at " + server_url_
        );
    }
    
    // Verify model is ready
    bool is_ready = false;
    err = client_->IsModelReady(&is_ready, model_name_);
    
    if (!err.IsOk() || !is_ready) {
        throw std::runtime_error(
            "Model '" + model_name_ + "' not ready on Triton server"
        );
    }
    
    std::cout << "[InferenceClient] ✓ Connected successfully. Model '"
              << model_name_ << "' is ready." << std::endl;
}

#include <chrono>
#include <iostream>
#include <thread>

void InferenceClient::infer(
    const float* input,
    const float* mask,
    float* policy,
    float* value
) {
    using clock = std::chrono::steady_clock;

    total_requests_++;

    const auto total_start = clock::now();

    int attempt = 0;
    std::exception_ptr last_exception;

    while (attempt <= max_retries_) {
        try {
            if (attempt > 0) {
                total_retries_++;

                int backoff_ms = 10 * (1 << (attempt - 1));
                std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));

                std::cout << "[InferenceClient] Retry "
                          << attempt << "/" << max_retries_
                          << " (backoff: " << backoff_ms << " ms)\n";
            }

            const auto attempt_start = clock::now();

            // Perform inference (throws on error)
            infer_internal(input, mask, policy, value);

            const auto attempt_end = clock::now();
            const auto attempt_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    attempt_end - attempt_start).count();

            const auto total_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    attempt_end - total_start).count();

            std::cout << "[InferenceClient] Inference succeeded "
                      << "(attempt " << (attempt + 1) << ") "
                      << "attempt_time=" << attempt_ms << " ms, "
                      << "total_time=" << total_ms << " ms\n";

            return; // Success

        } catch (const std::exception& e) {
            last_exception = std::current_exception();

            std::cout << "[InferenceClient] Inference failed "
                      << "(attempt " << (attempt + 1) << "): "
                      << e.what() << "\n";

            attempt++;
        }
    }

    failed_requests_++;

    const auto total_end = clock::now();
    const auto total_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            total_end - total_start).count();

    std::cout << "[InferenceClient] Inference failed after "
              << max_retries_ << " retries "
              << "(total_time=" << total_ms << " ms)\n";

    if (last_exception) {
        std::rethrow_exception(last_exception);
    }

    throw std::runtime_error(
        "Inference failed after " + std::to_string(max_retries_) + " retries"
    );
}


void InferenceClient::infer_internal(
    const float* input,
    const float* mask,
    float* policy,
    float* value
) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // ========================================================================
    // Prepare input tensors
    // ========================================================================
    
    std::vector<int64_t> board_shape = {1, INPUT_PLANES, X_, Y_};  // [1, 3, 4, 4]
    std::vector<int64_t> mask_shape = {1, POLICY_SIZE};             // [1, 16]
    
    InferInput* board_input_ptr = nullptr;
    InferInput* mask_input_ptr = nullptr;
    
    Error err = InferInput::Create(&board_input_ptr, "boards", board_shape, "FP32");
    if (!err.IsOk()) {
        throw std::runtime_error("Failed to create 'boards' input: " + err.Message());
    }
    std::shared_ptr<InferInput> board_input(board_input_ptr);
    
    err = InferInput::Create(&mask_input_ptr, "mask", mask_shape, "FP32");
    if (!err.IsOk()) {
        throw std::runtime_error("Failed to create 'mask' input: " + err.Message());
    }
    std::shared_ptr<InferInput> mask_input(mask_input_ptr);
    
    // Reshape input from [48] to [1, 3, 4, 4]
    std::vector<float> board_data(INPUT_SIZE);
    std::memcpy(board_data.data(), input, INPUT_SIZE * sizeof(float));
    
    err = board_input->AppendRaw(
        reinterpret_cast<const uint8_t*>(board_data.data()),
        board_data.size() * sizeof(float)
    );
    if (!err.IsOk()) {
        throw std::runtime_error("Failed to append board data: " + err.Message());
    }
    
    // Copy mask [16] → [1, 16]
    std::vector<float> mask_data(POLICY_SIZE);
    std::memcpy(mask_data.data(), mask, POLICY_SIZE * sizeof(float));
    
    err = mask_input->AppendRaw(
        reinterpret_cast<const uint8_t*>(mask_data.data()),
        mask_data.size() * sizeof(float)
    );
    if (!err.IsOk()) {
        throw std::runtime_error("Failed to append mask data: " + err.Message());
    }
    
    // ========================================================================
    // Prepare output tensors
    // ========================================================================
    
    InferRequestedOutput* policy_output_ptr = nullptr;
    InferRequestedOutput* value_output_ptr = nullptr;
    
    err = InferRequestedOutput::Create(&policy_output_ptr, "policy");
    if (!err.IsOk()) {
        throw std::runtime_error("Failed to create 'policy' output: " + err.Message());
    }
    std::shared_ptr<InferRequestedOutput> policy_output(policy_output_ptr);
    
    err = InferRequestedOutput::Create(&value_output_ptr, "value");
    if (!err.IsOk()) {
        throw std::runtime_error("Failed to create 'value' output: " + err.Message());
    }
    std::shared_ptr<InferRequestedOutput> value_output(value_output_ptr);
    
    // ========================================================================
    // Send inference request
    // CRITICAL: NO MUTEX HERE
    // Each client instance has its own gRPC connection
    // Multiple clients can send requests in parallel
    // ========================================================================
    
    auto network_start = std::chrono::high_resolution_clock::now();
    
    std::vector<InferInput*> inputs = {board_input.get(), mask_input.get()};
    std::vector<const InferRequestedOutput*> outputs = {
        policy_output.get(), 
        value_output.get()
    };
    
    InferOptions options(model_name_);
    options.client_timeout_ = static_cast<uint64_t>(timeout_ms_) * 1000;
    
    InferResult* result_ptr = nullptr;
    
    // Direct call - no locking needed since each client is independent
    err = client_->Infer(&result_ptr, options, inputs, outputs);
    
    if (!err.IsOk()) {
        throw std::runtime_error("Inference request failed: " + err.Message());
    }
    
    std::unique_ptr<InferResult> result(result_ptr);
    
    auto network_end = std::chrono::high_resolution_clock::now();
    double network_time = std::chrono::duration<double, std::milli>(network_end - network_start).count();
    
    // ========================================================================
    // Extract results
    // ========================================================================
    
    // Get policy output [1, 16]
    const uint8_t* policy_raw = nullptr;
    size_t policy_byte_size = 0;
    
    err = result->RawData("policy", &policy_raw, &policy_byte_size);
    if (!err.IsOk()) {
        throw std::runtime_error("Failed to get policy output: " + err.Message());
    }
    
    if (policy_byte_size != POLICY_SIZE * sizeof(float)) {
        throw std::runtime_error(
            "Policy output size mismatch: expected " + 
            std::to_string(POLICY_SIZE * sizeof(float)) + 
            " bytes, got " + std::to_string(policy_byte_size)
        );
    }
    
    std::memcpy(policy, policy_raw, policy_byte_size);
    
    // Get value output [1]
    const uint8_t* value_raw = nullptr;
    size_t value_byte_size = 0;
    
    err = result->RawData("value", &value_raw, &value_byte_size);
    if (!err.IsOk()) {
        throw std::runtime_error("Failed to get value output: " + err.Message());
    }
    
    if (value_byte_size != sizeof(float)) {
        throw std::runtime_error(
            "Value output size mismatch: expected " + 
            std::to_string(sizeof(float)) + 
            " bytes, got " + std::to_string(value_byte_size)
        );
    }
    
    std::memcpy(value, value_raw, value_byte_size);
    
    // ========================================================================
    // Update statistics
    // ========================================================================
    
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    
    // Atomic update of average latency
    double current_total = total_latency_ms_.load();
    total_latency_ms_.store(current_total + total_time);
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
    double total_latency = total_latency_ms_.load();
    
    Stats stats;
    stats.total_requests = total;
    stats.failed_requests = failed;
    stats.total_retries = retries;
    stats.avg_latency_ms = (total > 0) ? (total_latency / total) : 0.0;
    
    return stats;
}

void InferenceClient::reset_stats() {
    total_requests_ = 0;
    failed_requests_ = 0;
    total_retries_ = 0;
    total_latency_ms_ = 0.0;
}