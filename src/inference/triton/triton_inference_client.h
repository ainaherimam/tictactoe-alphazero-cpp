#pragma once
#include <string>
#include <memory>
#include <atomic>
#include <mutex>

// Forward declarations
namespace triton { namespace client {
    class InferenceServerGrpcClient;
}}

class InferenceClient {
public:
    struct Stats {
        uint64_t total_requests;
        uint64_t failed_requests;
        uint64_t total_retries;
    };

    /**
     * Create inference client.
     *
     * @param server_url  gRPC endpoint (e.g., "localhost:8001")
     * @param model_name  Model name on server
     * @param timeout_ms  Request timeout in milliseconds
     * @param max_retries Maximum retry attempts
     */
    InferenceClient(
        const std::string& server_url,
        const std::string& model_name = "AlphaZero",
        int timeout_ms = 5000,
        int max_retries = 3
    );
    ~InferenceClient();

    /**
     * Run inference.
     *
     * @param input  Board state [48] (flattened 3x4x4)
     * @param mask   Legal move mask [16]
     * @param policy Output policy [16]
     * @param value  Output value [1]
     */
    void infer(
        const float* input,
        const float* mask,
        float* policy,
        float* value
    );

    /** Send warmup request. */
    void warmup();

    /** Check if server is alive. */
    bool is_server_alive() const;

    /** Get statistics. */
    Stats get_stats() const;

    /** Reset statistics. */
    void reset_stats();

private:
    void connect();
    void infer_internal(
        const float* input,
        const float* mask,
        float* policy,
        float* value
    );

    std::string server_url_;
    std::string model_name_;
    int timeout_ms_;
    int max_retries_;

    std::unique_ptr<triton::client::InferenceServerGrpcClient> client_;
    mutable std::mutex connection_mutex_;

    // Statistics
    std::atomic<uint64_t> total_requests_{0};
    std::atomic<uint64_t> failed_requests_{0};
    std::atomic<uint64_t> total_retries_{0};
};