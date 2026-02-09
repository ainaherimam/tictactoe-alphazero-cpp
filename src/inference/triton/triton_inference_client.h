#ifndef TRITON_INFERENCE_CLIENT_H
#define TRITON_INFERENCE_CLIENT_H

#include <string>
#include <memory>
#include <atomic>
#include <mutex>

// Forward declare Triton types
namespace triton { namespace client {
    class InferenceServerGrpcClient;
    class Error;
}}

/**
 * Triton Inference Client for AlphaZero
 * 
 * DESIGNED FOR INDEPENDENT INSTANCES:
 * - Each worker should create its own client instance
 * - No shared state between clients
 * - Thread-safe for use by single thread or with internal locking
 * 
 * Usage pattern (RECOMMENDED):
 *   // Each worker thread creates its own client
 *   void worker_thread() {
 *       InferenceClient my_client("localhost:8001");
 *       for (int i = 0; i < iterations; i++) {
 *           my_client.infer(input, mask, policy, value);
 *       }
 *   }
 */
class InferenceClient {
public:
    struct Stats {
        uint64_t total_requests;
        uint64_t failed_requests;
        uint64_t total_retries;
        double avg_latency_ms;
    };
    
    /**
     * Create inference client
     * 
     * @param server_url  Triton server URL (e.g., "localhost:8001")
     * @param model_name  Model name on Triton server
     * @param timeout_ms  Request timeout in milliseconds
     * @param max_retries Maximum retry attempts on failure
     */
    InferenceClient(
        const std::string& server_url,
        const std::string& model_name = "alphazero",
        int timeout_ms = 1000,
        int max_retries = 3
    );
    
    ~InferenceClient();
    
    /**
     * Run synchronous inference
     * 
     * @param input  Board state [48] (reshaped to [1,3,4,4] internally)
     * @param mask   Action mask [16]
     * @param policy Output policy distribution [16]
     * @param value  Output value estimate (single float)
     */
    void infer(
        const float* input,
        const float* mask,
        float* policy,
        float* value
    );
    
    /**
     * Check if server is alive
     */
    bool is_server_alive() const;
    
    /**
     * Get client statistics
     */
    Stats get_stats() const;
    
    /**
     * Reset statistics
     */
    void reset_stats();

private:
    void connect();
    void infer_internal(const float* input, const float* mask, float* policy, float* value);
    
    // Connection details
    std::string server_url_;
    std::string model_name_;
    int timeout_ms_;
    int max_retries_;
    
    // Triton client (NOT shared between instances)
    std::unique_ptr<triton::client::InferenceServerGrpcClient> client_;
    
    // Statistics (atomic for thread-safety)
    std::atomic<uint64_t> total_requests_{0};
    std::atomic<uint64_t> failed_requests_{0};
    std::atomic<uint64_t> total_retries_{0};
    std::atomic<double> total_latency_ms_{0.0};
    
    // Mutex ONLY for connection management (rarely used)
    // NOT used during inference - each client instance is independent
    mutable std::mutex connection_mutex_;
};

#endif // TRITON_INFERENCE_CLIENT_H