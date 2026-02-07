#include "triton_inference_client.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <mutex>

/**
 * INDEPENDENT CLIENT INSTANCES TEST (Synchronous)
 * 
 * Each worker creates its own InferenceClient:
 * - No shared gRPC connection
 * - No mutex contention between workers
 * - True parallel requests to server
 */

std::mutex cout_mutex;

void safe_print(const std::string& msg) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << msg << std::flush;
}

void test_independent_sync_clients(
    const std::string& server_url,
    const std::string& model_name,
    int num_workers = 8,
    int iterations_per_worker = 100
) {
    std::cout << "\n=== INDEPENDENT SYNC CLIENTS TEST ===" << std::endl;
    std::cout << "Workers: " << num_workers << std::endl;
    std::cout << "Iterations per worker: " << iterations_per_worker << std::endl;
    std::cout << "Each worker has its own InferenceClient\n" << std::endl;
    
    std::vector<std::thread> threads;
    std::atomic<int> completed{0};
    std::atomic<int> failed{0};
    std::vector<double> all_latencies;
    std::mutex latency_mutex;
    
    auto worker = [&](int worker_id) {
        std::vector<double> worker_latencies;
        
        try {
            // Each worker creates its OWN client
            std::ostringstream oss;
            oss << "[Worker " << worker_id << "] Creating client...\n";
            safe_print(oss.str());
            
            InferenceClient client(server_url, model_name, 1000, 3);
            
            oss.str("");
            oss << "[Worker " << worker_id << "] ✓ Client ready\n";
            safe_print(oss.str());
            
            float input[48], mask[16], policy[16], value;
            
            // Each worker uses different board states
            for (int i = 0; i < 48; ++i) {
                input[i] = 0.01f * ((worker_id * 7 + i) % 100);
            }
            for (int i = 0; i < 16; ++i) {
                mask[i] = 1.0f;
            }
            
            // Tight inference loop
            for (int iter = 0; iter < iterations_per_worker; ++iter) {
                auto start = std::chrono::high_resolution_clock::now();
                
                // Synchronous inference on worker's own client
                client.infer(input, mask, policy, &value);
                
                auto end = std::chrono::high_resolution_clock::now();
                double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
                
                worker_latencies.push_back(latency_ms);
                completed++;
                
                if (iter % 25 == 0) {
                    std::ostringstream oss;
                    oss << "[Worker " << worker_id << "] Iter " << iter 
                        << "/" << iterations_per_worker 
                        << " - Latency: " << std::fixed << std::setprecision(2) 
                        << latency_ms << "ms\n";
                    safe_print(oss.str());
                }
            }
            
            {
                std::lock_guard<std::mutex> lock(latency_mutex);
                all_latencies.insert(all_latencies.end(),
                                   worker_latencies.begin(),
                                   worker_latencies.end());
            }
            
            std::ostringstream summary;
            summary << "[Worker " << worker_id << "] ✓ Completed all iterations\n";
            
            auto stats = client.get_stats();
            summary << "[Worker " << worker_id << "] Stats: "
                    << "Total=" << stats.total_requests << ", "
                    << "Failed=" << stats.failed_requests << ", "
                    << "Avg=" << std::fixed << std::setprecision(2)
                    << stats.avg_latency_ms << "ms\n";
            safe_print(summary.str());
            
        } catch (const std::exception& e) {
            failed++;
            std::ostringstream oss;
            oss << "[Worker " << worker_id << "] ✗ Failed: " << e.what() << "\n";
            safe_print(oss.str());
        }
    };
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch all workers
    for (int i = 0; i < num_workers; ++i) {
        threads.emplace_back(worker, i);
    }
    
    // Wait for completion
    for (auto& t : threads) {
        t.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Results
    std::cout << "\n========================================" << std::endl;
    std::cout << "Results Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    
    int total_requests = num_workers * iterations_per_worker;
    std::cout << "Total requests: " << total_requests << std::endl;
    std::cout << "Completed: " << completed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << "Success rate: " << std::fixed << std::setprecision(1)
              << (100.0 * completed / total_requests) << "%" << std::endl;
    
    std::cout << "\nTotal wall time: " << duration.count() << "ms" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(1)
              << (1000.0 * completed / duration.count()) << " inferences/sec" << std::endl;
    
    if (!all_latencies.empty()) {
        std::sort(all_latencies.begin(), all_latencies.end());
        
        double sum = 0.0;
        for (double lat : all_latencies) sum += lat;
        double avg = sum / all_latencies.size();
        
        double p50 = all_latencies[all_latencies.size() / 2];
        double p95 = all_latencies[(int)(all_latencies.size() * 0.95)];
        double p99 = all_latencies[(int)(all_latencies.size() * 0.99)];
        double min = all_latencies.front();
        double max = all_latencies.back();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Latency Statistics (ms)" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Min:  " << min << std::endl;
        std::cout << "  P50:  " << p50 << std::endl;
        std::cout << "  Avg:  " << avg << std::endl;
        std::cout << "  P95:  " << p95 << std::endl;
        std::cout << "  P99:  " << p99 << std::endl;
        std::cout << "  Max:  " << max << std::endl;
        
        int under_2ms = 0, under_5ms = 0, under_10ms = 0, over_10ms = 0;
        for (double lat : all_latencies) {
            if (lat < 2.0) under_2ms++;
            else if (lat < 5.0) under_5ms++;
            else if (lat < 10.0) under_10ms++;
            else over_10ms++;
        }
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Latency Distribution" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "  < 2ms:   " << under_2ms << " ("
                  << std::setprecision(1) << (100.0 * under_2ms / all_latencies.size()) << "%)" << std::endl;
        std::cout << "  2-5ms:   " << under_5ms << " ("
                  << (100.0 * under_5ms / all_latencies.size()) << "%)" << std::endl;
        std::cout << "  5-10ms:  " << under_10ms << " ("
                  << (100.0 * under_10ms / all_latencies.size()) << "%)" << std::endl;
        std::cout << "  > 10ms:  " << over_10ms << " ("
                  << (100.0 * over_10ms / all_latencies.size()) << "%)" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "  INDEPENDENT SYNC CLIENTS Test" << std::endl;
    std::cout << "  (Each worker owns its client)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::string server_url = (argc > 1) ? argv[1] : "localhost:8001";
    int num_workers = (argc > 2) ? std::atoi(argv[2]) : 1;
    int iterations = (argc > 3) ? std::atoi(argv[3]) : 100;
    std::string model_name = "alphazero";
    
    std::cout << "Server: " << server_url << std::endl;
    std::cout << "Model: " << model_name << std::endl;
    std::cout << "Workers: " << num_workers << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    
    try {
        test_independent_sync_clients(server_url, model_name, num_workers, iterations);
        std::cout << "\n✓ Test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}