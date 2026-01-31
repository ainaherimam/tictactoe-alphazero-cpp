#include "inference_queue_shm.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <thread>
#include <cassert>
#include <cstring>

SharedMemoryInferenceQueue::SharedMemoryInferenceQueue(
    const std::string& shm_name)
    : shm_name_(shm_name), 
      shm_fd_(-1),
      buffer_(nullptr),
      next_job_id_(1) {
    
    // Ensure shm_name starts with '/'
    if (!shm_name_.empty() && shm_name_[0] != '/') {
        shm_name_ = "/" + shm_name_;
    }
    
    // Open shared memory (Python creates it)
    shm_fd_ = shm_open(shm_name_.c_str(), O_RDWR, 0666);
    if (shm_fd_ == -1) {
        throw std::runtime_error(
            "Failed to open shared memory '" + shm_name_ + "'. "
            "Error: " + std::string(strerror(errno)) + ". "
            "Is inference_server.py running?");
    }
    
    // Map into our address space
    size_t size = sizeof(SharedMemoryBuffer);
    void* addr = mmap(nullptr, size, 
                     PROT_READ | PROT_WRITE, 
                     MAP_SHARED, 
                     shm_fd_, 0);
    
    if (addr == MAP_FAILED) {
        close(shm_fd_);
        throw std::runtime_error("Failed to mmap shared memory: " + 
                               std::string(strerror(errno)));
    }
    
    buffer_ = static_cast<SharedMemoryBuffer*>(addr);
    
    // Verify sizes
    verify_buffer_sizes();
    
    std::cout << "[InferenceQueue] Connected to shared memory: " 
              << shm_name_ << std::endl;
}

SharedMemoryInferenceQueue::~SharedMemoryInferenceQueue() {
    if (buffer_) {
        munmap(buffer_, sizeof(SharedMemoryBuffer));
        buffer_ = nullptr;
    }
    if (shm_fd_ >= 0) {
        close(shm_fd_);
        shm_fd_ = -1;
    }
}

// ============================================================================
// CORE OPERATIONS
// ============================================================================

uint64_t SharedMemoryInferenceQueue::submit(
    const float* board_state,
    const float* legal_mask) {
    
    if (!board_state || !legal_mask) {
        throw std::invalid_argument("Null pointer passed to submit()");
    }
    
    // Allocate a slot (may block if full)
    int slot = allocate_slot();
    
    // Generate unique job ID
    uint64_t job_id = next_job_id_.fetch_add(1, std::memory_order_relaxed);
    
    // Get references
    auto& req = buffer_->requests[slot];
    
    // State should already be WRITING (from allocate_slot's CAS)
    JobState current_state = req.state.load(std::memory_order_acquire);
    if (current_state != JobState::WRITING) {
        std::cerr << "[InferenceQueue::submit] ERROR: Slot state corruption! "
                  << "Expected WRITING (1), got " << static_cast<int>(current_state) << std::endl;
        throw std::runtime_error("Slot state corruption in submit()!");
    }
    
    // Write job ID
    req.job_id.store(job_id, std::memory_order_relaxed);
    
    // Copy input data
    std::memcpy(req.board_state, board_state, INPUT_SIZE * sizeof(float));
    std::memcpy(req.legal_mask, legal_mask, POLICY_SIZE * sizeof(float));
    
    // Memory barrier: ensure all writes are visible
    std::atomic_thread_fence(std::memory_order_release);
    
    // Transition: WRITING -> READY
    req.state.store(JobState::READY, std::memory_order_release);
    
    // Update statistics
    buffer_->total_requests_submitted.fetch_add(1, std::memory_order_relaxed);
    
    return job_id;
}

bool SharedMemoryInferenceQueue::is_done(uint64_t job_id) {
    int slot = find_slot_for_job(job_id);
    if (slot == -1) {
        return false;
    }
    
    auto& req = buffer_->requests[slot];
    JobState state = req.state.load(std::memory_order_acquire);
    
    return state == JobState::DONE;
}

bool SharedMemoryInferenceQueue::get_response(
    uint64_t job_id,
    float* out_policy,
    float* out_value) {
    
    if (!out_policy || !out_value) {
        throw std::invalid_argument("Null pointer passed to get_response()");
    }
    
    int slot = find_slot_for_job(job_id);
    if (slot == -1) {
        return false;
    }
    
    auto& req = buffer_->requests[slot];
    auto& resp = buffer_->responses[slot];
    
    // Check if done
    JobState state = req.state.load(std::memory_order_acquire);
    if (state != JobState::DONE) {
        return false;
    }
    
    // Verify job ID matches
    uint64_t resp_job_id = resp.job_id.load(std::memory_order_acquire);
    if (resp_job_id != job_id) {
        throw std::runtime_error(
            "Response job ID mismatch! Expected " + std::to_string(job_id) + 
            ", got " + std::to_string(resp_job_id));
    }
    
    // Copy response data
    std::memcpy(out_policy, resp.policy, POLICY_SIZE * sizeof(float));
    *out_value = resp.value;
    
    // Mark slot as free
    req.state.store(JobState::FREE, std::memory_order_release);
    
    // Update statistics
    buffer_->total_requests_completed.fetch_add(1, std::memory_order_relaxed);
    
    return true;
}

void SharedMemoryInferenceQueue::wait(
    uint64_t job_id,
    float* out_policy,
    float* out_value,
    int timeout_ms) {
    
    auto start = std::chrono::steady_clock::now();
    auto timeout = std::chrono::milliseconds(timeout_ms);
    
    int sleep_us = 50;  // Start with 50 microseconds
    const int max_sleep_us = 1000;  // Max 1ms
    
    while (true) {
        // Try to get response
        if (get_response(job_id, out_policy, out_value)) {
            return;  // Success!
        }
        
        // Check timeout
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed > timeout) {
            throw std::runtime_error(
                "Inference timeout after " + 
                std::to_string(timeout_ms) + "ms for job " + 
                std::to_string(job_id));
        }
        
        // Exponential backoff for sleep
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        sleep_us = std::min(sleep_us * 2, max_sleep_us);
    }
}

// ============================================================================
// SLOT ALLOCATION
// ============================================================================

int SharedMemoryInferenceQueue::allocate_slot() {
    // Round-robin allocation with spinning
    uint32_t start_hint = buffer_->next_slot_hint.load(
        std::memory_order_relaxed);
    
    int attempts = 0;
    const int max_attempts_before_sleep = 1000;
    
    while (true) {
        // Try all slots starting from hint
        for (uint32_t i = 0; i < MAX_BATCH_SIZE; i++) {
            uint32_t slot = (start_hint + i) % MAX_BATCH_SIZE;
            auto& req = buffer_->requests[slot];
            
            JobState current_state = req.state.load(std::memory_order_acquire);
            
            // Try to claim FREE slot
            JobState expected = JobState::FREE;
            if (req.state.compare_exchange_strong(
                    expected, JobState::WRITING,
                    std::memory_order_acquire,
                    std::memory_order_relaxed)) {
                
                // Update hint for next allocation
                buffer_->next_slot_hint.store(
                    (slot + 1) % MAX_BATCH_SIZE,
                    std::memory_order_relaxed);
                
                return slot;
            }
        }
        
        // All slots full
        attempts++;
        
        if (attempts == 1 || attempts % 100 == 0) {
            std::cout << "[allocate_slot] All slots full, attempt " << attempts 
                      << ", pending=" << count_pending() << std::endl;
        }
        
        if (attempts % max_attempts_before_sleep == 0) {
            // After many attempts, sleep longer
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else {
            // Brief spin
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        
        // Update start hint for next iteration
        start_hint = buffer_->next_slot_hint.load(std::memory_order_relaxed);
    }
}

int SharedMemoryInferenceQueue::find_slot_for_job(uint64_t job_id) const {
    // Linear scan (could be optimized with a map, but this is fine for now)
    for (size_t slot = 0; slot < MAX_BATCH_SIZE; slot++) {
        auto& req = buffer_->requests[slot];
        
        // Skip free slots
        JobState state = req.state.load(std::memory_order_acquire);
        if (state == JobState::FREE) {
            continue;
        }
        
        // Check job ID
        if (req.job_id.load(std::memory_order_acquire) == job_id) {
            return static_cast<int>(slot);
        }
    }
    return -1;  // Not found
}

// ============================================================================
// STATUS API
// ============================================================================

bool SharedMemoryInferenceQueue::is_server_ready() const {
    return buffer_->server_ready.load(std::memory_order_acquire);
}

void SharedMemoryInferenceQueue::request_shutdown() {
    buffer_->shutdown_requested.store(true, std::memory_order_release);
    std::cout << "[InferenceQueue] Shutdown requested" << std::endl;
}

size_t SharedMemoryInferenceQueue::count_pending() const {
    size_t count = 0;
    for (size_t i = 0; i < MAX_BATCH_SIZE; i++) {
        JobState state = buffer_->requests[i].state.load(
            std::memory_order_acquire);
        if (state != JobState::FREE) {
            count++;
        }
    }
    return count;
}

SharedMemoryInferenceQueue::Stats 
SharedMemoryInferenceQueue::get_stats() const {
    return {
        buffer_->total_requests_submitted.load(std::memory_order_relaxed),
        buffer_->total_requests_completed.load(std::memory_order_relaxed),
        buffer_->total_batches_processed.load(std::memory_order_relaxed),
        count_pending()
    };
}

void SharedMemoryInferenceQueue::verify_buffer_sizes() const {
    std::cout << "[InferenceQueue] Structure sizes:" << std::endl;
    std::cout << "  EvalRequest:         " << sizeof(EvalRequest) << " bytes" << std::endl;
    std::cout << "  EvalResponse:        " << sizeof(EvalResponse) << " bytes" << std::endl;
    std::cout << "  SharedMemoryBuffer:  " << sizeof(SharedMemoryBuffer) << " bytes" << std::endl;
    std::cout << "  INPUT_SIZE:          " << INPUT_SIZE << std::endl;
    std::cout << "  POLICY_SIZE:         " << POLICY_SIZE << std::endl;
    std::cout << "  MAX_BATCH_SIZE:      " << MAX_BATCH_SIZE << std::endl;
}

void SharedMemoryInferenceQueue::dump_shared_memory_state() const {
    std::cout << "\n======================================" << std::endl;
    std::cout << "SHARED MEMORY STATE DUMP" << std::endl;
    std::cout << "======================================" << std::endl;
    
    std::cout << "Server ready: " 
              << buffer_->server_ready.load(std::memory_order_acquire) << std::endl;
    std::cout << "Shutdown requested: " 
              << buffer_->shutdown_requested.load(std::memory_order_acquire) << std::endl;
    std::cout << "Next slot hint: " 
              << buffer_->next_slot_hint.load(std::memory_order_acquire) << std::endl;
    std::cout << "Total submitted: " 
              << buffer_->total_requests_submitted.load(std::memory_order_acquire) << std::endl;
    std::cout << "Total completed: " 
              << buffer_->total_requests_completed.load(std::memory_order_acquire) << std::endl;
    std::cout << "Total batches: " 
              << buffer_->total_batches_processed.load(std::memory_order_acquire) << std::endl;
    
    std::cout << "\nREQUEST SLOTS:" << std::endl;
    for (size_t i = 0; i < MAX_BATCH_SIZE; i++) {
        auto& req = buffer_->requests[i];
        JobState state = req.state.load(std::memory_order_acquire);
        
        if (state != JobState::FREE) {
            uint64_t job_id = req.job_id.load(std::memory_order_acquire);
            std::cout << "  Slot " << i << ": state=" << static_cast<int>(state) 
                      << " job_id=" << job_id;
            
            // Show sample data
            std::cout << " board[0]=" << req.board_state[0]
                      << " legal[0]=" << req.legal_mask[0];
            std::cout << std::endl;
        }
    }
    
    std::cout << "======================================\n" << std::endl;
}