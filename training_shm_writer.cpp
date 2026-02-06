#include "training_shm_writer.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <cstring>
#include <iostream>

TrainingShmWriter::TrainingShmWriter(size_t max_capacity, 
                                     const std::string& segment_name)
    : segment_name_(segment_name),
      max_capacity_(max_capacity),
      shm_fd_(-1),
      shm_base_(nullptr),
      header_(nullptr),
      positions_(nullptr)
{
    // Calculate segment size
    shm_size_ = training_segment_size(max_capacity_);
    
    std::cout << "[TrainingShmWriter] Creating segment: " << segment_name_ 
              << " (capacity=" << max_capacity_ 
              << ", size=" << (shm_size_ / 1024.0 / 1024.0) << " MB)" << std::endl;
    
    // Create shared memory segment
    shm_fd_ = shm_open(segment_name_.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd_ == -1) {
        throw std::runtime_error("Failed to create training SHM segment: " + 
                               segment_name_ + " - " + std::string(strerror(errno)));
    }
    
    // Set segment size
    if (ftruncate(shm_fd_, shm_size_) == -1) {
        close(shm_fd_);
        shm_unlink(segment_name_.c_str());
        throw std::runtime_error("Failed to set training SHM size: " + 
                               std::string(strerror(errno)));
    }
    
    // Map segment into memory
    shm_base_ = mmap(nullptr, shm_size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
    if (shm_base_ == MAP_FAILED) {
        close(shm_fd_);
        shm_unlink(segment_name_.c_str());
        throw std::runtime_error("Failed to mmap training SHM: " + 
                               std::string(strerror(errno)));
    }
    
    // Zero-initialize the segment
    std::memset(shm_base_, 0, shm_size_);
    
    // Set up pointers
    header_ = static_cast<TrainingBufferHeader*>(shm_base_);
    positions_ = training_positions_ptr(shm_base_);
    
    std::cout << "[TrainingShmWriter] Segment created successfully" << std::endl;
}

TrainingShmWriter::~TrainingShmWriter() {
    if (shm_base_ != nullptr && shm_base_ != MAP_FAILED) {
        munmap(shm_base_, shm_size_);
    }
    
    if (shm_fd_ != -1) {
        close(shm_fd_);
        shm_unlink(segment_name_.c_str());
        std::cout << "[TrainingShmWriter] Segment unlinked: " << segment_name_ << std::endl;
    }
}

void TrainingShmWriter::flush_game(const PositionPool& pool) {
    size_t num_moves = pool.size();
    if (num_moves == 0) return;
    
    std::lock_guard<std::mutex> lock(write_mutex_);
    
    // Load write index once
    uint32_t write_idx = header_->write_index.load(std::memory_order_relaxed);
    
    for (size_t i = 0; i < num_moves; i++) {
        const Position& src = pool.get_position(i);
        
        uint32_t slot = write_idx % max_capacity_;
        TrainingPosition& dst = positions_[slot];

        
        // Copy data
        std::memcpy(dst.board, src.board.data(), TRAINING_BOARD_SIZE * sizeof(float));
        std::memcpy(dst.pi, src.policy.data(), TRAINING_POLICY_SIZE * sizeof(float));
        dst.z = src.z;
        std::memcpy(dst.mask, src.mask.data(), TRAINING_POLICY_SIZE * sizeof(float));
        
        write_idx = (write_idx + 1) % max_capacity_;
    }
    
    // Store updated index once
    header_->write_index.store(write_idx, std::memory_order_release);
    
    // Update current_size
    uint32_t old_size = header_->current_size.load(std::memory_order_relaxed);
    uint32_t new_size = std::min(old_size + static_cast<uint32_t>(num_moves), 
                                  static_cast<uint32_t>(max_capacity_));
    header_->current_size.store(new_size, std::memory_order_release);
    
    // Increment generation
    header_->generation.fetch_add(1, std::memory_order_release);

    static std::atomic<uint64_t> total_games_flushed{0};
    uint64_t current_count = total_games_flushed.fetch_add(1, std::memory_order_relaxed) + 1;

    if (current_count % 20 == 0) {  // Use current_count instead of total_games_flushed
        std::cout << "[TrainingShmWriter] Flushed " << current_count 
        << " Games - Buffer: " << new_size << "/" << max_capacity_ 
        << " positions" << std::endl;
    }
}

void TrainingShmWriter::shutdown() {
    header_->shutdown.store(true, std::memory_order_release);
    std::cout << "[TrainingShmWriter] Shutdown signal sent" << std::endl;
}

bool TrainingShmWriter::is_shutdown() const {
    return header_->shutdown.load(std::memory_order_acquire);
}

uint64_t TrainingShmWriter::generation() const {
    return header_->generation.load(std::memory_order_acquire);
}

uint32_t TrainingShmWriter::current_size() const {
    return header_->current_size.load(std::memory_order_acquire);
}

uint32_t TrainingShmWriter::write_index() const {
    return header_->write_index.load(std::memory_order_acquire);
}