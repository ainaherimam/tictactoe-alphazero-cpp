#include "training/training_shm_writer.h"
// training_shm_protocol.h is transitively included via training_shm_writer.h
// and brings in constants.h (INPUT_SIZE, POLICY_SIZE) — no need to repeat.
#include "core/mcts/position_pool.h"   // PositionPool + Position (full definitions)

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

// ============================================================================
//  Constants shared between the writer (C++) and the reader (Python)
// ============================================================================

namespace {

constexpr char     kMagic[8]    = { 'A','Z','T','R','N','\x01','\x00','\x00' };
constexpr uint32_t kBoardFloats = static_cast<uint32_t>(INPUT_SIZE);   // all planes
constexpr uint32_t kPiFloats    = static_cast<uint32_t>(POLICY_SIZE);  // 16
constexpr uint32_t kMaskFloats  = static_cast<uint32_t>(POLICY_SIZE);  // 16

/// The fixed path used for both loading (on startup) and saving (dump_to_disk).
constexpr const char* kLastDataPath = "data/last_data.bin";

} // namespace

// ============================================================================
//  Constructor / destructor
// ============================================================================

TrainingShmWriter::TrainingShmWriter(size_t             max_capacity,
                                     const std::string& segment_name)
    : segment_name_(segment_name),
      max_capacity_(max_capacity),
      shm_fd_(-1),
      shm_base_(nullptr),
      header_(nullptr),
      positions_(nullptr)
{
    shm_size_ = training_segment_size(max_capacity_);

    std::cout << "[TrainingShmWriter] Creating segment: " << segment_name_
              << "  (capacity=" << max_capacity_
              << ", size="      << (shm_size_ / 1024.0 / 1024.0) << " MB)"
              << std::endl;

    // ── Create the POSIX SHM segment ────────────────────────────────────────
    shm_fd_ = shm_open(segment_name_.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd_ == -1) {
        throw std::runtime_error(
            "[TrainingShmWriter] shm_open failed for '" + segment_name_ +
            "': " + std::string(strerror(errno)));
    }

    if (ftruncate(shm_fd_, static_cast<off_t>(shm_size_)) == -1) {
        close(shm_fd_);
        shm_unlink(segment_name_.c_str());
        throw std::runtime_error(
            "[TrainingShmWriter] ftruncate failed: " +
            std::string(strerror(errno)));
    }

    shm_base_ = mmap(nullptr, shm_size_, PROT_READ | PROT_WRITE,
                     MAP_SHARED, shm_fd_, 0);
    if (shm_base_ == MAP_FAILED) {
        close(shm_fd_);
        shm_unlink(segment_name_.c_str());
        throw std::runtime_error(
            "[TrainingShmWriter] mmap failed: " +
            std::string(strerror(errno)));
    }

    // Zero-initialise so the reader never sees garbage.
    std::memset(shm_base_, 0, shm_size_);

    // ── Wire up typed pointers ───────────────────────────────────────────────
    header_    = static_cast<TrainingBufferHeader*>(shm_base_);
    positions_ = training_positions_ptr(shm_base_);

    std::cout << "[TrainingShmWriter] Segment created successfully" << std::endl;

    // ── Attempt to resume from the last persistent snapshot ─────────────────
    load_from_disk();
}

TrainingShmWriter::~TrainingShmWriter() {
    if (shm_base_ != nullptr && shm_base_ != MAP_FAILED) {
        munmap(shm_base_, shm_size_);
    }
    if (shm_fd_ != -1) {
        close(shm_fd_);
        shm_unlink(segment_name_.c_str());
        std::cout << "[TrainingShmWriter] Segment unlinked: "
                  << segment_name_ << std::endl;
    }
}

// ============================================================================
//  flush_game
// ============================================================================

void TrainingShmWriter::flush_game(const PositionPool& pool) {
    const size_t num_moves = pool.size();
    if (num_moves == 0) return;

    // Compile-time guards — sizes must match training_shm_protocol.h exactly.
    // TrainingPosition is 384 bytes (192 board + 64 pi + 4 z + 64 mask + 60 padding).
    // The _padding does NOT affect the field offsets, only total struct size.
    static_assert(sizeof(TrainingPosition) == TRAINING_POSITION_BYTES,
                  "TrainingPosition size mismatch — check training_shm_protocol.h");
    static_assert(sizeof(TrainingPosition::board) == TRAINING_BOARD_SIZE  * sizeof(float),
                  "TrainingPosition::board size mismatch");
    static_assert(sizeof(TrainingPosition::pi)    == TRAINING_POLICY_SIZE * sizeof(float),
                  "TrainingPosition::pi size mismatch");
    static_assert(sizeof(TrainingPosition::mask)  == TRAINING_POLICY_SIZE * sizeof(float),
                  "TrainingPosition::mask size mismatch");

    std::lock_guard<std::mutex> lock(write_mutex_);

    uint32_t write_idx =
        header_->write_index.load(std::memory_order_relaxed);

    for (size_t i = 0; i < num_moves; ++i) {
        const Position&   src  = pool.get_position(i);
        uint32_t          slot = write_idx % max_capacity_;
        TrainingPosition& dst  = positions_[slot];

        std::memcpy(dst.board, src.board.data(),  sizeof(dst.board));
        std::memcpy(dst.pi,    src.policy.data(), sizeof(dst.pi));
        dst.z = src.z;
        std::memcpy(dst.mask,  src.mask.data(),   sizeof(dst.mask));

        write_idx = (write_idx + 1) % static_cast<uint32_t>(max_capacity_);
    }

    header_->write_index.store(write_idx, std::memory_order_release);

    const uint32_t old_size =
        header_->current_size.load(std::memory_order_relaxed);
    const uint32_t new_size = std::min(
        old_size + static_cast<uint32_t>(num_moves),
        static_cast<uint32_t>(max_capacity_));
    header_->current_size.store(new_size, std::memory_order_release);

    header_->generation.fetch_add(1, std::memory_order_release);

    // Periodic console progress
    static std::atomic<uint64_t> total_games_flushed{0};
    const uint64_t n =
        total_games_flushed.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n % 20 == 0) {
        std::cout << "[TrainingShmWriter] Flushed " << n
                  << " games — buffer: " << new_size
                  << "/" << max_capacity_ << " positions" << std::endl;
    }
}

// ============================================================================
//  Persistence — public entry-point
// ============================================================================

void TrainingShmWriter::dump_to_disk() {
    std::lock_guard<std::mutex> lock(write_mutex_);
    dump_to_disk_locked();
}

// ============================================================================
//  dump_to_disk_locked   (caller must hold write_mutex_)
// ============================================================================

void TrainingShmWriter::dump_to_disk_locked() {
    // Ensure the output directory exists.
    std::error_code ec;
    std::filesystem::create_directories(
        std::filesystem::path(kLastDataPath).parent_path(), ec);
    if (ec) {
        std::cerr << "[TrainingShmWriter] WARNING: could not create directory for "
                  << kLastDataPath << ": " << ec.message() << std::endl;
        // Non-fatal — attempt the write anyway.
    }

    std::ofstream out(kLastDataPath, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "[TrainingShmWriter] ERROR: could not open dump file: "
                  << kLastDataPath << std::endl;
        return;
    }

    const uint32_t num_positions =
        header_->current_size.load(std::memory_order_relaxed);
    const uint32_t wi =
        header_->write_index.load(std::memory_order_relaxed);
    const uint32_t capacity = static_cast<uint32_t>(max_capacity_);

    // ── File header (48 bytes) ───────────────────────────────────────────────
    //
    //  Offset  Bytes  Field
    //  ------  -----  -----
    //     0      8    magic          ('A','Z','T','R','N',0x01,0x00,0x00)
    //     8      4    capacity       (ring-buffer size at write time)
    //    12      4    num_positions  (entries actually stored)
    //    16      4    board_floats   (should be INPUT_SIZE = 48)
    //    20      4    pi_floats      (should be POLICY_SIZE = 16)
    //    24      4    mask_floats    (should be POLICY_SIZE = 16)
    //    28      8    timestamp      (unix epoch, uint64)
    //    36      4    dump_idx       (monotonically increasing counter)
    //    40      4    write_index    (ring-buffer write cursor at save time)
    //    44      4    reserved       (0)
    //  Total: 48 bytes

    static std::atomic<uint32_t> s_dump_idx{0};
    const uint32_t dump_idx  = s_dump_idx.fetch_add(1, std::memory_order_relaxed);
    const uint64_t timestamp = static_cast<uint64_t>(std::time(nullptr));
    constexpr uint32_t kReserved = 0;

    out.write(kMagic,                                                  8);
    out.write(reinterpret_cast<const char*>(&capacity),                4);
    out.write(reinterpret_cast<const char*>(&num_positions),           4);
    out.write(reinterpret_cast<const char*>(&kBoardFloats),            4);
    out.write(reinterpret_cast<const char*>(&kPiFloats),               4);
    out.write(reinterpret_cast<const char*>(&kMaskFloats),             4);
    out.write(reinterpret_cast<const char*>(&timestamp),               8);
    out.write(reinterpret_cast<const char*>(&dump_idx),                4);
    out.write(reinterpret_cast<const char*>(&wi),                      4);
    out.write(reinterpret_cast<const char*>(&kReserved),               4);
    // ── Position records in chronological order ──────────────────────────────
    // The oldest live entry sits at write_index (the next-overwrite slot), so
    // we walk forward from there modulo capacity.
    for (uint32_t i = 0; i < num_positions; ++i) {
        const uint32_t      slot = (wi + i) % capacity;
        const TrainingPosition& pos  = positions_[slot];

        out.write(reinterpret_cast<const char*>(pos.board), sizeof(pos.board));
        out.write(reinterpret_cast<const char*>(pos.pi),    sizeof(pos.pi));
        out.write(reinterpret_cast<const char*>(&pos.z),    sizeof(pos.z));
        out.write(reinterpret_cast<const char*>(pos.mask),  sizeof(pos.mask));
    }

    out.flush();
    if (!out) {
        std::cerr << "[TrainingShmWriter] ERROR: write failure on: "
                  << kLastDataPath << std::endl;
    } else {
        std::cout << "[TrainingShmWriter] Snapshot → " << kLastDataPath
                  << "  (" << num_positions << " positions, "
                  << (static_cast<double>(out.tellp()) / 1024.0 / 1024.0)
                  << " MB)" << std::endl;
    }
}

// ============================================================================
//  load_from_disk   (called from constructor, no lock needed yet)
// ============================================================================

void TrainingShmWriter::load_from_disk() {
    std::ifstream in(kLastDataPath, std::ios::binary);
    if (!in) {
        std::cout << "[TrainingShmWriter] No existing data at '"
                  << kLastDataPath << "' — starting with empty buffer."
                  << std::endl;
        return;
    }

    // ── Read and validate header ─────────────────────────────────────────────
    char     magic[8]{};
    uint32_t capacity      = 0;
    uint32_t num_positions = 0;
    uint32_t board_floats  = 0;
    uint32_t pi_floats     = 0;
    uint32_t mask_floats   = 0;
    uint64_t timestamp     = 0;
    uint32_t dump_idx      = 0;
    uint32_t wi            = 0;
    uint32_t reserved      = 0;

    in.read(magic,                                               8);
    in.read(reinterpret_cast<char*>(&capacity),                  4);
    in.read(reinterpret_cast<char*>(&num_positions),             4);
    in.read(reinterpret_cast<char*>(&board_floats),              4);
    in.read(reinterpret_cast<char*>(&pi_floats),                 4);
    in.read(reinterpret_cast<char*>(&mask_floats),               4);
    in.read(reinterpret_cast<char*>(&timestamp),                 8);
    in.read(reinterpret_cast<char*>(&dump_idx),                  4);
    in.read(reinterpret_cast<char*>(&wi),                        4);
    in.read(reinterpret_cast<char*>(&reserved),                  4);

    if (!in) {
        std::cerr << "[TrainingShmWriter] WARNING: '" << kLastDataPath
                  << "' header is truncated — ignoring, starting empty."
                  << std::endl;
        return;
    }

    // Magic check
    if (std::memcmp(magic, kMagic, 8) != 0) {
        std::cerr << "[TrainingShmWriter] WARNING: '" << kLastDataPath
                  << "' has wrong magic — ignoring, starting empty."
                  << std::endl;
        return;
    }

    // Dimension sanity check
    if (board_floats != kBoardFloats ||
        pi_floats    != kPiFloats    ||
        mask_floats  != kMaskFloats) {
        std::cerr << "[TrainingShmWriter] WARNING: '" << kLastDataPath
                  << "' was built with different board dimensions — "
                  << "ignoring, starting empty." << std::endl;
        return;
    }

    // How many positions can we actually load given the current capacity?
    const uint32_t positions_to_load = std::min(
        num_positions,
        static_cast<uint32_t>(max_capacity_));

    if (positions_to_load == 0) {
        std::cout << "[TrainingShmWriter] Data file is empty — starting fresh."
                  << std::endl;
        return;
    }

    // Guard: struct size must match the protocol constant.  If padding or
    // field sizes ever change, the per-field reads below go out of sync.
    static_assert(sizeof(TrainingPosition) == TRAINING_POSITION_BYTES,
                  "TrainingPosition size mismatch — fix serialisation and protocol");

    // ── Skip oldest entries when the file is larger than our capacity ─────────
    // The file stores positions chronologically (oldest first).  We always want
    // the NEWEST positions, so seek past the ones we cannot fit.
    if (positions_to_load < num_positions) {
        const uint64_t skip_bytes =
            static_cast<uint64_t>(num_positions - positions_to_load)
            * sizeof(TrainingPosition);
        in.seekg(static_cast<std::streamoff>(skip_bytes), std::ios::cur);
        if (!in) {
            std::cerr << "[TrainingShmWriter] WARNING: '" << kLastDataPath
                      << "' seek failed while skipping old entries — "
                      << "ignoring, starting empty." << std::endl;
            return;
        }
        std::cout << "[TrainingShmWriter] File has " << num_positions
                  << " positions but capacity is " << max_capacity_
                  << " — skipping oldest " << (num_positions - positions_to_load)
                  << ", loading newest " << positions_to_load << std::endl;
    }

    // ── Read positions straight into the ring buffer ─────────────────────────
    // We fill slots 0 … positions_to_load-1 and set write_index to the next
    // slot, mirroring the invariant maintained by flush_game().
    uint32_t loaded = 0;
    for (uint32_t i = 0; i < positions_to_load; ++i) {
        TrainingPosition& dst = positions_[i % max_capacity_];

        in.read(reinterpret_cast<char*>(dst.board), sizeof(dst.board));
        in.read(reinterpret_cast<char*>(dst.pi),    sizeof(dst.pi));
        in.read(reinterpret_cast<char*>(&dst.z),    sizeof(dst.z));
        in.read(reinterpret_cast<char*>(dst.mask),  sizeof(dst.mask));

        if (!in) {
            std::cerr << "[TrainingShmWriter] WARNING: read error at position "
                      << i << " — loaded " << loaded
                      << " positions before truncation." << std::endl;
            break;
        }
        ++loaded;
    }

    if (loaded == 0) return;

    // Update the SHM header so the Python reader sees the pre-loaded data.
    // generation = 1 (non-zero) signals "there is already data here".
    header_->write_index.store(
        loaded % static_cast<uint32_t>(max_capacity_),
        std::memory_order_relaxed);
    header_->current_size.store(loaded,  std::memory_order_relaxed);
    header_->generation.store(1,         std::memory_order_relaxed);

    const std::time_t ts = static_cast<std::time_t>(timestamp);
    char ts_buf[32]{};
    std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%d %H:%M:%S",
                  std::localtime(&ts));

    std::cout << "[TrainingShmWriter] Resumed from '" << kLastDataPath << "':"
              << "\n  positions loaded : " << loaded
              << "\n  original capacity: " << capacity
              << "\n  saved on         : " << ts_buf
              << std::endl;
}

// ============================================================================
//  Control signals / queries
// ============================================================================

void TrainingShmWriter::shutdown() {
    header_->shutdown.store(true, std::memory_order_release);
    std::cout << "[TrainingShmWriter] Shutdown signal sent." << std::endl;
}

bool     TrainingShmWriter::is_shutdown()  const {
    return header_->shutdown.load(std::memory_order_acquire);
}
uint64_t TrainingShmWriter::generation()   const {
    return header_->generation.load(std::memory_order_acquire);
}
uint32_t TrainingShmWriter::current_size() const {
    return header_->current_size.load(std::memory_order_acquire);
}
uint32_t TrainingShmWriter::write_index()  const {
    return header_->write_index.load(std::memory_order_acquire);
}