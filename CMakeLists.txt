cmake_minimum_required(VERSION 3.15)
project(AlphaZero_TTT LANGUAGES CXX)

# ============================================================
# === Global Configuration ==================================
# ============================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# ============================================================
# === Dependencies ==========================================
# ============================================================

# ---- Torch -------------------------------------------------
# NOTE: Still needed for tensor operations in MCTS, just not for inference
if(NOT Torch_DIR AND DEFINED ENV{TORCH_HOME})
    set(Torch_DIR "$ENV{TORCH_HOME}/share/cmake/Torch")
endif()
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ---- Boost -------------------------------------------------
find_package(Boost REQUIRED COMPONENTS system)

# ---- Threads ----------------------------------------------
find_package(Threads REQUIRED)

# ---- OpenSSL (if needed) ----------------------------------
find_package(OpenSSL)

# ============================================================
# === Shared Memory Inference Library =======================
# ============================================================
# Header-only library for shared memory communication with JAX

add_library(shared_memory_inference INTERFACE)

target_include_directories(shared_memory_inference INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(shared_memory_inference INTERFACE
    ${TORCH_LIBRARIES}
    ${Boost_LIBRARIES}
    rt          # POSIX shared memory (shm_open, shm_unlink)
    pthread     # POSIX threads
)

# Optional: Link OpenSSL if found (may be needed for some Boost features)
if(OpenSSL_FOUND)
    target_link_libraries(shared_memory_inference INTERFACE
        OpenSSL::SSL
        OpenSSL::Crypto
    )
endif()

# ============================================================
# === Main Executable ========================================
# ============================================================

add_executable(AlphaZero_TTT
    main.cpp
    board.cpp
    game.cpp
    position_pool.cpp
    cell_state.cpp
    player.cpp
    logger.cpp
    mcts_agent_selfplay.cpp
    inference_client.cpp
    inference_queue_shm.cpp
    training_shm_writer.cpp
)

target_include_directories(AlphaZero_TTT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(AlphaZero_TTT PRIVATE
    shared_memory_inference
    ${TORCH_LIBRARIES}
    Threads::Threads
)

# ============================================================
# === Compiler Flags (Performance) ==========================
# ============================================================

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(AlphaZero_TTT PRIVATE
            -O3
            -march=native
            -DNDEBUG
        )
    endif()
endif()

# ============================================================
# === RPATH (Linux/macOS) ===================================
# ============================================================

set_property(TARGET AlphaZero_TTT PROPERTY
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# ============================================================
# === Windows Torch DLL fix =================================
# ============================================================

if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET AlphaZero_TTT POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:AlphaZero_TTT>
    )
endif()

# ============================================================
# === Installation (Optional) ===============================
# ============================================================

install(TARGETS AlphaZero_TTT
    RUNTIME DESTINATION bin
)

# Install Python inference server alongside binary
install(FILES
    inference_server.py
    alphazero_model.py
    DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
)

# ============================================================
# === Configuration Summary =================================
# ============================================================

message(STATUS "")
message(STATUS "=== AlphaZero JAX Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Torch_DIR: ${Torch_DIR}")
message(STATUS "Boost version: ${Boost_VERSION}")
message(STATUS "Threads: ${CMAKE_THREAD_LIBS_INIT}")
message(STATUS "Output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "")
message(STATUS "Shared Memory System: ENABLED")
message(STATUS "  - Header-only shared memory library")
message(STATUS "  - JAX Python inference server required")
message(STATUS "  - POSIX shared memory (rt library)")
message(STATUS "")
message(STATUS "Important Files:")
message(STATUS "  - shared_memory_protocol.h")
message(STATUS "  - inference_queue_shm.h")
message(STATUS "  - selfplay_manager_shm.h")
message(STATUS "  - inference_server.py")
message(STATUS "  - alphazero_model.py")
message(STATUS "====================================")
message(STATUS "")

# ============================================================
# === Build Instructions ====================================
# ============================================================

message(STATUS "Build Instructions:")
message(STATUS "  1. mkdir build && cd build")
message(STATUS "  2. cmake ..")
message(STATUS "  3. make -j$(nproc)")
message(STATUS "")
message(STATUS "Run Instructions:")
message(STATUS "  1. Start JAX server:")
message(STATUS "     python3 inference_server.py --model1 checkpoints/model")
message(STATUS "  2. Run C++ program:")
message(STATUS "     ./AlphaZero_TTT")
message(STATUS "")
