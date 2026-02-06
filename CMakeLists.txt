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

# ---- Boost -------------------------------------------------
find_package(Boost REQUIRED COMPONENTS system)

# ---- Threads ----------------------------------------------
find_package(Threads REQUIRED)

# ---- OpenSSL (optional) -----------------------------------
find_package(OpenSSL)

# ============================================================
# === Shared Memory Inference (Header-only) =================
# ============================================================

add_library(shared_memory_inference INTERFACE)

target_include_directories(shared_memory_inference INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(shared_memory_inference INTERFACE
    ${Boost_LIBRARIES}
    rt
    pthread
)

if(OpenSSL_FOUND)
    target_link_libraries(shared_memory_inference INTERFACE
        OpenSSL::SSL
        OpenSSL::Crypto
    )
endif()

# ============================================================
# === Core AlphaZero Library ================================
# ============================================================

add_library(az_core
    board.cpp
    game.cpp
    position_pool.cpp
    cell_state.cpp
    player.cpp
    logger.cpp
    mcts_agent_selfplay.cpp
    inference_queue_shm.cpp
    training_shm_writer.cpp
)

target_include_directories(az_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(az_core PUBLIC
    shared_memory_inference
    Threads::Threads
)

# ============================================================
# === Main Executable =======================================
# ============================================================

add_executable(AlphaZero_TTT
    main.cpp
)

target_link_libraries(AlphaZero_TTT PRIVATE
    az_core
)

# ============================================================
# === Eval Executable =======================================
# ============================================================

add_executable(AlphaZero_TTT_Eval
    eval.cpp
)

target_link_libraries(AlphaZero_TTT_Eval PRIVATE
    az_core
)

# ============================================================
# === Compiler Flags (Performance) ==========================
# ============================================================

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(az_core PRIVATE
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

set_property(TARGET AlphaZero_TTT_Eval PROPERTY
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# ============================================================
# === Installation (Optional) ===============================
# ============================================================

install(TARGETS
    AlphaZero_TTT
    AlphaZero_TTT_Eval
    RUNTIME DESTINATION bin
)

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
message(STATUS "Boost version: ${Boost_VERSION}")
message(STATUS "Threads: ${CMAKE_THREAD_LIBS_INIT}")
message(STATUS "Output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "")
message(STATUS "Shared Memory System: ENABLED")
message(STATUS "  - Header-only shared memory library")
message(STATUS "  - JAX Python inference server required")
message(STATUS "")
message(STATUS "Executables:")
message(STATUS "  - AlphaZero_TTT (main self-play)")
message(STATUS "  - AlphaZero_TTT_
 (eval harness)")
message(STATUS "====================================")
message(STATUS "")
