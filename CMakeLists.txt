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
# === Triton Client =========================================
# ============================================================

# Auto-detect Triton client directory (relative to project root)
# Priority: 1. ENV variable, 2. Local triton_client/, 3. User home, 4. Fallback
if(DEFINED ENV{TRITON_CLIENT_DIR})
    set(TRITON_CLIENT_DIR $ENV{TRITON_CLIENT_DIR})
    message(STATUS "Using Triton from environment: ${TRITON_CLIENT_DIR}")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/triton_client")
    set(TRITON_CLIENT_DIR "${CMAKE_SOURCE_DIR}/triton_client")
    message(STATUS "Using local Triton installation: ${TRITON_CLIENT_DIR}")
elseif(EXISTS "$ENV{HOME}/triton_client")
    set(TRITON_CLIENT_DIR "$ENV{HOME}/triton_client")
    message(STATUS "Using Triton from home directory: ${TRITON_CLIENT_DIR}")
else()
    message(WARNING "Triton client directory not found. Set TRITON_CLIENT_DIR or place triton_client/ in the project root.")
    message(WARNING "Run install_and_setup.sh to auto-configure Triton client")
endif()

# Verify Triton client exists
if(NOT EXISTS "${TRITON_CLIENT_DIR}/include")
    message(FATAL_ERROR 
        "Triton client not found at: ${TRITON_CLIENT_DIR}\n"
        "Please run: ./install_and_setup.sh\n"
        "Or set TRITON_CLIENT_DIR environment variable"
    )
endif()

include_directories(${TRITON_CLIENT_DIR}/include)
link_directories(${TRITON_CLIENT_DIR}/lib)

# ============================================================
# === Dependencies ==========================================
# ============================================================

find_package(Boost REQUIRED COMPONENTS system)
find_package(Threads REQUIRED)
find_package(OpenSSL)

# ============================================================
# === Shared Memory Inference (INTERFACE) ===================
# ============================================================

add_library(shared_memory_inference INTERFACE)

target_include_directories(shared_memory_inference INTERFACE
    ${CMAKE_SOURCE_DIR}/src
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
    # ---- Game ------------------------------------------------
    src/core/game/board.cpp
    src/core/game/game.cpp
    src/core/game/cell_state.cpp
    src/core/game/player.cpp

    # ---- MCTS ------------------------------------------------
    src/core/mcts/mcts_agent_selfplay.cpp
    src/core/mcts/mcts_agent_triton.cpp
    src/core/mcts/position_pool.cpp

    # ---- Utils ----------------------------------------------
    src/core/utils/logger.cpp

    # ---- Inference ------------------------------------------
    src/inference/shared_memory/inference_queue_shm.cpp
    src/inference/triton/triton_inference_client.cpp

    # ---- Training -------------------------------------------
    src/training/training_shm_writer.cpp
  
)

target_include_directories(az_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(az_core PUBLIC
    shared_memory_inference
    Threads::Threads
    grpcclient
)

# ============================================================
# === Executables ===========================================
# ============================================================

add_executable(AlphaZero_TTT
    executables/main.cpp
)

add_executable(AZ_Triton_TTT
    executables/main_triton.cpp
)

add_executable(AlphaZero_TTT_Eval
    executables/eval.cpp
)

add_executable(AZ_Triton_TTT_Eval
    executables/eval_triton.cpp
)


target_link_libraries(AlphaZero_TTT PRIVATE az_core)
target_link_libraries(AZ_Triton_TTT PRIVATE az_core)
target_link_libraries(AlphaZero_TTT_Eval PRIVATE az_core)
target_link_libraries(AZ_Triton_TTT_Eval PRIVATE az_core)

# ============================================================
# === Compiler Flags ========================================
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
# === RPATH =================================================
# ============================================================

foreach(target
    AlphaZero_TTT
    AlphaZero_TTT_Eval
    AZ_Triton_TTT
    AZ_Triton_TTT_Eval
)
    set_property(TARGET ${target}
        PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endforeach()

# ============================================================
# === Installation (Optional) ===============================
# ============================================================

install(TARGETS
    AlphaZero_TTT
    AlphaZero_TTT_Eval
    AZ_Triton_TTT
    AZ_Triton_TTT_Eval
)

# ============================================================
# === Summary ===============================================
# ============================================================

message(STATUS "")
message(STATUS "=== AlphaZero Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Triton Client: ${TRITON_CLIENT_DIR}")
message(STATUS "Boost version: ${Boost_VERSION}")
message(STATUS "Runtime output: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "Executables:")
message(STATUS "  - AlphaZero_TTT")
message(STATUS "  - AlphaZero_TTT_Eval")
message(STATUS "  - AZ_Triton_TTT")
message(STATUS "  - AZ_Triton_TTT_Eval")
message(STATUS "")