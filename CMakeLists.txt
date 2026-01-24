cmake_minimum_required(VERSION 3.10)
project(AlphaZero_TTT)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ============================================================
# === BUILD TYPE CONFIGURATION ===============================
# ============================================================
# Allow user to specify build type: Debug or Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
    message(STATUS "Build type not specified, defaulting to Debug")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Debug flags for segfault diagnosis
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -fno-omit-frame-pointer")
    
    # AddressSanitizer and UndefinedBehaviorSanitizer (optional but highly recommended)
    # Uncomment these if your compiler supports them:
    # set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined")
    # set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined")
    
    message(STATUS "Debug flags: ${CMAKE_CXX_FLAGS_DEBUG}")
endif()

# Release flags for production
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    message(STATUS "Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

if(NOT DEFINED Torch_DIR AND DEFINED ENV{TORCH_HOME})
    set(Torch_DIR "$ENV{TORCH_HOME}/share/cmake/Torch")
endif()

find_package(Torch REQUIRED)

if(NOT Torch_FOUND)
    message(FATAL_ERROR 
        "LibTorch not found. Please install LibTorch and specify its location:\n"
        "  Option 1: cmake -DTorch_DIR=/path/to/libtorch/share/cmake/Torch ..\n"
        "  Option 2: Set environment variable TORCH_HOME=/path/to/libtorch\n"
        "Download from: https://pytorch.org/get-started/locally/")
endif()

message(STATUS "Torch found: ${TORCH_LIBRARIES}")

# ============================================================
# === Source files ===========================================
# ============================================================
set(SOURCES
    main.cpp
    board.cpp
    cell_state.cpp
    console_interface.cpp
    game.cpp
    player.cpp
    mcts_agent.cpp
    logger.cpp
    alphaz_model.cpp
    game_dataset.cpp
    minimax_agent.cpp
    mcts_agent_parallel.cpp
    tensor_pool.cpp
    mcts_agent_selfplay.cpp
    inference_queue.cpp
    inference_worker.cpp
    selfplay_manager.cpp
)

# ============================================================
# === Executable =============================================
# ============================================================
add_executable(AlphaZero_TTT ${SOURCES})
target_include_directories(AlphaZero_TTT PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# ============================================================
# === Link LibTorch ==========================================
# ============================================================
target_link_libraries(AlphaZero_TTT "${TORCH_LIBRARIES}")
set_property(TARGET AlphaZero_TTT PROPERTY CXX_STANDARD 20)

# ============================================================
# === Threading Support ======================================
# ============================================================
find_package(Threads REQUIRED)
target_link_libraries(AlphaZero_TTT Threads::Threads)

# ============================================================
# === Runtime Library Path Fix ===============================
# ============================================================
set_property(TARGET AlphaZero_TTT PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

# Copy torch DLLs to binary directory on Windows
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET AlphaZero_TTT POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:AlphaZero_TTT>)
endif()

# ============================================================
# === Print Configuration Summary ===========================
# ============================================================
message(STATUS "")
message(STATUS "=== Configuration Summary ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "============================")
message(STATUS "")
