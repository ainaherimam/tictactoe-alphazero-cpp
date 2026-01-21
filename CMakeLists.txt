cmake_minimum_required(VERSION 3.10)
project(AlphaZero_TTT)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)
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
