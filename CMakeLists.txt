cmake_minimum_required(VERSION 3.10)
project(MCTS_Fanorona)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# ============================================================
# === Locate LibTorch (adjust this path to your install) ====
# ============================================================
# Example: you extracted libtorch to /home/aina/libtorch
set(Torch_DIR "/home/aina/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

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
    nn_model.cpp
)

# ============================================================
# === Executable =============================================
# ============================================================
add_executable(MCTS_Fanorona ${SOURCES})

target_include_directories(MCTS_Fanorona PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# ============================================================
# === Link LibTorch ==========================================
# ============================================================
target_link_libraries(MCTS_Fanorona "${TORCH_LIBRARIES}")
set_property(TARGET MCTS_Fanorona PROPERTY CXX_STANDARD 20)

# ============================================================
# === Runtime Library Path Fix (optional) ====================
# ============================================================
# This ensures the executable finds libtorch.so without LD_LIBRARY_PATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

