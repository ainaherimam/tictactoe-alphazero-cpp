#!/bin/bash

set -e  # Exit on any error

# ============================================================
# === Configuration =========================================
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRITON_INSTALL_DIR="${SCRIPT_DIR}/triton_client"
TRITON_VERSION="2.50.0"

# Parse command line arguments
FORCE_INSTALL=false
SKIP_VENV_ARG=false

for arg in "$@"; do
    if [[ "$arg" == "--force" ]] || [[ "$arg" == "-f" ]]; then
        FORCE_INSTALL=true
        echo "⚠ Force mode enabled - will reinstall everything"
        echo ""
    elif [[ "$arg" == "--no-venv" ]]; then
        SKIP_VENV_ARG=true
        echo "ℹ️  Skipping virtual environment creation (--no-venv flag)"
        echo ""
    fi
done

# Detect Google Colab environment
IS_COLAB=false
if [ -n "$COLAB_GPU" ] || [ -d "/content" ] && [ -f "/usr/local/lib/python*/dist-packages/google/colab/__init__.py" ] 2>/dev/null; then
    IS_COLAB=true
    SKIP_VENV_ARG=true
    echo "✓ Google Colab environment detected - skipping venv creation"
    echo ""
fi

# ============================================================
# === Check What's Already Installed ========================
# ============================================================

SKIP_TRITON=false
SKIP_VENV=false
SKIP_BUILD=false

if [ "$FORCE_INSTALL" = false ]; then
    # Check if Triton is already installed
    if [ -d "${TRITON_INSTALL_DIR}/lib" ] && [ -d "${TRITON_INSTALL_DIR}/include" ]; then
        if [ -f "${TRITON_INSTALL_DIR}/lib/libgrpcclient.so" ]; then
            echo "✓ Triton Client already installed at: ${TRITON_INSTALL_DIR}"
            SKIP_TRITON=true
        fi
    fi

    # Check if venv exists and has required packages (skip check if --no-venv or Colab)
    if [ "$SKIP_VENV_ARG" = false ]; then
        if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
            if python -c "import jax" 2>/dev/null; then
                echo "✓ Python virtual environment already set up with JAX"
                SKIP_VENV=true
            else
                echo "⚠ Virtual environment exists but missing packages"
                deactivate 2>/dev/null || true
            fi
            deactivate 2>/dev/null || true
        fi
    else
        # Skip venv entirely if --no-venv flag or Colab
        SKIP_VENV=true
    fi

    # Check if build artifacts exist
    if [ -f "AlphaZero_TTT" ] && [ -f "AZ_Triton_TTT" ] && [ -d "build" ]; then
        echo "✓ Executables already built"
        SKIP_BUILD=true
    fi
fi

echo ""
echo "================================================================"
echo "AlphaZero Tic-Tac-Toe - Complete Setup"
echo "================================================================"
echo ""
echo "Setup plan:"
if [ "$SKIP_TRITON" = true ]; then
    echo "  1. Install NVIDIA Triton Client libraries [SKIP - Already installed]"
else
    echo "  1. Install NVIDIA Triton Client libraries [WILL INSTALL]"
fi
if [ "$SKIP_VENV" = true ]; then
    echo "  2. Set up Python virtual environment [SKIP - Already configured]"
else
    echo "  2. Set up Python virtual environment (Python 3.11 or 3.12) [WILL INSTALL]"
fi
if [ "$SKIP_BUILD" = true ]; then
    echo "  3. Build C++ executables with CMake [SKIP - Already built]"
else
    echo "  3. Build C++ executables with CMake [WILL BUILD]"
fi
echo ""
echo "Installation directory: ${SCRIPT_DIR}"
echo "Triton client path: ${TRITON_INSTALL_DIR}"
echo ""

# If everything is already done, inform the user
if [ "$SKIP_TRITON" = true ] && [ "$SKIP_VENV" = true ] && [ "$SKIP_BUILD" = true ]; then
    echo "✓✓✓ Everything is already installed and built!"
    echo ""
    echo "To rebuild or reinstall anyway:"
    echo "  • Run with force flag: ./install_and_setup.sh --force"
    echo "  • Or manually delete components:"
    echo "    - Delete triton_client/ to reinstall Triton"
    echo "    - Delete venv/ to recreate Python environment"
    echo "    - Delete build/ and executables to rebuild"
    echo ""
    exit 0
fi

read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# ============================================================
# === Step 1: Install Triton Client =========================
# ============================================================

if [ "$SKIP_TRITON" = false ]; then
    echo "================================================================"
    echo "STEP 1/3: Installing NVIDIA Triton Client Libraries"
    echo "================================================================"
    echo ""
    echo "Version: ${TRITON_VERSION}"
    echo "Install location: ${TRITON_INSTALL_DIR}"
    echo ""

    mkdir -p "$TRITON_INSTALL_DIR"
    cd "$TRITON_INSTALL_DIR"

    echo "→ Downloading Triton client libraries..."
    TARBALL="v${TRITON_VERSION}_ubuntu2204.clients.tar.gz"
    URL="https://github.com/triton-inference-server/server/releases/download/v${TRITON_VERSION}/${TARBALL}"

    if [ ! -f "$TARBALL" ]; then
        echo "  Downloading from: $URL"
        wget "$URL"
    else
        echo "  ✓ Tarball already downloaded"
    fi

    echo ""
    echo "→ Extracting..."
    tar -xzf "$TARBALL"

    # Organize the extracted files
    if [ -d "clients" ]; then
        echo "  ✓ Extracted to clients/"
        
        # Create standard lib and include directories
        mkdir -p lib include
        
        # Move libraries
        if [ -d "clients/lib" ]; then
            cp -r clients/lib/* lib/ 2>/dev/null || true
        fi
        
        # Move headers
        if [ -d "clients/include" ]; then
            cp -r clients/include/* include/ 2>/dev/null || true
        fi
        
        echo "  ✓ Organized into lib/ and include/"
    fi

    # Create environment setup script
    cat > setup_triton_env.sh << EOF
#!/bin/bash
# Source this file to use Triton client libraries
export TRITON_CLIENT_DIR="${TRITON_INSTALL_DIR}"
export LD_LIBRARY_PATH="${TRITON_INSTALL_DIR}/lib:\$LD_LIBRARY_PATH"
export PATH="${TRITON_INSTALL_DIR}/bin:\$PATH"
export PKG_CONFIG_PATH="${TRITON_INSTALL_DIR}/lib/pkgconfig:\$PKG_CONFIG_PATH"
EOF

    chmod +x setup_triton_env.sh

    echo ""
    echo "✓ Triton Client installation complete!"
    echo ""
else
    echo "================================================================"
    echo "STEP 1/3: Triton Client [SKIPPED - Already Installed]"
    echo "================================================================"
    echo ""
    cd "$TRITON_INSTALL_DIR"
fi

# Source the Triton environment for the rest of the script
if [ -f "setup_triton_env.sh" ]; then
    source setup_triton_env.sh
fi

# ============================================================
# === Step 2: Python Virtual Environment ====================
# ============================================================

cd "$SCRIPT_DIR"

if [ "$SKIP_VENV" = false ]; then
    echo "================================================================"
    echo "STEP 2/3: Setting up Python Virtual Environment"
    echo "================================================================"
    echo ""

    # Check Python version
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    major_version=$(echo $python_version | cut -d. -f1)
    minor_version=$(echo $python_version | cut -d. -f2)

    echo "→ Detected Python version: $python_version"

    # Check if Python version is 3.11 or 3.12
    if [ "$major_version" -ne 3 ] || [ "$minor_version" -lt 11 ] || [ "$minor_version" -gt 12 ]; then
        echo ""
        echo "✗ Error: Python version must be 3.11 or 3.12"
        echo "  Current version: $python_version"
        echo ""
        echo "Please install Python 3.11 or 3.12 and try again."
        exit 1
    fi

    echo "  ✓ Python version check passed!"
    echo ""

    # Remove existing virtual environment if it exists but is incomplete
    if [ -d "venv" ]; then
        echo "→ Removing incomplete virtual environment..."
        rm -rf venv
    fi

    # Create new virtual environment
    echo "→ Creating new virtual environment..."
    python3 -m venv venv

    # Activate virtual environment
    echo "→ Activating virtual environment..."
    source venv/bin/activate

    # Upgrade pip
    echo "→ Upgrading pip..."
    python -m pip install --upgrade pip --quiet

    # Install packages from requirements.txt
    if [ -f "requirements.txt" ]; then
        echo "→ Installing Python packages from requirements.txt..."
        
        # Check which packages are missing
        missing_packages=()
        while IFS= read -r package || [ -n "$package" ]; do
            # Skip empty lines and comments
            [[ -z "$package" || "$package" =~ ^[[:space:]]*# ]] && continue
            
            package_name=$(echo "$package" | cut -d'=' -f1 | cut -d'>' -f1 | cut -d'<' -f1 | tr -d ' ')
            
            if ! python -c "import $package_name" 2>/dev/null; then
                missing_packages+=("$package")
            else
                echo "  ✓ $package_name already installed"
            fi
        done < requirements.txt
        
        # Install only missing packages
        if [ ${#missing_packages[@]} -gt 0 ]; then
            echo ""
            echo "  Installing missing packages: ${missing_packages[*]}"
            pip install "${missing_packages[@]}"
        fi
        
        echo "  ✓ Python packages installed!"
    else
        echo "  ⚠ Warning: requirements.txt not found, skipping Python package installation"
    fi

    echo ""
    echo "✓ Python virtual environment setup complete!"
    echo ""
else
    echo "================================================================"
    echo "STEP 2/3: Python Virtual Environment [SKIPPED]"
    echo "================================================================"
    echo ""
    
    if [ "$IS_COLAB" = true ]; then
        echo "  Google Colab detected - using system Python environment"
        echo ""
        
        # Install packages directly in Colab
        if [ -f "requirements.txt" ]; then
            echo "→ Installing Python packages in Colab environment..."
            pip install -q -r requirements.txt
            echo "  ✓ Python packages installed!"
        fi
    elif [ "$SKIP_VENV_ARG" = true ]; then
        echo "  Using system Python environment (--no-venv flag)"
        echo ""
        
        # Install packages in system Python if requirements.txt exists
        if [ -f "requirements.txt" ]; then
            echo "→ Installing Python packages in system environment..."
            pip install -r requirements.txt
            echo "  ✓ Python packages installed!"
        fi
    else
        echo "  Already configured"
    fi
    echo ""
fi

# ============================================================
# === Step 3: Build C++ Project =============================
# ============================================================

if [ "$SKIP_BUILD" = false ]; then
    echo "================================================================"
    echo "STEP 3/3: Building C++ Project with CMake"
    echo "================================================================"
    echo ""

    # Update CMakeLists.txt with the correct Triton path
    if [ -f "CMakeLists.txt" ]; then
        echo "→ Updating CMakeLists.txt with Triton path..."
        
        # Create a backup only if one doesn't exist
        if [ ! -f "CMakeLists.txt.backup" ]; then
            cp CMakeLists.txt CMakeLists.txt.backup
        fi
        
        # Replace the hardcoded path with our dynamic path
        sed -i "s|set(TRITON_CLIENT_DIR \".*\")|set(TRITON_CLIENT_DIR \"${TRITON_INSTALL_DIR}\")|g" CMakeLists.txt
        
        echo "  ✓ CMakeLists.txt updated"
        echo "    TRITON_CLIENT_DIR set to: ${TRITON_INSTALL_DIR}"
    else
        echo "  ✗ Error: CMakeLists.txt not found!"
        exit 1
    fi

    echo ""
    echo "→ Creating build directory..."
    mkdir -p build
    cd build

    echo "→ Running CMake configuration..."
    cmake .. -DCMAKE_BUILD_TYPE=Release

    echo ""
    echo "→ Building project (using 2 parallel jobs)..."
    make -j2

    echo ""
    echo "✓ Build complete!"
    echo ""
else
    echo "================================================================"
    echo "STEP 3/3: C++ Project Build [SKIPPED - Already Built]"
    echo "================================================================"
    echo ""
    echo "To rebuild, run:"
    echo "  cd build && make -j2"
    echo ""
fi

# ============================================================
# === Summary ===============================================
# ============================================================

cd "$SCRIPT_DIR"

echo "================================================================"
echo "✓✓✓ SETUP COMPLETE ✓✓✓"
echo "================================================================"
echo ""

# Show what was done vs skipped
echo "Actions taken:"
if [ "$SKIP_TRITON" = true ]; then
    echo "  • Triton Client: [SKIPPED - Already installed]"
else
    echo "  ✓ Triton Client: Installed"
fi

if [ "$SKIP_VENV" = true ]; then
    echo "  • Python venv: [SKIPPED - Already configured]"
else
    echo "  ✓ Python venv: Created and configured"
fi

if [ "$SKIP_BUILD" = true ]; then
    echo "  • C++ Build: [SKIPPED - Already built]"
else
    echo "  ✓ C++ Build: Compiled successfully"
fi

echo ""
echo "Installation paths:"
echo "  • Triton Client: ${TRITON_INSTALL_DIR}"
echo "  • Python venv: ${SCRIPT_DIR}/venv"
echo "  • Build directory: ${SCRIPT_DIR}/build"
echo ""
echo "Executables:"
if [ -f "AlphaZero_TTT" ]; then
    echo "  ✓ AlphaZero_TTT"
fi
if [ -f "AlphaZero_TTT_Eval" ]; then
    echo "  ✓ AlphaZero_TTT_Eval"
fi
if [ -f "AZ_Triton_TTT" ]; then
    echo "  ✓ AZ_Triton_TTT"
fi
if [ -f "AZ_Triton_TTT_Eval" ]; then
    echo "  ✓ AZ_Triton_TTT_Eval"
fi
echo ""
echo ""
echo "To use the environment:"
echo "  1. Run 'run_triton.sh' or 'run_inference_server.sh' as server"
echo ""
echo "  2. Run executables: ./AZ_Triton_TTT or ./AlphaZero_TTT based on server"
echo ""
echo "  3. Run 'run_train.sh' "
echo ""
echo "To rebuild after code changes:"
echo "make -j2"
echo ""
echo "To force reinstallation of any component:"
echo "  • Delete triton_client/ to reinstall Triton"
echo "  • Delete venv/ to recreate Python environment"
echo "  • Delete build/ and executables to rebuild from scratch"
echo ""
echo "================================================================"
