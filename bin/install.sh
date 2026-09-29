#!/usr/bin/env bash

set -euo pipefail

# Install and configure the COS487 Search Party application.
#
# This script:
#   1. Detects the operating system and architecture.
#   2. Finds or installs a working Conda installation.
#   3. Creates the 'cos487_env' Conda environment with Python 3.12.
#   4. Installs dependencies.
#   5. Preprocesses the application data.
#   6. Creates the retrieval indexes.
#
# Supported platforms:
#   - macOS
#   - Linux
#
# Prerequisites:
# - Download course data files and place them as follows:
#   data/
#   ├── JSON Files/     Unzipped data corpus
#   ├── Study.json      Query file
#   ├── processed/      Written by the preprocessing script
#   └── indexes/        Written by the retrieval index script
#
# After installation, activate the environment with:
#   conda activate cos487_env
#
# The application can then be started with:
#   ./bin/run.sh

ENV_NAME="cos487_env"
PYTHON_VERSION="3.12"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "Detecting operating system..."

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Darwin)
        OS_NAME="macOS"
        ;;
    Linux)
        OS_NAME="Linux"
        ;;
    *)
        echo "Error: Unsupported operating system: $OS"
        exit 1
        ;;
esac

echo "    OS: $OS_NAME"
echo "    Architecture: $ARCH"

# Find or install Conda
echo
echo "Checking for Conda..."

CONDA_EXE=""

# Test whether Conda works
is_working_conda() {
    local candidate="$1"

    [[ -x "$candidate" ]] || return 1

    "$candidate" --version >/dev/null 2>&1
}

# Check if Conda is already on PATH
if command -v conda >/dev/null 2>&1; then
    PATH_CONDA="$(command -v conda)"

    # "conda" can be a shell function rather than an executable.
    # In that case, find the actual executable using `type -P`.
    if [[ -x "$PATH_CONDA" ]] && is_working_conda "$PATH_CONDA"; then
        CONDA_EXE="$PATH_CONDA"
        echo "Conda found on PATH: $CONDA_EXE"
    elif type -P conda >/dev/null 2>&1; then
        PATH_CONDA="$(type -P conda)"

        if is_working_conda "$PATH_CONDA"; then
            CONDA_EXE="$PATH_CONDA"
            echo "Conda found on PATH: $CONDA_EXE"
        fi
    fi
fi

# Check common installation locations if Conda was not found on PATH
if [[ -z "$CONDA_EXE" ]]; then

    POSSIBLE_CONDA_PATHS=(
        "$HOME/miniconda3/bin/conda"
        "$HOME/anaconda3/bin/conda"
        "/opt/miniconda3/bin/conda"
        "/opt/anaconda3/bin/conda"
    )

    for POSSIBLE_CONDA in "${POSSIBLE_CONDA_PATHS[@]}"; do
        if is_working_conda "$POSSIBLE_CONDA"; then
            CONDA_EXE="$POSSIBLE_CONDA"
            echo "Found working Conda installation:"
            echo "    $CONDA_EXE"
            break
        fi
    done
fi

# Install Miniconda if no working Conda installation was found
if [[ -z "$CONDA_EXE" ]]; then
    echo
    echo "No working Conda installation found."
    echo "Installing Miniconda..."

    case "$OS" in
        Darwin)
            case "$ARCH" in
                arm64)
                    INSTALLER="Miniconda3-latest-MacOSX-arm64.sh"
                    ;;
                x86_64)
                    INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh"
                    ;;
                *)
                    echo "Error: Unsupported macOS architecture: $ARCH"
                    exit 1
                    ;;
            esac
            ;;
        Linux)
            case "$ARCH" in
                x86_64)
                    INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
                    ;;
                aarch64|arm64)
                    INSTALLER="Miniconda3-latest-Linux-aarch64.sh"
                    ;;
                *)
                    echo "Error: Unsupported Linux architecture: $ARCH"
                    exit 1
                    ;;
            esac
            ;;
    esac

    MINICONDA_URL="https://repo.anaconda.com/miniconda/$INSTALLER"
    INSTALLER_PATH="/tmp/$INSTALLER"
    CONDA_DIR="$HOME/miniconda3"

    # Check if the Conda installation directory already exists
    if [[ -d "$CONDA_DIR" ]]; then
        echo
        echo "ERROR: A Conda installation directory already exists:"
        echo "    $CONDA_DIR"
        echo
        echo "However, Conda could not be executed from that location."
        echo
        echo "Please inspect the installation before removing it."
        echo "If you are certain it is safe to remove, run:"
        echo
        echo "    rm -rf \"$CONDA_DIR\""
        echo
        exit 1
    fi

    # Download and install Miniconda
    echo "Downloading Miniconda..."
    curl -fL "$MINICONDA_URL" -o "$INSTALLER_PATH"
    echo "Installing Miniconda to:"
    echo "    $CONDA_DIR"
    bash "$INSTALLER_PATH" -b -p "$CONDA_DIR"
    rm -f "$INSTALLER_PATH"
    CONDA_EXE="$CONDA_DIR/bin/conda"
    echo "Miniconda installed successfully."
fi

# Verify that Conda can be executed
if ! is_working_conda "$CONDA_EXE"; then
    echo
    echo "Error: Conda was found but could not be executed:"
    echo "    $CONDA_EXE"
    exit 1
fi
echo
echo "Using Conda:"
echo "    $CONDA_EXE"

# Initialize Conda
CONDA_BASE="$("$CONDA_EXE" info --base)"
set +u
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
set -u
echo
echo "Conda version:"
conda --version

# Create Conda environment if necessary
echo
echo "Checking Conda environment '$ENV_NAME'..."

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating Conda environment '$ENV_NAME'..."
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi

# Activate environment
echo
echo "Using Conda environment '$ENV_NAME'..."

conda run -n "$ENV_NAME" python --version

# Install dependencies
echo
echo "Installing dependencies..."
conda run -n "$ENV_NAME" python -m pip install -r requirements.txt

cd apps/prod/frontend
npm install
cd "$REPO_ROOT"

echo "Dependencies installed successfully in '$ENV_NAME'."

# Preprocess data
echo
echo "Preprocessing data (this may take a few minutes)..."
conda run -n "$ENV_NAME" python -m apps.processing.preprocess
echo "Data preprocessing complete."

# Create retrieval indexes
echo
echo "Creating retrieval indexes (this may take a few minutes)..."
conda run -n "$ENV_NAME" python -m apps.retrieval.index
echo "Retrieval indexes created successfully."

# Final message
echo
echo "Installation/setup complete!"
echo
echo "Conda environment:"
echo "    $ENV_NAME"
echo
echo "To activate it later:"
echo "    conda activate $ENV_NAME"
