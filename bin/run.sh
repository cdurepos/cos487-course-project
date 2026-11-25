#!/usr/bin/env bash

set -euo pipefail

# Run the COS487 Search Party project.
#
# Usage:
#   bash bin/run.sh
#
# Prerequisites:
#   - Run the repository's install script first. It creates the Conda
#     environment, and installs the necessary dependencies:
#       - bash bin/install.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

ENV_NAME="cos487_env"

cd "$REPO_ROOT"

# Find Conda and check the project's environment.
CONDA_EXE=""

if command -v conda >/dev/null 2>&1; then
    if type -P conda >/dev/null 2>&1; then
        CONDA_EXE="$(type -P conda)"
    fi
fi

if [[ -z "$CONDA_EXE" ]]; then
    for candidate in \
        "$HOME/miniconda3/bin/conda" \
        "$HOME/anaconda3/bin/conda" \
        "/opt/miniconda3/bin/conda" \
        "/opt/anaconda3/bin/conda"
    do
        if [[ -x "$candidate" ]] && "$candidate" --version >/dev/null 2>&1; then
            CONDA_EXE="$candidate"
            break
        fi
    done
fi

if [[ -z "$CONDA_EXE" ]]; then
    echo "Error: Conda was not found. Run the install script first."
    exit 1
fi

if ! "$CONDA_EXE" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Error: Conda environment '$ENV_NAME' was not found."
    echo "Run the install script first."
    exit 1
fi

echo "Using Conda environment '$ENV_NAME'..."

CONDA_BASE="$("$CONDA_EXE" info --base)"
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

python --version >/dev/null 2>&1 || {
    echo "Error: Python was not found on PATH."
    exit 1
}

# Verify that dependencies have been installed.
python -c "import ranx, yaml, torch, transformers" >/dev/null 2>&1 || {
    echo "Error: Dependencies are not installed."
    echo "Run: python -m pip install -r requirements.txt"
    exit 1
}

if [[ ! -d "data/qrels" ]]; then
    echo "Error: No relevance judgments found at data/qrels/."
    echo "Download the course data files first (see the README)."
    exit 1
elif [[ ! -d "data/JSON Files" ]]; then
    echo "Error: No data corpus found at data/JSON Files/."
    echo "Download the course data files first (see the README)."
    exit 1
elif [[ ! -f "data/Study.json" ]]; then
    echo "Error: No Study queries found at data/Study.json."
    echo "Download the course data files first (see the README)."
    exit 1
elif [[ ! -f "data/Train.json" ]]; then
    echo "Error: No Train queries found at data/Train.json."
    echo "Download the course data files first (see the README)."
    exit 1
elif [[ ! -f "data/Test.json" ]]; then
    echo "Error: No Test queries found at data/Test.json."
    echo "Download the course data files first (see the README)."
    exit 1
fi

echo "Repository: $REPO_ROOT"
echo

echo "Preprocessing the corpus..."
python -m apps.processing.preprocess

echo "Creating the BM25 indexes..."
python -m apps.baseline.index

echo "Performing retrieval and evaluation..."
python -m apps.evaluate

echo "Indexing, retrieval, and evaluation completed successfully!"
