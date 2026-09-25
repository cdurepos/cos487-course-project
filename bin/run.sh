#!/usr/bin/env bash

set -euo pipefail
set -m

# Run the COS487 Search Party application.
#
# This script:
#   1. Runs from the repository root regardless of the caller's current directory.
#   2. Starts the FastAPI backend on port 8000.
#   3. Starts the Vite/React frontend on port 5173.
#
# Prerequisites:
#   - Run the repository's install script first.
#     - This can be done by running the following commands from the repository root:
#       - chmod +x ./bin/install.sh
#       - ./bin/install.sh
#
# The frontend is configured to proxy /search to http://localhost:8000.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

cd "$REPO_ROOT"

cleanup() {
    status=$?
    echo
    echo "Stopping application..."
    
    # Stop backend process and any children it spawned
    if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    # Stop frontend process and any children it spawned
    if [[ -n "${FRONTEND_PID:-}" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi

    # Give processes a moment to exit cleanly
    sleep 1

    # Force anything still running
    if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill -TERM "-$BACKEND_PID" 2>/dev/null || true
    fi
    if [[ -n "${FRONTEND_PID:-}" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        kill -9 "$FRONTEND_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    exit "$status"
}
trap cleanup INT TERM EXIT

# Find Conda and activate the project's environment.
ENV_NAME="cos487_env"
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

# Check required commands before starting anything.
"$CONDA_EXE" run -n "$ENV_NAME" python --version >/dev/null 2>&1 || {
    echo "Error: Python was not found on PATH."
    exit 1
}
command -v npm >/dev/null 2>&1 || {
    echo "Error: npm was not found on PATH."
    exit 1
}

# Verify that dependencies have been installed.
"$CONDA_EXE" run -n "$ENV_NAME" python -c "import fastapi, uvicorn" >/dev/null 2>&1 || {
    echo "Error: Python dependencies are not installed."
    echo "Run: python -m pip install -r requirements.txt"
    exit 1
}

if [[ ! -d "apps/prod/frontend/node_modules" ]]; then
    echo "Error: Frontend dependencies are not installed."
    echo "Run: (cd apps/prod/frontend && npm install)"
    exit 1
fi

echo "Repository: $REPO_ROOT"
echo "Backend:    http://${BACKEND_HOST}:${BACKEND_PORT}"
echo "Frontend:   http://${FRONTEND_HOST}:${FRONTEND_PORT}"
echo

echo "Starting backend..."
# "$CONDA_EXE" run -n "$ENV_NAME" \
#     python -m uvicorn apps.prod.backend.main:app \
#     --host "$BACKEND_HOST" \
#     --port "$BACKEND_PORT" \
#     --reload &
"$CONDA_EXE" run -n "$ENV_NAME" uvicorn apps.prod.backend.main:app --reload --port "$BACKEND_PORT" &
BACKEND_PID=$!

echo "Starting frontend..."
(
    cd "$REPO_ROOT/apps/prod/frontend"
    "$CONDA_EXE" run -n "$ENV_NAME" npm run dev
    # npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
) &
FRONTEND_PID=$!
echo
echo "Application is running."
echo "Open: http://${FRONTEND_HOST}:${FRONTEND_PORT}"
echo "Press Ctrl+C to stop both services."
echo

# Keep this script alive while either service is running.
while true; do
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "Error: Backend process exited."
        exit 1
    fi
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo "Error: Frontend process exited."
        exit 1
    fi
    sleep 1
done
