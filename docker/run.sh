#!/usr/bin/env bash
# Run the Triton ROCm dev container using the same docker flags as the
# 'drun' alias in ~/.profile:
#
#   sudo docker run -it --network=host --ipc=host \
#     --device=/dev/kfd --device=/dev/dri \
#     --group-add video --cap-add=SYS_PTRACE \
#     --security-opt seccomp=unconfined \
#     -v $HOME:/home --shm-size=16G \
#     --ulimit memlock=-1 --ulimit stack=67108864
#
# Usage:
#   ./docker/run.sh                    # interactive shell
#   ./docker/run.sh bash -c "pip install -r python/requirements.txt && pip install -e . --no-build-isolation"
#
# Two-step build:
#   1. pip install -r python/requirements.txt   (cmake, ninja, pybind11, lit, …)
#   2. pip install -e . --no-build-isolation    (build Triton, reuse ROCm torch)
#
# setup.py imports pybind11 at module level, so step 1 must run before step 2
# even when using --no-build-isolation.
#
# GPU isolation:
#   HIP_VISIBLE_DEVICES=0 ./docker/run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE="${TRITON_DOCKER_IMAGE:-xguo-triton-dev}"

HOST_USER="$(id -un)"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

# ── Build the image if it doesn't exist yet ────────────────────────────────────
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "[run.sh] Image '$IMAGE' not found — building..."
    docker build \
        --build-arg USERNAME="$HOST_USER" \
        --build-arg USER_UID="$HOST_UID" \
        --build-arg USER_GID="$HOST_GID" \
        -f "$REPO_ROOT/docker/Dockerfile.rocm" \
        -t "$IMAGE" \
        "$REPO_ROOT"
fi

# ── Docker run — matches drun alias exactly, plus Triton-specific mounts ───────
DOCKER_ARGS=(
    # From drun alias
    --network=host
    --ipc=host
    --device=/dev/kfd
    --device=/dev/dri
    --group-add video
    --group-add kvm
    --cap-add=SYS_PTRACE
    --security-opt seccomp=unconfined
    --volume "$HOME:/home"
    --shm-size=16G
    --ulimit memlock=-1
    --ulimit stack=67108864

    # Triton source tree — mounted separately so it's accessible at /triton
    # regardless of where $HOME points inside the container.
    --volume "$REPO_ROOT:/triton"

    # GPU isolation env vars — forwarded only if set by the caller.
    ${HIP_VISIBLE_DEVICES:+--env HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"}
    ${ROCR_VISIBLE_DEVICES:+--env ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"}
)

# Default to an interactive shell if no command is given.
if [[ $# -eq 0 ]]; then
    DOCKER_ARGS+=(-it)
    set -- /bin/bash
fi

echo "[run.sh] Running as $HOST_USER (uid=$HOST_UID gid=$HOST_GID)"
exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" "$@"
