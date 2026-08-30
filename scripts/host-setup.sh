#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# host-setup.sh — one-time WSL2 host preparation for the TRELLIS.2 container.
#
# Installs Docker Engine (from Docker's official apt repo) and the NVIDIA
# Container Toolkit, wires the toolkit into Docker, and verifies that both
# GPUs are visible from inside a container.
#
# Run from WSL2 (Ubuntu):   sudo bash scripts/host-setup.sh
# Idempotent — safe to re-run.
# ---------------------------------------------------------------------------
set -euo pipefail

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[fail] %s\033[0m\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "Run with sudo: sudo bash scripts/host-setup.sh"

TARGET_USER="${SUDO_USER:-$USER}"

# --- 0. Sanity checks ------------------------------------------------------
log "Checking environment"
grep -qi microsoft /proc/version || warn "This does not look like WSL2 — continuing anyway."
command -v systemctl >/dev/null || die "systemd is not available. Add '[boot]\\nsystemd=true' to /etc/wsl.conf, then run 'wsl --shutdown' from Windows and reopen."
systemctl is-system-running --quiet 2>/dev/null || warn "systemd is not fully running (degraded is usually fine)."
[[ -e /dev/dxg ]] || warn "/dev/dxg missing — WSL GPU passthrough may not be active."

. /etc/os-release
CODENAME="${UBUNTU_CODENAME:-$VERSION_CODENAME}"
log "Ubuntu ${VERSION_ID} (${CODENAME})"

# --- 1. Docker Engine ------------------------------------------------------
if command -v docker >/dev/null; then
  log "Docker already installed: $(docker --version)"
else
  log "Installing Docker Engine"
  apt-get update -qq
  apt-get install -y -qq ca-certificates curl gnupg

  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg --yes
  chmod a+r /etc/apt/keyrings/docker.gpg

  # Docker publishes a channel for this release (verified: resolute -> HTTP 200).
  # If a future release is missing, fall back to the newest LTS channel.
  DOCKER_SUITE="$CODENAME"
  if ! curl -fsI "https://download.docker.com/linux/ubuntu/dists/${CODENAME}/Release" >/dev/null; then
    warn "No Docker channel for '${CODENAME}' — falling back to 'noble'."
    DOCKER_SUITE="noble"
  fi

  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${DOCKER_SUITE} stable" \
    > /etc/apt/sources.list.d/docker.list

  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io \
                        docker-buildx-plugin docker-compose-plugin
fi

# --- 2. NVIDIA Container Toolkit -------------------------------------------
if command -v nvidia-ctk >/dev/null; then
  log "NVIDIA Container Toolkit already installed: $(nvidia-ctk --version | head -1)"
else
  log "Installing NVIDIA Container Toolkit"
  # This repo is distribution-agnostic, so the Ubuntu codename does not matter.
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg --yes
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq
  apt-get install -y -qq nvidia-container-toolkit
fi

log "Configuring Docker to use the NVIDIA runtime"
nvidia-ctk runtime configure --runtime=docker

# --- 3. Start Docker -------------------------------------------------------
log "Enabling and starting Docker"
systemctl enable --now docker
systemctl restart docker
sleep 2

if ! getent group docker | grep -q "\b${TARGET_USER}\b"; then
  log "Adding '${TARGET_USER}' to the docker group"
  usermod -aG docker "$TARGET_USER"
  warn "Group change needs a new login shell. Run 'exit' then reopen WSL (or 'wsl --shutdown' from Windows)."
fi

# --- 4. Verify -------------------------------------------------------------
log "Verifying GPU passthrough into a container"
if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L; then
  printf '\n\033[1;32m==> Host setup complete. Both GPUs are visible to Docker.\033[0m\n'
else
  die "GPU passthrough failed. Check that the Windows NVIDIA driver is current and /dev/dxg exists."
fi

cat <<'EOF'

Next steps:
  1. Open a NEW WSL shell (so docker group membership applies).
  2. cd into your clone of the repository.
  3. cp .env.example .env    # then set HF_TOKEN
  4. bash scripts/build.sh nvidia-cuda        # or dgx-spark on a DGX Spark
  5. bash scripts/smoke-test.sh geppetto:nvidia-cuda
  6. bash scripts/fetch-models.sh
  7. docker compose up -d                     # UI at http://localhost:8080/app

On WSL2, also set networkingMode=mirrored in %USERPROFILE%\.wslconfig, and arrange for something
to hold the VM open (e.g. a logon task running: wsl.exe -d Ubuntu -u root -e sleep infinity).
Otherwise WSL shuts the VM down when idle and takes Docker — and this service — with it.

EOF
