# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# TRELLIS.2 inference image
#
# Mirrors upstream setup.sh exactly, but layer-by-layer so a failing extension
# does not invalidate the whole build cache. Ubuntu 22.04 already ships
# Python 3.10, which is what upstream's `--new-env` creates, so a plain venv
# replaces conda.
#
# Model weights are NOT baked in — they live in the `models` volume (HF_HOME).
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Pinned upstream commit (main @ 2026-08-16). Bump deliberately, not by accident.
ARG TRELLIS2_COMMIT=75fbf0183001ed9876c8dbb35de6b68552ee08bd
# sm_89 = Ada Lovelace (RTX 6000 Ada / RTX 4090). Restricting this cuts the
# CUDA extension build time by roughly an order of magnitude.
ARG CUDA_ARCH=8.9
ARG MAX_JOBS=16

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

# --- System packages -------------------------------------------------------
# libegl/libgles/libglvnd: nvdiffrast needs a real GL/EGL toolchain.
# sudo: upstream setup.sh --basic shells out to `sudo apt install libjpeg-dev`.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3.10-dev python3-pip \
      build-essential ninja-build cmake pkg-config git curl ca-certificates sudo \
      libjpeg-dev zlib1g-dev libpng-dev \
      libglvnd-dev libgl1 libgl1-mesa-dev libegl1 libegl1-mesa-dev \
      libgles2 libgles2-mesa-dev libglib2.0-0 libsm6 libxext6 libxrender1 \
      ffmpeg unzip \
    && rm -rf /var/lib/apt/lists/*

# EGL vendor config so nvdiffrast can create a headless GL context in-container.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
 && printf '{\n  "file_format_version" : "1.0.0",\n  "ICD" : {\n    "library_path" : "libEGL_nvidia.so.0"\n  }\n}\n' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# --- Python environment ----------------------------------------------------
RUN python3.10 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip setuptools wheel packaging

# Torch first — every extension below builds against it with --no-build-isolation.
RUN pip install torch==2.6.0 torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cu124

# --- TRELLIS.2 source ------------------------------------------------------
# --recursive matters: o-voxel/third_party/eigen is a submodule and the
# o-voxel build fails without it.
RUN git clone https://github.com/microsoft/TRELLIS.2.git /opt/TRELLIS.2 \
 && cd /opt/TRELLIS.2 \
 && git checkout ${TRELLIS2_COMMIT} \
 && git submodule update --init --recursive
WORKDIR /opt/TRELLIS.2

ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${CUDA_ARCH} \
    MAX_JOBS=${MAX_JOBS}

# --- Base Python dependencies (upstream `--basic`) -------------------------
# pillow-simd is a drop-in speed optimization that frequently fails to compile;
# pre-satisfying the requirement lets setup.sh's own `pip install pillow-simd`
# no-op instead of aborting the layer.
RUN pip install pillow-simd || pip install "pillow>=10.4,<12"
RUN apt-get update \
 && bash setup.sh --basic \
 && rm -rf /var/lib/apt/lists/*

# --- CUDA extensions, one layer each ---------------------------------------
# flash-attn tries a prebuilt wheel matching torch2.6/cu12/cp310 first and
# falls back to a source build (slow but functional). If it fails outright,
# run the container with ATTN_BACKEND=xformers.
RUN bash setup.sh --flash-attn
RUN bash setup.sh --nvdiffrast
RUN bash setup.sh --nvdiffrec
RUN bash setup.sh --cumesh
RUN bash setup.sh --flexgemm
RUN bash setup.sh --o-voxel

# --- Inference service -----------------------------------------------------
COPY server/requirements.txt /opt/server/requirements.txt
RUN pip install -r /opt/server/requirements.txt
COPY server/ /opt/server/

# --- Runtime configuration -------------------------------------------------
ENV PYTHONPATH=/opt/TRELLIS.2:/opt/server \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/models \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    TRELLIS2_MODEL=microsoft/TRELLIS.2-4B \
    TRELLIS2_OUTPUT_DIR=/outputs \
    TRELLIS2_ENVMAP=/opt/TRELLIS.2/assets/hdri/forest.exr

VOLUME ["/models", "/outputs"]
EXPOSE 8000 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15m --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/opt/server"]
