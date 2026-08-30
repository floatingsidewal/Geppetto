# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Geppetto — TRELLIS.2 inference image
#
# One Dockerfile, two release targets. Everything that differs between them is
# an ARG, so the targets cannot silently drift apart:
#
#   nvidia-cuda  x86_64 / sm_89  / CUDA 12.4 / torch 2.6+cu124  (verified)
#   dgx-spark    aarch64 / sm_121 / CUDA 13  / torch cu130      (UNVERIFIED)
#
# Build args for each live in docker/targets/*.args — pass one with
# `docker build $(sed 's/^/--build-arg /' docker/targets/<target>.args)` or let
# compose/CI do it.
#
# Mirrors upstream setup.sh, but layer-by-layer so a failing extension does not
# invalidate the whole build cache. A plain venv replaces upstream's conda.
#
# Model weights are NOT baked in — they live in the `models` volume (HF_HOME).
# ---------------------------------------------------------------------------
ARG BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# Pinned upstream commit (main @ 2026-08-16). Bump deliberately, not by accident.
ARG TRELLIS2_COMMIT=75fbf0183001ed9876c8dbb35de6b68552ee08bd

# --- Target-dependent knobs ------------------------------------------------
# CUDA_ARCH: 8.9 = Ada (RTX 6000 Ada / 4090); 12.1 = GB10 Blackwell (DGX Spark).
# Restricting to one arch cuts extension build time by ~an order of magnitude.
ARG CUDA_ARCH=8.9
# Ubuntu 22.04 ships 3.10; 24.04 ships 3.12. Upstream targets 3.10.
ARG PYTHON_VERSION=3.10
ARG TORCH_SPEC="torch==2.6.0 torchvision==0.21.0"
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG MAX_JOBS=16

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

# --- System packages -------------------------------------------------------
# libegl/libgles/libglvnd: nvdiffrast needs a real GL/EGL toolchain.
# sudo: upstream setup.sh --basic shells out to `sudo apt install libjpeg-dev`.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev python3-pip \
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
RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip setuptools wheel packaging

# Torch first — every extension below builds against it with --no-build-isolation.
# The sm_121 caveat for DGX Spark: stock wheels compile kernels only to sm_120,
# leaving GB10 on PTX JIT, which has been reported to hang during diffusion
# sampling. Point TORCH_INDEX_URL at an sm_121-native build if that bites.
RUN pip install ${TORCH_SPEC} --index-url ${TORCH_INDEX_URL} \
 && python -c "import torch; print('torch', torch.__version__)"

# --- TRELLIS.2 source ------------------------------------------------------
# --recursive matters: o-voxel/third_party/eigen is a submodule and the
# o-voxel build fails without it.
#
# Retried with backoff: the submodule chain reaches gitlab.com for eigen, which
# regularly returns "unable to handle this request due to load". git's own
# single retry is not enough, and a third-party outage should not fail a
# release build. Each attempt resumes what previous ones fetched.
RUN git clone https://github.com/microsoft/TRELLIS.2.git /opt/TRELLIS.2 \
 && cd /opt/TRELLIS.2 \
 && git checkout ${TRELLIS2_COMMIT} \
 && for attempt in 1 2 3 4 5; do \
      echo "submodule fetch attempt ${attempt}/5"; \
      git submodule update --init --recursive && break; \
      if [ "${attempt}" = "5" ]; then \
        echo "submodule fetch failed after 5 attempts" >&2; exit 1; \
      fi; \
      if [ "${attempt}" = "2" ]; then \
        echo "switching eigen to the GitHub mirror"; \
        git config --global url."https://github.com/libeigen/eigen.git".insteadOf "https://gitlab.com/libeigen/eigen.git"; \
      fi; \
      sleep $((attempt * 20)); \
    done
WORKDIR /opt/TRELLIS.2

ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${CUDA_ARCH} \
    MAX_JOBS=${MAX_JOBS}

# --- Build-time nvidia-smi stub --------------------------------------------
# setup.sh picks its platform with `command -v nvidia-smi` and aborts with
# "No supported GPU found" otherwise. The GPU is only present at run time, not
# during `docker build`, so every setup.sh layer would fail. The stub lives in
# its own directory and is prepended to PATH per-RUN, never at runtime — the
# NVIDIA container runtime injects the real nvidia-smi into /usr/bin, and this
# must not shadow or collide with it.
RUN mkdir -p /opt/buildstub \
 && echo '#!/bin/sh' > /opt/buildstub/nvidia-smi \
 && echo 'echo "stub nvidia-smi (build only)"' >> /opt/buildstub/nvidia-smi \
 && chmod +x /opt/buildstub/nvidia-smi

# --- Base Python dependencies (upstream `--basic`) -------------------------
# pillow-simd is a drop-in speed optimization that frequently fails to compile;
# pre-satisfying the requirement lets setup.sh's own `pip install pillow-simd`
# no-op instead of aborting the layer. It is replaced with stock Pillow later.
RUN pip install pillow-simd || pip install "pillow>=10.4,<12"
RUN apt-get update \
 && PATH=/opt/buildstub:$PATH bash setup.sh --basic \
 && rm -rf /var/lib/apt/lists/*

# --- CUDA extensions, one layer each ---------------------------------------
# Upstream's flash-attn line is `pip install flash-attn==2.7.3` with no
# --no-build-isolation, so its build env has no torch and metadata generation
# dies with ModuleNotFoundError. pip swallows that into a zero exit, so the
# layer "succeeds" with nothing installed. It is left here to mirror upstream;
# the real install is the explicit layer after the extensions, which is also
# where the failure becomes loud.
RUN PATH=/opt/buildstub:$PATH bash setup.sh --flash-attn
RUN PATH=/opt/buildstub:$PATH bash setup.sh --nvdiffrast
RUN PATH=/opt/buildstub:$PATH bash setup.sh --nvdiffrec
RUN PATH=/opt/buildstub:$PATH bash setup.sh --cumesh
RUN PATH=/opt/buildstub:$PATH bash setup.sh --flexgemm
RUN PATH=/opt/buildstub:$PATH bash setup.sh --o-voxel

# --- Attention backend ------------------------------------------------------
# trellis2/modules/sparse/config.py accepts ONLY xformers / flash_attn /
# flash_attn_3 for sparse attention — there is no sdpa or naive fallback there,
# unlike dense attention. So each target must ship a working one of those two,
# and the import check below is what proves it.
#
#   nvidia-cuda: prebuilt flash-attn wheel for this exact stack
#                (cp310 / torch 2.6 / cu12 / pre-2.7 cxx11 ABI).
#   dgx-spark:   no flash-attn wheel exists for aarch64+sm_121, and FA2 targets
#                sm_80/sm_90, so xformers is built from source instead.
#
# ATTN_PACKAGE is what pip installs; ATTN_IMPORT is what must then import.
ARG ATTN_PACKAGE=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
ARG ATTN_PACKAGE_FALLBACK="flash-attn==2.7.3"
ARG ATTN_IMPORT=flash_attn
ARG DEFAULT_ATTN_BACKEND=flash_attn
RUN pip install "${ATTN_PACKAGE}" \
 || pip install ${ATTN_PACKAGE_FALLBACK} --no-build-isolation \
 ; python -c "import ${ATTN_IMPORT} as m; print('${ATTN_IMPORT}', getattr(m, '__version__', 'unknown'))"

# --- Pillow: drop pillow-simd -----------------------------------------------
# pillow-simd is stuck at 9.5 and its WebP plugin is broken against modern
# libwebp — `PIL._webp` has no HAVE_WEBPANIM, so *importing* the plugin throws
# and every WebP upload 500s. Its SIMD speedup is irrelevant here (decode is
# microseconds next to a 60s generation), so stock Pillow wins on correctness.
# Done late so the expensive extension layers stay cached.
RUN pip uninstall -y pillow-simd pillow >/dev/null 2>&1 || true
RUN pip install --no-cache-dir "pillow>=10.4,<12" \
 && python -c "from PIL import Image, features; assert features.check('webp'), 'webp support missing'; print('pillow', Image.__version__, 'webp ok')"

# --- OpenCV: pin to 4.x so OpenEXR still decodes ----------------------------
# setup.sh installs opencv-python-headless unpinned, which now resolves to 5.x.
# OpenCV 5 no longer decodes OpenEXR, so cv2.imread() on the HDRI env map
# returns None even with OPENCV_IO_ENABLE_OPENEXR=1. The service treats that as
# "preview renders unavailable" and carries on, so the only symptom is that
# every preview_video request quietly produces no video. Pinning to 4.x
# restores it; the assertion makes a silent regression impossible.
RUN pip install --no-cache-dir "opencv-python-headless>=4.9,<5" \
 && OPENCV_IO_ENABLE_OPENEXR=1 python -c "import cv2; im = cv2.imread('/opt/TRELLIS.2/assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED); assert im is not None, 'OpenEXR decode failed — preview renders would be silently disabled'; print('opencv', cv2.__version__, 'exr ok', im.shape)"

# --- transformers: pin to the DINOv3 layout upstream was written against ----
# setup.sh installs transformers unpinned. trellis2/modules/image_feature_extractor.py
# iterates `self.model.layer`, which is the 4.56 DINOv3ViTModel layout; in 5.x
# the blocks moved into a nested DINOv3ViTEncoder and every generation dies with
# "'DINOv3ViTModel' object has no attribute 'layer'". The assertion turns a
# future version drift into a build failure instead of a runtime one.
RUN pip install --no-cache-dir "transformers==4.56.2" \
 && python -c "from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel; import inspect; assert 'self.layer' in inspect.getsource(DINOv3ViTModel.__init__), 'DINOv3ViTModel.layer missing — image_feature_extractor.py will fail'; print('transformers pinned, DINOv3 layout ok')"

# --- Inference service -----------------------------------------------------
COPY server/requirements.txt /opt/server/requirements.txt
RUN pip install -r /opt/server/requirements.txt
COPY server/ /opt/server/

# --- Runtime configuration -------------------------------------------------
# NVIDIA_VISIBLE_DEVICES is deliberately NOT set here. Baking `=all` into the
# image overrides the per-container device selection the caller asks for
# (compose `device_ids`, or `docker run --gpus`), so every container sees every
# GPU and they all pile onto device 0 — silently, because each one still reports
# a healthy GPU. Leaving it unset lets the runtime inject exactly what was
# requested, which is what makes GPU_ID and the `scale` profile work at all.
ENV PYTHONPATH=/opt/TRELLIS.2:/opt/server \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/models \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    NVIDIA_DRIVER_CAPABILITIES=all \
    ATTN_BACKEND=${DEFAULT_ATTN_BACKEND} \
    TRELLIS2_MODEL=microsoft/TRELLIS.2-4B \
    TRELLIS2_OUTPUT_DIR=/outputs \
    TRELLIS2_ENVMAP=/opt/TRELLIS.2/assets/hdri/forest.exr

# Recorded so a running container can say what it was built for, and so the
# release workflow can label the image without guessing.
ARG GEPPETTO_TARGET=nvidia-cuda
ENV GEPPETTO_TARGET=${GEPPETTO_TARGET} \
    GEPPETTO_CUDA_ARCH=${CUDA_ARCH}
LABEL org.opencontainers.image.title="Geppetto" \
      org.opencontainers.image.description="TRELLIS.2 image-to-3D inference service" \
      org.opencontainers.image.source="https://github.com/floatingsidewal/geppetto" \
      org.opencontainers.image.licenses="MIT" \
      io.geppetto.target="${GEPPETTO_TARGET}" \
      io.geppetto.cuda-arch="${CUDA_ARCH}"

VOLUME ["/models", "/outputs"]
EXPOSE 8000 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15m --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/opt/server"]
