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
# no-op instead of aborting the layer.
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

# --- flash-attn (the install that actually works) --------------------------
# Prebuilt wheel matching this exact stack (cp310 / torch 2.6 / cu12 / pre-2.7
# cxx11 ABI). Falls back to a source build with --no-build-isolation, which is
# correct but slow. The import check makes a silent no-op impossible: if
# flash-attn is missing, the build stops here instead of at first inference.
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN pip install "${FLASH_ATTN_WHEEL}" \
 || pip install flash-attn==2.7.3 --no-build-isolation \
 ; python -c "import flash_attn; print('flash-attn', flash_attn.__version__)"

# --- Pillow: drop pillow-simd -----------------------------------------------
# pillow-simd is stuck at 9.5 and its WebP plugin is broken against modern
# libwebp — `PIL._webp` has no HAVE_WEBPANIM, so *importing* the plugin throws
# and every WebP upload 500s. Its SIMD speedup is irrelevant here (decode is
# microseconds next to a 60s generation), so stock Pillow wins on correctness.
# Done late so the expensive extension layers stay cached.
RUN pip uninstall -y pillow-simd pillow >/dev/null 2>&1 || true
RUN pip install --no-cache-dir "pillow>=10.4,<12" \
 && python -c "from PIL import Image, features; assert features.check('webp'), 'webp support missing'; print('pillow', Image.__version__, 'webp ok')"

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
