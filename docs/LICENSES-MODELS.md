# Licence register: models, weights and libraries

Every model, weight file and library Geppetto loads at runtime, its licence, whether it allows
commercial use, and which code paths load it. Each licence was read on the component's own page or
licence file on **2026-11-16**. **This is a description of the dependency graph, not legal advice.**

**Summary**

- With the defaults (`export_backend: open`, `matting: birefnet`, `preview_video: false`), a job runs
  **no non-commercial code and loads no non-commercial weights**. Every job records what it used in
  `noncommercial_components` (empty means none), and `/healthz` lists what is loaded in the process
  (`noncommercial_loaded`).
- Three components are **non-commercial**, and each is now **opt-in**: nvdiffrast, nvdiffrec and
  briaai/RMBG-2.0 (see [Non-commercial components](#non-commercial-components)).
- **Open question: TRELLIS.2's research-only language.** The code and the weights are MIT, but the
  project page says the materials are for research only (see [below](#open-question-trellis2s-research-only-language)).
  It applies to every job, on every setting. Until counsel resolves it, treat all output as
  research use.
- **DINOv3** is required on every job. Commercial use appears to be allowed, with conditions,
  including "Built with DINOv3" attribution.

## Models and weights

| Weights | Role | Licence | Commercial use | Loaded by |
|---|---|---|---|---|
| [`microsoft/TRELLIS.2-4B`](https://huggingface.co/microsoft/TRELLIS.2-4B) | the generator (flow models, decoders) | MIT (model card) | **Open question**: see below | every job; loaded at start |
| [`microsoft/TRELLIS-image-large`](https://huggingface.co/microsoft/TRELLIS-image-large) (`ss_dec_conv3d_16l8_fp16` only) | sparse-structure decoder | MIT (model card) | Yes (same open question as the project) | every job; loaded at start |
| [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | image conditioning encoder | [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/) | Yes, with conditions: ship the licence with any redistribution, display "Built with DINOv3", and follow the restricted-use list (trade controls, military and weapons uses, among others). Gated: each operator must accept the terms. | every job; loaded at start |
| [`ZhengPeng7/BiRefNet`](https://huggingface.co/ZhengPeng7/BiRefNet) (code: [GitHub](https://github.com/ZhengPeng7/BiRefNet)) | background removal, **default** | MIT (weights and code) | Yes | first job with `preprocess_image: true` and an opaque input; never before |
| [`briaai/RMBG-2.0`](https://huggingface.co/briaai/RMBG-2.0) | background removal, **opt-in** (`matting: rmbg2` or `GEPPETTO_MATTING=rmbg2`) | CC BY-NC 4.0 | **No**: commercial use needs an agreement with BRIA | only when chosen and needed; never before. Upstream's `pipeline.json` names it, but `server/matting.py` replaces it with a loader that does nothing until first use. |
| HDRI env maps (`TRELLIS.2/assets/hdri`, from [Poly Haven](https://polyhaven.com)) | preview lighting | CC0 (`assets/hdri/license.txt`) | Yes | read at start; used only by `preview_video` |

BiRefNet weights are MIT. They were trained on public research datasets (DIS5K, HRSOD and others),
which have terms of their own. The MIT licence on the weights is the licence that applies to using
them. Mention the datasets if the weights' provenance is ever audited.

## Code: the generation and export path (always runs)

| Library | Role | Licence | Commercial use |
|---|---|---|---|
| [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (`trellis2`, `o_voxel`) | pipeline, mesh extraction, upstream export | MIT | Yes (open question below) |
| Geppetto `server/open_export.py` | `open` export backend; adapted from `o_voxel.postprocess.to_glb` (MIT) | MIT | Yes |
| [CuMesh](https://github.com/JeffreyXiang/CuMesh) | hole filling, remeshing, simplification, UV unwrap | MIT; bundles [cubvh](https://github.com/ashawkey/cubvh) (MIT) and [xatlas](https://github.com/jpcy/xatlas) (MIT). The simplifier follows PaMO's published algorithm; the code is CuMesh's own. | Yes |
| [FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM) | sparse convolution; trilinear volume sampling for the bake | MIT | Yes |
| [utils3d](https://github.com/EasternJournalist/utils3d) | geometry helpers | MIT | Yes |
| flash-attn (`nvidia-cuda`) / xformers (`dgx-spark`) | attention kernels | BSD-3-Clause | Yes |
| PyTorch, torchvision | tensors, CUDA | BSD-3-Clause | Yes |
| transformers, huggingface_hub, safetensors, accelerate, timm, kornia | model loading, DINOv3 and BiRefNet backbones | Apache-2.0 | Yes |
| einops, trimesh, easydict, pydantic, FastAPI | BiRefNet code, GLB writing, config, API | MIT | Yes |
| OpenCV 4.x | seam inpainting, EXR decode | Apache-2.0 | Yes |
| Pillow | images | MIT-CMU (HPND) | Yes |
| NumPy, SciPy, uvicorn | | BSD | Yes |

## Non-commercial components

| Component | Licence | Loaded by | Default |
|---|---|---|---|
| [nvdiffrast](https://github.com/NVlabs/nvdiffrast) | [NVIDIA Source Code License](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt), §3.3: "may be used or intended for use non-commercially", meaning "for research or evaluation purposes only" | `export_backend: nvdiffrast` (upstream's texture bake) and `preview_video: true` | **off**. `o_voxel` imports it at module level, but `server/licence_guard.py` defers the import, so its code does not run unless one of these is used. |
| [nvdiffrec](https://github.com/JeffreyXiang/nvdiffrec) (`nvdiffrec_render`, JeffreyXiang fork) | NVIDIA Source Code License (non-commercial) | `preview_video: true` (split-sum PBR lighting) | **off** |
| [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) | CC BY-NC 4.0 | `matting: rmbg2` | **off**; weights are fetched only with `fetch-models.sh --with-rmbg2` |

To keep a deployment free of all three: leave `GEPPETTO_EXPORT_BACKEND` and `GEPPETTO_MATTING`
at their defaults, do not request `preview_video`, and check `noncommercial_components` on each
job. The libraries stay **installed** in the image, because `o_voxel` imports nvdiffrast and the
other paths need them, but their code does not run. Removing them from the image is possible with
a stub `nvdiffrast` package; that has not been done yet.

### Owner's position (2026-11-16)

The downstream consumer (the Byways game) is a non-commercial research project for now.
Research-licensed components may be used while it stays non-commercial. Everything non-commercial is
listed above so it can be replaced or licensed before any commercial use.

## Open question: TRELLIS.2's research-only language

- The [repository](https://github.com/microsoft/TRELLIS.2) `LICENSE` is **MIT** (Copyright (c)
  Microsoft Corporation).
- The [model card](https://huggingface.co/microsoft/TRELLIS.2-4B) says "This model is released under
  the MIT License".
- The [project page](https://microsoft.github.io/TRELLIS.2/) says: "TRELLIS.2 is purely a research
  project", and its Material Disclaimer says: "The materials made available on this page are
  provided solely for academic and research purposes" and "These materials are not intended for
  commercial exploitation or use."

Whether that disclaimer narrows the MIT grant for the code and weights, or only covers the project
page's own materials (demo videos, renders), is a question for counsel. It applies to **every**
Geppetto output, whatever the switches are set to. No Geppetto switch can resolve it.

## Not loaded at runtime

These are in the image or the repo, but not in the service's code path:

- **gradio** (Apache-2.0): only for the opt-in upstream demo (`--profile ui`), which also runs
  upstream's own `app.py` with nvdiffrast and RMBG-2.0 as upstream ships it.
- **ffmpeg / imageio-ffmpeg**: only writes the `preview_video` MP4. The ffmpeg binaries are
  LGPL/GPL builds; a concern only if the image is redistributed.
- **The base image** `nvidia/cuda:*-devel`: the NVIDIA Deep Learning Container License and the
  CUDA EULA cover redistributing the **image**. They do not cover generated output.
