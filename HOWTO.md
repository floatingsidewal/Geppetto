# Geppetto — how to build, install and use it

A practical walkthrough: get it running on your hardware, then turn photos into 3D models from a
browser, a script, or an agent.

Everything here has been run end to end on both supported platforms. Times and memory figures are
measured, not estimated.

- [Before you start](#before-you-start)
- [Step 1 — request model access](#step-1--request-model-access-do-this-first)
- [Step 2 — prepare the host](#step-2--prepare-the-host)
- [Step 3 — build the container](#step-3--build-the-container)
- [Step 4 — download the weights](#step-4--download-the-weights)
- [Step 5 — start it](#step-5--start-it)
- [Using it: the web UI](#using-it-the-web-ui)
- [Using it: the command line](#using-it-the-command-line)
- [Using it: the REST API](#using-it-the-rest-api)
- [Using it: from Claude or Copilot](#using-it-from-claude-or-copilot)
- [Sharing the built container](#sharing-the-built-container)
- [Getting good results](#getting-good-results)
- [Troubleshooting](#troubleshooting)

---

## Before you start

You need one of these:

| Platform | Hardware | Build target |
|---|---|---|
| **Workstation / server** | x86_64 + NVIDIA Ada GPU (RTX 6000 Ada, RTX 4090, L40S) | `nvidia-cuda` |
| **DGX Spark** | aarch64 + GB10 Grace Blackwell | `dgx-spark` |

Plus:

- **Linux, or Windows with WSL2.** Native Linux is simpler; WSL2 works with two extra settings
  covered [below](#windows--wsl2-only-two-settings-that-matter).
- **~50 GB of disk** — 26 GB image, 21 GB weights.
- **12 GB of GPU memory** for default settings; 16 GB clears every setting. Far less than "4B
  parameters" suggests — peak measured is 11.3 GB at maximum quality.
- **A Hugging Face account.**

> **Running a different NVIDIA generation?** (Ampere, Hopper, a 5090.) Edit `CUDA_ARCH` in
> `docker/targets/nvidia-cuda.args` — Ampere `8.6`, Ada `8.9`, Hopper `9.0`, Blackwell `12.0`/`12.1`
> — and build as normal. Only that architecture is compiled, which is why the build takes minutes
> instead of hours.

---

## Step 1 — request model access (do this first)

Geppetto needs two gated Hugging Face repositories, and **one of them is reviewed by a human at
Meta, which can take hours to days.** Request it before you do anything else, or you will finish
every other step and then wait.

1. **[facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)**
   — **required.** The image conditioning encoder. Manual approval. Nothing generates without it.
2. **[briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)** — optional, automatic approval.
   Background removal. Skip it entirely by feeding pre-cut RGBA images (see
   [Getting good results](#getting-good-results)).

Accept the licences with the account that will own your token, then create a **read** token at
<https://huggingface.co/settings/tokens>.

Approval is **per account**. Each collaborator requests their own — you cannot run Geppetto on
someone else's grant.

Check your access at any time, without Docker or a built image:

```bash
HF_TOKEN=hf_xxx bash scripts/check-hf-access.sh
```

---

## Step 2 — prepare the host

```bash
git clone https://github.com/floatingsidewal/Geppetto.git
cd Geppetto
```

### Docker + NVIDIA Container Toolkit

If the host already runs GPU containers, skip this. Otherwise:

```bash
sudo bash scripts/host-setup.sh
```

It installs Docker Engine and the NVIDIA Container Toolkit, wires the runtime, and verifies GPU
passthrough by running `nvidia-smi` inside a container. **Then open a new shell** so your Docker
group membership applies.

Confirm it worked:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L
```

You should see your GPU listed. If not, stop here — nothing downstream will work.

### Configure

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```bash
HF_TOKEN=hf_your_token_here
```

Two more worth knowing:

- `GPU_ID=0` — which GPU to use. `nvidia-smi -L` lists them. On a multi-GPU machine where GPU 0
  drives a display, point it at another index so a long generation does not fight your desktop.
- `GEPPETTO_TARGET=dgx-spark` — **set this on a DGX Spark.** It selects the right image. Leave it
  unset on a workstation.

### Windows / WSL2 only: two settings that matter

Both of these cost hours if you discover them the hard way.

**1. Mirrored networking**, or the service is only reachable on a NAT address that changes every
restart. Create `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
dnsTunneling=true
```

Apply with `wsl --shutdown` (from PowerShell). After this, `localhost:8080` works from Windows.

**2. Something must hold the WSL VM open.** With no client attached, WSL shuts the VM down and takes
Docker — and Geppetto — with it. The symptom is maddening: the service is reachable whenever you are
actively typing commands, and dead whenever you actually try to use it. `vmIdleTimeout` does **not**
prevent this; we tested it.

Create a logon task that keeps a session attached (PowerShell, once):

```powershell
$action  = New-ScheduledTaskAction -Execute 'wsl.exe' -Argument '-d Ubuntu -u root -e sleep infinity'
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERNAME"
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -Hidden
Register-ScheduledTask -TaskName 'WSL-Keepalive' -Action $action -Trigger $trigger `
  -Settings $settings -Description 'Keeps the WSL2 VM (and Docker) alive.' -Force
Start-ScheduledTask -TaskName 'WSL-Keepalive'
```

Verified: 24/24 health checks pass with this running, 0/20 without.

---

## Step 3 — build the container

Pick the target for your hardware:

```bash
bash scripts/build.sh nvidia-cuda     # x86_64 + Ada
bash scripts/build.sh dgx-spark       # DGX Spark
```

**Roughly 10–20 minutes.** It compiles six CUDA extensions from source. The script refuses to build
a target on the wrong CPU architecture rather than wasting the time.

Then verify it before trusting it:

```bash
bash scripts/smoke-test.sh geppetto:nvidia-cuda      # or geppetto:dgx-spark
```

This asserts that torch sees the GPU, that the GPU matches what the image was built for, that all six
CUDA extensions import, and that the attention backend is present. It takes seconds and catches the
failures that otherwise appear minutes into your first generation. Expect:

```
torch 2.6.0+cu124
gpu   NVIDIA RTX 6000 Ada Generation
arch  sm_89
ok    o_voxel
ok    cumesh
ok    flex_gemm
ok    nvdiffrast
ok    utils3d
ok    trellis2
ok    attention backend flash_attn (2.7.3)
ok    pillow webp
level 1 ok
```

---

## Step 4 — download the weights

```bash
bash scripts/fetch-models.sh
```

~21 GB into a Docker volume named `trellis2-models`, once per machine. Rebuilding the image does not
re-download it.

On a DGX Spark, tell it which image to use:

```bash
GEPPETTO_IMAGE=geppetto:dgx-spark bash scripts/fetch-models.sh
```

If DINOv3 access has not been granted yet, this stops and tells you which link to click.

---

## Step 5 — start it

```bash
docker compose up -d
```

The model takes **about 90 seconds** to load. Watch for it:

```bash
curl -s localhost:8080/healthz
```

`"model_loaded": true` means ready. Then open:

**<http://localhost:8080/app>**

To reach it from other machines on your network, use the host's name or address —
`http://your-host.local:8080/app`. The port is published on all interfaces.

> **No authentication.** Anything that can reach port 8080 can drive your GPU. Keep it on a trusted
> network. If it needs to be exposed more widely, put a reverse proxy with authentication in front
> and restrict `GEPPETTO_CORS_ORIGINS` in `.env`.

Stop it with `docker compose down`. It restarts automatically on boot (`restart: unless-stopped`).

---

## Using it: the web UI

The simplest path, and it is built into the container — no separate install.

1. Open **<http://localhost:8080/app>**
2. Drag an image in, or click to pick one (PNG, JPEG, WebP)
3. Choose **Detail** and **Texture**, tick the formats you want
4. Press **Make it real**
5. Download the results when it finishes

It shows queue position and elapsed time while working. Generation takes minutes, not seconds — that
is normal.

**The page also runs from a file.** Copy `server/static/index.html` anywhere — a laptop, a USB
stick — open it in a browser, expand *Server settings*, and point it at your server
(`http://your-host.local:8080`). It remembers. Nothing to install, useful for collaborators who
should not have to touch the terminal.

### Which format do you want?

| Format | Use it for | Carries texture? |
|---|---|---|
| **GLB** | viewing, sharing, game engines, Windows 3D Viewer | **Yes** — the only one that does |
| **STL** | 3D printing, slicers | No — geometry only |
| **OBJ** / **PLY** | tools that want vertex colours | Vertex colours, no texture map |

GLB is always produced. The others are converted from the same mesh, so asking for several costs
seconds, not another generation.

---

## Using it: the command line

`scripts/convert.py` needs **only the Python standard library** — no `pip install`. It works from any
machine that can reach the server, including Windows.

```bash
# one image
python3 scripts/convert.py photo.png --format stl --out results/

# a whole folder
python3 scripts/convert.py ~/photos/ --format stl --out results/

# against a remote server
python3 scripts/convert.py photo.png --url http://spark.home.lan:8080 --out results/
```

Useful flags:

| Flag | What it does |
|---|---|
| `--format stl` | also write STL (repeatable: `--format stl --format obj`) |
| `--resolution 512\|1024\|1536` | detail; default 1024 |
| `--texture-size 1024\|2048\|4096` | texture resolution; default 2048 |
| `--no-preprocess` | skip background removal — for pre-cut RGBA images |
| `--preview` | also render a turntable MP4 (slow) |
| `--skip-existing` | resume a batch without redoing finished images |
| `--seed 12345` | reproduce an exact result |

It waits for the model to load, submits each image, downloads the results, and writes
`conversion-summary.json` with timings, seeds and peak memory for every image.

---

## Using it: the REST API

For your own scripts, in any language.

```bash
# submit -> {"job_id":"a1b2c3d4e5f6", ...}
curl -X POST localhost:8080/v1/generate \
  -F image=@photo.png \
  -F 'params={"resolution":1024,"export_formats":["stl"]}'

# poll until "completed" or "failed"
curl -s localhost:8080/v1/jobs/a1b2c3d4e5f6

# download
curl -o model.glb localhost:8080/v1/jobs/a1b2c3d4e5f6/asset.glb
curl -o model.stl localhost:8080/v1/jobs/a1b2c3d4e5f6/asset.stl
```

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | status, GPU, queue depth |
| `POST` | `/v1/generate` | submit an image → `202` + job id |
| `GET` | `/v1/jobs` | recent jobs |
| `GET` | `/v1/jobs/{id}` | job state, timings, warnings, errors |
| `GET` | `/v1/jobs/{id}/asset.glb` | the textured mesh |
| `GET` | `/v1/jobs/{id}/asset.{stl,obj,ply}` | other formats, if requested |
| `GET` | `/v1/jobs/{id}/preview.mp4` | turntable render, if requested |
| `DELETE` | `/v1/jobs/{id}` | delete the job and its files |

Interactive docs, with every parameter: **<http://localhost:8080/docs>**

Jobs are written to `outputs/{job_id}/` and survive a restart. Work that was mid-flight when the
service stopped is marked failed rather than lost silently.

---

## Using it: from Claude or Copilot

`skills/geppetto/SKILL.md` is an agent skill. Copy it into your skills directory:

```bash
mkdir -p ~/.claude/skills/geppetto
cp skills/geppetto/SKILL.md ~/.claude/skills/geppetto/
```

Then ask in plain language — *"turn this photo into something I can print"* — and the agent will
check the server is healthy, submit the job, wait, and tell you where the files are. Set
`GEPPETTO_URL` if the server is not on `localhost:8080`.

---

## Sharing the built container

No prebuilt images are published. Each machine builds its own — but you can hand a built image to a
collaborator with the same hardware instead of making them rebuild:

```bash
# on the machine that has it
docker save geppetto:nvidia-cuda | gzip > geppetto-nvidia-cuda.tar.gz    # ~9 GB compressed

# on theirs
gunzip -c geppetto-nvidia-cuda.tar.gz | docker load
```

**The image must match their hardware** — an `nvidia-cuda` image will not run on a DGX Spark, and
vice versa. They still need their own Hugging Face access and their own `scripts/fetch-models.sh`
run; weights are not in the image.

Honestly, on a decent machine `bash scripts/build.sh` is ~15 minutes against a ~9 GB transfer.
Building is usually the better answer.

---

## Getting good results

Geppetto infers 3D from the depth cues in a photograph — shading, occlusion, perspective. That
dictates what works.

**Works well**

- Photographs of real objects, especially one clear subject
- Even lighting that reveals form; hard shadows and blown highlights both hide geometry
- Three-quarter angles, product-style shots, objects on a table or held up

**Works poorly**

- **Line art, flat illustrations, diagrams, logos.** No depth cues to read. We tried a 19th-century
  pen-and-ink illustration and got a shapeless blob wearing the drawing's colours.
- **Scenes rather than objects.** Background removal has to choose one subject; a landscape
  containing several things gives it nothing to lock onto.
- Motion blur, very low resolution, or a subject mostly hidden behind something else

**Practical advice**

- A better photograph beats a higher `resolution` setting almost every time.
- Cut the background yourself when you can. A pre-cut RGBA PNG with `--no-preprocess` is faster,
  more predictable, and never loads the non-commercial RMBG-2.0 weights.
- The seed changes the result. The same image at the same settings varies noticeably between runs —
  if one attempt disappoints, run it again before changing anything.
- Meshes are **not watertight**. For anything structural, expect a repair pass in your slicer.

### What to expect, measured

| Settings | RTX 6000 Ada | DGX Spark | Peak GPU memory |
|---|---|---|---|
| 512³, texture 1024 | 90 s | 171 s | 2.6 GB |
| 1024³, texture 2048 (default) | 110–190 s | 294 s | 4.1–6.2 GB |
| 1536³, texture 4096 (maximum) | 300 s | not measured | 11.3 GB |

The Spark is slower despite newer silicon: its unified memory runs ~273 GB/s against the RTX 6000
Ada's ~960 GB/s, and this work is bandwidth-bound. What it offers instead is 121 GB of unified
memory on a quiet, dedicated box.

---

## Troubleshooting

**`/healthz` says `model_loaded: false` and never changes**
First load takes ~90 seconds. Longer than a few minutes means something is wrong: check
`docker compose logs api` for a `load_error`. Most often the weights were never fetched, or DINOv3
access was not granted.

**A job fails with `'DINOv3ViTModel' object has no attribute 'layer'`**
Your image was built with an incompatible `transformers`. Rebuild — the Dockerfile pins 4.56.2 and
asserts the layout at build time.

**Every WebP upload returns an error**
An old Pillow with a broken WebP plugin. Rebuild; stock Pillow is pinned and verified during the
build.

**`preview_video` produces no video**
OpenCV without OpenEXR support cannot load the lighting environment map, and preview renders are
disabled with a warning. Rebuild — OpenCV is pinned to 4.x and the env map is decoded during the
build to prove it.

**The service is reachable sometimes and dead other times (WSL2)**
The WSL VM is idling out and taking Docker with it. See
[the keepalive task](#windows--wsl2-only-two-settings-that-matter).

**Two containers, but both land on the same GPU**
Set `GPU_ID` (and `GPU_ID_2` for the `scale` profile) in `.env`. Compose pins with
`CUDA_VISIBLE_DEVICES`, because under WSL2 the runtime-level device selection is silently ignored
and every container sees every GPU.

**`docker compose up` starts a long build on a DGX Spark**
`GEPPETTO_TARGET=dgx-spark` is missing from `.env`, so compose defaulted to the x86 image and began
building it.

**Permission denied writing to `outputs/`**
Docker created the directory as root. Fix with
`docker run --rm -v "$PWD:/w" alpine chown -R "$(id -u):$(id -g)" /w/outputs`.

**Out of memory during generation**
Lower `--resolution`, then `--texture-size`. 512³ peaks at 2.6 GB.

---

## Running more than one GPU

One container owns one GPU; generation is serialized behind a single worker because diffusion
inference is not safely concurrent on a device. Throughput comes from more containers:

```bash
# .env
GPU_ID=0
GPU_ID_2=1

docker compose --profile scale up -d     # second worker on :8081
```

Then point different clients at `:8080` and `:8081`, or put a least-connections proxy in front of
both. For more than two cards, copy the `api2` block per GPU with its own `device_ids` and port.
