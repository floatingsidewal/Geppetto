#!/usr/bin/env python3
"""Compare export backends and matting models on the same generations.

Runs INSIDE the image (it loads the pipeline itself; it does not use the API),
so it never disturbs a running service:

    bash scripts/compare-backends.sh outputs/compare-01 img1.png img2.png:nomatte ...

For each input it generates ONE mesh (fixed seed), then exports it three
times: `open` twice (to measure run-to-run noise: the UV unwrap is not
deterministic) and `nvdiffrast` once. The open exports run first, and the
script records whether nvdiffrast had been loaded by then.

It also runs both matting models (BiRefNet, RMBG-2.0) on every input's RGB
and writes the cut-outs side by side with their mask IoU.

Suffix an input with `:nomatte` to generate without preprocessing (as a
pre-cut or deliberately unmatted input would be), or `:alpha` to keep its own
alpha channel.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/server")

import licence_guard  # noqa: E402

licence_guard.install()

import matting  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

SEED = 42


def checker(size, n=16):
    w, h = size
    y, x = np.mgrid[0:h, 0:w]
    c = (((x // n) + (y // n)) % 2) * 60 + 160
    return Image.fromarray(np.stack([c, c, c], -1).astype(np.uint8))


def matte_compare(rembg, img: Image.Image, out: Path, stem: str) -> dict:
    rgb = img.convert("RGB")
    result, masks = {}, {}
    tiles = [rgb]
    for choice in ("birefnet", "rmbg2"):
        rembg.choice = choice
        rembg.to("cuda")
        rembg(rgb.copy())  # warm-up: first call includes the load
        torch.cuda.synchronize()
        t0 = time.time()
        cut = rembg(rgb.copy())
        torch.cuda.synchronize()
        result[f"{choice}_seconds"] = round(time.time() - t0, 3)
        rembg.cpu()
        a = np.array(cut)[..., 3].astype(np.float32) / 255
        masks[choice] = a
        bg = checker(cut.size)
        bg.paste(cut, (0, 0), cut)
        tiles.append(bg)
    inter = np.minimum(masks["birefnet"], masks["rmbg2"]).sum()
    union = np.maximum(masks["birefnet"], masks["rmbg2"]).sum()
    result["mask_iou"] = round(float(inter / max(union, 1e-6)), 4)
    result["mask_mean_abs_diff"] = round(float(np.abs(masks["birefnet"] - masks["rmbg2"]).mean()), 4)
    w, h = rgb.size
    sheet = Image.new("RGB", (w * 3, h), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * w, 0))
    sheet.save(out / f"{stem}_matting_input-birefnet-rmbg2.png")
    return result


def export(to_glb, mesh, p):
    return to_glb(
        vertices=mesh.vertices.clone(),
        faces=mesh.faces.clone(),
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=500_000,
        texture_size=2048,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )


def main() -> int:
    out = Path(sys.argv[1])
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty; use a fresh folder")
    # Write report.json after every input, so a failure keeps what finished.
    out.mkdir(parents=True, exist_ok=True)
    inputs = sys.argv[2:]

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    matting.install()
    t0 = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    report = {
        "load_seconds": round(time.time() - t0, 1),
        "matting_loaded_after_model_load": pipeline.rembg_model.loaded,
        "noncommercial_loaded_after_model_load": licence_guard.noncommercial_loaded(),
        "inputs": [],
    }
    print(json.dumps(report), flush=True)

    from open_export import to_glb as open_to_glb

    meshes = []
    for spec in inputs:
        path, _, mode = spec.partition(":")
        stem = Path(path).parent.name + "_" + Path(path).stem
        img = Image.open(path)
        entry = {"input": path, "mode": mode or "matte", "stem": stem}

        # Generation (once).
        if mode == "alpha":
            gen_img, pre = img, True  # upstream keeps an existing alpha, no matting
        elif mode == "nomatte":
            gen_img, pre = img, False
        else:
            gen_img, pre = img.convert("RGB"), True
        pipeline.rembg_model.choice = "birefnet"
        pipeline.rembg_model.last_used = None
        t0 = time.time()
        mesh = pipeline.run(gen_img, seed=SEED, preprocess_image=pre)[0]
        entry["generate_seconds"] = round(time.time() - t0, 1)
        entry["matting_used"] = pipeline.rembg_model.last_used
        mesh.simplify(16_777_216)
        meshes.append((stem, mesh))

        for tag in ("open_a", "open_b"):
            torch.cuda.synchronize()
            t0 = time.time()
            glb = export(open_to_glb, mesh, None)
            torch.cuda.synchronize()
            entry[f"{tag}_seconds"] = round(time.time() - t0, 1)
            glb.export(str(out / f"{stem}_{tag}.glb"), extension_webp=True)
            entry[f"{tag}_faces"] = int(len(glb.faces))
        entry["noncommercial_loaded_after_open_exports"] = licence_guard.noncommercial_loaded()

        entry.update(matte_compare(pipeline.rembg_model, img, out, stem))
        report["inputs"].append(entry)
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(entry), flush=True)

    # nvdiffrast exports last, so the open runs above were provably without it.
    from o_voxel.postprocess import to_glb as nv_to_glb

    for (stem, mesh), entry in zip(meshes, report["inputs"]):
        torch.cuda.synchronize()
        t0 = time.time()
        glb = export(nv_to_glb, mesh, None)
        torch.cuda.synchronize()
        entry["nvdiffrast_seconds"] = round(time.time() - t0, 1)
        entry["nvdiffrast_faces"] = int(len(glb.faces))
        glb.export(str(out / f"{stem}_nvdiffrast.glb"), extension_webp=True)
        print(json.dumps({"stem": stem, "nvdiffrast_seconds": entry["nvdiffrast_seconds"]}), flush=True)
    report["noncommercial_loaded_at_end"] = licence_guard.noncommercial_loaded()

    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("done", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
