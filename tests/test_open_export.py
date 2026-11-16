"""The open UV rasterizer must agree with nvdiffrast on what the bake needs.

GPU tests: run inside the image with `bash scripts/test.sh` (HOWTO.md, "Tests"). They
skip on a machine without CUDA. The nvdiffrast comparison is a test-only use of
nvdiffrast for evaluation; the server's open path never imports it.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from open_export import rasterize_uv  # noqa: E402


def _unwrapped_sphere(subdivisions: int = 5):
    trimesh = pytest.importorskip("trimesh")
    cumesh = pytest.importorskip("cumesh")
    s = trimesh.creation.icosphere(subdivisions=subdivisions)
    m = cumesh.CuMesh()
    m.init(
        torch.tensor(s.vertices, dtype=torch.float32, device="cuda"),
        torch.tensor(s.faces, dtype=torch.int32, device="cuda"),
    )
    v, f, uv, _ = m.uv_unwrap(return_vmaps=True)
    return v.cuda().float(), f.cuda().int(), uv.cuda().float()


def _layout_from_env():
    """GEPPETTO_TEST_GLB=path/to/asset.glb tests a real Geppetto UV layout."""
    path = os.environ.get("GEPPETTO_TEST_GLB")
    if not path:
        return None
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.load(path, force="mesh", process=False)
    uv = mesh.visual.uv.copy()
    return (
        torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda"),
        torch.tensor(mesh.faces, dtype=torch.int32, device="cuda"),
        torch.tensor(uv, dtype=torch.float32, device="cuda"),
    )


def test_single_triangle_covers_expected_texels():
    uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], device="cuda")
    faces = torch.tensor([[0, 1, 2]], device="cuda")
    fid, bary = rasterize_uv(uv, faces, 8)
    covered = fid >= 0
    # Texel centres (j+.5, i+.5)/8 inside the lower-left half: j + i <= 7.
    i, j = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    assert torch.equal(covered.cpu(), (i + j) <= 7)
    assert torch.allclose(bary[covered].sum(-1), torch.ones(int(covered.sum()), device="cuda"))
    # Barycentric weights reproduce the texel centre.
    pos = (uv[faces[0]].unsqueeze(0) * bary[covered].unsqueeze(-1)).sum(1)
    centres = torch.stack([(j[covered.cpu()] + 0.5) / 8, (i[covered.cpu()] + 0.5) / 8], -1).cuda()
    assert torch.allclose(pos, centres, atol=1e-5)


def test_degenerate_and_out_of_range_faces_are_ignored():
    uv = torch.tensor([[0.2, 0.2], [0.2, 0.2], [0.2, 0.2], [1.5, 1.5], [2.0, 1.5], [1.5, 2.0]], device="cuda")
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]], device="cuda")
    fid, _ = rasterize_uv(uv, faces, 16)
    assert (fid < 0).all()


def test_chunking_matches_single_pass(monkeypatch):
    import open_export

    v, f, uv = _unwrapped_sphere(4)
    fid_a, bary_a = rasterize_uv(uv, f, 256)
    monkeypatch.setattr(open_export, "_MAX_PAIRS", 997)
    fid_b, bary_b = rasterize_uv(uv, f, 256)
    assert torch.equal(fid_a >= 0, fid_b >= 0)
    pa = (v[f[fid_a[fid_a >= 0]].long()] * bary_a[fid_a >= 0].unsqueeze(-1)).sum(1)
    pb = (v[f[fid_b[fid_b >= 0]].long()] * bary_b[fid_b >= 0].unsqueeze(-1)).sum(1)
    assert torch.allclose(pa, pb, atol=1e-5)


@pytest.mark.parametrize("size", [512, 2048])
def test_agrees_with_nvdiffrast(size):
    dr = pytest.importorskip("nvdiffrast.torch")
    layout = _layout_from_env() or _unwrapped_sphere(6)
    v, f, uv = layout

    # Upstream's exact call (o_voxel.postprocess.to_glb).
    ctx = dr.RasterizeCudaContext()
    clip = torch.cat([uv * 2 - 1, torch.zeros_like(uv[:, :1]), torch.ones_like(uv[:, :1])], -1).unsqueeze(0)
    rast = torch.zeros((1, size, size, 4), device="cuda")
    for i in range(0, f.shape[0], 100000):
        chunk, _ = dr.rasterize(ctx, clip, f[i:i + 100000], resolution=[size, size])
        m = chunk[..., 3:4] > 0
        chunk[..., 3:4] += i
        rast = torch.where(m, chunk, rast)
    ref_mask = rast[0, ..., 3] > 0
    ref_pos = dr.interpolate(v.unsqueeze(0), rast, f)[0][0]

    fid, bary = rasterize_uv(uv, f, size)
    mask = fid >= 0
    pos = torch.zeros_like(ref_pos)
    pos[mask] = (v[f[fid[mask]].long()] * bary[mask].unsqueeze(-1)).sum(1)

    both = mask & ref_mask
    n_ref = ref_mask.sum().item()
    # nvdiffrast uses a strict fill rule on snapped coordinates; this rasterizer
    # includes edges. On a dense layout that moves ~0.5-3% of texels, all on chart
    # borders and evenly in both directions, so total coverage barely changes.
    assert abs(mask.sum().item() - n_ref) <= 0.005 * n_ref
    union = (mask | ref_mask).float()[None, None]
    interior = (-torch.nn.functional.max_pool2d(-union, 3, 1, 1))[0, 0] > 0.5
    holes = ((mask ^ ref_mask) & interior).sum().item()
    assert holes <= 5e-4 * n_ref, f"{holes} interior texels differ in coverage"
    err = (pos[both] - ref_pos[both]).norm(dim=-1)
    extent = (v.max(0).values - v.min(0).values).norm()
    # Same surface point to float precision (relative to the mesh size) for
    # almost every texel. The exceptions: texels where the unwrap folded a
    # triangle over its neighbour (two triangles cover it; either answer is as
    # good), and sub-texel triangles, where nvdiffrast's fixed-point snapping
    # of vertex positions moves its own answer. A real bug (flipped V, a
    # half-texel offset) moves nearly every texel, so 1% still catches it.
    off = (err > 1e-4 * extent).sum().item()
    assert off <= 1e-2 * both.sum().item(), f"{off} of {both.sum().item()} texels map elsewhere"
    assert torch.median(err).item() <= 1e-6 * extent.item()
