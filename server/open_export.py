"""GLB export with no non-commercial components (`export_backend: "open"`).

This is upstream's `o_voxel.postprocess.to_glb` (TRELLIS.2, MIT, Copyright (c)
Microsoft Corporation) with one change: the texture bake's UV-space
rasterization, which upstream does with nvdiffrast (NVIDIA Source Code License,
non-commercial), is done by `rasterize_uv()` below in plain PyTorch.

Everything else is the same code path and the same libraries: CuMesh (MIT) for
hole filling, remeshing, simplification and the xatlas-based UV unwrap,
FlexGEMM (MIT) for trilinear sampling of the attribute volume, OpenCV for seam
inpainting and trimesh for the GLB. So the mesh, the UV layout and the texel
values match the nvdiffrast backend; only which texels count as covered can
differ, on triangle edges (see `rasterize_uv`).

nvdiffrast's rasterizer is general-purpose and differentiable. The bake needs
neither: it renders a flat UV layout once, with no depth test, and only wants
"which triangle covers this texel centre, and where". That is a few lines of
vectorised barycentric tests.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import torch

# Upper bound on candidate (triangle, texel) pairs tested at once. 24M pairs is
# roughly 1.2 GB of temporaries; a 2048² bake of a 500k-face mesh needs ~2-4
# rounds.
_MAX_PAIRS = 24_000_000


def rasterize_uv(uvs: torch.Tensor, faces: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize a UV layout into a `size`×`size` texel grid.

    Uses the same texel convention as upstream's nvdiffrast call (clip
    coordinates `uv * 2 - 1`): texel (row i, column j) has its centre at
    `u = (j + 0.5) / size`, `v = (i + 0.5) / size`.

    A texel is covered when its centre lies inside or on the edge of a
    triangle. nvdiffrast applies a strict top-left fill rule instead, so a
    texel centre exactly on an edge may be assigned to the neighbouring
    triangle, or (on a chart border) be uncovered there and inpainted. Either
    way the position it maps to is the same point on the surface.

    Returns `(face_id, bary)`: `face_id` is `(size, size)` int64, -1 where no
    triangle covers the texel; `bary` is `(size, size, 3)` float32 weights of
    the triangle's three vertices.
    """
    device = uvs.device
    n_texels = size * size
    face_id = torch.full((n_texels,), -1, dtype=torch.int64, device=device)
    bary = torch.zeros((n_texels, 3), dtype=torch.float32, device=device)

    # Texel-centre coordinates: centre of column j sits at x == j.
    p = uvs.float() * size - 0.5
    tri = p[faces.long()]  # (F, 3, 2)
    lo = tri.min(dim=1).values
    hi = tri.max(dim=1).values
    x0 = torch.ceil(lo[:, 0]).clamp(min=0).long()
    y0 = torch.ceil(lo[:, 1]).clamp(min=0).long()
    x1 = torch.floor(hi[:, 0]).clamp(max=size - 1).long()
    y1 = torch.floor(hi[:, 1]).clamp(max=size - 1).long()
    w = (x1 - x0 + 1).clamp(min=0)
    h = (y1 - y0 + 1).clamp(min=0)
    counts = w * h

    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    counts = torch.where(area.abs() > 1e-12, counts, torch.zeros_like(counts))

    # Split the faces into groups whose candidate texels fit in _MAX_PAIRS.
    ends = torch.cumsum(counts, 0)
    n_faces = faces.shape[0]
    start = 0
    while start < n_faces:
        base = ends[start - 1].item() if start > 0 else 0
        stop = int(torch.searchsorted(ends, torch.tensor(base + _MAX_PAIRS, device=device), right=True).item())
        stop = max(stop, start + 1)  # a single huge triangle still gets its turn
        stop = min(stop, n_faces)
        _cover(start, stop, counts, x0, y0, w, a, b, c, area, size, face_id, bary)
        start = stop

    return face_id.view(size, size), bary.view(size, size, 3)


def _cover(start, stop, counts, x0, y0, w, a, b, c, area, size, face_id, bary) -> None:
    device = counts.device
    cnt = counts[start:stop]
    total = int(cnt.sum().item())
    if total == 0:
        return
    local = torch.repeat_interleave(torch.arange(stop - start, device=device), cnt)
    first = torch.cumsum(cnt, 0) - cnt
    offs = torch.arange(total, device=device) - first[local]
    f = local + start
    px = x0[f] + offs % w[f]
    py = y0[f] + offs // w[f]
    qx = px.float()
    qy = py.float()

    ax, ay = a[f, 0], a[f, 1]
    bx, by = b[f, 0], b[f, 1]
    cx, cy = c[f, 0], c[f, 1]
    inv = 1.0 / area[f]
    # Barycentric weights from signed sub-triangle areas.
    l0 = ((bx - qx) * (cy - qy) - (by - qy) * (cx - qx)) * inv
    l1 = ((cx - qx) * (ay - qy) - (cy - qy) * (ax - qx)) * inv
    l2 = 1.0 - l0 - l1
    eps = -1e-6
    inside = (l0 >= eps) & (l1 >= eps) & (l2 >= eps)

    idx = (py * size + px)[inside]
    face_id[idx] = f[inside]
    bary[idx] = torch.stack([l0, l1, l2], dim=-1)[inside].clamp(0.0, 1.0)


def to_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose: bool = False,
):
    """Same signature and result as `o_voxel.postprocess.to_glb`."""
    import cv2
    import cumesh
    import trimesh
    import trimesh.visual
    from flex_gemm.ops.grid_sample import grid_sample_3d
    from PIL import Image

    # --- AABB, voxel size, grid size (unchanged from upstream) ---
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)
    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size

    vertices = vertices.cuda()
    faces = faces.cuda()

    # --- Clean, then remesh or simplify (unchanged from upstream) ---
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    vertices, faces = mesh.read()
    bvh = cumesh.cuBVH(vertices, faces)

    if not remesh:
        mesh.simplify(decimation_target * 3, verbose=verbose)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.simplify(decimation_target, verbose=verbose)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.unify_face_orientations()
    else:
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        resolution = grid_size.max().item()
        mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
            vertices, faces,
            center=center,
            scale=(resolution + 3 * remesh_band) / resolution * scale,
            resolution=resolution,
            band=remesh_band,
            project_back=remesh_project,
            verbose=verbose,
            bvh=bvh,
        ))
        mesh.simplify(decimation_target, verbose=verbose)

    # --- UV unwrap (CuMesh / xatlas, unchanged from upstream) ---
    out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
            "refine_iterations": mesh_cluster_refine_iterations,
            "global_iterations": mesh_cluster_global_iterations,
            "smooth_strength": mesh_cluster_smooth_strength,
        },
        return_vmaps=True,
        verbose=verbose,
    )
    out_vertices = out_vertices.cuda()
    out_faces = out_faces.cuda()
    out_uvs = out_uvs.cuda()
    out_vmaps = out_vmaps.cuda()
    mesh.compute_vertex_normals()
    out_normals = mesh.read_vertex_normals()[out_vmaps]

    # --- Texture bake: the only change from upstream ---
    face_id, bary = rasterize_uv(out_uvs, out_faces, texture_size)
    mask = face_id >= 0
    fid = face_id[mask]
    w = bary[mask]
    tri = out_vertices[out_faces[fid].long()]  # (N, 3, 3)
    valid_pos = (tri * w.unsqueeze(-1)).sum(dim=1)

    # Project back onto the original high-res mesh, then sample the volume
    # (unchanged from upstream).
    _, hit_face, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    orig_tri_verts = vertices[faces[hit_face.long()]]
    valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device="cuda")
    attrs[mask] = grid_sample_3d(
        attr_volume,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
        grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
        mode="trilinear",
    )

    # --- Material (unchanged from upstream) ---
    mask = mask.cpu().numpy()
    base_color = np.clip(attrs[..., attr_layout["base_color"]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs[..., attr_layout["metallic"]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs[..., attr_layout["roughness"]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs[..., attr_layout["alpha"]].cpu().numpy() * 255, 0, 255).astype(np.uint8)

    mask_inv = (~mask).astype(np.uint8)
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(
            np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)
        ),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode="OPAQUE",
        doubleSided=not remesh,
    )

    vertices_np = out_vertices.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    uvs_np = out_uvs.cpu().numpy()
    normals_np = out_normals.cpu().numpy()
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
    uvs_np[:, 1] = 1 - uvs_np[:, 1]

    return trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material),
    )
