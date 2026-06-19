#!/usr/bin/env python3
"""FastAPI bridge for the MedSAM organ segmentation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MEDSAM_PATH = Path("/Users/petersun/brain-network-chart")
sys.path.insert(0, str(MEDSAM_PATH))

from app.agent.orchestrator import SegmentationOrchestrator
from app.io.image_writer import save_mask_nifti, save_overlay_grid, save_prompts_json
from app.models.medsam_adapter import MedSAMAdapter
from app.models.organ_detector import OrganDetector


app = FastAPI(title="MedSAM Segmentation Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SegmentRequest(BaseModel):
    organ: str
    scan_path: str
    modality: Optional[str] = "auto"


ORGANS = yaml.safe_load((MEDSAM_PATH / "config/organs.yaml").read_text())["organs"]
OUTPUT_DIR = MEDSAM_PATH / "output"
CHECKPOINTS = {
    "ct_medsam": MEDSAM_PATH / "checkpoints/medsam_vit_b.pth",
    "mri_medsam": MEDSAM_PATH / "checkpoints/medsam_kidney_mri_weights.pth",
    "ct_detector": MEDSAM_PATH / "checkpoints/organ_bbox_unet.pth",
    "mri_detector": MEDSAM_PATH / "checkpoints/organ_bbox_unet_mri.pth",
}

_ADAPTERS: Dict[str, MedSAMAdapter] = {}
_DETECTORS: Dict[str, OrganDetector] = {}

SHMOLLI_BOXES_1024 = {
    "right_kidney": [580, 260, 860, 700],
    "left_kidney": [160, 260, 420, 700],
}
SHMOLLI_BOX_ENSEMBLES_1024 = {
    "right_kidney": [
        [580, 260, 860, 700],
        [600, 285, 845, 680],
        [615, 305, 830, 660],
    ],
    "left_kidney": [
        [160, 260, 420, 700],
        [180, 285, 400, 680],
        [195, 305, 385, 660],
    ],
}


@app.on_event("startup")
async def startup() -> None:
    _ensure_mri_weights()
    _get_adapter("ct")
    _get_adapter("mri")


@app.post("/segment")
async def segment(req: SegmentRequest) -> Dict[str, Any]:
    try:
        scan = nib.load(req.scan_path)
        volume = scan.get_fdata(dtype=np.float32)
        if volume.ndim == 4:
            volume = volume[..., volume.shape[3] // 2]
        if volume.ndim == 2:
            volume = volume[:, :, np.newaxis]

        organ_key = _match_organ(req.organ)
        modality = _detect_modality(req.modality, scan, volume)
        is_2d = volume.shape[2] == 1

        if is_2d and modality == "mri" and organ_key in SHMOLLI_BOXES_1024:
            return _segment_shmolli_mri(req, scan, volume, organ_key)
        return _segment_volume(req, volume, organ_key, modality)
    except Exception as exc:
        import traceback

        return {
            "status": "error",
            "organ": req.organ,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


def _segment_shmolli_mri(
    req: SegmentRequest,
    scan: nib.Nifti1Image,
    volume: np.ndarray,
    organ_key: str,
) -> Dict[str, Any]:
    """Mirror Monica's 04_infer.py preprocessing for 2D ShMoLLI kidneys."""
    from skimage.transform import resize

    raw_slice = volume[:, :, 0].astype(np.float32)
    norm_slice = _normalize_mr_nonzero(raw_slice)
    img_1024 = resize(
        norm_slice,
        (1024, 1024),
        order=3,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)
    img_1024 = _minmax(img_1024)

    box_1024 = SHMOLLI_BOXES_1024[organ_key]
    ensemble_boxes = SHMOLLI_BOX_ENSEMBLES_1024[organ_key]
    adapter = _get_adapter("mri")
    masks_1024 = [
        adapter.predict_preprocessed_1024(img_1024, box, slice_idx=0).mask
        for box in ensemble_boxes
    ]
    mask_1024 = (np.mean(np.stack(masks_1024, axis=0), axis=0) >= 0.5).astype(np.uint8)

    rotated = np.rot90(mask_1024, k=3, axes=(0, 1)).astype(np.uint8)
    mask_native = resize(
        rotated,
        raw_slice.shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.uint8)
    mask_native, postprocess_meta = _postprocess_shmolli_mask(
        mask_native, raw_slice.shape, box_1024
    )
    mask_volume = mask_native[:, :, np.newaxis]

    case_dir = OUTPUT_DIR / Path(req.scan_path).stem.replace(".nii", "")
    case_dir.mkdir(parents=True, exist_ok=True)
    mask_path = case_dir / "pred_mask.nii.gz"
    overlay_path = case_dir / "overlay_grid.png"
    prompts_path = case_dir / "prompts.json"
    save_mask_nifti(mask_volume, mask_path, scan.affine, scan.header)
    save_overlay_grid(
        _normalize_mr_nonzero(volume).astype(np.float32),
        mask_volume,
        overlay_path,
        axis=2,
        num_slices=1,
    )
    save_prompts_json(
        {
            "organ": organ_key,
            "axis": 2,
            "prompts": [
                {
                    "slice_idx": 0,
                    "axis": 2,
                    "prompt_type": "box",
                    "box": box_1024,
                    "meta": {
                        "source": "monica_shmolli_04_infer",
                        "box_space": "1024_after_resize",
                        "ensemble_boxes": ensemble_boxes,
                        "output_rotation": "np.rot90(mask, k=3, axes=(0,1))",
                        "postprocess": postprocess_meta,
                    },
                }
            ],
        },
        prompts_path,
    )

    return {
        "status": "success",
        "organ": req.organ,
        "organ_key": organ_key,
        "modality": "mri",
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "voxel_count": int(mask_volume.sum()),
        "seeds_used": [0],
        "box_used": box_1024,
        "ensemble_boxes": ensemble_boxes,
        "box_space": "1024",
        "postprocess": postprocess_meta,
        "scan_shape": list(volume.shape),
        "mode": "2D ShMoLLI Monica preprocessing",
    }


def _segment_volume(
    req: SegmentRequest,
    volume: np.ndarray,
    organ_key: str,
    modality: str,
) -> Dict[str, Any]:
    organ_data = ORGANS.get(organ_key, ORGANS["custom"])
    axial_range = organ_data.get("typical_axial_range", [0.0, 1.0])
    volume_norm = _normalize_mr_nonzero(volume) if modality == "mri" else _normalize_ct(volume)
    H, W, D = volume.shape[:3]

    detector = _get_detector(modality)
    bbox_frac = detector.predict_bbox(volume_norm, organ_key, axial_range)
    bbox_frac = _pad_bbox_frac(bbox_frac, pad=0.15)
    box = [
        int(bbox_frac[0] * W),
        int(bbox_frac[1] * H),
        int(bbox_frac[2] * W),
        int(bbox_frac[3] * H),
    ]
    seed_slices = _seed_slices(D, axial_range)

    orch = SegmentationOrchestrator(
        adapter=_get_adapter(modality),
        output_dir=OUTPUT_DIR,
        propagation_strategy="sam_guided",
    )
    norm_kwargs = {"low": 0.5, "high": 99.5} if modality == "mri" else {"low": 1, "high": 99}
    orch.new_session(req.scan_path, organ=organ_key, axis=2, norm_kwargs=norm_kwargs)

    for slice_idx in seed_slices:
        orch.add_box_prompt(slice_idx=slice_idx, box=box)
        orch.segment_slice(slice_idx)

    z_min = int(axial_range[0] * D)
    z_max = int(axial_range[1] * D)
    if D > 1:
        orch.propagate_to_volume(slice_range=(z_min, z_max))
        correction = orch.self_correct_sparse_slices(slice_range=(z_min, z_max))
    else:
        correction = {"corrected": [], "median_area": 0, "threshold_area": 0}

    outputs = orch.save_outputs()
    return {
        "status": "success",
        "organ": req.organ,
        "organ_key": organ_key,
        "modality": modality,
        "mask_path": str(outputs.get("mask", "")),
        "overlay_path": str(outputs.get("overlay_grid", "")),
        "voxel_count": int(orch.mask_volume.sum()) if orch.mask_volume is not None else 0,
        "seeds_used": seed_slices,
        "box_used": box,
        "scan_shape": list(volume.shape),
        "mode": "3D" if D > 1 else "2D",
        "self_correction": correction,
    }


def _get_adapter(modality: str) -> MedSAMAdapter:
    key = "mri" if modality == "mri" else "ct"
    if key not in _ADAPTERS:
        adapter = MedSAMAdapter(device="cpu")
        adapter.load_model(
            checkpoint=str(CHECKPOINTS[f"{key}_medsam"]),
            segment_anything_path=str(MEDSAM_PATH),
        )
        _ADAPTERS[key] = adapter
    return _ADAPTERS[key]


def _get_detector(modality: str) -> OrganDetector:
    key = "mri" if modality == "mri" else "ct"
    if key not in _DETECTORS:
        _DETECTORS[key] = OrganDetector(
            checkpoint_path=str(CHECKPOINTS[f"{key}_detector"]),
            modality=key,
            device="cpu",
        )
    return _DETECTORS[key]


def _match_organ(organ: str) -> str:
    organ_lower = organ.lower().strip()
    for key, data in ORGANS.items():
        aliases = data.get("aliases", [])
        if any(organ_lower in alias or alias in organ_lower for alias in aliases):
            return key
    return "custom"


def _detect_modality(requested: Optional[str], scan: nib.Nifti1Image, volume: np.ndarray) -> str:
    modality = requested.lower() if requested else "auto"
    if modality in {"ct", "mri"}:
        return modality

    descrip = str(scan.header.get("descrip", b"")).lower()
    if any(token in descrip for token in ("mri", "mr", "t1", "t2", "shmolli")):
        return "mri"
    if volume.shape[0] <= 320 and volume.shape[1] <= 320:
        return "mri"
    return "ct"


def _normalize_mr_nonzero(volume: np.ndarray) -> np.ndarray:
    nonzero = volume[volume > 0]
    if nonzero.size:
        lo, hi = np.percentile(nonzero, [0.5, 99.5])
    else:
        lo, hi = float(volume.min()), float(volume.max())
    return np.clip((volume - lo) / max(float(hi - lo), 1e-8), 0, 1).astype(np.float32)


def _normalize_ct(volume: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(volume, [1, 99])
    return np.clip((volume - p1) / max(float(p99 - p1), 1e-8), 0, 1).astype(np.float32)


def _minmax(image: np.ndarray) -> np.ndarray:
    lo = float(image.min())
    hi = float(image.max())
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - lo) / (hi - lo)).astype(np.float32)


def _postprocess_shmolli_mask(
    mask: np.ndarray,
    native_shape: tuple[int, int],
    box_1024: List[int],
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Clean ShMoLLI kidney masks using the rotated native bbox as a prior."""
    from scipy import ndimage as ndi

    H, W = native_shape
    x1, y1, x2, y2 = _rotated_box_to_native(box_1024, native_shape)
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1

    # Monica's boxes are intentionally generous. After rotation, trim them to
    # the kidney-bearing center and prevent inferior spill into adjacent tissue.
    trim_x = int(0.18 * bw)
    trim_y_top = int(0.08 * bh)
    trim_y_bottom = int(0.26 * bh)
    tx1 = max(0, x1 + trim_x)
    tx2 = min(W - 1, x2 - trim_x)
    ty1 = max(0, y1 + trim_y_top)
    ty2 = min(H - 1, y2 - trim_y_bottom)

    prior = np.zeros((H, W), dtype=bool)
    prior[ty1 : ty2 + 1, tx1 : tx2 + 1] = True

    yy, xx = np.ogrid[:H, :W]
    cx = (tx1 + tx2) / 2.0
    cy = (ty1 + ty2) / 2.0
    rx = max(1.0, (tx2 - tx1 + 1) * 0.56)
    ry = max(1.0, (ty2 - ty1 + 1) * 0.64)
    oval = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0

    cleaned = (mask > 0) & prior & oval
    cleaned = ndi.binary_opening(cleaned, structure=np.ones((3, 3), dtype=bool))
    cleaned = ndi.binary_closing(cleaned, structure=np.ones((5, 5), dtype=bool))
    cleaned = ndi.binary_fill_holes(cleaned)
    cleaned = _largest_component(cleaned)

    # If the model output was fragmented after prior clipping, fall back to the
    # conservative oval-intersection before returning an empty mask.
    if not cleaned.any():
        cleaned = _largest_component((mask > 0) & prior)

    cleaned = _ellipse_regularized_mask(cleaned, prior & oval)
    cleaned = ndi.binary_opening(cleaned, structure=np.ones((3, 3), dtype=bool))
    cleaned = ndi.binary_closing(cleaned, structure=np.ones((5, 5), dtype=bool))
    cleaned = ndi.binary_fill_holes(cleaned)
    return cleaned.astype(np.uint8), {
        "native_box_after_rotation": [x1, y1, x2, y2],
        "tight_native_box": [tx1, ty1, tx2, ty2],
        "strategy": "largest_component_covariance_ellipse_prior",
    }


def _largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    labels, n_labels = ndi.label(mask)
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(counts.argmax())


def _ellipse_regularized_mask(component: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Return a smooth ellipse fitted to a component, clipped by allowed prior."""
    ys, xs = np.nonzero(component)
    if len(xs) < 20:
        return component

    coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    center = coords.mean(axis=0)
    cov = np.cov(coords, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    centered = coords - center
    projected = centered @ vecs
    radii = np.array(
        [
            np.percentile(np.abs(projected[:, 0]), 94),
            np.percentile(np.abs(projected[:, 1]), 93),
        ],
        dtype=np.float64,
    )
    radii = np.maximum(radii * np.array([1.38, 1.45]), [8.0, 8.0])

    yy, xx = np.indices(component.shape)
    grid = np.stack([xx - center[0], yy - center[1]], axis=-1)
    proj0 = grid[..., 0] * vecs[0, 0] + grid[..., 1] * vecs[1, 0]
    proj1 = grid[..., 0] * vecs[0, 1] + grid[..., 1] * vecs[1, 1]
    ellipse = (proj0 / radii[0]) ** 2 + (proj1 / radii[1]) ** 2 <= 1.0

    # Keep the fitted shape anchored to where MedSAM was confident; this avoids
    # growing into the adjacent bright structures while removing jagged edges.
    return ellipse & allowed


def _rotated_box_to_native(
    box_1024: List[int],
    native_shape: tuple[int, int],
) -> List[int]:
    H, W = native_shape
    x1, y1, x2, y2 = [float(v) for v in box_1024]
    rotated_x1 = 1023.0 - y2
    rotated_x2 = 1023.0 - y1
    rotated_y1 = x1
    rotated_y2 = x2
    return [
        max(0, min(W - 1, int(round(rotated_x1 / 1024.0 * W)))),
        max(0, min(H - 1, int(round(rotated_y1 / 1024.0 * H)))),
        max(0, min(W - 1, int(round(rotated_x2 / 1024.0 * W)))),
        max(0, min(H - 1, int(round(rotated_y2 / 1024.0 * H)))),
    ]


def _pad_bbox_frac(bbox: List[float], pad: float) -> List[float]:
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return [
        max(0.0, x1 - w * pad),
        max(0.0, y1 - h * pad),
        min(1.0, x2 + w * pad),
        min(1.0, y2 + h * pad),
    ]


def _seed_slices(depth: int, axial_range: List[float]) -> List[int]:
    if depth <= 1:
        return [0]
    seeds = [
        int((axial_range[0] * 0.7 + axial_range[1] * 0.3) * depth),
        int((axial_range[0] * 0.5 + axial_range[1] * 0.5) * depth),
        int((axial_range[0] * 0.3 + axial_range[1] * 0.7) * depth),
    ]
    return sorted({max(0, min(depth - 1, seed)) for seed in seeds})


def _ensure_mri_weights() -> None:
    import torch

    raw_path = MEDSAM_PATH / "checkpoints/medsam_kidney_mri.pth"
    weights_path = CHECKPOINTS["mri_medsam"]
    if raw_path.exists() and not weights_path.exists():
        raw = torch.load(raw_path, map_location="cpu")
        weights = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        torch.save(weights, weights_path)


if __name__ == "__main__":
    _ensure_mri_weights()
    print("Starting MedSAM server on port 8099...")
    uvicorn.run(app, host="0.0.0.0", port=8099)
