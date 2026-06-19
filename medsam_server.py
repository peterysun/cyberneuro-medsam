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
    adapter = _get_adapter("mri")
    result = adapter.predict_preprocessed_1024(img_1024, box_1024, slice_idx=0)

    rotated = np.rot90(result.mask, k=3, axes=(0, 1)).astype(np.uint8)
    mask_native = resize(
        rotated,
        raw_slice.shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.uint8)
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
                        "output_rotation": "np.rot90(mask, k=3, axes=(0,1))",
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
        "box_space": "1024",
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
