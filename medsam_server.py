#!/usr/bin/env python3
"""
MedSAM FastAPI server — bridge between CyberNeuro and MedSAM pipeline.
Run: python3 medsam_server.py
"""
import sys
import os
from fastapi import FastAPI
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yaml
import uvicorn

MEDSAM_PATH = "/Users/petersun/brain-network-chart"
sys.path.insert(0, MEDSAM_PATH)

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

@app.post("/segment")
async def segment(req: SegmentRequest):
    try:
        import nibabel as nib
        from app.agent.orchestrator import SegmentationOrchestrator
        from app.models.medsam_adapter import MedSAMAdapter

        # Load organ config
        config_path = os.path.join(MEDSAM_PATH, "config/organs.yaml")
        with open(config_path) as f:
            organs = yaml.safe_load(f)["organs"]

        # Match organ name from natural language
        organ_key = "custom"
        organ_lower = req.organ.lower().strip()
        for key, data in organs.items():
            aliases = data.get("aliases", [])
            if any(organ_lower in alias or alias in organ_lower for alias in aliases):
                organ_key = key
                break

        # Load scan to get dimensions
        scan = nib.load(req.scan_path)
        shape = scan.shape[:3]
        H, W, D = shape

        # Get organ priors
        organ_data = organs.get(organ_key, organs["custom"])
        axial_range = organ_data.get("typical_axial_range", [0.0, 1.0])
        from app.models.organ_detector import OrganDetector
        detector = OrganDetector(
            checkpoint_path="/Users/petersun/brain-network-chart/checkpoints/organ_bbox_unet.pth",
            device="cpu"
        )
        img_vol = scan.get_fdata().astype(np.float32)
        p1, p99 = np.percentile(img_vol, [1, 99])
        img_vol_norm = np.clip((img_vol - p1) / (p99 - p1 + 1e-8), 0, 1)

        bbox_frac = detector.predict_bbox(img_vol_norm, organ_key, axial_range)
        box = [
            int(bbox_frac[0] * W),
            int(bbox_frac[1] * H),
            int(bbox_frac[2] * W),
            int(bbox_frac[3] * H),
        ]
        print(f"DEBUG: UNet predicted bbox_frac={bbox_frac}, box={box}")

        # Prompt at 3 slices: 30%, 50%, 70% through the organ's axial range
        seed_slices = [
            int((axial_range[0] * 0.7 + axial_range[1] * 0.3) * D),
            int((axial_range[0] * 0.5 + axial_range[1] * 0.5) * D),
            int((axial_range[0] * 0.3 + axial_range[1] * 0.7) * D),
        ]
        print(f"DEBUG: D={D}, axial_range={axial_range}, seeds={seed_slices}, box={box}")

        # Expand bbox by 15% for better recall
        pad = 0.15
        w = bbox_frac[2] - bbox_frac[0]
        h = bbox_frac[3] - bbox_frac[1]
        bbox_frac = [
            max(0.0, bbox_frac[0] - w * pad),
            max(0.0, bbox_frac[1] - h * pad),
            min(1.0, bbox_frac[2] + w * pad),
            min(1.0, bbox_frac[3] + h * pad),
        ]
        box = [
            int(bbox_frac[0] * W),
            int(bbox_frac[1] * H),
            int(bbox_frac[2] * W),
            int(bbox_frac[3] * H),
        ]
        print(f"DEBUG: Padded box={box}")

        # Run MedSAM pipeline
        adapter = MedSAMAdapter(device="cpu")
        adapter.load_model(
            checkpoint="/Users/petersun/brain-network-chart/checkpoints/medsam_vit_b.pth",
            segment_anything_path="/Users/petersun/brain-network-chart"
        )
        orch = SegmentationOrchestrator(
            adapter=adapter,
            output_dir=os.path.join(MEDSAM_PATH, "output")
        )
        orch.new_session(req.scan_path, organ=organ_key, axis=2)

        for slice_idx in seed_slices:
            orch.add_box_prompt(slice_idx=slice_idx, box=box)
            orch.segment_slice(slice_idx)

        # Limit propagation to organ's known axial range
        z_min = int(axial_range[0] * D)
        z_max = int(axial_range[1] * D)
        print(f"DEBUG: z_min={z_min}, z_max={z_max}")
        orch.propagate_to_volume(slice_range=(z_min, z_max))

        outputs = orch.save_outputs()
        voxels = int(orch.mask_volume.sum()) if orch.mask_volume is not None else 0

        return {
            "status": "success",
            "organ": req.organ,
            "organ_key": organ_key,
            "mask_path": str(outputs.get("mask", "")),
            "overlay_path": str(outputs.get("overlay_grid", "")),
            "voxel_count": voxels,
            "seeds_used": seed_slices,
            "box_used": box,
            "scan_shape": list(shape)
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "organ": req.organ,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    print("Starting MedSAM server on port 8099...")
    uvicorn.run(app, host="0.0.0.0", port=8099)
