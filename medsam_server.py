#!/usr/bin/env python3
"""
MedSAM FastAPI server — bridge between CyberNeuro and MedSAM pipeline.
Run: python3 medsam_server.py
"""

import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yaml
import uvicorn

# Point to your brain-network-chart medsam branch
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
        from app.models.mock_adapter import MockAdapter

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
        slice_idx = int(((axial_range[0] + axial_range[1]) / 2) * D)

        bbox_frac = organ_data.get("typical_bbox_fraction", [0.1, 0.1, 0.9, 0.9])
        box = [
            int(bbox_frac[0] * W),
            int(bbox_frac[1] * H),
            int(bbox_frac[2] * W),
            int(bbox_frac[3] * H),
        ]

        # Run MedSAM pipeline
        adapter = MockAdapter()
        orch = SegmentationOrchestrator(
            adapter=adapter,
            output_dir=os.path.join(MEDSAM_PATH, "output")
        )
        orch.new_session(req.scan_path, organ=organ_key, axis=0)
        orch.add_box_prompt(slice_idx=slice_idx, box=box)
        orch.segment_slice(slice_idx)
        orch.propagate_to_volume()
        outputs = orch.save_outputs()

        voxels = int(orch.mask_volume.sum()) if orch.mask_volume is not None else 0

        return {
            "status": "success",
            "organ": req.organ,
            "organ_key": organ_key,
            "mask_path": str(outputs.get("mask", "")),
            "overlay_path": str(outputs.get("overlay_grid", "")),
            "voxel_count": voxels,
            "slice_used": slice_idx,
            "box_used": box,
            "scan_shape": list(shape)
        }

    except Exception as e:
        return {
            "status": "error",
            "organ": req.organ,
            "error": str(e)
        }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    print("Starting MedSAM server on port 8099...")
    uvicorn.run(app, host="0.0.0.0", port=8099)
