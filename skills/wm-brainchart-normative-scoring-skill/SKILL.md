---
name: medsam-segmentation-skill
version: 1.0.0
author: ACMLab
homepage: https://github.com/acmlab/brain-network-chart
description: Use when the user wants to segment an organ or tumor in a medical scan using MedSAM. Triggers on requests mentioning "segment", "segmentation", "outline", "delineate", "organ", "kidney", "liver", "heart", "spleen", "tumor", or any NIfTI file path (.nii.gz). Automatically calculates slice location and bounding box from anatomical priors — no manual coordinate input required.
---

# MedSAM Organ Segmentation Skill

Segments organs and tumors in medical CT and MRI scans using MedSAM (Medical Segment Anything Model). Given a NIfTI scan file and an organ name in plain English, this skill automatically determines the correct slice location and bounding box coordinates, runs MedSAM inference, propagates the segmentation across the full 3D volume, and saves the output mask and overlay image.

This skill:

- Accepts plain English organ names (right kidney, left kidney, liver, heart, spleen, myocardium)
- Automatically calculates slice index from anatomical priors — no manual slice number needed
- Automatically calculates bounding box coordinates from organ location priors — no manual coordinate input needed
- Runs MedSAM segmentation on the target slice
- Propagates segmentation across the full 3D volume
- Saves pred_mask.nii.gz and overlay_grid.png to the output directory

## Dataset Inspection Requirements

Before attempting segmentation, always inspect the scan file.

Accepted input formats:
- NIfTI files (.nii.gz, .nii)
- Must be a 3D or 4D volume

If the file does not exist or is not a valid NIfTI file, stop and report the issue clearly.

If no organ name is provided, ask the user which organ to segment before proceeding.

Supported organs:
- right_kidney
- left_kidney
- heart
- myocardium
- liver
- spleen
- custom (fallback for unsupported organs — uses full image as search region)

## Workflow Overview

Follow these steps in order. Each step must be completed before the next.

**Step 1 — Verify environment**
Check that Python 3.8+ is available and required packages are installed (nibabel, fastapi, uvicorn, pyyaml).

```bash
python3 --version
python3 -c "import nibabel, fastapi, uvicorn, yaml; print('OK')"
```

If packages are missing, install them:
```bash
pip install nibabel fastapi uvicorn pyyaml
```

**Step 2 — Locate the MedSAM pipeline**

The MedSAM pipeline lives in the medsam branch of the brain-network-chart repo. Check it exists:

```bash
ls /Users/petersun/brain-network-chart/app/agent/orchestrator.py
ls /Users/petersun/brain-network-chart/config/organs.yaml
```

If the path is different on the current machine, ask the user for the correct path to the brain-network-chart medsam branch.

**Step 3 — Parse organ name from user input**

Match the user's natural language organ name to a key in organs.yaml:

```python
import yaml

with open('/Users/petersun/brain-network-chart/config/organs.yaml') as f:
    organs = yaml.safe_load(f)['organs']

organ_lower = user_input.lower().strip()
organ_key = 'custom'
for key, data in organs.items():
    aliases = data.get('aliases', [])
    if any(organ_lower in alias or alias in organ_lower for alias in aliases):
        organ_key = key
        break

print(f"Matched organ: {organ_key}")
```

**Step 4 — Load scan and calculate parameters automatically**

```python
import nibabel as nib
import numpy as np

scan = nib.load(scan_path)
shape = scan.shape[:3]
H, W, D = shape

organ_data = organs[organ_key]
axial_range = organ_data.get('typical_axial_range', [0.0, 1.0])
slice_idx = int(((axial_range[0] + axial_range[1]) / 2) * D)

bbox_frac = organ_data.get('typical_bbox_fraction', [0.1, 0.1, 0.9, 0.9])
box = [
    int(bbox_frac[0] * W),
    int(bbox_frac[1] * H),
    int(bbox_frac[2] * W),
    int(bbox_frac[3] * H),
]

print(f"Scan shape: {shape}")
print(f"Slice: {slice_idx}")
print(f"Bounding box: {box}")
```

**Step 5 — Run MedSAM segmentation**

```python
import sys
sys.path.insert(0, '/Users/petersun/brain-network-chart')

from app.agent.orchestrator import SegmentationOrchestrator
from app.models.mock_adapter import MockAdapter  # swap for real adapter when available

adapter = MockAdapter()
orch = SegmentationOrchestrator(
    adapter=adapter,
    output_dir=output_dir
)

orch.new_session(scan_path, organ=organ_key, axis=0)
orch.add_box_prompt(slice_idx=slice_idx, box=box)
orch.segment_slice(slice_idx)
orch.propagate_to_volume()
outputs = orch.save_outputs()

voxels = int(orch.mask_volume.sum()) if orch.mask_volume is not None else 0
print(f"Voxels segmented: {voxels}")
print(f"Mask saved to: {outputs.get('mask', '')}")
print(f"Overlay saved to: {outputs.get('overlay_grid', '')}")
```

**Step 6 — Report results**

Report the following to the user:
- Organ segmented
- Slice used
- Bounding box used
- Voxel count
- Path to mask file (pred_mask.nii.gz)
- Path to overlay image (overlay_grid.png)

If voxel count is 0 and using MockAdapter, note that this is expected behavior with the placeholder model. Real voxel counts require the real MedSAM checkpoint.

## Privacy and PHI

WARNING: Medical scan files may contain Protected Health Information (PHI) in the NIfTI header or filename. This skill does NOT de-identify scan data.

Before running segmentation on patient data:
- Confirm the scan has been de-identified
- Do not save outputs to shared or public directories
- Follow your institution's IRB and HIPAA guidelines

If the user confirms the data is de-identified, proceed without further checks.

## Execution Rules

- All segmentation uses the MedSAM pipeline only
- Do NOT call any external API or cloud service — runs 100% locally for HIPAA compliance
- Do NOT modify the original scan file — only write to the output directory
- If the output directory does not exist, create it before running

## Output Rules

All outputs MUST be written to the user-provided output directory.

If the user has not provided an output directory, use:
```
/Users/petersun/brain-network-chart/output/<scan_name>/
```

Outputs produced:
- `pred_mask.nii.gz` — 3D binary mask of the segmented organ
- `overlay_grid.png` — visual grid showing 8 slices with segmentation overlay in gold
- `prompts.json` — record of slice and bounding box used
- `session_log.json` — full session log

## Organ Anatomical Priors

The skill uses pre-defined anatomical priors for automatic parameter calculation:

| Organ | Axial Range | Bbox Fraction [x1, y1, x2, y2] |
|---|---|---|
| right_kidney | [0.3, 0.7] | [0.55, 0.30, 0.88, 0.70] |
| left_kidney | [0.3, 0.7] | [0.12, 0.30, 0.45, 0.70] |
| heart | [0.2, 0.6] | [0.30, 0.20, 0.70, 0.65] |
| myocardium | [0.2, 0.6] | [0.35, 0.25, 0.65, 0.60] |
| liver | [0.25, 0.6] | [0.40, 0.20, 0.90, 0.65] |
| spleen | [0.3, 0.6] | [0.10, 0.30, 0.40, 0.65] |
| custom | [0.0, 1.0] | [0.10, 0.10, 0.90, 0.90] |

These are approximate values based on typical human anatomy in standard axial CT/MRI scans. Accuracy can be improved by calibrating against real scan data.

## Adapter Notes

**MockAdapter (current):** Draws approximate ellipses for testing the pipeline. Produces 0 or near-0 voxel counts on synthetic data. Use for pipeline validation only.

**Real MedSAM adapter:** Requires the MedSAM checkpoint file on the lab's frontier GPU server. Contact Ziquan Wei for checkpoint location. Swap MockAdapter for MedSAMAdapter in Step 5 when the checkpoint is available.

## Requirements

| Tool | Install |
|---|---|
| Python ≥ 3.8 | — |
| nibabel | `pip install nibabel` |
| fastapi | `pip install fastapi` |
| uvicorn | `pip install uvicorn` |
| pyyaml | `pip install pyyaml` |
| MedSAM pipeline | medsam branch of brain-network-chart repo |

## References

- MedSAM: https://github.com/bowang-lab/MedSAM
- brain-network-chart medsam branch: https://github.com/acmlab/brain-network-chart
- Peter Sun's implementation: https://github.com/peterysun/cyberneuro-medsam
