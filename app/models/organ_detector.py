"""
UNet-based organ bounding box detector.
Replaces hardcoded organs.yaml bbox fractions in medsam_server.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
from pathlib import Path


ORGAN_TO_IDX = {
    'liver': 0,
    'right_kidney': 1,
    'spleen': 2,
    'left_kidney': 3,
}

IDX_TO_ORGAN = {v: k for k, v in ORGAN_TO_IDX.items()}


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class OrganBBoxUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.organ_embed = nn.Linear(4, 64)
        self.head = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def forward(self, img, organ_vec):
        x = self.enc1(img)
        x = self.pool(x)
        x = self.enc2(x)
        x = self.pool(x)
        x = self.enc3(x)
        x = self.pool(x)
        x = self.enc4(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        organ_feat = F.relu(self.organ_embed(organ_vec))
        combined = torch.cat([x, organ_feat], dim=1)
        return self.head(combined)


class OrganDetector:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        self.model = OrganBBoxUNet().to(device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
        self.model.eval()

    def predict_bbox(self, volume, organ, axial_range, img_size=256):
        if organ not in ORGAN_TO_IDX:
            return [0.1, 0.1, 0.9, 0.9]

        H, W, D = volume.shape
        z = int((axial_range[0] * 0.5 + axial_range[1] * 0.5) * D)
        z = max(0, min(z, D - 1))
        img_slice = volume[:, :, z].astype(np.float32)

        p1, p99 = np.percentile(img_slice, [1, 99])
        img_slice = np.clip((img_slice - p1) / (p99 - p1 + 1e-8), 0, 1)

        img_resized = zoom(img_slice, (img_size / H, img_size / W), order=1)

        img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        organ_vec = torch.zeros(1, 4, dtype=torch.float32).to(self.device)
        organ_vec[0, ORGAN_TO_IDX[organ]] = 1.0

        with torch.no_grad():
            bbox = self.model(img_tensor, organ_vec).squeeze(0).cpu().numpy()

        return bbox.tolist()
