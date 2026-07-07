from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import functional as tvf


ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
FOLD_OUTPUT_DIR = OUTPUT_DIR / "folds"

SOURCE_OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "002_image_contour_landmark_guided_fusion_classification"
    / "002002_image_contour_prediction"
    / "002002002_image_contour_auxiliary_supervision"
    / "outputs"
)

MODEL_ROOT = ROOT / "Models_saved"
PRETRAINED_CACHE_DIR = MODEL_ROOT / "Downloaded_Pretrained_Models"
BASELINE_CHECKPOINT_DIR = MODEL_ROOT / "Generated_Models" / "002002002_contour_auxiliary_resnet18"
CHECKPOINT_ROOT = MODEL_ROOT / "Generated_Models" / "002002003010003_final_module_ablation"
CHECKPOINT_DIR = CHECKPOINT_ROOT / "full"

EXPERIMENT_ID = "002002003010003"
EXPERIMENT_LABEL = "Final module ablation for baseline-preserving fractal-fractional residual F2MI"

VARIANT_DESCRIPTIONS = {
    "full": "Full 010 architecture control rerun",
    "no_fractal": "Ablation without fractal box-counting/multifractal contour tokens",
    "no_fractional": "Ablation without fractional contour operator",
    "no_residual": "Ablation without baseline-preserving residual fusion",
}

TARGET_NAME = "contour_targets_224_uint8.npz"
REQUIRED_INPUTS = [
    "metadata.csv",
    "folds.csv",
    "outer_splits.csv",
    TARGET_NAME,
    "contour_target_manifest.json",
]

TARGET_PATH = OUTPUT_DIR / TARGET_NAME
METADATA_PATH = OUTPUT_DIR / "metadata.csv"
FOLDS_PATH = OUTPUT_DIR / "folds.csv"
OUTER_SPLITS_PATH = OUTPUT_DIR / "outer_splits.csv"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CONTOUR_CHANNELS = 2
CONTOUR_POINTS = 128
LANDMARK_POINTS = 4
MORPHOLOGY_DIM = 10


@dataclass(frozen=True)
class RunConfig:
    experiment_id: str
    experiment_label: str
    variant: str
    use_fractal_token: bool
    use_fractional_operator: bool
    use_residual_fusion: bool
    seed: int
    n_splits: int
    image_size: int
    batch_size: int
    epochs: int
    patience: int
    monitor: str
    lr: float
    weight_decay: float
    d_model: int
    n_heads: int
    fusion_hidden_dim: int
    dropout: float
    alpha_min: float
    alpha_max: float
    lambda_mask: float
    lambda_points: float
    lambda_landmark: float
    lambda_morphology: float
    lambda_fractal: float
    lambda_baseline_cls: float
    lambda_residual_l2: float
    lambda_router_div: float
    lambda_router_entropy: float
    contour_pos_weight: float
    fractal_gamma: float
    router_temperature: float
    router_div_target: float
    residual_scale: float
    load_baseline_checkpoint: bool
    baseline_checkpoint_dir: str
    pretrained: bool
    amp: bool
    use_pos_weight: bool
    num_workers: int
    device: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_model_storage() -> None:
    PRETRAINED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(PRETRAINED_CACHE_DIR)
    torch.hub.set_dir(str(PRETRAINED_CACHE_DIR))


def configure_variant_paths(variant: str) -> None:
    global OUTPUT_DIR, FOLD_OUTPUT_DIR, CHECKPOINT_DIR
    global TARGET_PATH, METADATA_PATH, FOLDS_PATH, OUTER_SPLITS_PATH

    OUTPUT_DIR = EXPERIMENT_DIR / "outputs" / variant
    FOLD_OUTPUT_DIR = OUTPUT_DIR / "folds"
    CHECKPOINT_DIR = CHECKPOINT_ROOT / variant
    TARGET_PATH = OUTPUT_DIR / TARGET_NAME
    METADATA_PATH = OUTPUT_DIR / "metadata.csv"
    FOLDS_PATH = OUTPUT_DIR / "folds.csv"
    OUTER_SPLITS_PATH = OUTPUT_DIR / "outer_splits.csv"


def variant_flags(variant: str) -> dict[str, bool]:
    if variant not in VARIANT_DESCRIPTIONS:
        raise ValueError(f"Unknown variant: {variant}")
    return {
        "use_fractal_token": variant != "no_fractal",
        "use_fractional_operator": variant != "no_fractional",
        "use_residual_fusion": variant != "no_residual",
    }


def prepare_local_inputs() -> dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in REQUIRED_INPUTS:
        src = SOURCE_OUTPUT_DIR / name
        dst = OUTPUT_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"Missing required source file from 002002002: {src}")
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)
            copied[name] = "copied"
        else:
            copied[name] = "exists"
    return copied


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, np.ndarray]]:
    prepare_local_inputs()
    metadata = pd.read_csv(METADATA_PATH, encoding="utf-8-sig")
    folds = pd.read_csv(FOLDS_PATH, encoding="utf-8-sig")
    outer_splits = pd.read_csv(OUTER_SPLITS_PATH, encoding="utf-8-sig")
    target_data = np.load(TARGET_PATH, allow_pickle=True)
    masks = target_data["masks"]
    row_ids = target_data["row_ids"].astype(int)
    masks_by_row_id = {int(row_id): masks[index] for index, row_id in enumerate(row_ids)}
    return metadata, folds, outer_splits, masks_by_row_id


def split_frames(folds: pd.DataFrame, outer_splits: pd.DataFrame, fold: int):
    split_table = outer_splits[outer_splits["outer_fold"] == fold]
    frames = {}
    for split_name in ["train", "validation", "test"]:
        row_ids = split_table.loc[split_table["split"] == split_name, "row_id"].astype(int)
        frame = folds[folds["row_id"].astype(int).isin(set(row_ids))].copy()
        frames[split_name] = frame.reset_index(drop=True)
    return frames["train"], frames["validation"], frames["test"]


def resample_polyline(points: np.ndarray, num_points: int = CONTOUR_POINTS) -> np.ndarray:
    if points.ndim != 2 or points.shape[0] == 0:
        return np.zeros((num_points, 2), dtype=np.float32)
    if points.shape[0] == 1:
        return np.repeat(points.astype(np.float32), num_points, axis=0)
    deltas = np.diff(points, axis=0)
    lengths = np.sqrt((deltas**2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    if total <= 1e-6:
        return np.repeat(points[:1].astype(np.float32), num_points, axis=0)
    target = np.linspace(0.0, total, num_points)
    x = np.interp(target, cumulative, points[:, 0])
    y = np.interp(target, cumulative, points[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def polyline_length(points: np.ndarray) -> float:
    if points.ndim != 2 or points.shape[0] < 2:
        return 0.0
    return float(np.sqrt((np.diff(points, axis=0) ** 2).sum(axis=1)).sum())


def point_line_distance(point: np.ndarray, line: np.ndarray) -> float:
    if line.shape[0] < 2:
        return 0.0
    a, b = line[0], line[1]
    denom = np.linalg.norm(b - a)
    if denom <= 1e-6:
        return float(np.linalg.norm(point - a))
    return float(abs(np.cross(b - a, a - point)) / denom)


def normalize_points(points: np.ndarray, width: float, height: float) -> np.ndarray:
    out = points.astype(np.float32).copy()
    out[:, 0] = out[:, 0] / max(width, 1.0) * 2.0 - 1.0
    out[:, 1] = out[:, 1] / max(height, 1.0) * 2.0 - 1.0
    return out


def parse_annotation(label_path: str | Path, a_n: float) -> dict[str, np.ndarray]:
    path = Path(label_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    width = float(data.get("imageWidth", 512))
    height = float(data.get("imageHeight", 512))
    diag = float(np.sqrt(width * width + height * height))
    shapes = {}
    for shape in data.get("shapes", []):
        label = shape.get("label")
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if points.size == 0:
            continue
        if label not in shapes:
            shapes[label] = points

    y_raw = shapes.get("y", np.zeros((1, 2), dtype=np.float32))
    e_raw = shapes.get("e", np.zeros((1, 2), dtype=np.float32))
    y_points = normalize_points(resample_polyline(y_raw), width, height)
    e_points = normalize_points(resample_polyline(e_raw), width, height)
    contour_points = np.stack([y_points, e_points], axis=0)

    landmark_points = np.zeros((LANDMARK_POINTS, 2), dtype=np.float32)
    landmark_valid = np.zeros((LANDMARK_POINTS,), dtype=np.float32)
    landmark_specs = [("b", 0), ("Y", 1)]
    for label, index in landmark_specs:
        if label in shapes:
            landmark_points[index] = normalize_points(shapes[label][:1], width, height)[0]
            landmark_valid[index] = 1.0
    if "B" in shapes and shapes["B"].shape[0] >= 2:
        b_line = normalize_points(shapes["B"][:2], width, height)
        landmark_points[2] = b_line[0]
        landmark_points[3] = b_line[1]
        landmark_valid[2:] = 1.0

    y_len = polyline_length(y_raw) / max(diag, 1.0)
    e_len = polyline_length(e_raw) / max(diag, 1.0)
    y_chord = float(np.linalg.norm(y_raw[-1] - y_raw[0])) / max(diag, 1.0) if y_raw.shape[0] >= 2 else 0.0
    e_chord = float(np.linalg.norm(e_raw[-1] - e_raw[0])) / max(diag, 1.0) if e_raw.shape[0] >= 2 else 0.0
    b_line_len = float(np.linalg.norm(shapes["B"][1] - shapes["B"][0])) / max(diag, 1.0) if "B" in shapes and shapes["B"].shape[0] >= 2 else 0.0
    b_to_line = point_line_distance(shapes["b"][0], shapes["B"][:2]) / max(diag, 1.0) if "b" in shapes and "B" in shapes else 0.0
    start_gap = float(np.linalg.norm(y_raw[0] - e_raw[0])) / max(diag, 1.0) if y_raw.size and e_raw.size else 0.0
    end_gap = float(np.linalg.norm(y_raw[-1] - e_raw[-1])) / max(diag, 1.0) if y_raw.size and e_raw.size else 0.0
    morphology = np.asarray(
        [
            float(a_n),
            b_line_len,
            b_to_line,
            y_len,
            e_len,
            y_chord,
            e_chord,
            y_len / max(y_chord, 1e-6),
            e_len / max(e_chord, 1e-6),
            0.5 * (start_gap + end_gap),
        ],
        dtype=np.float32,
    )
    morphology = np.nan_to_num(morphology, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "contour_points": contour_points,
        "landmark_points": landmark_points,
        "landmark_valid": landmark_valid,
        "morphology": morphology,
    }


class XrayMultimodalDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, masks_by_row_id: dict[int, np.ndarray], image_size: int, training: bool):
        self.frame = frame.reset_index(drop=True)
        self.masks_by_row_id = masks_by_row_id
        self.image_size = image_size
        self.training = training
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.annotation_cache = [
            parse_annotation(row["label_path"], float(row.get("A_N", 0.0)))
            for _, row in self.frame.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        row_id = int(row["row_id"])
        image = Image.open(row["image_path"]).convert("RGB").resize((self.image_size, self.image_size))
        if self.training:
            image = tvf.adjust_brightness(image, random.uniform(0.88, 1.12))
            image = tvf.adjust_contrast(image, random.uniform(0.88, 1.12))
        mask = self.masks_by_row_id[row_id].astype(np.float32) / 255.0
        ann = self.annotation_cache[index]
        return {
            "image": self.normalize(tvf.to_tensor(image)),
            "mask": torch.from_numpy(mask),
            "contour_points": torch.from_numpy(ann["contour_points"]),
            "landmark_points": torch.from_numpy(ann["landmark_points"]),
            "landmark_valid": torch.from_numpy(ann["landmark_valid"]),
            "morphology": torch.from_numpy(ann["morphology"]),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "row_id": row_id,
            "sample_id": str(row["sample_id"]),
            "image_file": str(row["image_file"]),
            "fold": int(row["fold"]),
        }


def up_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class FractalTokenEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        scales: tuple[int, ...] = (2, 4, 8, 16, 32),
        q_values: tuple[float, ...] = (0.5, 1.0, 2.0),
        gamma: float = 0.05,
    ):
        super().__init__()
        self.scales = scales
        self.q_values = q_values
        self.gamma = gamma
        self.token_feature_dim = CONTOUR_CHANNELS + CONTOUR_CHANNELS * len(q_values) + 1
        self.projection = nn.Sequential(
            nn.Linear(self.token_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def extract_token_features(self, probs: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        features = []
        for scale in self.scales:
            box_sum = F.avg_pool2d(probs, kernel_size=scale, stride=scale) * float(scale * scale)
            occupancy = 1.0 - torch.exp(-self.gamma * box_sum)
            count = torch.log(occupancy.sum(dim=(2, 3)).clamp_min(eps))
            mass = box_sum + eps
            mass = mass / mass.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
            q_parts = []
            for q in self.q_values:
                q_parts.append(torch.log(torch.pow(mass, q).sum(dim=(2, 3)).clamp_min(eps)))
            scale_code = torch.full(
                (probs.shape[0], 1),
                np.log(1.0 / float(scale)),
                dtype=probs.dtype,
                device=probs.device,
            )
            features.append(torch.cat([count, *q_parts, scale_code], dim=1))
        return torch.stack(features, dim=1)

    def forward(self, probs: torch.Tensor) -> dict[str, torch.Tensor]:
        raw_tokens = self.extract_token_features(probs)
        tokens = self.projection(raw_tokens)
        return {
            "raw_tokens": raw_tokens,
            "tokens": tokens,
            "pooled": tokens.mean(dim=1),
        }


class FractionalContourOperator(nn.Module):
    def __init__(self, d_model: int, alpha_min: float, alpha_max: float):
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, CONTOUR_CHANNELS),
        )

    def forward(self, points: torch.Tensor, fractal_pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.alpha_net(fractal_pooled))
        a = alpha[:, :, None, None]
        out = points.clone()
        out[:, :, 1:] = out[:, :, 1:] - a * points[:, :, :-1]
        if points.shape[2] > 2:
            coeff2 = a * (a - 1.0) / 2.0
            out[:, :, 2:] = out[:, :, 2:] + coeff2 * points[:, :, :-2]
        if points.shape[2] > 3:
            coeff3 = -a * (a - 1.0) * (a - 2.0) / 6.0
            out[:, :, 3:] = out[:, :, 3:] + coeff3 * points[:, :, :-3]
        return out, alpha


class ContourTokenEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.value_proj = nn.Linear(4, d_model)
        self.channel_embed = nn.Embedding(CONTOUR_CHANNELS, d_model)
        self.position_embed = nn.Parameter(torch.zeros(1, CONTOUR_POINTS, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, points: torch.Tensor, fractional_points: torch.Tensor) -> torch.Tensor:
        x = torch.cat([points, fractional_points], dim=-1)
        b = x.shape[0]
        tokens = self.value_proj(x)
        channel_ids = torch.arange(CONTOUR_CHANNELS, device=x.device)
        tokens = tokens + self.channel_embed(channel_ids).view(1, CONTOUR_CHANNELS, 1, -1)
        tokens = tokens + self.position_embed.view(1, 1, CONTOUR_POINTS, -1)
        return self.dropout(self.norm(tokens.view(b, CONTOUR_CHANNELS * CONTOUR_POINTS, -1)))


class LandmarkTokenEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.value_proj = nn.Linear(2, d_model)
        self.type_embed = nn.Embedding(LANDMARK_POINTS, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        ids = torch.arange(LANDMARK_POINTS, device=landmarks.device)
        tokens = self.value_proj(landmarks) + self.type_embed(ids).view(1, LANDMARK_POINTS, -1)
        return self.dropout(self.norm(tokens))


class MorphologyTokenEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model)
        self.type_embed = nn.Embedding(MORPHOLOGY_DIM, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, morphology: torch.Tensor) -> torch.Tensor:
        ids = torch.arange(MORPHOLOGY_DIM, device=morphology.device)
        tokens = self.value_proj(morphology.unsqueeze(-1)) + self.type_embed(ids).view(1, MORPHOLOGY_DIM, -1)
        return self.dropout(self.norm(tokens))


class FractalGuidedInteraction(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, router_temperature: float):
        super().__init__()
        self.router_temperature = max(float(router_temperature), 1e-6)
        self.image_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.contour_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.landmark_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.morphology_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.router = nn.Sequential(
            nn.LayerNorm(d_model * 6),
            nn.Linear(d_model * 6, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 4),
        )
        self.norm = nn.LayerNorm(d_model)

    def attend(self, attn: nn.MultiheadAttention, fractal_tokens: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        out, _ = attn(query=fractal_tokens, key=tokens, value=tokens, need_weights=False)
        return out.mean(dim=1)

    def forward(
        self,
        fractal_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        contour_tokens: torch.Tensor,
        landmark_tokens: torch.Tensor,
        morphology_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fractal_pooled = fractal_tokens.mean(dim=1)
        summaries = torch.stack(
            [
                self.attend(self.image_attn, fractal_tokens, image_tokens),
                self.attend(self.contour_attn, fractal_tokens, contour_tokens),
                self.attend(self.landmark_attn, fractal_tokens, landmark_tokens),
                self.attend(self.morphology_attn, fractal_tokens, morphology_tokens),
            ],
            dim=1,
        )
        summary_std = summaries.std(dim=1, unbiased=False)
        router_context = torch.cat([fractal_pooled, summaries.flatten(1), summary_std], dim=1)
        router_logits = self.router(router_context)
        router_weights = torch.softmax(router_logits / self.router_temperature, dim=1)
        router_entropy = -(router_weights * torch.log(router_weights.clamp_min(1e-8))).sum(dim=1)
        fused = (summaries * router_weights.unsqueeze(-1)).sum(dim=1)
        return {
            "fused": self.norm(fused),
            "summaries": summaries,
            "router_weights": router_weights,
            "router_logits": router_logits,
            "router_entropy": router_entropy,
        }


class F2MITransformer(nn.Module):
    def __init__(
        self,
        config: RunConfig,
        allow_random_fallback: bool = True,
    ):
        super().__init__()
        configure_model_storage()
        weights = None
        if config.pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        try:
            resnet = models.resnet18(weights=weights)
        except Exception as exc:
            if config.pretrained and allow_random_fallback:
                print(f"[warn] Could not initialize pretrained ResNet18; using random weights: {exc}")
                resnet = models.resnet18(weights=None)
            else:
                raise

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.residual_scale = float(config.residual_scale)
        self.variant = config.variant
        self.use_fractal_token = bool(config.use_fractal_token)
        self.use_fractional_operator = bool(config.use_fractional_operator)
        self.use_residual_fusion = bool(config.use_residual_fusion)

        self.dec4 = up_block(512, 256)
        self.dec3 = up_block(256, 128)
        self.dec2 = up_block(128, 64)
        self.dec1 = up_block(64, 32)
        self.dec0 = up_block(32, 16)
        self.contour_head = nn.Conv2d(16, CONTOUR_CHANNELS, kernel_size=1)

        self.image_token_proj = nn.Conv2d(512, config.d_model, kernel_size=1)
        self.image_global_proj = nn.Sequential(
            nn.Linear(512, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(inplace=True),
        )
        self.baseline_classifier = nn.Linear(512, 1)
        self.point_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(512, CONTOUR_CHANNELS * CONTOUR_POINTS * 2),
            nn.Tanh(),
        )
        self.landmark_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(256, LANDMARK_POINTS * 2),
            nn.Tanh(),
        )
        self.morphology_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(256, MORPHOLOGY_DIM),
        )

        self.fractal = FractalTokenEncoder(config.d_model, gamma=config.fractal_gamma)
        self.nonfractal_tokens = nn.Parameter(torch.zeros(1, len(self.fractal.scales), config.d_model))
        nn.init.normal_(self.nonfractal_tokens, mean=0.0, std=0.02)
        self.fractional = FractionalContourOperator(config.d_model, config.alpha_min, config.alpha_max)
        self.contour_encoder = ContourTokenEncoder(config.d_model, config.dropout)
        self.landmark_encoder = LandmarkTokenEncoder(config.d_model, config.dropout)
        self.morphology_encoder = MorphologyTokenEncoder(config.d_model, config.dropout)
        self.interaction = FractalGuidedInteraction(
            config.d_model,
            config.n_heads,
            config.dropout,
            config.router_temperature,
        )
        self.residual_classifier = nn.Sequential(
            nn.Linear(config.d_model * 3, config.fusion_hidden_dim),
            nn.LayerNorm(config.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim, 1),
        )

    @staticmethod
    def gap(x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        d4 = self.dec4(c5)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        d0 = self.dec0(d1)
        contour_logits = self.contour_head(d0)
        contour_probs = torch.sigmoid(contour_logits)

        image_tokens = self.image_token_proj(c5).flatten(2).transpose(1, 2)
        image_global_raw = self.gap(c5)
        image_global = self.image_global_proj(image_global_raw)

        pred_points = self.point_head(image_global_raw).view(-1, CONTOUR_CHANNELS, CONTOUR_POINTS, 2)
        pred_landmarks = self.landmark_head(image_global_raw).view(-1, LANDMARK_POINTS, 2)
        pred_morphology = self.morphology_head(image_global_raw)

        if self.use_fractal_token:
            fractal_out = self.fractal(contour_probs)
        else:
            tokens = self.nonfractal_tokens.expand(images.shape[0], -1, -1)
            raw_tokens = torch.zeros(
                images.shape[0],
                len(self.fractal.scales),
                self.fractal.token_feature_dim,
                dtype=tokens.dtype,
                device=tokens.device,
            )
            fractal_out = {
                "tokens": tokens,
                "raw_tokens": raw_tokens,
                "pooled": tokens.mean(dim=1),
            }
        if self.use_fractional_operator:
            fractional_points, alpha = self.fractional(pred_points, fractal_out["pooled"])
        else:
            fractional_points = pred_points
            alpha = torch.zeros(
                pred_points.shape[0],
                CONTOUR_CHANNELS,
                dtype=pred_points.dtype,
                device=pred_points.device,
            )
        contour_tokens = self.contour_encoder(pred_points, fractional_points)
        landmark_tokens = self.landmark_encoder(pred_landmarks)
        morphology_tokens = self.morphology_encoder(pred_morphology)

        interaction = self.interaction(
            fractal_out["tokens"],
            image_tokens,
            contour_tokens,
            landmark_tokens,
            morphology_tokens,
        )
        cls_input = torch.cat([interaction["fused"], fractal_out["pooled"], image_global], dim=1)
        baseline_logits = self.baseline_classifier(image_global_raw).squeeze(1)
        residual_logits = self.residual_classifier(cls_input).squeeze(1)
        if self.use_residual_fusion:
            logits = baseline_logits + self.residual_scale * residual_logits
        else:
            logits = residual_logits
        extras = {
            "baseline_logits": baseline_logits,
            "residual_logits": residual_logits,
            "pred_points": pred_points,
            "fractional_points": fractional_points,
            "pred_landmarks": pred_landmarks,
            "pred_morphology": pred_morphology,
            "fractal_tokens": fractal_out["tokens"],
            "fractal_raw_tokens": fractal_out["raw_tokens"],
            "fractional_alpha": alpha,
            "router_weights": interaction["router_weights"],
            "router_entropy": interaction["router_entropy"],
            "image_tokens": image_tokens,
            "contour_tokens": contour_tokens,
            "landmark_tokens": landmark_tokens,
            "morphology_tokens": morphology_tokens,
        }
        return logits, contour_logits, extras


def amp_autocast(device, use_amp: bool):
    enabled = use_amp and device.type == "cuda"
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(device, use_amp: bool):
    enabled = use_amp and device.type == "cuda"
    if hasattr(torch, "amp"):
        return torch.amp.GradScaler(device.type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def make_cls_criterion(train_frame: pd.DataFrame, config: RunConfig, device) -> nn.Module:
    if not config.use_pos_weight:
        return nn.BCEWithLogitsLoss()
    labels = train_frame["label"].to_numpy()
    positives = max(int((labels == 1).sum()), 1)
    negatives = max(int((labels == 0).sum()), 1)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def make_contour_bce(config: RunConfig, device):
    pos_weight = torch.tensor([config.contour_pos_weight, config.contour_pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight.view(1, CONTOUR_CHANNELS, 1, 1))


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def contour_loss_fn(logits: torch.Tensor, targets: torch.Tensor, bce) -> tuple[torch.Tensor, float, float]:
    bce_loss = bce(logits, targets)
    dice_loss = soft_dice_loss(logits, targets)
    return bce_loss + dice_loss, float(bce_loss.detach().cpu()), float(dice_loss.detach().cpu())


def compute_cls_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = (probs >= 0.5).astype(int)
    metrics = {
        "auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
        "balanced_acc": float(balanced_accuracy_score(labels, pred)),
        "acc": float(accuracy_score(labels, pred)),
        "pre": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
    }
    tn, fp, _, _ = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    metrics["specificity"] = float(tn / max(tn + fp, 1))
    return metrics


def compute_losses(model, batch, cls_criterion, contour_bce, config: RunConfig, device):
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    target_points = batch["contour_points"].to(device, non_blocking=True)
    target_landmarks = batch["landmark_points"].to(device, non_blocking=True)
    landmark_valid = batch["landmark_valid"].to(device, non_blocking=True)
    target_morphology = batch["morphology"].to(device, non_blocking=True)

    logits, contour_logits, extras = model(images)
    cls_loss = cls_criterion(logits, labels)
    baseline_cls_loss = cls_criterion(extras["baseline_logits"], labels)
    if model.use_residual_fusion:
        residual_l2 = extras["residual_logits"].pow(2).mean()
    else:
        residual_l2 = logits.new_tensor(0.0)
    residual_abs_logit = extras["residual_logits"].abs().mean()
    mask_loss, mask_bce, mask_dice_loss = contour_loss_fn(contour_logits, masks, contour_bce)
    points_loss = F.smooth_l1_loss(extras["pred_points"], target_points)

    landmark_error = F.smooth_l1_loss(extras["pred_landmarks"], target_landmarks, reduction="none").mean(dim=2)
    landmark_loss = (landmark_error * landmark_valid).sum() / landmark_valid.sum().clamp_min(1.0)
    morphology_loss = F.smooth_l1_loss(extras["pred_morphology"], target_morphology)

    if model.use_fractal_token:
        with torch.no_grad():
            target_fractal = model.fractal(masks)["tokens"]
        fractal_loss = F.smooth_l1_loss(extras["fractal_tokens"], target_fractal)
    else:
        fractal_loss = logits.new_tensor(0.0)

    router_weights = extras["router_weights"]
    router_entropy = extras["router_entropy"].mean() / np.log(router_weights.shape[1])
    router_batch_std = router_weights.std(dim=0, unbiased=False).mean()
    router_div_loss = F.relu(torch.as_tensor(config.router_div_target, device=device) - router_batch_std).pow(2)
    router_entropy_loss = -router_entropy

    total = (
        cls_loss
        + config.lambda_mask * mask_loss
        + config.lambda_points * points_loss
        + config.lambda_landmark * landmark_loss
        + config.lambda_morphology * morphology_loss
        + config.lambda_fractal * fractal_loss
        + config.lambda_baseline_cls * baseline_cls_loss
        + config.lambda_residual_l2 * residual_l2
        + config.lambda_router_div * router_div_loss
        + config.lambda_router_entropy * router_entropy_loss
    )
    loss_parts = {
        "loss": total,
        "cls_loss": cls_loss,
        "baseline_cls_loss": baseline_cls_loss,
        "residual_l2": residual_l2,
        "residual_abs_logit": residual_abs_logit,
        "mask_loss": mask_loss,
        "mask_bce": mask_bce,
        "mask_dice_loss": mask_dice_loss,
        "points_loss": points_loss,
        "landmark_loss": landmark_loss,
        "morphology_loss": morphology_loss,
        "fractal_loss": fractal_loss,
        "router_div_loss": router_div_loss,
        "router_entropy_loss": router_entropy_loss,
        "router_batch_std": router_batch_std,
        "router_entropy": router_entropy,
    }
    return logits, contour_logits, extras, loss_parts


def make_loaders(train_frame, val_frame, test_frame, masks_by_row_id, config: RunConfig, device):
    kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    return (
        DataLoader(XrayMultimodalDataset(train_frame, masks_by_row_id, config.image_size, True), shuffle=True, **kwargs),
        DataLoader(XrayMultimodalDataset(val_frame, masks_by_row_id, config.image_size, False), shuffle=False, **kwargs),
        DataLoader(XrayMultimodalDataset(test_frame, masks_by_row_id, config.image_size, False), shuffle=False, **kwargs),
    )


def build_model(config: RunConfig, allow_random_fallback: bool) -> F2MITransformer:
    return F2MITransformer(config, allow_random_fallback=allow_random_fallback)


def load_contour_aux_checkpoint(model: F2MITransformer, fold: int, config: RunConfig) -> dict[str, object]:
    if not config.load_baseline_checkpoint:
        return {"enabled": False, "loaded": False, "reason": "disabled"}

    checkpoint_dir = Path(config.baseline_checkpoint_dir)
    checkpoint_path = checkpoint_dir / f"contour_aux_resnet18_fold{fold}.pt"
    if not checkpoint_path.exists():
        return {"enabled": True, "loaded": False, "reason": f"missing: {checkpoint_path}"}

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint.get("state_dict", checkpoint)
    target_state = model.state_dict()
    mapped = {}

    direct_prefixes = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
    for key, value in source_state.items():
        if any(key.startswith(prefix + ".") for prefix in direct_prefixes):
            mapped[key] = value

    for source_prefix, target_prefix in [
        ("decoder.0", "dec4"),
        ("decoder.1", "dec3"),
        ("decoder.2", "dec2"),
        ("decoder.3", "dec1"),
        ("decoder.4", "dec0"),
    ]:
        for key, value in source_state.items():
            if key.startswith(source_prefix + "."):
                mapped[target_prefix + key[len(source_prefix) :]] = value

    for source_key, target_key in [
        ("decoder.5.weight", "contour_head.weight"),
        ("decoder.5.bias", "contour_head.bias"),
        ("classifier.weight", "baseline_classifier.weight"),
        ("classifier.bias", "baseline_classifier.bias"),
    ]:
        if source_key in source_state:
            mapped[target_key] = source_state[source_key]

    compatible = {
        key: value
        for key, value in mapped.items()
        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape)
    }
    target_state.update(compatible)
    model.load_state_dict(target_state)
    return {
        "enabled": True,
        "loaded": True,
        "checkpoint": str(checkpoint_path),
        "mapped_tensors": len(compatible),
        "skipped_tensors": len(mapped) - len(compatible),
    }


def run_epoch(model, loader, cls_criterion, contour_bce, optimizer, scaler, device, config: RunConfig):
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "cls_loss": 0.0,
        "baseline_cls_loss": 0.0,
        "residual_l2": 0.0,
        "residual_abs_logit": 0.0,
        "mask_loss": 0.0,
        "points_loss": 0.0,
        "landmark_loss": 0.0,
        "morphology_loss": 0.0,
        "fractal_loss": 0.0,
        "router_div_loss": 0.0,
        "router_entropy_loss": 0.0,
        "router_batch_std": 0.0,
        "router_entropy": 0.0,
    }
    labels_all, probs_all, baseline_probs_all = [], [], []
    router_all, alpha_all = [], []
    for batch in loader:
        labels = batch["label"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with amp_autocast(device, config.amp):
                logits, _, extras, loss_parts = compute_losses(model, batch, cls_criterion, contour_bce, config, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss_parts["loss"]).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_parts["loss"].backward()
                    optimizer.step()
        batch_size = len(labels)
        for key in totals:
            totals[key] += float(loss_parts[key].detach().cpu()) * batch_size
        labels_all.append(labels.detach().cpu().numpy())
        probs_all.append(torch.sigmoid(logits.detach()).cpu().numpy())
        baseline_probs_all.append(torch.sigmoid(extras["baseline_logits"].detach()).cpu().numpy())
        router_all.append(extras["router_weights"].detach().cpu().numpy())
        alpha_all.append(extras["fractional_alpha"].detach().cpu().numpy())

    n = max(sum(len(item) for item in labels_all), 1)
    labels_np = np.concatenate(labels_all)
    probs_np = np.concatenate(probs_all)
    baseline_probs_np = np.concatenate(baseline_probs_all)
    metrics = compute_cls_metrics(labels_np, probs_np)
    baseline_metrics = compute_cls_metrics(labels_np, baseline_probs_np)
    metrics.update({f"baseline_{key}": value for key, value in baseline_metrics.items()})
    metrics["delta_auc_vs_baseline_head"] = metrics["auc"] - baseline_metrics["auc"]
    metrics["delta_f1_vs_baseline_head"] = metrics["f1"] - baseline_metrics["f1"]
    metrics["delta_recall_vs_baseline_head"] = metrics["recall"] - baseline_metrics["recall"]
    metrics["delta_pre_vs_baseline_head"] = metrics["pre"] - baseline_metrics["pre"]
    metrics.update({key: value / n for key, value in totals.items()})
    router = np.concatenate(router_all, axis=0)
    alpha = np.concatenate(alpha_all, axis=0)
    metrics.update(
        {
            "router_image": float(router[:, 0].mean()),
            "router_contour": float(router[:, 1].mean()),
            "router_landmark": float(router[:, 2].mean()),
            "router_morphology": float(router[:, 3].mean()),
            "router_image_std": float(router[:, 0].std()),
            "router_contour_std": float(router[:, 1].std()),
            "router_landmark_std": float(router[:, 2].std()),
            "router_morphology_std": float(router[:, 3].std()),
            "fractional_alpha_mean": float(alpha.mean()),
            "fractional_alpha_std": float(alpha.std()),
        }
    )
    return metrics


def evaluate_with_ids(model, loader, cls_criterion, contour_bce, device, config: RunConfig, split_name: str, outer_fold: int):
    model.eval()
    records = []
    metrics = run_epoch(model, loader, cls_criterion, contour_bce, None, None, device, config)
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with amp_autocast(device, config.amp):
                logits, _, extras = model(images)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            baseline_probs = torch.sigmoid(extras["baseline_logits"]).detach().cpu().numpy()
            residual_logits = extras["residual_logits"].detach().cpu().numpy()
            final_logits = logits.detach().cpu().numpy()
            labels = batch["label"].numpy().astype(int)
            router = extras["router_weights"].detach().cpu().numpy()
            router_entropy = extras["router_entropy"].detach().cpu().numpy()
            alpha = extras["fractional_alpha"].detach().cpu().numpy()
            for idx, (row_id, sample_id, image_file, fold, label, prob, baseline_prob, residual_logit, final_logit) in enumerate(
                zip(
                    batch["row_id"],
                    batch["sample_id"],
                    batch["image_file"],
                    batch["fold"],
                    labels,
                    probs,
                    baseline_probs,
                    residual_logits,
                    final_logits,
                )
            ):
                records.append(
                    {
                        "row_id": int(row_id),
                        "sample_id": str(sample_id),
                        "image_file": str(image_file),
                        "experiment_id": config.experiment_id,
                        "experiment_label": config.experiment_label,
                        "variant": config.variant,
                        "outer_fold": int(outer_fold),
                        "source_fold": int(fold),
                        "split": split_name,
                        "label": int(label),
                        "prob_hypertrophy": float(prob),
                        "pred_label": int(prob >= 0.5),
                        "baseline_prob_hypertrophy": float(baseline_prob),
                        "baseline_pred_label": int(baseline_prob >= 0.5),
                        "residual_logit": float(residual_logit),
                        "final_logit": float(final_logit),
                        "router_image": float(router[idx, 0]),
                        "router_contour": float(router[idx, 1]),
                        "router_landmark": float(router[idx, 2]),
                        "router_morphology": float(router[idx, 3]),
                        "router_entropy": float(router_entropy[idx]),
                        "fractional_alpha_y": float(alpha[idx, 0]),
                        "fractional_alpha_e": float(alpha[idx, 1]),
                    }
                )
    return metrics, records


def train_fold(folds, outer_splits, masks_by_row_id, fold: int, args, config: RunConfig):
    device = torch.device(config.device)
    fold_output_dir = FOLD_OUTPUT_DIR / f"f{fold}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    train_frame, val_frame, test_frame = split_frames(folds, outer_splits, fold)
    train_loader, val_loader, test_loader = make_loaders(train_frame, val_frame, test_frame, masks_by_row_id, config, device)
    model = build_model(config, allow_random_fallback=args.allow_random_fallback)
    baseline_init_status = load_contour_aux_checkpoint(model, fold, config)
    print(f"fold={fold} baseline_init={baseline_init_status}", flush=True)
    model = model.to(device)
    cls_criterion = make_cls_criterion(train_frame, config, device)
    contour_bce = make_contour_bce(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = make_grad_scaler(device, config.amp)

    best_score = -np.inf
    best_epoch = 0
    best_state = None
    history = []
    epochs_without_improvement = 0
    start = time.time()
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, cls_criterion, contour_bce, optimizer, scaler, device, config)
        val_metrics = run_epoch(model, val_loader, cls_criterion, contour_bce, None, None, device, config)
        test_metrics = run_epoch(model, test_loader, cls_criterion, contour_bce, None, None, device, config)
        row = {"fold": fold, "epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(
            fold_output_dir / "history_live.csv",
            index=False,
            encoding="utf-8-sig",
        )
        monitor_value = row.get(config.monitor, val_metrics["auc"])
        if monitor_value > best_score:
            best_score = monitor_value
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"fold={fold} epoch={epoch:03d} "
            f"train_auc={train_metrics['auc']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"test_auc={test_metrics['auc']:.4f} val_recall={val_metrics['recall']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} baseline_f1={val_metrics['baseline_f1']:.4f} "
            f"delta_f1={val_metrics['delta_f1_vs_baseline_head']:.4f} residual={val_metrics['residual_abs_logit']:.3f} "
            f"router=[{val_metrics['router_image']:.2f},{val_metrics['router_contour']:.2f},"
            f"{val_metrics['router_landmark']:.2f},{val_metrics['router_morphology']:.2f}] "
            f"patience={epochs_without_improvement}/{config.patience}",
            flush=True,
        )
        if epochs_without_improvement >= config.patience:
            print(f"fold={fold} early stopping at epoch={epoch} with best_{config.monitor}={best_score:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    best_val_metrics, val_records = evaluate_with_ids(model, val_loader, cls_criterion, contour_bce, device, config, "validation", fold)
    best_test_metrics, test_records = evaluate_with_ids(model, test_loader, cls_criterion, contour_bce, device, config, "test", fold)
    best_metrics = {"fold": fold, "best_epoch": best_epoch, "runtime_sec": time.time() - start}
    best_metrics.update({f"val_{key}": value for key, value in best_val_metrics.items()})
    best_metrics.update({f"test_{key}": value for key, value in best_test_metrics.items()})
    best_metrics.update(best_test_metrics)
    best_metrics["baseline_init_loaded"] = bool(baseline_init_status.get("loaded", False))
    best_metrics["baseline_init_mapped_tensors"] = int(baseline_init_status.get("mapped_tensors", 0) or 0)
    best_metrics["baseline_init_checkpoint"] = str(baseline_init_status.get("checkpoint", ""))

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{config.experiment_id}_fold{fold}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "fold": fold,
            "best_epoch": best_epoch,
            "metrics": best_metrics,
        },
        ckpt_path,
    )
    best_metrics["checkpoint"] = str(ckpt_path)
    prediction_records = val_records + test_records
    pd.DataFrame([best_metrics]).to_csv(
        fold_output_dir / "metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(prediction_records).to_csv(
        fold_output_dir / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(history).to_csv(
        fold_output_dir / "history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return best_metrics, prediction_records, history


def summarize_metrics(metrics_by_fold: pd.DataFrame) -> pd.DataFrame:
    row = {}
    metrics = [
        "auc",
        "balanced_acc",
        "acc",
        "pre",
        "recall",
        "specificity",
        "f1",
        "loss",
        "cls_loss",
        "baseline_cls_loss",
        "residual_l2",
        "residual_abs_logit",
        "mask_loss",
        "points_loss",
        "landmark_loss",
        "morphology_loss",
        "fractal_loss",
        "router_div_loss",
        "router_entropy_loss",
        "router_batch_std",
        "router_entropy",
        "router_image",
        "router_contour",
        "router_landmark",
        "router_morphology",
        "router_image_std",
        "router_contour_std",
        "router_landmark_std",
        "router_morphology_std",
        "fractional_alpha_mean",
        "fractional_alpha_std",
        "baseline_auc",
        "baseline_acc",
        "baseline_pre",
        "baseline_recall",
        "baseline_specificity",
        "baseline_f1",
        "delta_auc_vs_baseline_head",
        "delta_f1_vs_baseline_head",
        "delta_recall_vs_baseline_head",
        "delta_pre_vs_baseline_head",
    ]
    for metric in metrics:
        row[f"{metric}_mean"] = metrics_by_fold[metric].mean()
        row[f"{metric}_std"] = metrics_by_fold[metric].std(ddof=1)
    return pd.DataFrame([row])


def write_summary(config: RunConfig, folds: pd.DataFrame, trained: bool, input_status=None, metrics=None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = folds["group"].value_counts().to_dict()
    lines = [
        "# 002002003010003 Final Module Ablation Run Summary",
        "",
        "## Status",
        "",
        f"- Experiment: `{config.experiment_id}`",
        f"- Experiment label: {config.experiment_label}",
        f"- Variant: `{config.variant}`",
        f"- Variant description: {VARIANT_DESCRIPTIONS.get(config.variant, '')}",
        f"- Use fractal token: {config.use_fractal_token}",
        f"- Use fractional contour operator: {config.use_fractional_operator}",
        f"- Use baseline-preserving residual fusion: {config.use_residual_fusion}",
        "- Inference input: raw X-ray image only.",
        "- Training supervision: contour mask, 128-point y/e contours, b/Y/B landmarks, morphology vector, fractal token consistency.",
        "- Core mechanism: initialize from the 002002002 contour-auxiliary baseline and learn a bounded fractal/fractional F2MI residual on top of the baseline logit.",
        "- Final decision: `final_logit = baseline_logit + residual_scale * residual_logit`.",
        "- Purpose: keep the strong contour-auxiliary decision surface while allowing sample-adaptive fractal/fractional residual correction.",
        f"- Samples: {len(folds)}",
        f"- Unique sample_id count: {folds['sample_id'].nunique()}",
        f"- Normal samples: {counts.get('normal', 0)}",
        f"- Hypertrophy samples: {counts.get('hypertrophy', 0)}",
        f"- Folds: {config.n_splits}",
        f"- Image/mask size: {config.image_size} x {config.image_size}",
        f"- Trained: {'yes' if trained else 'no'}",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Input Reuse",
        "",
        f"- Source outputs: `{SOURCE_OUTPUT_DIR}`",
    ]
    if input_status:
        for name, status in input_status.items():
            lines.append(f"- `{name}`: {status}")
    if trained and metrics is not None and not metrics.empty:
        lines += ["", "## Cross-Validation Summary", "", "| Metric | Mean | Std |", "|---|---:|---:|"]
        for metric in [
            "auc",
            "acc",
            "pre",
            "recall",
            "specificity",
            "f1",
            "baseline_auc",
            "baseline_acc",
            "baseline_pre",
            "baseline_recall",
            "baseline_specificity",
            "baseline_f1",
            "delta_auc_vs_baseline_head",
            "delta_f1_vs_baseline_head",
            "delta_recall_vs_baseline_head",
            "delta_pre_vs_baseline_head",
            "residual_abs_logit",
            "router_image",
            "router_contour",
            "router_landmark",
            "router_morphology",
            "router_image_std",
            "router_contour_std",
            "router_landmark_std",
            "router_morphology_std",
        ]:
            lines.append(f"| {metric} | {metrics[f'{metric}_mean'].iloc[0]:.4f} | {metrics[f'{metric}_std'].iloc[0]:.4f} |")
    (OUTPUT_DIR / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_training_progress(config: RunConfig, selected_folds: list[int], metrics_by_fold: pd.DataFrame, summary: pd.DataFrame) -> None:
    completed_folds = [int(value) for value in metrics_by_fold["fold"].tolist()]
    remaining_folds = [fold for fold in selected_folds if fold not in completed_folds]
    lines = [
        f"# {config.experiment_id} Training Progress",
        "",
        f"- Status: {'complete' if len(remaining_folds) == 0 else 'running'}",
        f"- Completed folds: {completed_folds}",
        f"- Remaining folds: {remaining_folds}",
        f"- Updated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Fold Metrics",
        "",
        "| Fold | AUC | Acc | Pre | Recall | F1 | Baseline F1 | Delta F1 | Residual | Best epoch |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics_by_fold.sort_values("fold").iterrows():
        lines.append(
            f"| {int(row['fold'])} | {row['auc']:.4f} | {row['acc']:.4f} | "
            f"{row['pre']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
            f"{row['baseline_f1']:.4f} | {row['delta_f1_vs_baseline_head']:.4f} | "
            f"{row['residual_abs_logit']:.4f} | {int(row['best_epoch'])} |"
        )
    lines += [
        "",
        "## Current Summary",
        "",
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
    ]
    for metric in [
        "auc",
        "acc",
        "pre",
        "recall",
        "specificity",
        "f1",
        "baseline_f1",
        "delta_f1_vs_baseline_head",
        "delta_recall_vs_baseline_head",
        "residual_abs_logit",
    ]:
        mean_value = summary[f"{metric}_mean"].iloc[0]
        std_value = summary[f"{metric}_std"].iloc[0]
        std_text = "NA" if pd.isna(std_value) else f"{std_value:.4f}"
        lines.append(f"| {metric} | {mean_value:.4f} | {std_text} |")
    (OUTPUT_DIR / "training_progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_incremental_outputs(
    config: RunConfig,
    selected_folds: list[int],
    all_metrics: list[dict],
    all_predictions: list[dict],
    all_history: list[dict],
) -> pd.DataFrame:
    metrics_by_fold = pd.DataFrame(all_metrics)
    predictions = pd.DataFrame(all_predictions)
    history = pd.DataFrame(all_history)
    summary = summarize_metrics(metrics_by_fold)

    metrics_by_fold.to_csv(OUTPUT_DIR / "cv_metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUT_DIR / "cv_predictions.csv", index=False, encoding="utf-8-sig")
    history.to_csv(OUTPUT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "cv_summary.csv", index=False, encoding="utf-8-sig")
    write_training_progress(config, selected_folds, metrics_by_fold, summary)
    return summary


def run_smoke_test(folds, outer_splits, masks_by_row_id, args, config: RunConfig) -> None:
    device = torch.device(config.device)
    train_frame, _, _ = split_frames(folds, outer_splits, fold=0)
    smoke_frame = train_frame.head(min(4, len(train_frame))).copy()
    loader = DataLoader(
        XrayMultimodalDataset(smoke_frame, masks_by_row_id, config.image_size, training=False),
        batch_size=min(4, len(smoke_frame)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = build_model(config, allow_random_fallback=args.allow_random_fallback)
    baseline_init_status = load_contour_aux_checkpoint(model, 0, config)
    model = model.to(device)
    cls_criterion = make_cls_criterion(train_frame, config, device)
    contour_bce = make_contour_bce(config, device)
    batch = next(iter(loader))
    with amp_autocast(device, config.amp):
        logits, contour_logits, extras, loss_parts = compute_losses(model, batch, cls_criterion, contour_bce, config, device)
    print(
        json.dumps(
            {
                "smoke_test": "ok",
                "experiment_id": config.experiment_id,
                "variant": config.variant,
                "use_fractal_token": config.use_fractal_token,
                "use_fractional_operator": config.use_fractional_operator,
                "use_residual_fusion": config.use_residual_fusion,
                "image_shape": list(batch["image"].shape),
                "mask_shape": list(batch["mask"].shape),
                "target_contour_points_shape": list(batch["contour_points"].shape),
                "target_landmark_shape": list(batch["landmark_points"].shape),
                "target_morphology_shape": list(batch["morphology"].shape),
                "cls_logits_shape": list(logits.shape),
                "baseline_logits_shape": list(extras["baseline_logits"].shape),
                "residual_logits_shape": list(extras["residual_logits"].shape),
                "contour_logits_shape": list(contour_logits.shape),
                "pred_points_shape": list(extras["pred_points"].shape),
                "pred_landmarks_shape": list(extras["pred_landmarks"].shape),
                "pred_morphology_shape": list(extras["pred_morphology"].shape),
                "fractal_tokens_shape": list(extras["fractal_tokens"].shape),
                "image_tokens_shape": list(extras["image_tokens"].shape),
                "contour_tokens_shape": list(extras["contour_tokens"].shape),
                "landmark_tokens_shape": list(extras["landmark_tokens"].shape),
                "morphology_tokens_shape": list(extras["morphology_tokens"].shape),
                "router_weights_mean": extras["router_weights"].detach().cpu().mean(dim=0).tolist(),
                "router_weights_std": extras["router_weights"].detach().cpu().std(dim=0).tolist(),
                "router_entropy_mean": float(extras["router_entropy"].detach().cpu().mean()),
                "fractional_alpha_mean": float(extras["fractional_alpha"].detach().cpu().mean()),
                "loss": float(loss_parts["loss"].detach().cpu()),
                "cls_loss": float(loss_parts["cls_loss"].detach().cpu()),
                "baseline_cls_loss": float(loss_parts["baseline_cls_loss"].detach().cpu()),
                "residual_l2": float(loss_parts["residual_l2"].detach().cpu()),
                "residual_abs_logit": float(loss_parts["residual_abs_logit"].detach().cpu()),
                "mask_loss": float(loss_parts["mask_loss"].detach().cpu()),
                "points_loss": float(loss_parts["points_loss"].detach().cpu()),
                "landmark_loss": float(loss_parts["landmark_loss"].detach().cpu()),
                "morphology_loss": float(loss_parts["morphology_loss"].detach().cpu()),
                "fractal_loss": float(loss_parts["fractal_loss"].detach().cpu()),
                "router_div_loss": float(loss_parts["router_div_loss"].detach().cpu()),
                "router_entropy_loss": float(loss_parts["router_entropy_loss"].detach().cpu()),
                "router_batch_std": float(loss_parts["router_batch_std"].detach().cpu()),
                "baseline_init_status": baseline_init_status,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANT_DESCRIPTIONS), default="full")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--alpha-min", type=float, default=0.2)
    parser.add_argument("--alpha-max", type=float, default=1.8)
    parser.add_argument("--lambda-mask", type=float, default=0.25)
    parser.add_argument("--lambda-points", type=float, default=0.15)
    parser.add_argument("--lambda-landmark", type=float, default=0.05)
    parser.add_argument("--lambda-morphology", type=float, default=0.05)
    parser.add_argument("--lambda-fractal", type=float, default=0.05)
    parser.add_argument("--lambda-baseline-cls", type=float, default=0.5)
    parser.add_argument("--lambda-residual-l2", type=float, default=0.02)
    parser.add_argument("--lambda-router-div", type=float, default=0.0)
    parser.add_argument("--lambda-router-entropy", type=float, default=0.0)
    parser.add_argument("--contour-pos-weight", type=float, default=50.0)
    parser.add_argument("--fractal-gamma", type=float, default=0.05)
    parser.add_argument("--router-temperature", type=float, default=1.2)
    parser.add_argument("--router-div-target", type=float, default=0.03)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--baseline-checkpoint-dir", default=str(BASELINE_CHECKPOINT_DIR))
    parser.add_argument("--load-baseline-checkpoint", dest="load_baseline_checkpoint", action="store_true")
    parser.add_argument("--no-load-baseline-checkpoint", dest="load_baseline_checkpoint", action="store_false")
    parser.add_argument("--monitor", default="val_f1")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--allow-random-fallback", action="store_true")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--use-pos-weight", dest="use_pos_weight", action="store_true")
    parser.add_argument("--no-pos-weight", dest="use_pos_weight", action="store_false")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.set_defaults(pretrained=True, amp=True, use_pos_weight=True, load_baseline_checkpoint=True)
    args = parser.parse_args()
    configure_variant_paths(args.variant)
    flags = variant_flags(args.variant)

    config = RunConfig(
        experiment_id=f"{EXPERIMENT_ID}_{args.variant}",
        experiment_label=f"{EXPERIMENT_LABEL}: {VARIANT_DESCRIPTIONS[args.variant]}",
        variant=args.variant,
        use_fractal_token=flags["use_fractal_token"],
        use_fractional_operator=flags["use_fractional_operator"],
        use_residual_fusion=flags["use_residual_fusion"],
        seed=args.seed,
        n_splits=args.n_splits,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        monitor=args.monitor,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        n_heads=args.n_heads,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        lambda_mask=args.lambda_mask,
        lambda_points=args.lambda_points,
        lambda_landmark=args.lambda_landmark,
        lambda_morphology=args.lambda_morphology,
        lambda_fractal=args.lambda_fractal,
        lambda_baseline_cls=args.lambda_baseline_cls,
        lambda_residual_l2=args.lambda_residual_l2,
        lambda_router_div=args.lambda_router_div,
        lambda_router_entropy=args.lambda_router_entropy,
        contour_pos_weight=args.contour_pos_weight,
        fractal_gamma=args.fractal_gamma,
        router_temperature=args.router_temperature,
        router_div_target=args.router_div_target,
        residual_scale=args.residual_scale,
        load_baseline_checkpoint=args.load_baseline_checkpoint,
        baseline_checkpoint_dir=args.baseline_checkpoint_dir,
        pretrained=args.pretrained,
        amp=args.amp,
        use_pos_weight=args.use_pos_weight,
        num_workers=args.num_workers,
        device=args.device,
    )

    set_seed(config.seed)
    configure_model_storage()
    input_status = prepare_local_inputs()
    _, folds, outer_splits, masks_by_row_id = load_inputs()
    write_summary(config, folds, trained=False, input_status=input_status)
    print(f"Loaded {len(folds)} {config.experiment_id} samples for {config.experiment_label} from {OUTPUT_DIR}")
    print(folds.groupby(["fold", "group"]).size().unstack(fill_value=0).to_string())

    if args.smoke_test:
        run_smoke_test(folds, outer_splits, masks_by_row_id, args, config)
        return
    if args.prepare_only:
        return

    selected_folds = args.folds if args.folds is not None and len(args.folds) > 0 else list(range(config.n_splits))
    all_metrics, all_predictions, all_history = [], [], []
    for fold in selected_folds:
        fold_metrics, prediction_records, history = train_fold(folds, outer_splits, masks_by_row_id, fold, args, config)
        all_metrics.append(fold_metrics)
        all_predictions.extend(prediction_records)
        all_history.extend(history)
        summary = write_incremental_outputs(config, selected_folds, all_metrics, all_predictions, all_history)
        print(f"Saved incremental outputs after fold={fold}", flush=True)

    summary = write_incremental_outputs(config, selected_folds, all_metrics, all_predictions, all_history)
    write_summary(config, folds, trained=True, input_status=input_status, metrics=summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
