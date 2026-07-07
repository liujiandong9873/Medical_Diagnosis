from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
PREVIEW_DIR = OUTPUT_DIR / "previews"

BASELINE_DIR = (
    ROOT
    / "experiments"
    / "002_image_contour_landmark_guided_fusion_classification"
    / "002002_image_contour_prediction"
    / "002002001_matched_image_only_baseline"
    / "outputs"
)


@dataclass(frozen=True)
class PrepConfig:
    target_size: int
    line_width: int
    preview_count: int
    baseline_outputs: str
    output_dir: str


def load_label_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="gb18030"))


def find_linestrip(data: dict, label: str) -> list[list[float]]:
    for shape in data.get("shapes", []):
        if shape.get("label") == label and shape.get("shape_type") == "linestrip":
            return [[float(x), float(y)] for x, y in shape.get("points", [])]
    return []


def scaled_points(points: list[list[float]], image_width: int, image_height: int, target_size: int) -> list[tuple[float, float]]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Invalid source image dimensions.")
    sx = target_size / float(image_width)
    sy = target_size / float(image_height)
    return [(float(x) * sx, float(y) * sy) for x, y in points]


def draw_contour(points: list[tuple[float, float]], target_size: int, line_width: int) -> np.ndarray:
    mask = Image.new("L", (target_size, target_size), 0)
    if len(points) >= 2:
        draw = ImageDraw.Draw(mask)
        xy = [(int(round(x)), int(round(y))) for x, y in points]
        draw.line(xy, fill=255, width=line_width, joint="curve")
        radius = max(1, line_width // 2)
        for x, y in xy:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    return np.asarray(mask, dtype=np.uint8)


def polyline_length(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return float(total)


def bbox_from_points(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    if not points:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))


def overlay_preview(image_path: Path, y_mask: np.ndarray, e_mask: np.ndarray, target_size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB").resize((target_size, target_size))
    base = np.asarray(image, dtype=np.uint8).copy()
    y = y_mask > 0
    e = e_mask > 0
    base[y] = np.array([255, 32, 32], dtype=np.uint8)
    base[e] = np.array([32, 160, 255], dtype=np.uint8)
    both = y & e
    base[both] = np.array([255, 220, 32], dtype=np.uint8)
    return Image.fromarray(base)


def copy_baseline_tables(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata_path = BASELINE_DIR / "metadata.csv"
    folds_path = BASELINE_DIR / "folds.csv"
    splits_path = BASELINE_DIR / "outer_splits.csv"
    for path in [metadata_path, folds_path, splits_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing baseline table: {path}")

    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
    folds = pd.read_csv(folds_path, encoding="utf-8-sig")
    splits = pd.read_csv(splits_path, encoding="utf-8-sig")

    metadata.to_csv(output_dir / "metadata.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(output_dir / "folds.csv", index=False, encoding="utf-8-sig")
    splits.to_csv(output_dir / "outer_splits.csv", index=False, encoding="utf-8-sig")
    return metadata, folds, splits


def generate_targets(config: PrepConfig) -> tuple[pd.DataFrame, dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    metadata, folds, splits = copy_baseline_tables(OUTPUT_DIR)
    masks = np.zeros((len(metadata), 2, config.target_size, config.target_size), dtype=np.uint8)
    audit_rows = []

    for index, row in metadata.iterrows():
        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label: {label_path}")

        with Image.open(image_path) as image:
            width, height = image.size
        data = load_label_json(label_path)

        y_points_raw = find_linestrip(data, "y")
        e_points_raw = find_linestrip(data, "e")
        if len(y_points_raw) < 2 or len(e_points_raw) < 2:
            raise RuntimeError(f"Missing usable y/e linestrip in {label_path}")

        y_points = scaled_points(y_points_raw, width, height, config.target_size)
        e_points = scaled_points(e_points_raw, width, height, config.target_size)
        y_mask = draw_contour(y_points, config.target_size, config.line_width)
        e_mask = draw_contour(e_points, config.target_size, config.line_width)
        masks[index, 0] = y_mask
        masks[index, 1] = e_mask

        y_bbox = bbox_from_points(y_points)
        e_bbox = bbox_from_points(e_points)
        audit_rows.append(
            {
                "row_id": int(row["row_id"]),
                "sample_id": row["sample_id"],
                "image_file": row["image_file"],
                "label_file": row["label_file"],
                "group": row["group"],
                "label": int(row["label"]),
                "source_width": int(width),
                "source_height": int(height),
                "target_size": int(config.target_size),
                "line_width": int(config.line_width),
                "y_num_points": len(y_points_raw),
                "e_num_points": len(e_points_raw),
                "y_mask_pixels": int((y_mask > 0).sum()),
                "e_mask_pixels": int((e_mask > 0).sum()),
                "y_length_224": polyline_length(y_points),
                "e_length_224": polyline_length(e_points),
                "y_bbox_x0": y_bbox[0],
                "y_bbox_y0": y_bbox[1],
                "y_bbox_x1": y_bbox[2],
                "y_bbox_y1": y_bbox[3],
                "e_bbox_x0": e_bbox[0],
                "e_bbox_y0": e_bbox[1],
                "e_bbox_x1": e_bbox[2],
                "e_bbox_y1": e_bbox[3],
            }
        )

        if index < config.preview_count:
            preview = overlay_preview(image_path, y_mask, e_mask, config.target_size)
            preview.save(PREVIEW_DIR / f"preview_{index:03d}_{row['row_id']}.png")

    target_path = OUTPUT_DIR / f"contour_targets_{config.target_size}_uint8.npz"
    np.savez_compressed(
        target_path,
        masks=masks,
        row_ids=metadata["row_id"].to_numpy(dtype=np.int64),
        labels=metadata["label"].to_numpy(dtype=np.int64),
        sample_ids=metadata["sample_id"].astype(str).to_numpy(),
        image_files=metadata["image_file"].astype(str).to_numpy(),
        channels=np.array(["y", "e"]),
    )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUTPUT_DIR / "contour_target_audit.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "target_path": str(target_path),
        "mask_shape": list(masks.shape),
        "mask_dtype": str(masks.dtype),
        "channels": ["y", "e"],
        "target_size": config.target_size,
        "line_width": config.line_width,
        "metadata_rows": int(len(metadata)),
        "fold_rows": int(len(folds)),
        "outer_split_rows": int(len(splits)),
        "preview_dir": str(PREVIEW_DIR),
    }
    (OUTPUT_DIR / "contour_target_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit, manifest


def write_summary(config: PrepConfig, audit: pd.DataFrame, manifest: dict) -> None:
    group_counts = audit["group"].value_counts().to_dict()
    lines = [
        "# 002002002 Preparation Summary",
        "",
        "## Status",
        "",
        "- Prepared contour auxiliary targets: yes",
        "- Reused matched metadata/folds/outer_splits from 002002001: yes",
        f"- Samples: {len(audit)}",
        f"- Normal samples: {group_counts.get('normal', 0)}",
        f"- Hypertrophy samples: {group_counts.get('hypertrophy', 0)}",
        f"- Target mask shape: `{manifest['mask_shape']}`",
        f"- Target channels: `{', '.join(manifest['channels'])}`",
        f"- Target size: {config.target_size} x {config.target_size}",
        f"- Line width: {config.line_width}",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Target Audit",
        "",
        f"- y mask pixels mean: {audit['y_mask_pixels'].mean():.2f}",
        f"- y mask pixels min/max: {audit['y_mask_pixels'].min()} / {audit['y_mask_pixels'].max()}",
        f"- e mask pixels mean: {audit['e_mask_pixels'].mean():.2f}",
        f"- e mask pixels min/max: {audit['e_mask_pixels'].min()} / {audit['e_mask_pixels'].max()}",
        f"- y length mean: {audit['y_length_224'].mean():.2f}",
        f"- e length mean: {audit['e_length_224'].mean():.2f}",
        "",
        "## Output Files",
        "",
        "- `metadata.csv`",
        "- `folds.csv`",
        "- `outer_splits.csv`",
        f"- `contour_targets_{config.target_size}_uint8.npz`",
        "- `contour_target_audit.csv`",
        "- `contour_target_manifest.json`",
        "- `previews/`",
        "",
        "## Next Step",
        "",
        "Create the 002002002 training script with a shared ResNet18 encoder, classification head, and contour decoder.",
    ]
    (OUTPUT_DIR / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--preview-count", type=int, default=12)
    args = parser.parse_args()

    config = PrepConfig(
        target_size=args.target_size,
        line_width=args.line_width,
        preview_count=args.preview_count,
        baseline_outputs=str(BASELINE_DIR),
        output_dir=str(OUTPUT_DIR),
    )
    audit, manifest = generate_targets(config)
    write_summary(config, audit, manifest)
    print(f"Prepared {len(audit)} contour targets at {OUTPUT_DIR}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
