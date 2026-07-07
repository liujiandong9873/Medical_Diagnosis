from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_ROOT = PROJECT_ROOT / "Dataset" / "ZYQ_Dataset"
IMAGE_DIR = DATASET_ROOT / "images"
LABEL_DIR = DATASET_ROOT / "labels"
METADATA_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "002_image_contour_landmark_guided_fusion_classification"
    / "002002_image_contour_prediction"
    / "002002001_matched_image_only_baseline"
    / "outputs"
    / "metadata.csv"
)

OUTPUT_DIR = Path(__file__).resolve().parent
CASE_DIR = OUTPUT_DIR / "case_examples"
MANIFEST_PATH = OUTPUT_DIR / "case_examples_manifest.csv"

TARGET_A_N = {
    "normal": [0.30, 0.40, 0.48, 0.54, 0.62],
    "hypertrophy": [0.56, 0.64, 0.70, 0.82, 0.94],
}

COLORS = {
    "y": (0, 220, 255, 230),
    "e": (255, 205, 0, 230),
    "B": (255, 0, 180, 230),
    "b": (255, 70, 70, 240),
    "Y": (80, 255, 120, 240),
}


def ascii_path(value: str) -> bool:
    return all(ord(char) < 128 for char in str(value))


def select_cases(metadata: pd.DataFrame) -> pd.DataFrame:
    frame = metadata.copy()
    frame = frame[frame["image_file"].map(ascii_path)]
    frame = frame[frame["A_N"].notna()]
    frame = frame[frame[["has_y", "has_e", "has_b", "has_Y", "has_B"]].all(axis=1)]

    rows = []
    used_row_ids: set[int] = set()
    for group, targets in TARGET_A_N.items():
        group_frame = frame[frame["group"] == group].copy()
        for target in targets:
            candidate = (
                group_frame[~group_frame["row_id"].isin(used_row_ids)]
                .assign(distance=lambda item: (item["A_N"] - target).abs())
                .sort_values(["distance", "A_N", "row_id"])
                .iloc[0]
            )
            rows.append(candidate)
            used_row_ids.add(int(candidate["row_id"]))

    selected = pd.DataFrame(rows).reset_index(drop=True)
    selected.insert(0, "case_id", [f"case_{index:02d}" for index in range(1, len(selected) + 1)])
    return selected


def points_from_shape(shape: dict) -> list[tuple[float, float]]:
    points = shape.get("points", [])
    return [(float(point[0]), float(point[1])) for point in points]


def draw_annotation(image: Image.Image, label_data: dict, row: pd.Series, case_index: int) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width = max(3, int(max(base.size) / 220))
    point_radius = max(5, int(max(base.size) / 130))

    for shape in label_data.get("shapes", []):
        label = shape.get("label")
        shape_type = shape.get("shape_type")
        if label not in COLORS:
            continue
        points = points_from_shape(shape)
        if not points:
            continue
        color = COLORS[label]
        if shape_type in {"linestrip", "line"} and len(points) >= 2:
            draw.line(points, fill=color, width=width, joint="curve")
        elif shape_type == "point":
            x, y = points[0]
            draw.ellipse(
                (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
                fill=color,
                outline=(255, 255, 255, 240),
                width=max(1, width // 2),
            )

    annotated = Image.alpha_composite(base, overlay).convert("RGB")
    max_width = 1200
    if annotated.width > max_width:
        new_height = int(annotated.height * max_width / annotated.width)
        annotated = annotated.resize((max_width, new_height), Image.Resampling.LANCZOS)

    header_height = 76
    canvas = Image.new("RGB", (annotated.width, annotated.height + header_height), (18, 18, 18))
    canvas.paste(annotated, (0, header_height))
    header = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    title = (
        f"Case {case_index:02d} | {row['group']} | label={int(row['label'])} | "
        f"A/N={float(row['A_N']):.3f} | row_id={int(row['row_id'])}"
    )
    legend = "Overlay: y=cyan contour, e=yellow contour, B=magenta line, b=red point, Y=green point"
    header.text((14, 14), title, fill=(255, 255, 255), font=font)
    header.text((14, 42), legend, fill=(220, 220, 220), font=font)
    return canvas


def main() -> None:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata: {METADATA_PATH}")

    CASE_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(METADATA_PATH, encoding="utf-8-sig")
    selected = select_cases(metadata)

    manifest_rows = []
    for index, row in selected.iterrows():
        image_path = IMAGE_DIR / row["image_file"]
        label_path = LABEL_DIR / row["label_file"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label: {label_path}")

        with label_path.open("r", encoding="utf-8") as handle:
            label_data = json.load(handle)

        image = Image.open(image_path)
        figure = draw_annotation(image, label_data, row, index + 1)
        output_name = f"case_{index + 1:02d}_{row['group']}_row{int(row['row_id']):03d}_AN{float(row['A_N']):.3f}.png"
        output_path = CASE_DIR / output_name
        figure.save(output_path)

        manifest_rows.append(
            {
                "case_id": f"case_{index + 1:02d}",
                "row_id": int(row["row_id"]),
                "sample_id": row["sample_id"],
                "group": row["group"],
                "label": int(row["label"]),
                "A_N": float(row["A_N"]),
                "image_file": row["image_file"],
                "label_file": row["label_file"],
                "case_figure": f"case_examples/{output_name}",
            }
        )

    pd.DataFrame(manifest_rows).to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(manifest_rows)} case figures to {CASE_DIR}")
    print(f"Wrote manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
