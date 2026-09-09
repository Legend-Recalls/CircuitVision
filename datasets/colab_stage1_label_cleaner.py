from pathlib import Path
import ast
import argparse
import re
from PIL import Image, ImageOps
import yaml


DEFAULT_SRC_ROOT = Path("/content/circuitvision/ALL_COMPONENTS.merged.yolov8")
DEFAULT_OUT_ROOT = Path("/content/circuitvision/ALL_COMPONENTS.cleaned.rgb.yolov8")
DEFAULT_APPLY_EXIF_TRANSPOSE = True
DEFAULT_FORCE_GRAYSCALE = False
MIN_BOX_WH = 1e-4


def load_names(dataset_root: Path):
    text = (dataset_root / "data.yaml").read_text(encoding="utf-8")
    return ast.literal_eval(re.search(r"names:\s*(\[.*\])", text, re.S).group(1))


def validate_yolo_label_line(line: str, nc: int):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    if len(parts) > 5:
        parts = parts[:5]
    try:
        cls = int(float(parts[0]))
        x, y, w, h = [float(v) for v in parts[1:5]]
    except Exception:
        return None
    if not (0 <= cls < nc):
        return None
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    if w < MIN_BOX_WH or h < MIN_BOX_WH:
        return None
    return f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def clean_dataset(src_root: Path, out_root: Path, apply_exif_transpose: bool, force_grayscale: bool):
    names = load_names(src_root)
    nc = len(names)
    if out_root.exists():
        print(f"Output already exists: {out_root}")
        return

    for split in ("train", "valid", "test"):
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

        for img_path in sorted((src_root / split / "images").glob("*")):
            out_img = out_root / split / "images" / img_path.name
            with Image.open(img_path) as im:
                if apply_exif_transpose:
                    im = ImageOps.exif_transpose(im)
                im = im.convert("L" if force_grayscale else "RGB")
                im.save(out_img)

            src_label = src_root / split / "labels" / f"{img_path.stem}.txt"
            out_label = out_root / split / "labels" / src_label.name
            cleaned = []
            if src_label.exists():
                for raw in src_label.read_text(encoding="utf-8").splitlines():
                    fixed = validate_yolo_label_line(raw, nc)
                    if fixed is not None:
                        cleaned.append(fixed)
            out_label.write_text(
                "\n".join(cleaned) + ("\n" if cleaned else ""),
                encoding="utf-8",
            )

    yaml.safe_dump(
        {
            "path": str(out_root.resolve()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(names),
            "names": names,
        },
        (out_root / "data.yaml").open("w", encoding="utf-8"),
        sort_keys=False,
    )
    print(f"Wrote cleaned dataset to {out_root}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--no-exif-transpose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clean_dataset(
        src_root=args.src,
        out_root=args.out,
        apply_exif_transpose=not args.no_exif_transpose,
        force_grayscale=args.grayscale,
    )
