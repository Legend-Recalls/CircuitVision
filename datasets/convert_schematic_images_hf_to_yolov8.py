import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "schematic_images.hf"
IMAGES_ZIP = SRC / "images.zip"
LABELS_ZIP = SRC / "components.zip"
CLASS_FILE = SRC / "comoponent_id.txt"
OUT = ROOT / "schematic_images.hf.yolov8"
SPLITS = ("train", "valid", "test")
AUTHORITATIVE_NAMES = [
    "gnd",
    "pmos",
    "nmos",
    "pnp",
    "npn",
    "resistor",
    "capacity",
    "voltage",
    "current",
    "diode",
    "inductor",
    "and",
    "or",
    "xor",
    "not",
    "func",
    "op",
    "tgate",
]


def reset_output():
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in SPLITS:
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)


def load_names():
    names = [
        line.strip()
        for line in CLASS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # The published class file is missing entries, but the linked Netlistify repo
    # uses the full 18-class order for this dataset.
    if names == [
        "gnd",
        "pmos",
        "nmos",
        "pnp",
        "npn",
        "resistor",
        "capacitor",
        "voltage",
        "current",
        "diode",
        "inductor",
        "and",
        "xor",
        "inverter",
        "dflipflop",
        "opamp",
        "tgate",
    ]:
        return AUTHORITATIVE_NAMES
    return names


def split_for_stem(stem: str) -> str:
    bucket = int(hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "valid"
    return "test"


def main():
    for path in (IMAGES_ZIP, LABELS_ZIP, CLASS_FILE):
        if not path.exists():
            raise SystemExit(f"Missing source file: {path}")

    names = load_names()
    reset_output()

    stats = {
        "images_copied": 0,
        "labels_copied": 0,
        "missing_labels": 0,
        "label_lines": 0,
    }
    split_counts = {split: 0 for split in SPLITS}

    with zipfile.ZipFile(IMAGES_ZIP) as images_zip, zipfile.ZipFile(LABELS_ZIP) as labels_zip:
        label_members = {
            Path(member.filename).stem: member
            for member in labels_zip.infolist()
            if not member.is_dir() and member.filename.lower().endswith(".txt")
        }
        image_members = [
            member
            for member in images_zip.infolist()
            if not member.is_dir() and Path(member.filename).suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        for image_member in sorted(image_members, key=lambda item: item.filename.lower()):
            image_name = Path(image_member.filename).name
            stem = Path(image_name).stem
            split = split_for_stem(stem)
            split_counts[split] += 1

            out_stem = f"schematic_images_hf_{stem}"
            out_image = OUT / split / "images" / f"{out_stem}{Path(image_name).suffix.lower()}"
            out_label = OUT / split / "labels" / f"{out_stem}.txt"

            with images_zip.open(image_member) as src, out_image.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            stats["images_copied"] += 1

            label_member = label_members.get(stem)
            if label_member is None:
                out_label.write_text("", encoding="utf-8")
                stats["missing_labels"] += 1
                continue

            with labels_zip.open(label_member) as src:
                label_text = src.read().decode("utf-8")
            out_label.write_text(label_text, encoding="utf-8", newline="\n")
            stats["labels_copied"] += 1
            stats["label_lines"] += sum(1 for line in label_text.splitlines() if line.strip())

    data_yaml = (
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        f"nc: {len(names)}\n"
        f"names: {names!r}\n\n"
        "source: schematic_images.hf\n"
        "conversion_notes: Hugging Face flat images/components ZIPs converted to YOLOv8 split directories.\n"
    )
    (OUT / "data.yaml").write_text(data_yaml, encoding="utf-8")

    report = [
        "schematic_images.hf to YOLOv8 report",
        f"source: {SRC}",
        f"output: {OUT}",
        f"classes: {len(names)}",
        f"images copied: {stats['images_copied']}",
        f"labels copied: {stats['labels_copied']}",
        f"missing labels: {stats['missing_labels']}",
        f"label lines: {stats['label_lines']}",
        f"train images: {split_counts['train']}",
        f"valid images: {split_counts['valid']}",
        f"test images: {split_counts['test']}",
        "names: " + json.dumps(names),
    ]
    (OUT / "CONVERSION_REPORT.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
