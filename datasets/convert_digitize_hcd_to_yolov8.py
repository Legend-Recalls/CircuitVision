import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "Digitize-HCD Dataset" / "Component Symbol and Text Label Data"
OUT = ROOT / "digitize_hcd.component_symbols.yolov8"
IMAGE_DIR = SRC / "Circuit Diagram Images"
ANNOTATIONS = SRC / "component_annotations.json"
SPLITS = (("train", 0.80), ("valid", 0.10), ("test", 0.10))
SEED = 42


def reset_output():
    if OUT.exists():
        shutil.rmtree(OUT)
    for split, _ in SPLITS:
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)


def split_images(images):
    rng = random.Random(SEED)
    shuffled = images[:]
    rng.shuffle(shuffled)
    train_count = int(len(shuffled) * SPLITS[0][1])
    valid_count = int(len(shuffled) * SPLITS[1][1])
    return {
        "train": shuffled[:train_count],
        "valid": shuffled[train_count : train_count + valid_count],
        "test": shuffled[train_count + valid_count :],
    }


def yolo_line(annotation, image, category_to_yolo):
    x, y, width, height = [float(value) for value in annotation["bbox"]]
    img_width = float(image["width"])
    img_height = float(image["height"])
    x = max(0.0, min(img_width, x))
    y = max(0.0, min(img_height, y))
    width = max(0.0, min(img_width - x, width))
    height = max(0.0, min(img_height - y, height))
    if width <= 0.0 or height <= 0.0:
        return None
    x_center = (x + width / 2.0) / img_width
    y_center = (y + height / 2.0) / img_height
    return (
        f"{category_to_yolo[annotation['category_id']]} "
        f"{x_center:.6f} {y_center:.6f} "
        f"{width / img_width:.6f} {height / img_height:.6f}"
    )


def main():
    if not ANNOTATIONS.exists():
        raise SystemExit(f"Missing annotation file: {ANNOTATIONS}")
    if not IMAGE_DIR.exists():
        raise SystemExit(f"Missing image directory: {IMAGE_DIR}")

    data = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    categories = sorted(data["categories"], key=lambda item: item["id"])
    names = [category["name"] for category in categories]
    category_to_yolo = {category["id"]: index for index, category in enumerate(categories)}
    images = sorted(data["images"], key=lambda item: item["id"])
    image_by_id = {image["id"]: image for image in images}
    annotations_by_image = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    reset_output()
    splits = split_images(images)
    stats = {
        "images_converted": 0,
        "missing_images": 0,
        "labels_written": 0,
        "invalid_boxes": 0,
    }

    for split, split_images_ in splits.items():
        for image in split_images_:
            src_image = IMAGE_DIR / image["file_name"]
            if not src_image.exists():
                stats["missing_images"] += 1
                continue
            out_name = f"digitize_hcd_{src_image.name}"
            shutil.copy2(src_image, OUT / split / "images" / out_name)
            lines = []
            for annotation in annotations_by_image.get(image["id"], []):
                line = yolo_line(annotation, image_by_id[annotation["image_id"]], category_to_yolo)
                if line is None:
                    stats["invalid_boxes"] += 1
                    continue
                lines.append(line)
            (OUT / split / "labels" / f"{Path(out_name).stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )
            stats["images_converted"] += 1
            stats["labels_written"] += len(lines)

    data_yaml = (
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        f"nc: {len(names)}\n"
        f"names: {names!r}\n\n"
        "source: Digitize-HCD Dataset/Component Symbol and Text Label Data\n"
        "conversion_notes: COCO component symbol boxes converted to YOLOv8 boxes.\n"
    )
    (OUT / "data.yaml").write_text(data_yaml, encoding="utf-8")
    report = [
        "Digitize-HCD component symbols to YOLOv8 report",
        f"source: {SRC}",
        f"output: {OUT}",
        f"classes: {len(names)}",
        f"images converted: {stats['images_converted']}",
        f"labels written: {stats['labels_written']}",
        f"missing images: {stats['missing_images']}",
        f"invalid boxes: {stats['invalid_boxes']}",
        f"train images: {len(splits['train'])}",
        f"valid images: {len(splits['valid'])}",
        f"test images: {len(splits['test'])}",
        "names: " + json.dumps(names),
    ]
    (OUT / "CONVERSION_REPORT.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
