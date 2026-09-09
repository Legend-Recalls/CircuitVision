from __future__ import annotations

import ast
import hashlib
import os
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ALL_COMPONENTS.merged.yolov8"
SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

DATASETS = [
    ROOT / "cghd.yolov8",
    ROOT / "ci2n.device_identification.yolov8",
    ROOT / "digitize_hcd.component_symbols.yolov8",
    ROOT / "SCHEMATIC.merged.yolov8",
    ROOT / "schematic_images.hf.yolov8",
    ROOT / "v3qwe.v2i.yolov8",
]


ALIASES = {
    "-": "unknown",
    "7 segments": "display.7_segment",
    "AC source": "voltage.ac",
    "AC_Source": "voltage.ac",
    "DC source": "voltage.dc",
    "DC_Source": "voltage.dc",
    "Voltage_Source": "voltage",
    "voltage": "voltage",
    "voltage-lines": "voltage.lines",
    "voltage.dc": "voltage.dc",
    "voltage.ac": "voltage.ac",
    "voltage.battery": "voltage.battery",
    "Battery": "voltage.battery",
    "current": "current_source",
    "current source": "current_source",
    "Current_Source": "current_source",
    "dependent current source": "dependent_current_source",
    "dependent voltage source": "dependent_voltage_source",
    "GND": "gnd",
    "Ground": "gnd",
    "gnd": "gnd",
    "vdd": "vdd",
    "vss": "vss",
    "Capacitor": "capacitor",
    "capacitor": "capacitor",
    "capacity": "capacitor",
    "capacitor-3": "capacitor",
    "capacitor.unpolarized": "capacitor",
    "capacitor polarized": "capacitor.polarized",
    "capacitor.polarized": "capacitor.polarized",
    "VARIABLE CAPACITOR": "capacitor.adjustable",
    "capacitor.adjustable": "capacitor.adjustable",
    "Resistor": "resistor",
    "resistor": "resistor",
    "resistor2": "resistor",
    "resistor2_3": "resistor",
    "resistor.adjustable": "resistor.adjustable",
    "variable resistor": "resistor.adjustable",
    "potentiometer": "resistor.adjustable",
    "resistor.photo": "resistor.photo",
    "ldr": "resistor.photo",
    "Inductor": "inductor",
    "inductor": "inductor",
    "inductor-3": "inductor",
    "iron core inductor": "inductor.ferrite",
    "inductor.ferrite": "inductor.ferrite",
    "inductor.coupled": "inductor.coupled",
    "transformer": "transformer",
    "Diode": "diode",
    "diode": "diode",
    "led": "diode.light_emitting",
    "diode.light_emitting": "diode.light_emitting",
    "schottky-zener diode": "diode.zener",
    "diode.zener": "diode.zener",
    "diode.thyrector": "diode.thyrector",
    "BJT": "transistor.bjt",
    "BJT-NPN": "npn",
    "BJT-PNP": "pnp",
    "transistor": "transistor",
    "transistor.bjt": "transistor.bjt",
    "NPN transistor": "npn",
    "npn": "npn",
    "npn-cross": "npn.cross",
    "PNP transistor": "pnp",
    "pnp": "pnp",
    "pnp-cross": "pnp.cross",
    "MOSFET": "mosfet",
    "MOSFET-N": "nmos",
    "MOSFET-P": "pmos",
    "mosfet": "mosfet",
    "transistor.fet": "mosfet",
    "nmos": "nmos",
    "nmos-bulk": "nmos.bulk",
    "nmos-cross": "nmos.cross",
    "pmos": "pmos",
    "pmos-bulk": "pmos.bulk",
    "pmos-cross": "pmos.cross",
    "transistor.photo": "transistor.photo",
    "single-end-amp": "amplifier.single_end",
    "single-input-single-end-amp": "amplifier.single_input_single_end",
    "diff-amp": "amplifier.differential",
    "amplifier": "amplifier",
    "operational amplifier": "operational_amplifier",
    "operational_amplifier": "operational_amplifier",
    "operational_amplifier.schmitt_trigger": "operational_amplifier.schmitt_trigger",
    "Op-Amp": "operational_amplifier",
    "and gate": "and",
    "and": "and",
    "or gate": "or",
    "or": "or",
    "op": "operational_amplifier",
    "func": "dflipflop",
    "not gate": "not",
    "not": "not",
    "nand gate": "nand",
    "nand": "nand",
    "nor gate": "nor",
    "nor": "nor",
    "xor gate": "xor",
    "xor": "xor",
    "xnor gate": "xnor",
    "inverter": "not",
    "opamp": "operational_amplifier",
    "switch": "switch",
    "switch-3": "switch",
    "terminal": "terminal",
    "port": "port",
    "junction": "junction",
    "crossover": "crossover",
    "cross-line-curved": "crossover.curved",
    "Wire Crossover": "crossover",
    "I-AC": "current_source.ac",
    "I-DC": "current_source.dc",
    "V-AC": "voltage.ac",
    "V-DC": "voltage.dc",
    "V-DC (one port)": "voltage.dc.one_port",
    "Zener Diode": "diode.zener",
    "box": "block",
    "block": "block",
    "explanatory": "explanatory",
}


def lp(path: Path) -> str:
    resolved = str(path.resolve())
    return resolved if resolved.startswith("\\\\?\\") else "\\\\?\\" + resolved


def read_text(path: Path) -> str:
    with open(lp(path), "r", encoding="utf-8") as handle:
        return handle.read()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(lp(path), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def parse_names(dataset: Path) -> list[str]:
    text = read_text(dataset / "data.yaml")
    match = re.search(r"names:\s*(\[.*?\])", text, flags=re.S)
    if not match:
        raise ValueError(f"Could not find names in {dataset / 'data.yaml'}")
    return list(ast.literal_eval(match.group(1)))


def sanitize_class(name: str) -> str:
    cleaned = name.strip().replace("/", "_")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.replace("-", "_")
    cleaned = re.sub(r"[^0-9A-Za-z_.]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned.lower() or "unknown"


def canonical_class(name: str) -> str:
    return ALIASES.get(name, sanitize_class(name))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(lp(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    images = []
    with os.scandir(lp(path)) as entries:
        for entry in entries:
            if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_EXTS:
                images.append(path / entry.name)
    return sorted(images, key=lambda p: p.name.lower())


def collect_used_class_ids(dataset: Path) -> set[int]:
    used: set[int] = set()
    for split in SPLITS:
        label_dir = dataset / split / "labels"
        if not label_dir.exists():
            continue
        for label_path in label_dir.glob("*.txt"):
            for raw_line in read_text(label_path).splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    used.add(int(float(stripped.split()[0])))
                except (ValueError, IndexError):
                    continue
    return used


def label_for_image(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lp(src), lp(dst))


def short_output_stem(dataset_key: str, split: str, image: Path) -> str:
    source_id = hashlib.sha1(str(image).encode("utf-8")).hexdigest()[:12]
    dataset_part = sanitize_class(dataset_key)[:28]
    stem_part = sanitize_class(image.stem)[:48]
    return f"{dataset_part}_{split}_{source_id}_{stem_part}"


def remap_label_file(src: Path, dst: Path, class_map: dict[int, int]) -> tuple[int, int]:
    if not src.exists():
        write_text(dst, "")
        return 0, 0
    out_lines = []
    skipped = 0
    for raw_line in read_text(src).splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        try:
            old_id = int(float(parts[0]))
        except ValueError:
            skipped += 1
            continue
        if old_id not in class_map:
            skipped += 1
            continue
        parts[0] = str(class_map[old_id])
        out_lines.append(" ".join(parts))
    write_text(dst, "\n".join(out_lines) + ("\n" if out_lines else ""))
    return len(out_lines), skipped


def main() -> None:
    available = [dataset for dataset in DATASETS if dataset.exists()]
    if not available:
        raise SystemExit("No source datasets found.")

    source_names = {dataset.name: parse_names(dataset) for dataset in available}
    used_class_ids = {dataset.name: collect_used_class_ids(dataset) for dataset in available}
    all_names = sorted(
        {
            canonical_class(source_names[dataset.name][old_id])
            for dataset in available
            for old_id in used_class_ids[dataset.name]
            if 0 <= old_id < len(source_names[dataset.name])
        }
    )
    name_to_id = {name: index for index, name in enumerate(all_names)}
    maps = {
        dataset.name: {
            old_id: name_to_id[canonical_class(name)]
            for old_id, name in enumerate(names)
            if old_id in used_class_ids[dataset.name]
        }
        for dataset, names in ((d, source_names[d.name]) for d in available)
    }

    if OUT.exists():
        shutil.rmtree(lp(OUT))
    for split in SPLITS:
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    stats: dict[str, dict[str, int]] = {
        dataset.name: {
            "images_seen": 0,
            "duplicates_skipped": 0,
            "images_copied": 0,
            "labels_written": 0,
            "label_lines_skipped": 0,
        }
        for dataset in available
    }

    for dataset in available:
        dataset_key = dataset.name
        class_map = maps[dataset_key]
        for split in SPLITS:
            for image in iter_images(dataset / split / "images"):
                stats[dataset_key]["images_seen"] += 1
                image_hash = file_hash(image)
                if image_hash in seen_hashes:
                    stats[dataset_key]["duplicates_skipped"] += 1
                    continue
                seen_hashes.add(image_hash)

                out_stem = short_output_stem(dataset_key, split, image)
                out_image = OUT / split / "images" / f"{out_stem}{image.suffix.lower()}"
                out_label = OUT / split / "labels" / f"{out_stem}.txt"
                copy_file(image, out_image)
                written, skipped = remap_label_file(label_for_image(image), out_label, class_map)
                stats[dataset_key]["images_copied"] += 1
                stats[dataset_key]["labels_written"] += written
                stats[dataset_key]["label_lines_skipped"] += skipped

    data_yaml = (
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        f"nc: {len(all_names)}\n"
        f"names: {all_names!r}\n\n"
        "merged_from:\n"
        + "".join(f"  - {dataset.name}\n" for dataset in available)
        + "merge_notes: all local YOLO detection datasets merged; source class ids remapped to a canonical all-class schema.\n"
    )
    write_text(OUT / "data.yaml", data_yaml)

    mapping_lines = ["source_dataset,source_id,source_name,canonical_id,canonical_name"]
    for dataset in available:
        for old_id, source_name in enumerate(source_names[dataset.name]):
            if old_id not in used_class_ids[dataset.name]:
                continue
            canonical = canonical_class(source_name)
            mapping_lines.append(
                f"{dataset.name},{old_id},{source_name},{name_to_id[canonical]},{canonical}"
            )
    write_text(OUT / "CLASS_MAPPING.csv", "\n".join(mapping_lines) + "\n")

    report = [
        "All YOLO detection dataset merge report",
        f"output: {OUT}",
        f"classes: {len(all_names)}",
        "",
    ]
    total_images = 0
    total_labels = 0
    for dataset_name, dataset_stats in stats.items():
        report.append(dataset_name)
        for key, value in dataset_stats.items():
            report.append(f"  {key}: {value}")
        total_images += dataset_stats["images_copied"]
        total_labels += dataset_stats["labels_written"]
    report.extend(["", f"total images copied: {total_images}", f"total labels written: {total_labels}"])
    for split in SPLITS:
        images = len(iter_images(OUT / split / "images"))
        labels = len(list((OUT / split / "labels").glob("*.txt")))
        report.append(f"{split}: {images} images, {labels} label files")
    write_text(OUT / "MERGE_REPORT.txt", "\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
