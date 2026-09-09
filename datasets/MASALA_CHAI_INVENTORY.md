# Masala-CHAI Inventory

Local paths:

- Repository: `Masala-CHAI/`
- Downloaded dataset zip: `masala-chai-dataset-new.zip`
- Extracted dataset: `masala-chai-dataset-new/`

## Repository Contents

- `Masala-CHAI/Dataset/`: 1,815 schematic images from `data_1` through `data_5`
- `Masala-CHAI/trained_checkpoints/yolov8_best.pt`: YOLO checkpoint
- `Masala-CHAI/sample-images/`: sample inference images
- `Masala-CHAI/sample-output/`: sample Auto-SPICE outputs
- `Masala-CHAI/analoggenie.jsonl`: generated caption/netlist-style records

## Downloaded Dataset Contents

- `masala-chai-dataset-new/images/`: 6,371 JPG schematic images
- `masala-chai-dataset-new/captions/`: 6,403 caption text files
- `masala-chai-dataset-new/spice/`: 6,069 SPICE netlist text files
- `masala-chai-dataset-new/data_mapping.json`: 6,069 image-caption-SPICE records

Mapping check:

- Records: 6,069
- Missing mapped images: 32
- Missing mapped captions: 0
- Missing mapped SPICE files: 0

## Detection Dataset Status

The downloaded Masala-CHAI dataset does not include ground-truth component bounding-box labels in YOLO, COCO, VOC, or labelme format. It was therefore not merged into `ALL_COMPONENTS.merged.yolov8`, because adding these schematic images as empty-label YOLO samples would teach the detector that visible components are background.

## Practical Use

- Use now for later stages: image-to-caption, image-to-SPICE, graph/netlist validation, and end-to-end evaluation.
- Use for component detection only after one of these steps:
  - obtain the original detection annotations if available,
  - manually label a subset,
  - pseudo-label with `Masala-CHAI/trained_checkpoints/yolov8_best.pt` and review/correct outputs.
