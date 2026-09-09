# ALL_COMPONENTS.merged.yolov8

Unified YOLOv8 detection dataset for all available circuit/component symbol classes.

## Included Detection Datasets

- `cghd.yolov8`
- `ci2n.device_identification.yolov8`
- `digitize_hcd.component_symbols.yolov8`
- `SCHEMATIC.merged.yolov8`
- `schematic_images.hf.yolov8`
- `v3qwe.v2i.yolov8`

## Final Counts

- Images: 116,595
- Labels: 1,424,372
- Classes: 106
- Train images: 93,995
- Valid images: 11,642
- Test images: 10,958

## Important Files

- Dataset: `ALL_COMPONENTS.merged.yolov8/`
- Training config: `ALL_COMPONENTS.merged.yolov8/data.yaml`
- Source-to-final class mapping: `ALL_COMPONENTS.merged.yolov8/CLASS_MAPPING.csv`
- Merge report: `ALL_COMPONENTS.merged.yolov8/MERGE_REPORT.txt`

## Conversion Scripts

- `convert_cghd_to_yolov8.py`
- `convert_ci2n_to_existing_format.py`
- `convert_digitize_hcd_to_yolov8.py`
- `convert_schematic_images_hf_to_yolov8.py`
- `merge_all_yolo_detection.py`

## Notes

- Equivalent labels from different sources were mapped into a canonical schema.
- Zero-instance classes were pruned from the final schema.
- Long source filenames were shortened deterministically to avoid Windows path-length issues.
- Digitize-HCD was converted from COCO component annotations to YOLOv8 boxes.
- `schematic_images.hf` was converted from flat Hugging Face ZIP archives into YOLOv8 split directories before merging.
