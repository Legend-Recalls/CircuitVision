# Project Roadmap: Schematic Image to Netlist

## Papers Reviewed

- `papers/2405.09045v2.pdf` (`AMSNet`)
- `papers/autospice.pdf` (`Auto-SPICE`)
- `papers/masalachai.pdf` (`Masala-CHAI`)
- `papers/SINA_ A Circuit Schematic Image-to-Netlist Generator Using Artificial Intelligence - 2601.22114v1.pdf` (`SINA`)
- `papers/OmniSch_ A Multimodal PCB Schematic Benchmark For Structured Diagram Visual Reasoning - 2604.00270v1.pdf` (`OmniSch`)
- `papers/Executive Summary.pdf`
- `papers/Circuit Vision Project.pdf`
- Supporting local notes: `papers/README.md`, `papers/goals.txt`, `papers/circuitvision.md`, `papers/Pasted markdown.md`

## What The Papers Already Cover

### AMSNet

- Main value: dataset creation for transistor-level AMS schematics and schematic-to-netlist pairing.
- Strong point: makes textbook schematics usable for downstream learning.
- Weak point: GPT-level understanding of topology is still poor; paper itself reports much weaker net accuracy than component accuracy.

### Auto-SPICE

- Main value: fully automated dataset extraction pipeline using object detection, Hough priors, prompt tuning, and verification.
- Strong point: useful recipe for harvesting textbook schematics into SPICE examples.
- Weak point: still LLM-heavy and still brittle on topology.

### Masala-CHAI

- Main value: largest open-source SPICE-oriented corpus among the papers you collected.
- Strong point: strong for netlist-text supervision and fine-tuning language models.
- Weak point: heavy reliance on GPT-style generation and verification; topology validity remains a bottleneck.

### SINA

- Main value: strongest practical image-to-netlist pipeline among the papers here.
- Strong point: good decomposition: YOLO for components, CCL for wiring, OCR/VLM for labels.
- Weak point: it is still mostly a perception pipeline; it verifies labels/counts more than deep circuit semantics.

### Netlistify

- Main value: strong synthetic-data-driven pipeline with component detection, orientation classification, and connectivity modeling.
- Strong point: best evidence that a structured multi-stage pipeline beats direct VLM guessing.
- Weak point: centered on AMS components and synthetic setup; less emphasis on symbolic correction and semantic circuit reasoning.

### OmniSch

- Main value: benchmark paper, not a production pipeline.
- Strong point: shows current LMMs still fail on structured schematic grounding, graph parsing, and geometry-aware reasoning.
- Weak point for your purposes: it tells you what not to rely on; it does not solve the problem.

## Clear Research Gap

The strongest gap across these papers is not plain object detection anymore.

The gap is:

1. Reliable topology recovery after detection.
2. Constraint-based correction of wrong or incomplete graphs.
3. Semantic circuit reasoning above the raw graph.
4. A unified evaluation that measures not only detection, but also graph correctness, netlist validity, and simulation success.

Put differently:

- Existing work is strong on dataset harvesting.
- Existing work is improving on perception.
- Existing work is weak on symbolic validation, correction loops, and circuit-level reasoning.

That is the gap your project should own.

## Recommended Project Positioning

Do **not** position your project as:

- "another YOLO model for components"
- "an LLM that reads schematics"
- "a generic multimodal reasoning demo"

Position it as:

**A neuro-symbolic schematic-to-netlist system that combines vision, graph construction, circuit constraints, and semantic correction.**

Short version for a paper title:

**Constraint-Aware Schematic-to-Netlist Generation with Semantic Graph Correction**

## Scope You Should Choose

### Phase-1 Scope

Target one tractable domain first:

- Analog and mixed-signal textbook-style schematics
- SPICE-compatible output
- Component values optional at first
- OCR/reference designators optional in first milestone

### Do Not Start With

- Full handwritten circuits plus real photos plus PCB schematics all at once
- General-purpose VLM reasoning
- Rich textual explanation generation

Start narrow, then expand.

## How To Use Your Datasets

### 1. `ALL_COMPONENTS.merged.yolov8`

Use for:

- broad component detection pretraining
- canonical symbol vocabulary
- robustness to many symbol styles

Do not use it as your main end-to-end netlist benchmark because it does not provide full netlist ground truth.

### 2. `schematic_images.hf.yolov8`

Use for:

- component detection on analog-style synthetic circuits
- topology recovery experiments
- orientation-aware modeling
- end-to-end graph extraction

This should be your main visual benchmark for the first full pipeline.

### 3. Hugging Face `schematic_images` raw assets (`images.zip`, `components.zip`, `pkl.zip`, `sp.zip`)

Use for:

- net-level supervision
- graph supervision from `pkl`
- SPICE-level supervision from `sp`
- end-to-end evaluation

This is your highest-value dataset for actual schematic-to-netlist work.

### 4. Masala-CHAI assets

Use for:

- text/netlist supervision
- retrieval examples
- semantic circuit pattern mining
- optional LLM fine-tuning after your symbolic pipeline exists

This should not be your first dependency for topology extraction.

### 5. CGHD / CI2N / other hand-drawn or validation datasets

Use for:

- robustness testing
- domain shift experiments
- later-stage generalization claims

Do not mix them into the first milestone if they slow down topology work.

## The System You Should Build

### Stage 1: Canonical Detection Layer

Input:

- schematic image

Output:

- component boxes
- component class
- confidence
- optional orientation

Implementation:

- fine-tune YOLOv8/YOLO11 on the merged canonical schema
- train a separate orientation classifier only for orientation-sensitive symbols if needed

Goal:

- high component recall without exploding the ontology

### Stage 2: Pin Grounding Layer

Input:

- detected components

Output:

- pin coordinates for each component

Implementation:

- begin with rule templates per symbol class
- use orientation-aware pin templates
- only move to keypoint models if templates fail materially

This stage is critical. Most papers underemphasize explicit pin grounding, but topology quality depends on it.

### Stage 3: Wire / Net Extraction Layer

Input:

- image with masked component interiors

Output:

- wire segments
- junctions
- candidate nets

Implementation:

- start with classical CV: thresholding, thinning, connected components, line merging
- use `pkl` and `sp` supervision to evaluate net reconstruction
- only introduce learned wire segmentation if the classical approach becomes the bottleneck

### Stage 4: Graph Construction Layer

Build a hierarchical graph:

- component nodes
- pin nodes
- junction/net nodes
- edges: `component -> pin`, `pin -> net`

Do not directly connect components to components. You want pin-level structure so correction is possible.

### Stage 5: Constraint Validator

This is the first real research contribution layer.

Examples:

- resistor/capacitor/inductor should connect to exactly 2 nets
- transistor terminals must map to valid pin roles
- voltage/current source cannot float
- duplicate ground symbols should collapse into one electrical ground
- wire crossing without junction marker should not imply connectivity
- short circuits and isolated islands should be flagged

### Stage 6: Semantic Reasoning Layer

This is your second major contribution layer.

Operate on the recovered graph, not raw pixels.

Start with a small library of motifs:

- current mirror
- differential pair
- common source / common gate / common drain
- inverter
- transmission gate
- RC low-pass / high-pass
- voltage divider
- bias network

Use these motifs to:

- detect implausible local topology
- resolve ambiguous edges
- rank candidate corrections

### Stage 7: Graph Correction Engine

This is where your project becomes paper-worthy.

Correction actions:

- attach dangling pin to nearest plausible net
- split over-merged net clusters
- remove impossible pin-to-net assignments
- choose between ambiguous crossings
- recover missed component instances if graph constraints strongly imply one

Order of correction:

1. hard geometric consistency
2. hard circuit constraints
3. motif-level semantic priors
4. optional ML/GNN fallback

### Stage 8: Netlist Generation + Verification

Output:

- SPICE netlist

Verification:

- syntax parse
- structural validity checks
- optional ngspice run
- compare node/component structure to ground truth `sp`

The simulator should be a validator, not the primary reasoning engine.

## What To Build First

### Milestone 1

Use only `schematic_images` and produce:

- component detections
- pin grounding
- wire/net graph
- raw SPICE netlist

Success criterion:

- end-to-end netlist generation works on a clean subset

### Milestone 2

Add correction:

- constraint validator
- graph repair passes
- before/after metrics

Success criterion:

- clear lift in graph correctness and netlist validity over the uncorrected baseline

### Milestone 3

Add semantic reasoning:

- motif recognizer
- semantics-guided correction

Success criterion:

- measurable improvement on ambiguous or failure-case circuits

### Milestone 4

Generalize:

- evaluate on merged detection datasets and robustness sets
- add OCR/reference designators/values

## Baselines You Should Compare Against

### Baseline A: Detection + Naive Connectivity

- components from detector
- simple nearest-wire / CCL connectivity
- no correction

This is your minimum baseline.

### Baseline B: SINA-like Classical Pipeline

- detector
- CCL
- OCR if available
- no symbolic correction

### Baseline C: LLM/VLM Direct Netlist Generation

- prompt a multimodal model with the schematic
- use the paper-reported setup or simplified local analogue

This baseline is important mainly to show why direct multimodal reasoning is not enough.

### Baseline D: Optional Learned Graph Completion

- use a GNN only after graph extraction
- compare against your rule-based correction layer

## Metrics You Should Report

Do not stop at mAP.

### Detection

- mAP50
- per-class F1
- orientation accuracy

### Topology

- pin localization error
- net assignment F1
- connection edge F1
- node clustering quality
- graph edit distance to ground truth

### Netlist

- exact netlist match where feasible
- component match rate
- connectivity match rate
- SPICE syntax validity
- simulation success rate

### System

- runtime per image
- failure mode breakdown
- ablations of correction and semantics

## The Paper Claim You Can Defend

If executed well, your strongest defensible claim is:

**Symbolic constraint validation and semantic graph correction improve schematic-to-netlist accuracy beyond perception-only and LLM-heavy baselines.**

That is stronger than claiming:

- better YOLO detection
- better generic reasoning
- better OCR

## Concrete 8-Week Plan

### Week 1

- Freeze project scope and class subset.
- Build benchmark split from `schematic_images`.
- Define graph representation and evaluation scripts.

### Week 2

- Train/finalize component detector baseline.
- Build orientation-aware pin template library.

### Week 3

- Implement classical wire/net extraction.
- Produce first graph builder.

### Week 4

- Generate first raw netlists.
- Build evaluation pipeline against `pkl` / `sp`.
- Measure baseline failure modes.

### Week 5

- Implement hard constraint validator.
- Add first correction passes.

### Week 6

- Add semantic motif recognizer.
- Add semantics-guided correction and ranking.

### Week 7

- Run ablations and baseline comparisons.
- Test domain shift on other datasets.

### Week 8

- Lock experiments.
- Build figures/tables.
- Draft paper around the correction/semantic contribution.

## What Not To Waste Time On

- Full LLM fine-tuning before your graph pipeline works
- Fancy UI before the evaluator exists
- Huge ontology expansion before pin/net correctness is stable
- Claiming real-world PCB generalization from textbook-style analog data

## Immediate Next Actions

1. Freeze the first benchmark to `schematic_images` plus its `pkl` and `sp` supervision.
2. Define one canonical graph schema for all later modules.
3. Build the uncorrected end-to-end baseline first.
4. Instrument evaluation before adding semantic logic.
5. Make correction and semantic reasoning the paper contribution, not object detection alone.


---

## Execution Log (appended 2026-09-09, do not edit above)

### Stage 1 detector: DONE (presentable)

- Final weights: ../control/runs/round3_68_v1/best.pt -> stage2: ../control/runs/stage2_v1/best_stage2.pt (69->68 classes).
- Scores: mAP50 0.665, mAP50-95 0.485, P/R 0.69/0.69. Converged flat; detector at coarse ceiling.
- Taxonomy frozen at 68 (dropped 21 tail + dflipflop/tgate zeros + unknown). Tables in ../control/runs/taxonomy_v1|v2|v3/.
- Dataset: schematic_images.hf 100k OUT (off-mission bulk), CGHD back IN. Published once as versioned dataset circuitvision-focused69 (16,595 imgs, nc=69); later runs attach it.
- Source breakdown (116,595 imgs): schematic_images.hf 100k / v3qwe 6910 / cghd 3269 / schematic.merged 2945 / ci2n+digitize 3471. See ../control/runs/count_v1/notes.md.
- Negative result: 100-epoch 92-class backbone ~= COCO init beyond ~15 epochs here.

### Stage 3 pin grounding: v1 DONE, v2 flaws fixed

- Code: ../ (graph_schema.py contract, pin_templates.py 11 kinds covering all 68 classes, demo_smps_v2.py).
- SMPS demo (off-domain CAD): 113 components -> 205 pins, graph validates.
- Fixes: cross-class IoU>0.5 dedup; wire-pixel orientation scoring with 2x margin gate (0 random flips); IC border wire-stub detection (13/16 stubs on IC1, text-stroke FPs filtered by through-counts).
- Known gaps: detector misses large packages (IC1, T1/T2); stub finder validated on hand boxes only; transistor roles default without decisive margin.
- Full narrative: ../PROGRESS.md.

