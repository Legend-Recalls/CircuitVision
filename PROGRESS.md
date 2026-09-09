# CircuitVision progress log (execution record — appends only)

## 2026-09-08/09 — Stage 1 detector: DONE, presentable

- Weights: `control/runs/round3_68_v1/best.pt` (21.9 MB, 68 classes),
  then `control/runs/stage2_v1/best_stage2.pt` (22 MB, 1024px finetune).
- Scores: mAP50 **0.665**, mAP50-95 **0.485**, P/R 0.69/0.69 (was 0.556/0.444 on
  stale 92-class mix). Converged flat — detector at coarse ceiling.
- Decisions (frozen): taxonomy 106 → 92 (remap, pre-existing) → 71 → **68**
  (dropped 21 tail classes + `dflipflop`/`tgate` zeros + `unknown` noise class).
  Freeze tables: `control/runs/taxonomy_v1|v2|v3/`.
- Dataset call: schematic_images.hf 100k OUT (off-mission bulk), CGHD back IN.
  Training set ~13.6k images. Published once as versioned dataset
  `circuitvision-focused69` (16,595 imgs + labels, nc=69) — all later runs
  attach it, zero rebuilds. Breakdown: `control/runs/count_v1/`.
- Infra findings (all fixed in `control`): Kaggle API pushes only
  `code_file` (self-contained driver builder); `/kaggle/src` read-only;
  API default GPU is P100 + cu128-torch = incompatible → `machine_shape:
  NvidiaTeslaT4` gives 2×T4; kernels self-auth via ambient token.
- Negative result (paper-worthy): 100-epoch 92-class backbone ≈ COCO init
  beyond ~15 epochs on this task (recorded in `finetune69_v1/notes.md`).

## 2026-09-09 — Stage 3 pin grounding: v1 DONE, v2 flaws fixed

- `graph_schema.py`: component/pin/net contract + validation + IoU dedup.
- `pin_templates.py`: 11 template kinds, all 68 classes covered exactly once
  (verified programmatically).
- Demo on real SMPS schematic (off-domain CAD): 115 dets → **113 comps**,
  **205 pins**, graph validates.
- Flaw fixes:
  1. Duplicates → cross-class IoU>0.5 merge (measured: only 2 true stacks,
     npn/pnp variant confusion; visual clutter was label rendering, not boxes).
  2. Orientation → wire-pixel scoring with 2× margin gate (0 random flips;
     proven on adversarial synthetic cases).
  3. IC pins → border wire-stub detection (through-counts both sides of the
     border kill text-stroke FPs): 13 stubs on IC1's 16 pins vs pitch guess.
- Known gaps (not hidden): detector misses large packages (IC1 KA7500B,
  transformers T1/T2 — recall gap, detector-side); stub finder untested on
  true IC boxes (no IC detections to test on); transistor roles still default
  orientation unless margin is decisive.
- Next: Stage 4 wire/net extraction (classical CV on masked interiors),
  which consumes these pins + stubs.

## 2026-09-09 � vertical-pins bugfix (user-caught)

- Bug: axial templates branched on aspect AND applied rotation -> double-rotation put horizontal pins on vertical parts. Fixed: canonical-horizontal + rotation only; callers own orientation.
- Tie-break bug: score sort preferred largest orientation on exact ties -> canonical-first now.
- Verified: 4/4 synthetic wire cases + live tall box (1011,212,1035,265) -> top/bottom pins on leads (see demo/outputs/vert_debug/).


## 2026-09-09 � repo merge (user call)

- Single root: circuitvision/ with control/ (was kaggle-control) + datasets/ (was Datasets).
- All cross-refs rewritten (.. /kaggle-control -> control/, roadmap paths fixed); demo re-ran green post-move (113 comps, 205 pins).
- Casualty: README.md clobbered once by a bad shell one-liner, reconstructed + improved.


## 2026-09-09 - pin localisation overhaul (user-caught)

- Bug: axial symmetry tie in score_orientation (0==180 and 90==270 produced margin=1.0 identically), causing 35 components (capacitors, diodes, inductors) to default to aspect ratio and land sideways.
- Bug: BJT3 assumed pins at (0.5, 0.0) and (0.5, 1.0) instead of schematic leads (~0.85); square BJT boxes flipped to 90 on sub-pixel height noise, rotating collector/base/emitter 90 degrees.
- Bug: IC stubs required wire penetration inside package body, killing valid boundary stubs.
- Fixed:
  1. detect_edge_wires() scans outside boundary strips to locate exact 1D lead coordinates.
  2. score_orientation() scores orthogonal axes directly (H vs V) with decisive margin. Flipped 40 misoriented components.
  3. pins_for(cls, bbox, orient, dark=dark) snaps pin positions directly onto real wire strokes.
  4. Ground symbols locked to canonical top lead; BJT3 lead geometry corrected.
- Metrics: pin-on-wire accuracy jumped from 78.5% (161/205) to 99.5% (204/205). Graph validates.


## 2026-09-09 � Stage 3 v3: edge-wire snapping (merged + independently verified)

- External analysis correctly diagnosed: strict margin gate fired 0 flips while ~36 axial parts disagreed with aspect; verified by measurement (not trust).
- Merged: detect_edge_wires (outside-only strips), axis vote for axial, canonical transistors (c-top/b-left/e-bottom at nx 0.85), ground lock, snap-to-wire in pins_for; demo gate 1.5/0.1.
- Independent measure (measure_pins.py, same 113 boxes): pin-on-wire 178/205 (86.8%) -> 205/205 (100%), flips 40.
- Honesty note: 100% is partly self-consistency (snapping moves pins onto the map it is scored against). True correctness needs Stage 4 graph + constraints. Claimed 78.5% baseline differs by metric definition only.
- Visual check passes: Q1/Q2/Q3/Q4/Q7 upright triplets, IC1 bottom-edge stub pins, resistor/diode leads capped.


## 2026-09-09 � Stage 4 wire/net extraction v1: DONE

- wire_nets.py: mask interiors -> CC fragments (790) -> junction-dot merge (117 dots) -> 527 nets; pins assigned 192/205 (93.7%), 13 isolated flagged for Stage 7.
- Topology rules honored: dot-merge only; un-dotted crossings stay split; text dies by area filter.
- SMPS overlay (demo/outputs/smps_nets.jpg): rails/grounds/branches trace correctly by eye.
- Known noise: title text colored (unmasked text fragments); 300+ pin-less nets need pruning in Stage 5; 13 isolated pins (magenta rings) await correction.


## 2026-09-09 � net overlay readability fix

- Problem: overlay colored all 527 fragments incl. text noise. Fixed: color only the 65 pin-bearing nets; rest stays grayscale; N-index labels on 12 largest nets.
- Caught own bug mid-fix: union-find roots vs compact indices confused (blank overlay); fixed + verified visually.


## 2026-09-09 � maze routing rescue (user idea)

- User proposal: pixel routes + maps-style pathfinding for wires. Implemented as Dijkstra/Lee expansion (A* with zero heuristic � goal location unknown) over unmasked wire map, gap_cost 12, windowed.
- Result: 13/13 isolated pins bridged -> 205/205 assigned (100%), 0 isolated. Flood-fill finds regions; routing crosses the gaps it cannot.
- Honest note: bridges are cost-capped (~15px gaps); longer breaks correctly stay isolated for Stage 7.

## 2026-09-09 - anomaly rescue v2 (user call: abnormality check before Stage 5)

- User spotted undetected components in wires-only view. Built the asked layer: wires are linear, so abnormalities = missed components.
- anomaly_rescue.py v2: (A) dead-end endpoints min_pts 3->2 (a missed axial leaves exactly 2 ends), grid 64->80, box_margin 30->50; (B) thick-blob detector (5x5 opening kills thin wires; survivors minus dots/boxes = filled symbols; measured 8/9 unexplained = ground bars); (C) corroborated low-conf harvest (full-image at 0.15, keep novel IoU<0.5 only on endpoint/blob/wire-density abnormality, gates hallucinations).
- second_pass gains pad=30 context (tight clusters cut symbol edges).
- SMPS: 113 base -> 23 proposals (15 endpoint + 8 thick) -> 39 zoom + 19 harvest = 58 rescued -> 171 comps, 301 pins, 83 nets, isolated 3 (maze-bridged 54). Breakdown: gnd 18, capacitor 9, port 9, diode 6, resistor 5, transformer 3, inductor 3, + singles.
- Outputs: demo/outputs/smps_rescue.jpg (orange proposals, lime rescued), smps_wires_rescued.jpg (re-masked wires Stage 4 now sees).
- Honest noise: antenna 1 is hallucination; tiny port boxes may be dots; some ground doubles (zoom+harvest IoU<0.5 both kept) - Stage 5 pruning fodder.
- Remaining gap (model-blind, not proposal-blind): IC1 KA7500E, power circles Q1/Q2/Q3/Q4 D209L/C1815, transformer T1 coils - no fire even zoomed at 0.2. Needs loop-detector + 0.05/TTA crops, or accept for Stage 7.

## 2026-09-09 - loop-detector + Stage 5 constraint validator: DONE, verified

- Loop-detector (loop_detector.py): contour loops (closed outlines unboxed) = 0 proposals, correctly - Q cans already boxed, coils/IC have no closed outlines on the wire map. IC footprint rows (Hough pin-circle lines, both orientations): 1 footprint at IC1 (269,854,628,1174, 5 pins). Classifier blind there (ports/noise only even at 0.05+TTA) -> recorded as structural candidate for Stage 7, not merged.
- Stage 5 constraints.py: all 6 roadmap checks with PASS/FLAG/MAP verdicts. Verified via stage5_verify.py on SMPS (171 comps, 301 pins, 86 nets):
  1. two_terminal: 103 ok, 0 shorted, 2 OPEN (cap 132, vdc 146 - true wire breaks, Stage 7); 5 PARALLEL pairs (R2/D15+D16 antiparallel/LED1/C147 on shared live nets - legit, info only).
  2. transistor_roles: 6 ok, 3 dangling pins (T2/T1 p1, bjt 128 base - breaks, Stage 7).
  3. sources_live: 13 ok, 2 floating (vdc 146 break; port 163 noise box on a dot - Stage 7 prune).
  4. ground_collapse: MAP solved, 27 raw nets -> GND, 0 floating gnd pins.
  5. crossing_no_dot: PASS. Zhang-Suen skeleton (skeleton_x.py, 2s) -> 93 X-points partitioned 61 dotted-T / 30 symbol-geometry (coils, pin circles, cans, lead corners) / 2 text-touch / 0 unexplained. Mechanism unit-tested (split->PASS, merged->FLAG). T<=3 arms = connection by drafting convention; only bare 4-arm X's are hop candidates.
  6. shorts_islands: 0 rail shorts; 17 single-pin island nets + 5 dangling pins fully itemized with owners -> Stage 7 worklist.
- Validator bugs caught and fixed during verification (all re-verified): maze rescue faked 25 SHORTED by conducting through part bodies (sibling-net exclusion in wire_nets.extract_nets via comp_ids); title-text dot FPs; T-vs-X confusion; Hough recall on masked map (circles now found on unmasked minus validated dots).
- Stage 7 handoff: 2 opens + 3 dangling + 1 noise port + 17 islands + IC1 structural synthesis. No rail shorts. Netlist stage unblocked.

## 2026-09-09 - Stage 5 completeness audit (user asked: sure it's all?)

- Namespace scare, checked empirically: merge_map/island net ids (e.g. 448, 496) look out of range next to 86 pin-bearing nets, but net_index compacts ALL fragment-nets (557 incl. pin-less text/dead copper), so ids are valid compact labels in full space. No bug; merge_map directly usable for Stage 8 GND collapsing.
- Coverage sweep: all 68 classes now sit in >=1 check. Two gaps found and closed: AMP1 (amplifier.single_end, 2-pin in/out) added to TWO_T; new 7th check crossover_junction (hop-over boxes must contain no dot; junction glyphs must sit on a dot) - SMPS: 1 crossover examined, 0 junctions, PASS.
- Deliberately NOT added (documented): duplicate-box audit would only re-list known rescue doubles (IoU 0.3-0.5, Stage 7's first dedup pass); pin-on-copper audit is subsumed by isolated/dangling + cost-capped bridging.
- Final Stage 5: 7 checks, all verdicts definitive, all flags owned. DONE.

## 2026-09-09 - Stage 6 semantic motifs: DONE, verified

- motifs.py on electrical nets (Stage-5 GND map applied; without it nothing matches). Shared pipeline extracted to build_graph.py (stage5_verify reuses it, output byte-identical).
- SMPS yield: 9 dividers, 4 decoupling, 3 RC-lowpass, 0 highpass, 1 common-emitter (Q2), 0 follower, 4 parallel diode pairs. Matcher precision fixes: tap!=GND (163 junk), tap/node pin-count<=5 (rail-hub combinatorics), antiparallel renamed parallel_pair (a/k polarity unresolvable from templates - honesty).
- Photo-verified: divider R28/R27 tap -> Q7 base (bias divider, textbook); dual-diode pack = parallel pair; Q2 common-emitter (R8 pull-up, emitter on ground-return rail 217, base driven). Structurally verified: node-386 network (R105 feed + C56 shunt + R1||R25 = RC + divider, REF bias area). Unloaded divider taps (327/467/424) point at the missing IC1 - consistent with the footprint.
- rank_repairs(): all 22 troubled pins get scored live-net candidates (divider-tap/fat-rail bonus + distance), e.g. T2-p1 -> net 96 @18px; port163 -> no candidates (prune signal). Stage 7 consumes the ranking.
- Known limit: no OCR/refdes, so instances are coords not R-numbers; follower 0 and highpass 0 are true negatives on this sheet (no such stages), not recall failures.

## 2026-09-09 - Stage 7 correction + Stage 8 netlist: DONE (Milestone 1 end-to-end)

- correct.py, roadmap order (geometry -> hard constraints -> motif priors, no ML needed): P1 pruned noise port163 (rescued, all pins troubled, zero candidates); P2 ground homing (no moves needed - gnd islands already GND by merge); P3 motif-attach applied 11 reassigns (score>=1.0, dist<=80, own-net excluded); P4 synthesized IC1 (footprint + 13 terminal circles, 13/13 on nets via overlap-based _frag_at: pin copper beats nearby rails - proximity alone had piled 4 pins onto ground rail 235); P5 no-op (no overmerges/true shorts).
- Pixel-truth caveat (photo-verified): the 6 top-row IC pins share one copper bus in the image (pin-row guide line) - synthesis is pixel-faithful; shared-bus warning logged. Validator bugs fixed en route: P4 attached-counter, ghost flags from pruned pins (reindex output), rank suggesting own island net.
- Before/after (Milestone-2 lift): two_terminal FLAG(2 open)->PASS; transistor_roles 3->2 dangling (no-candidate only); sources_live FLAG->PASS; islands 17->7; dangling 5->2 (T115.p1, bjt128.b: no candidates within radius, correctly left flagged).
- netlist.py -> demo/outputs/smps.sp: 126 elements, 53 nodes, 0 bad lines, coverage 120 modeled + 51 node-labels = 171/171 comps. Transformers as coupled L+K, IC1 as X stub, dangling pins as FLT_* (2). Stubs = 2 FLT + 8 single-pin island nodes (dead copper, faithful). ngspice absent on this machine - simulation check not run (stated, not skipped silently). Values are placeholders (Milestone-4 OCR work).

## 2026-09-09 - polished tools: PySpice parses, LTspice solves OP (SKiDL deferred)

- Referenced instead of reinvented (user call): SKiDL (graph->netlist+ERC), PySpice (netlist parse), LTspice (free, Windows-native sim; installed via winget 26.0.2.1). Note: ngspice is free/open-source (GPL), not paid - but LTspice fits this box better anyway.
- PySpice SpiceParser (real grammar, replaces regex as external check): 124 elements + 6 models + ICSTUB subcircuit. Two upstream limits hit and bisected: (1) `SMOD OFF` suffix IndexErrors in non-first position -> suffix dropped (ngspice/LTspice default OFF); (2) CRLF line endings + Rser kwarg unsupported by PySpice Netlist builder (parse-only role; LTspice is the sim role). File now LF.
- LTspice batch OP on smps_lt.cir: SOLVED in 0.068s, warnings only (floating island stubs N129/N249/FLT_212 - all known Stage-5 islands). En route LTspice caught OUR bugs: X-vs-subckt pin miscount (comment tokens in pin split) and P3-shorted transformer winding (sibling exclusion added to P3; new transformer_windings check: T114/T115 p2+p3 SHORTED on 151 pre-existing at T2 secondary, p0+p1 OPEN left dangling honestly). Inductor Rser=10m added per LTspice's own prescription.
- SKiDL deferred with reason: needs KiCad symbol libs (no KICAD_SYMBOL_DIR on box; TEMPLATE + from-scratch both fail at lib lookup). Its ERC ground (unconnected pins, shorts) is already covered by Stage 5. Revisit at Milestone 4 (needs KiCad libs; then: graph->SKiDL Circuit for ERC + DOT output).

## 2026-09-09 - single generic pipeline (user call: any image, no per-circuit scripts)

- Deleted wien_pipeline.py/make_drive.py. New `pipeline.py <image> [--sim] [--ltspice]`: detect->pins->nets->validate->motifs->correct->plot->netlist for ANY image, outputs in demo/outputs/<stem>/. Verified: same entry on smps.jpg reproduces known results.

## 2026-09-09 - cir1 (user image): PCA pins + Otsu + Req=10 end-to-end proof

- All 10 resistors detected, 0 rescue needed. User spotted diagonal pins off-lead -> PCA axis (pca_axis + _axis_exits in pin_templates): R9 45.4deg, R4 143deg, H/V parts untouched (diagonal branch only if >20deg off-axis, ratio>=2.5). Opens healed, two_terminal PASS, flt=0.
- binarize is Otsu now (smps 138 / wien 156 / cir1 173): fixed-128 dropped whole thin runs on cir1. SMPS drift accepted with evidence: rescued 58->54 (2 caps + 1 port at the <=0.41 noise floor), rest identical; fragment-direct SHORTED-vs-PARALLEL replaced the <24px proxy (P6 removed with it).
- Req proof: printed values assigned, 1A A->B, LTspice OP = 10.000000 ohm = hand nodal analysis. Fixed en route: sim-deck node names missing N prefix (ITEST/VVCC/.ic on phantom nodes).
- Crossing check: 92/93 explained + T1-terminal (1192,875) photo-verified benign twice; FLAG with documented disposition, not a failure.

## 2026-09-09 - LTspice reconstruction from graph (user: see it in the viewer)

- asc_gen.py (pipeline-native): symbols at detection boxes (fixed LTspice sizes, R0/R90-CCW proven from Hartly.asc), stubs pin->symbol, FLAG net labels (our names, GND->0), Manhattan MST wire trees per net, refdes match netlist.py exactly. asc_render.py draws the same .asy art LTspice shows.
- Verification: LTspice -netlist incidence vs our pin->net table. cir1: 20/20 match with routed trees. Bugs killed: net-name prefix (N13 vs 13), float coords + zero-length wires HANG the netlister (integers only), title overlapping copper (moved below max extent).
- cir1.asc opens in the viewer as the ladder + diagonals with N-labels; cir1_ltspice.png is the same geometry previewed.

## 2026-09-09 - asc spacing + one-to-one stubs (user: components too close)

- gen_asc spread=1.6 about centroid (relative layout preserved, verification-safe). Fixed en route: two symbol pins claiming one our-pin stranded the other (R3.2 read NC_01) -> greedy one-to-one assignment; re-verified 20/20 with routed trees.

## 2026-09-09 - readable reconstruction: rails, stubs, labels (user: messy wires)

- Net-aware rail snap (same-net coords cluster to median; cross-net can never merge -> no false shorts), Manhattan stubs (no diagonal scars), FLAG every our-pin (one-per-net left symbol pins banking on tree fragments that MST had deduped away).
- Caught by the verifier each time, fixed, re-verified 20/20: float coords + zero-length wires hang the netlister (integers only); title-overlap moved below extent; render font-size token bug.
- cir1.asc is straight rails + labeled parts, viewable and simulatable as-is.

