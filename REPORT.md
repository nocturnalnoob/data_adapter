# Detection Quality Adapter — implementation, calibration, and validation report

Implements `instruction.md` Parts One and Two (generic post-processing adapter,
specialized for hands) and runs the §7 calibration / §8 validation it requires
against the 39-clip corpus in `downloads/`. No hand-level ground truth exists
anywhere in this corpus — every number below is measured off weak labels
mined from the raw data itself (see "Method" in each section), not picked by
inspection, and every place that couldn't be measured reliably is called out
rather than papered over.

**Scope note:** the assignment asked for false-positive rejection only
(§3); false-negative recovery (§4, interpolation) was called out as not
expected and is included below purely as an extension on top of the FP
work. See `README.md` for the scope breakdown and code-to-spec mapping.

## Code map

| File | Stage |
| --- | --- |
| `adapter/reject_geometric.py` | Stages 1–3: duplicate merge, size, shape (+ hand stereo reach, gated off — see below) |
| `adapter/association.py` | Track building: position/motion only, never the reported handedness label |
| `adapter/reject_temporal.py` | Stages 4–6: displacement, unsupported, static (VIO-correlated) |
| `adapter/selection.py` | Post-association cap-to-2, ranked by track support |
| `adapter/interpolate.py` | Gap-fill + leaving-frame test (border-weighted for hands) |
| `adapter/pipeline.py` | Orchestrates the mandated order; `check_completeness` verifies every raw box is accounted for |
| `adapter/calibrate.py` | Mines §7's quantities from the corpus |
| `adapter/validate.py` | §8 per-stage precision/recall + synthetic-dropout + interpolation-rate monitor |
| `scripts/run_pipeline.py`, `scripts/visualize.py`, `scripts/devset.py` | CLI runner, qualitative spot-check renderer, dev-set ranking |

Run `python scripts/run_pipeline.py --all` then `python -m adapter.validate` to
reproduce everything below (outputs land in `adapter_out/`).

## Calibration (§7)

**Method.** A "stable track" — built with an unconstrained pass (dedup only,
generous association gate, no size/shape/speed/static rejection) and then
filtered to tracks ≥30 frames long — stands in for "almost certainly a real
hand." Distributions of size/shape/speed/dropout-length/pool-rank are
measured off these tracks and off the README's observation that any frame
with ≥3 raw boxes holds ≥1 duplicate/bystander/FP.

| Quantity | Result | Used for |
| --- | --- | --- |
| Box area (px²), stable tracks | p1=7.4k, p50=44.1k, p99.5=307k | `min_area_px2`=3.7k, `max_area_px2`=460k (with margin) |
| Aspect ratio, stable tracks | p50=1.20, p99=2.80 | `max_aspect_ratio`=3.08 |
| Per-frame speed (px/frame), stable tracks | p50=9.0, p99.5=119.0 | `max_speed_px_per_frame`=142.8 |
| Dropout length (flicker gaps), stable tracks | p50=2, p90=22, p99=131 (n=1971 gaps) | `max_dropout_frames`=22 |
| True-box rank by raw confidence, ≥3-box frames | p95 = rank 3 (n=8376 samples) | `candidate_pool_width`=3 |
| Stereo disparity, wearer vs. bystander-candidate | see below | **not used** — see finding |

**Finding: the stereo arm's-reach threshold could not be safely calibrated
from this corpus.** The "top-2-longest-track = wearer" weak label, applied to
disparity computed via rectified epipolar template matching (confirmed
rectified: ORB match check gave median `|Δy|≈2px` on a 1200px frame), gives
wearer-labeled boxes a *lower* median disparity (59px) than bystander-labeled
boxes (66px) — backwards from the physical expectation that closer objects
have larger disparity. Best achievable split accuracy is 67%. Read plainly:
in real multi-box frames, "rank-3 candidate" is dominated by near-duplicates
and detector noise sitting at the *same* depth as the real hand, not by
genuine distant bystanders — the case instruction.md §6 flags as needing
stereo depth is comparatively rare in this corpus and gets drowned out by
that noise in a proxy-labeled measurement. `stereo.py`'s disparity computation
works mechanically (see `adapter/stereo.py`, `adapter/calibrate.py:pick_disparity_threshold`),
but the adapter ships with `stereo_enabled=False` rather than a threshold with
67% accuracy dressed up as calibrated. This needs labelled bystander examples,
exactly the instruction's own escape hatch (§6 last row: "needs labelled
examples").

Static-rule and camera-moving thresholds were also measured (not a primary
target per the plan, but needed for the pipeline to be usable):
`camera_speed_moving_thresh`=0.078 m/s and `camera_ang_moving_thresh_deg`=0.34°/frame
(40th percentile of the corpus's VIO speed/angular-rate distributions), and
`static_motion_px_threshold`=2.56px (10th percentile of stable-track speed
during camera-moving frames).

Full numbers: `adapter_out/calibration_report.json`. Final config:
`adapter_out/calibrated_config.json`.

## Validation (§8)

**Per-stage precision/recall (proxy).** Using the same stable-track weak
label as ground truth, averaged over all 39 clips:

| Stage | Survivors are "real" (proxy precision) | "Real" boxes retained (proxy recall) |
| --- | --- | --- |
| raw | 0.998 | 1.000 |
| after geometric (dedup/size/shape) | 0.998 | 0.986 |
| after temporal (displacement/unsupported/static) | 0.998 | 0.951 |
| after cap | 0.999 | 0.945 |

Recall drops monotonically as the rules get stricter, which is the direction
§8 requires for rejection stages. Precision barely moves because it's
already ≈1 at "raw" — a ceiling effect from the proxy label itself
(long-track membership already excludes most obvious junk before any rule
runs), not evidence the rejection stages aren't doing anything; the
multi-box-frame check below is the more informative precision signal.

**Multi-box frame resolution (label-free where it counts).** Across all
4,504 ≥3-box frames corpus-wide, the adapter brings the surviving count down
to ≤2 in 97.0% of them (worst individual clips still resolve 87.5%–92.3%;
see `adapter_out/validation_report.json` `multibox_resolution` per clip) —
via the track-support cap alone, with stereo disabled. A qualitative spot-check
(`adapter_out/viz/887c633e_t028/frame_00194.png`, `frame_00495.png`) shows a
genuine bystander's hand rejected while both wearer hands are kept, purely
because the bystander's track has less support than the two dominant tracks
— i.e. the candidate-pool-and-cap-by-track-support mechanism (§3) is
carrying real bystander-rejection weight on its own, which somewhat softens
the loss of the stereo signal for the common case, though not the guaranteed
per-frame case §5/§6 call for.

**Frame-coverage recall / interpolation's contribution.** On the dev-set
clip, recall (fraction of stable-track frames covered by *some* surviving
final detection) is 90.0% without interpolation and 91.4% with it — a +1.4pp
recall gain attributable specifically to trajectory recovery.

**Interpolation-rate monitor (fully label-free, §8's standing check).**
Mean 5.6% across the corpus, std 5.5%. One clip flagged (`beb348be_t000`,
20.3% interpolated). Its task label is `repair_backpack` — fine manual
manipulation (stitching/handling small parts) is exactly the scenario
instruction.md §6 calls "motion blur during rapid movement," which is
*supposed* to interpolate. The monitor did its job (flagged for review); it
isn't itself proof of a bug, and this specific case reads as a legitimate
high-dropout task rather than fabrication.

**Synthetic dropout injection (needs no ground truth — see method above).**
Real, contiguous observed spans on selected tracks were blanked for `k`
frames and re-interpolated; IoU against the withheld real boxes measures
recovery quality directly:

| gap length k | recovery rate | mean IoU | mean center error (px) |
| --- | --- | --- | --- |
| 2 | 100% | 0.976 | 1.5 |
| 5 | 100% | 0.938 | 4.6 |
| 10 | 73%* | 0.897 | 8.3 |
| 22 (= calibrated `max_dropout_frames`) | 93%* | 0.809 | 25.1 |
| 32 | 0% | — | — |
| 44 | 0% | — | — |

*rates at k=10/22 are noisy at n=15 trials; the qualitative pattern — quality
degrading smoothly with gap length, then a hard cutoff exactly at
`max_dropout_frames` — is the meaningful result and confirms the calibrated
cutoff is doing its job (refusing to fabricate across gaps longer than what
was calibrated as a safe flicker length).

## Known limitations (measured or explicitly flagged, not guessed)

- **Stereo arm's-reach is disabled**, per the finding above. Bystander
  rejection currently relies entirely on the track-support cap, which works
  when a bystander's track is weaker than both wearer tracks but has no
  guarantee in general (e.g. a bystander present as persistently as the
  wearer). Needs labelled bystander examples to fix properly.
- **Detector class-confusion is out of scope.** A foot detected as a "hand"
  by WiLoR and tracked continuously (visible in
  `adapter_out/viz/887c633e_t028/frame_00495.png`, `reported#0`) passes
  every rule because it behaves exactly like a real, stable hand-shaped box.
  This adapter corrects tracking/temporal errors on boxes the detector
  already labeled as hands; it cannot fix the detector's own
  object-identity mistakes.
- **Static rule is intrinsics-free by necessity** (no calibration data in
  the corpus — checked, no baseline/focal/intrinsic field in any
  `meta.json`). It uses box-motion magnitude gated on VIO ego-motion rather
  than a full parallax/depth-aware reprojection, so it will under-flag
  background objects during pure-translation camera motion with no
  rotation (rare for a head-mounted rig, per instruction.md §5, but not
  zero).
- **Gloved/partially-occluded hands** — instruction.md §6's own last row:
  behavior unmeasured, needs labelled examples. Nothing in this corpus's
  metadata distinguishes these cases, so no attempt was made to handle them
  specially.
- **The static rule over-concentrates on sustained-low-hand-mobility desk
  tasks.** `fcaf46ee_t001` (`copy_documents`) and `6cd0b236_t000`
  (`calculate_data`) reject far more than typical (752/2673 and 889/5362
  detections respectively), and in both clips essentially every rejection is
  `static` (735/752 and 889/889 — checked directly against
  `adapter_out/corrected/*.json`). Both clips' VIO speed medians (0.085,
  0.066 m/s) sit at or above the calibrated "camera moving" threshold
  (0.078 m/s), so the camera is nominally moving most of the time by that
  measure, while the wearer's hands stay resting on a printer control panel
  or calculator for extended stretches — a real violation of §5's
  "a hand is never stationary in the image for a sustained period" design
  assumption for desk-bound fine-motor tasks, not a threshold-tuning bug.
  The interpolation-rate monitor (§8, label-free) doesn't catch this class
  of problem, because it measures fabrication, not over-rejection; there's
  no equivalent standing check for a rejection-rate spike, and adding one
  was out of scope for this pass.
- **`reject_unsupported`'s reading of "no track before or after it" (§3 row
  5) is isolation of the *same* track's own continuation, not proximity to
  a different nearby track.** The literal text is ambiguous between the
  two. Corpus-wide there are 105 `unsupported` rejections total (a small
  fraction of ~185k detections); 33 of them land within 5 frames of a
  different surviving track's start/end. Spot-checking is consistent with
  these being genuine short-lived noise near a real hand's onset/offset
  rather than wrongly-split continuations of the same object (association
  and the displacement rule run before this stage and would have merged a
  true continuation if the position matched), but the alternative reading
  would suppress rejection near any nearby track regardless of identity,
  which is a different, stricter policy. Flagging the interpretation choice
  rather than silently picking one.
