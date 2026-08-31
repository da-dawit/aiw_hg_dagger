# Deploying `16d_30k/checkpoint-30000` — what changed since the last model

For the robot agent. Read the **Critical** section before running anything: two of the changes
fail *silently* — the policy loads, produces plausible actions, and is wrong.

Checkpoint: `dawity/groot_screwing35` → `16d_30k/checkpoint-30000` (public, 6.44 GiB, 15 files).
Verified intact: 1031/1031 tensors readable, `global_step: 30000`, LR annealed to 3.04e-13.

---

## What did NOT change

**Nothing about the robot, the cameras, or the control loop.**

- Same 3 cameras, same topics, same resolutions — `spec_sg2.CAMERAS` is unchanged:
  `cam_left_head 672x376`, `cam_left_wrist 240x424`, `cam_right_wrist 240x424`
- Same 16 controlled joints — `spec_sg2.MODEL_JOINTS` was already 16 and still is
- Same action semantics: absolute joint targets, 30 Hz
- Same velocity/acceleration clamps — **do not touch these, they are safety, not tuning**

So no rewiring, no recalibration, no new topics.

---

## Critical — two things that fail SILENTLY if you skip them

### 1. The image crop is a CODE PATCH, not part of the checkpoint

The model was trained with a **per-camera crop** that does not exist in a clean Isaac-GR00T
checkout. If you run without it, every frame gets letterboxed instead, the model sees an input
distribution it never trained on, and you get a quietly worse policy with no error message.

Required on the robot:

```bash
# a) apply the patch (adds class CropToSquare to image_augmentations.py)
cd <Isaac-GR00T>
git apply /path/to/crop_to_square.patch      # or: patch -p1 < crop_to_square.patch
grep -q "class CropToSquare" gr00t/model/gr00t_n1d7/image_augmentations.py || exit 1

# b) both env vars, in the SAME shell that runs inference
export GR00T_SQUARE_CROP=1
export GR00T_ROI_JSON=/path/to/roi_hybrid.json
```

`roi_hybrid.json` must ship alongside the checkpoint. What it does:

| camera | crop | padding |
|---|---|---|
| `cam_left_head` | full frame 672x376 → padded to 672² | 44% (letterbox, full field of view kept) |
| `cam_left_wrist` | bottom 240x240 of 240x424 | 0% |
| `cam_right_wrist` | bottom 240x240 of 240x424 | 0% |

The head keeps its full view because 50% of its motion energy lives in the side bands a crop
would delete. The wrists are cropped because their upper half is ceiling (88% of the frame at the
grasp instant). Each crop must be square; the squares need not match — the trailing resize
normalises all three to 256x256 before they are stacked.

**Verify before trusting it.** The ROI is matched by aspect ratio and stored as fractions, so it
survives a resolution change but NOT an aspect-ratio change. If a camera is ever swapped for a
different aspect, this config is invalid and the model must be retrained.

### 2. The instruction strings changed — they now name the arm

Byte-exact strings. The tokenizer sees these bytes; a paraphrase is a different input.

```
0  Grab the orange bolt with the left arm
1  Place the orange bolt into the hole with the left arm
2  Grab the driver with the right arm
3  Screw in the bolt by pushing down with the right arm
4  Return both arms to home
```

`infer.py --subtasks` reads them from `<--data>/meta/subtasks.parquet` (falling back to
`tasks.parquet`, where the string lives in the **index**, not a column). Point `--data` at a
dataset whose meta carries the NEW strings — `datasets/screwing35_follower` has them. Using an
old dataset silently feeds the previous sentences to a model trained on the new ones.

---

## Travels with the checkpoint — do NOT set these by hand

Already inside `processor_config.json`; overriding them will break the match with training:

```
crop_fraction        1.0      (random crop OFF)
use_percentiles      False    (mean/std, not q01/q99)
letter_box_transform False
shortest_image_edge  256      -> model input is 256x256
random_rotation_angle 0
color_jitter_params  {b .3, c .4, s .5, h .08}   (train-time only)
```

---

## Run it

```bash
export GR00T_SQUARE_CROP=1
export GR00T_ROI_JSON=/path/to/roi_hybrid.json

python3 aiworker_deploy/infer.py \
  --ckpt /path/to/16d_30k/checkpoint-30000 \
  --data /path/to/datasets/screwing35_follower \
  --subtasks \
  --router-ip <robot ip>
```

`--subtasks`: ENTER advances the stage, BACKSPACE goes back. **Keep using this.** There is ~1.2 s
of motionless arm labelled into the start of every subtask in the training data (4.8% of frames),
so the policy can dwell at a subtask boundary. Nudging past it is the intended workaround; that
was a deliberate decision, not an oversight.

**Dry-run first.** Never send `--live` on a path that has not been dry-run on this exact code.

---

## What to expect, and what these numbers are not

Held-out episodes 31–34, open-loop, same protocol for both models:

| | previous model | this model |
|---|---|---|
| Grab bolt (left) | 16.3 mm | **14.4 mm** |
| Grab driver (right) | 22.7 mm | **13.3 mm** |
| systematic fraction (right) | 0.87 | **0.40** |

The right-arm improvement (−41%) and the collapse in *systematic* error are the meaningful part:
before, nearly all the error was a fixed bias — the signature of stalling in the same place every
time. Now most of it is variance.

**These are open-loop, teacher-forced numbers.** Every predicted chunk starts from the human's
recorded observation, so they cannot show error accumulation, which is what closed-loop control
actually suffers from. They say the policy predicts better actions *given good states*. They do
not say it works on the robot. **This run is the first real test.**

If it still stalls short, capture `--trace` on one failed grasp — that is the one measurement that
separates "the policy commands short" from "the arm does not arrive", and it has never been taken.

---

## Rollback

The previous model is `follower/checkpoint-30000` in the same repo. It is **22-dim** and needs the
old `modality.json`; loading it against the current 16-dim dataset raises `KeyError: state.head`.
It also expects letterboxing, so unset `GR00T_SQUARE_CROP` if you fall back to it.
