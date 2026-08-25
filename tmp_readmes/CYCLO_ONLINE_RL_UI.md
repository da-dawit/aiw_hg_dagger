# Addendum: revert the gate, run GR00T on the robot, add an Online-RL Data UI

**For: the agent on `ffw-snpr48a1106`. Supersedes parts of `CYCLO_HIL_INTEGRATION.md`.**
**Date: 2026-08-21.**

Three changes, in this order. Read section 1 first — it undoes work, and the sooner it is undone the
less there is to unpick.

---

## 1. Revert the freeze/gate work to the original state

**Stop maintaining `arm_freeze_gate`, `safe_publish()`, and `/arm_freeze/ready`.** Restore the
teleop path to stock ROBOTIS.

This is not a judgement on the code — the `safe_publish()` limiter in particular was correct, and it
found a real bug on our side. It is that the whole subsystem is now redundant:

- **Teleop with freeze already exists.** ROBOTIS ships it, it is proven on this hardware, and it has
  the slow-start ramp we were reimplementing.
- **Recording already exists.** Cyclo Intelligence's UI does segmented recording with subtask
  structure.
- A custom relay between the leader and the controllers is the highest-risk component in the system
  and it is the one thing we do not need to own. It nearly broke the robot once.

**What to revert:**
- the gate node and its arbitration between `/leader/...` and `/policy/...`
- `safe_publish()` and its rate limiter
- `/arm_freeze/status` and `/arm_freeze/ready`
- the `/policy/...` topic indirection, if the stock path publishes directly

**What to keep:** the *findings*, written down somewhere durable. They are still true and we will hit
them again:
- the single-threaded rclpy relay deserialising ~400 msg/s it discarded, which queued our 18–21 Hz
  stream — that is why a Python relay in the control path is a bad idea, and why C++ was the honest
  fix
- `/leader/dynamic_joint_states` at 100 Hz being the heaviest message in the graph
- the easing flag latching true when the source went silent mid-ease
- the measured `18–21 Hz` command-stream rate — we still need this number

**Where the intervention signal comes from now.** `CYCLO_HIL_INTEGRATION.md` section 3.1 listed
`/arm_freeze/status` and `/arm_freeze/ready` as two of three sources. Those are going away, so the
label derives from **`InferenceStatus.inference_phase`** alone:

```
inference_phase == PAUSED   -> human driving
inference_phase == INFERENCING -> policy driving
```

Everything else in that document still stands — the per-frame column, the converter touch points, the
`-1` fallback, the `tasks.parquet` trap. The `-1` state still exists and still matters: whatever the
stock teleop does during its slow-start ramp must be excluded, so section 3 of this document adds a
UI-driven way to mark it.

---

## 2. Run GR00T on the robot with TensorRT

Move inference onto the robot to cut the round trip. **Most of this is already built — do not write
a TensorRT integration.**

### 2.1 What already exists

`interfaces/msg/TaskInfo.msg` already carries:

```
string acceleration_mode        # "" / "pytorch" = normal. "tensorrt_dit" = GR00T DiT-only TensorRT
string acceleration_engine_path # empty -> <model_path>/dit_model_bf16.trt
uint16 control_hz
uint16 inference_hz
float64 chunk_align_window_s
```

`InferenceCommand.srv` carries the same two fields at `LOAD` time. So the plumbing from UI → service
→ backend is done. Cyclo also already models the control-rate / inference-rate split, which is the
thing that actually matters (see 2.3).

The engine builder ships too:

```
cyclo_brain/policy/groot/Isaac-GR00T/scripts/deployment/build_tensorrt_engine.py

python scripts/deployment/build_tensorrt_engine.py \
    --mode full_pipeline \
    --onnx-dir ./gr00t_n1d7_onnx \
    --engine-dir ./gr00t_n1d7_engines \
    --precision bf16
```

It builds ViT, LLM, State Encoder, Action Encoder, DiT, and Action Decoder engines; shape profiles
are derived from the ONNX automatically.

### 2.2 The model

```
HF repo:   dawity/groot_screwing35   (PRIVATE)
path:      checkpoints/014400/pretrained_model
size:      12.58 GB
```

**Ask Dawit to run the download himself, or to provide a token he is willing to place on this
machine.** Do not ask for his token in chat and do not copy one from another host — he has declined
that before and it was the right call.

Note `acceleration_mode="tensorrt_dit"` accelerates the **DiT only**. Our checkpoint is GR00T
N1.7-3B: a frozen 1.52 B backbone and a 1.62 B action head, with the DiT running **4 denoising steps
per inference** against one backbone pass. So DiT-only acceleration targets the part that runs 4×,
which is the right first move — but it is not the whole model, and it will not turn 300 ms into 30 ms
on its own. Measure before and after and report both numbers.

### 2.3 Latency is not the thing to optimise — stalling is

Read this before tuning anything.

With receding-horizon chunking, one plan of `execute_steps` waypoints at `control_hz` covers
`execute_steps / control_hz` seconds of motion. At 25 waypoints and 30 Hz that is **833 ms**. An
inference taking 300 ms fits inside that window with room to spare — **as long as it happens
concurrently with execution**.

If inference is synchronous in the control loop, the loop stalls 300 ms every 833 ms. That is not
"slightly slower", it is a visible hitch three times a second, and it looks exactly like a bad
policy. `TaskInfo.action_request_mode` already exposes this:

```
"async"  request the next chunk while the current buffer is still executing   <- use this
"sync"   wait until the buffer is empty                                        <- stalls
```

**Use `async`.** Getting this right is worth more than TensorRT. Do it first, measure, then add
TensorRT and measure again — otherwise you cannot tell which change did what.

### 2.4 Report back

- inference latency: PyTorch vs `tensorrt_dit`, on the robot
- achieved `control_hz` under each, measured against a wall clock, not assumed
- whether `async` was already the default

---

## 3. An "Online-RL Data" section in the UI

### 3.1 Do the three methods need different formats? Mostly no — and that matters

Short answer: **one recorded format serves all three**, provided it captures two things Cyclo does
not capture today. It is worth being precise about why, because it decides how much UI to build.

| method | what it consumes | derivable from a per-frame dataset? |
|---|---|---|
| **HG-DAgger** | `(obs, action)` on human-driven frames, runs of ≥40 | yes — filter on the intervention label |
| **gate-as-potential (ours)** | `(obs, action)` on every frame **plus** who drove | yes — the label *is* the extra signal |
| **HIL-SERL** | transitions `(s, a, r, s', done)` | yes, **if** reward and outcome are recorded |

HIL-SERL looks like a different format because it is off-policy RL over transitions rather than
supervised learning over frames. But `s'` is just the next frame's state and `done` is the episode
boundary, so transitions can be *derived* offline from a per-frame dataset. What cannot be derived is
**reward**. There is no success signal anywhere in the recording today — I checked every
`interfaces/msg` and `interfaces/srv`, and there is no outcome, success, or reward field.

So the gap is two fields, not three formats:

1. **per-frame intervention label** — `CYCLO_HIL_INTEGRATION.md` section 3 (`0` / `1` / `-1`)
2. **per-episode outcome, plus when success occurred** — new, described below

Record both and all three methods are served by one dataset. Please do not build three recorders.

### 3.2 Why a separate UI section rather than a flag on Record

The *format* is nearly the same; the *workflow* is not. Normal teleop recording is: human drives,
save. Online-RL recording is:

```
policy drives  ->  human sees a failure  ->  human takes over  ->  human corrects
               ->  hands back            ->  episode ends      ->  human labels the outcome
```

That last step has no equivalent in teleop recording and it is mandatory for HIL-SERL. The operator
also needs to see, live, which of the three states is being recorded — otherwise a whole session can
be collected with everything labelled wrong and look completely normal. Hence a dedicated section.

### 3.3 What to build

The UI is React + Redux Toolkit, pages in `orchestrator/ui/src/pages/`, feature slices in
`orchestrator/ui/src/features/<name>/<name>Slice.js`. Follow the existing pattern:

```
orchestrator/ui/src/pages/OnlineRLPage.js
orchestrator/ui/src/features/onlineRL/onlineRLSlice.js
+ a nav entry in NavigationPage.js
```

`RecordPage.js` and `InferencePage.js` are the two to read first — this page is a composition of
both, because a run is an inference session that is being recorded.

**Live display, while a run is going:**
- current state, large and unambiguous: **POLICY** / **HUMAN** / **EXCLUDED**
- a running count of frames in each of the three states, so the operator sees at a glance whether the
  labels look sane before saving
- current subtask and its instruction (`RecordingStatus` already publishes these)
- measured control rate and inference latency

**Controls:**
- start / stop the run — `InferenceCommand.START` / `STOP`
- takeover and hand back — `InferenceCommand.PAUSE` / `RESUME`. Reuse these; do not invent a
  parallel mechanism.
- advance subtask — `InferenceCommand.UPDATE_INSTRUCTION`
- **mark excluded**, a manual `-1` toggle. With the gate gone there is no automatic easing signal, so
  the operator needs a way to say "this stretch is not a demonstration" — during the slow-start ramp,
  or after bumping something. Cheap to add and it is the safety valve for everything we cannot
  detect automatically.
- **dry-run switch** — `publish_to_robot=false` / `inference_mode="simulation"`. Default it to
  **off**, so the first click of a new session never moves the arm.

**At episode end, a mandatory outcome prompt:**
- `SUCCESS` / `FAILURE` / `DISCARD`
- for `SUCCESS`, the frame index where the task was achieved — this is the sparse reward, and
  without it HIL-SERL has nothing to learn from
- free-text note, optional

Make it **blocking**. An episode saved without an outcome is unusable for HIL-SERL and will not be
noticed until training.

### 3.4 Where the outcome is stored

Two additions, mirroring how `subtask_index` is already handled (see `CYCLO_HIL_INTEGRATION.md`
section 3.2 for the exact line numbers):

- **per-frame** `intervention` — `int8`, in the episode parquet
- **per-episode** `outcome` and `success_frame` — in `meta/episodes/`, alongside the existing
  per-episode metadata. Do **not** put a constant per-episode value in the per-frame parquet.

### 3.5 Order of work

1. per-frame `intervention` column, converter side (the previous document)
2. outcome capture and storage
3. the UI page

Send us **one converted episode after step 2**, before building the UI. If the columns are wrong, the
UI is built on sand — and a wrong column is invisible until training.

---

## 4. Summary of what changed

| | before | now |
|---|---|---|
| teleop / freeze | custom `arm_freeze_gate` | **stock ROBOTIS — revert** |
| intervention source | `/arm_freeze/{status,ready}` | `InferenceStatus.inference_phase` |
| excluded (`-1`) marking | automatic from easing flag | **manual toggle in the UI** |
| recording | our `online_rl.py` | Cyclo's recorder |
| inference | workstation RTX 4090 | **on the robot, `tensorrt_dit`** |
| outcome / reward | not recorded | **new, mandatory prompt** |

Unchanged and still required: the per-frame column, the `-1` fallback on length mismatch, the
`tasks.parquet` index check, and the wrist-orientation check. Those are in
`CYCLO_HIL_INTEGRATION.md` and none of them are superseded.
