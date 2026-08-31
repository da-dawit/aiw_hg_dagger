# Start here — documents for the robot-side agent

**Updated 2026-08-21.** Read in this order. Anything not listed under "Live" is dead — do not
implement from it.

---

## Live — read these three, in order

### 1. `RECORDING_MUST_FIX.md`  ← read first
Four requirements that decide whether a recorded session is usable at all. Every one of them fails
**silently** — the recording completes, the UI looks fine, the dataset converts, and the problem only
appears at training. Ends with a copy-paste acceptance test to run on **one** episode.

Read this before writing recorder code, not after.

### 2. `CYCLO_HIL_INTEGRATION.md`
The converter work: adding a per-frame `intervention` column to Cyclo's rosbag → LeRobot v3.0
pipeline. Exact file and line touch points in `base_converter.py` and `to_lerobot_v30.py`, using the
existing `subtask_index` column as the template. Also the `tasks.parquet` index trap and the wrist
orientation check.

*Superseded in one place:* section 3.1 lists `/arm_freeze/status` and `/arm_freeze/ready` as label
sources. Those are being reverted — the label now comes from `InferenceStatus.inference_phase`. See
doc 3 section 1.

### 3. `CYCLO_ONLINE_RL_UI.md`
Three things: revert the custom gate back to stock ROBOTIS teleop; run GR00T on the robot with
TensorRT (**already plumbed** — `acceleration_mode="tensorrt_dit"` plus the shipped engine builder);
and add the Online-RL Data page.

---

## Reference — accurate, but not a plan

### `RUN_ON_ROBOT.md`
Written when we thought we would port `online_rl.py` to the robot. We are not — Cyclo runs inference
instead. Sections 1–3, 7 and 11 no longer apply.

Still worth reading:
- **§4 loop rate** — more relevant now, not less. Inference latency on the robot is higher, and a
  control loop that stamps commands with a deadline shorter than its real tick interval produces
  sprint-and-wait jitter that looks exactly like a bad policy.
- **§5 camera orientation** — wrists are portrait 424×240 in the training set, landscape from the
  camera.
- **§8 control constants** — `MAX_ACC 2.0`, `MAX_VEL 0.6`, `EXECUTE_STEPS 25`, all measured. Changing
  them without re-measuring regresses the robot.
- **§9 known-bad things** — mistakes already made once each. `pkill -f` matching your own shell is in
  there; it has bitten twice.
- **§10 the right-arm dive** — still unexplained. Your effort data (−1.0/−0.9/−0.5 on
  `arm_r_joint1..3` while 4–7 tracked, against 460 on the stalled lift) is the best evidence so far
  and points away from table contact.

---

## Dead — do not implement

| file | why |
|---|---|
| `GATE_ARBITRATION_REQUEST.md` | the gate is being reverted |
| `GATE_MODE_SWITCH_BUG.md` | same |
| `ROBOT_SIDE_CONTRACT.md` | the `/policy/...` contract; Cyclo owns this path now |
| `SESSION_SUMMARY_FOR_ROBOT_AGENT.md` | superseded by this index |

The gate work was not wasted — its **findings** are still true and are preserved in
`CYCLO_ONLINE_RL_UI.md` section 1. Keep those; drop the code.

Older docs in this directory dated 2026-08-14 (`HIL_THEORY.md`, `QUEST_VS_PICO.md`, `SETUP.md`,
`STARTUP.md`, `AIWORKER_DAGGER_PORT.md`, `HIL_PIPELINE.md`, `OKAMI_NOTE.md`) are from the
Quest/Pico teleop phase and are not relevant to this task.

---

## The one-paragraph version

Cyclo Intelligence already does teleop, freeze, segmented recording, inference PAUSE/RESUME, mid-run
language re-conditioning, and TensorRT. We are not rebuilding any of it. The gap is that Cyclo
records *what happened* but not *who was driving*. Add one per-frame `int8` column
(`0` policy / `1` human / `-1` excluded) plus a per-episode outcome, expose it through a new
Online-RL Data page that attaches to a session started on the Inference page, and all three methods —
HG-DAgger, gate-as-potential, HIL-SERL — are served by one dataset.

**Do not collect a full session until the acceptance test in doc 1 passes on a single episode.**
A wrong intervention column is invisible until training.
