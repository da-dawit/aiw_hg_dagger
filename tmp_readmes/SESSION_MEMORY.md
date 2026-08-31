# Session memory — FFW SG2 / A2 leader, HG-DAgger + GR00T inference

Author of this note: Claude (session assistant), preserving work for Dawit Chun before a stop.
Date: 2026-08-25.

## How to port this back once the robot is back

Every repo below got ONE new branch, forked directly off the original branch's HEAD at the time
of the stop, with everything committed on top in one commit. Nothing on the original branch was
touched. That means porting back is a plain, conflict-free fast-forward as long as the original
branch hasn't moved:

```
git checkout <original branch>      # feature-a2 / main / feature-leader-interface
git merge <new branch>              # fast-forwards cleanly if original branch is unchanged
```

If the original branch *has* moved by the time you're back, use `git rebase <original branch>`
on the new branch first, or `git cherry-pick <commit>` for just the one commit — there's only one
commit per repo, so a rebase has exactly one thing to replay. The exact branch name and commit SHA
for each repo is printed in the terminal output right after these branches were created; re-run
`git log --oneline -5 <branch>` in each repo if you need to find it again.

This session spans **three separate git repos**. Each was forked to a new branch off its
original HEAD so the original branch is untouched (colleagues keep working from it normally):

| repo | original branch (untouched) | new branch with this session's work |
|---|---|---|
| `/home/robotis/ai_worker` (host) | `feature-a2` | see branch noted in that repo's own commit |
| `/home/robotis/cyclo_intelligence` (host) | `main` | see branch noted in that repo's own commit |
| `cyclo_teleoperation` (git repo *inside* the `ai_worker` docker container, at `/root/ros2_ws/src/cyclo_control/cyclo_teleoperation`, remote `github.com/ROBOTIS-GIT/cyclo_control.git`) | `feature-leader-interface` | see branch noted in that repo's own commit |

Run `git log --oneline -1` on the new branch in each repo to get the exact commit. To get back to
the clean original state in any of the three, just `git checkout <original branch>` — nothing on
those branches was touched.

---

## 1. Architecture diagram (`ai_worker/tmp_readmes/hg_dagger_architecture.tex`)

Full HG-DAgger / online-RL system architecture, LaTeX+TikZ, compiles to a 3-page PDF via
`docker run --rm --network none -v "$PWD":/work -w /work texlive/texlive:latest pdflatex ...`.

- Page 1: overall architecture (containers, nodes, topics). Cleaned up this session — fixed two
  text collisions, shortened edge labels to bare topic IDs.
- Page 2: **new** — 6 core pipelines as independent step-by-step chains (Teleoperation,
  Autonomous, Pause/resume gesture, Recording→dataset, Home Pose, UI task lifecycle).
- Page 3: Topic/Service Legend, rebuilt as a `longtable` with a **Kind** column
  (PUB/SUB / PUB/SUB latched / SERVICE / HTTP) and one-line notes — was 5 pages of dense
  paragraphs, now fits one page.

## 2. `cyclo_teleoperation` gripper snap-back fix — VERIFIED root cause, fix built, NOT yet
   live-tested (leader was down at time of fix)

**Bug**: on Home-pose release / disengage / preset-cancel, the gripper would snap back to its
exact pre-event value instead of holding the real (post-event) position.

**Root cause, confirmed via a debug log pulled from `~/.ros/log/cyclo_teleoperation_node_*.log`**:
`captureGripperPosition()` looked up the gripper joint in `follower_joint_names_`, which comes
from the **Pinocchio kinematics model** and only contains `lift_joint` + the 14 arm joints —
**zero grippers**. The lookup silently no-op'd every single call.

**Fix** (`ai_worker_profile.hpp/.cpp` inside the container repo above):
- Added `follower_left_gripper_position_` / `follower_right_gripper_position_` fields.
- `updateFollowerState()` (which already builds a full name→index map from the *real*
  `follower_joint_states_topic` message, unlike the Pinocchio-scoped list) now also populates
  these two fields.
- `captureGripperPosition()` rewritten to read from these fields instead of the broken lookup.
- Temporary debug log removed once root cause was confirmed.

Package rebuilds clean (`colcon build --packages-select cyclo_teleoperation`). **Never
live-verified on hardware** — leader was disconnected before a retest could happen. Do that
before trusting it.

## 3. Joystick pause/resume gesture + idle-publish contention fix (earlier in session, largely
   verified)

- `T14` (`/task/policy_active`, latched): `cyclo_teleoperation_node` stops publishing its idle
  `hold_target_` onto T6 while the policy is running, so leader + follower can be brought up
  simultaneously for a whole inference session without fighting the policy. On the Pause falling
  edge, `hold_target_` is re-captured from the follower's live position.
- `T16` (`/leader/teleoperation/toggle_pause_request`): fired by `leader_bridge.py` on a
  right-stick drag-hold ≥1.0s, `|value|>0.6`. Orchestrator owns the actual Pause↔Resume decision.
- `T15`/`T5`: raw joystick axes and the decodable `active_arms` translation, both in
  `leader_bridge.py` (still a standalone script, not packaged into any launch file).
- Full detail is in the architecture PDF's legend (page 3) and pipeline 3 (page 2).

## 4. `cyclo_data` converter fixes — code done + partially validated on real data, NOT yet run
   on the full 35-episode dataset

All in `cyclo_data/cyclo_data/converter/{base_converter.py,to_lerobot_v30.py}`.

### 4a. `sample_weight` — HG-DAgger weighting, matches Larchenko's LeHome Challenge paper exactly
(arXiv:2606.27163 §9.10, Figure 17, verified by reading the actual PDF, not just the abstract):
- Human-correction frames (`intervention=1`): weight **2.0**.
- Autonomous frames far from any takeover: baseline **0.3**.
- Autonomous frames in the 5.0s window before a takeover: linear ramp 0.3→0.0.
- Excluded frames (`intervention=-1`): neutral 1.0, untouched.
- `_PRE_PAUSE_RAMP_SECONDS` changed from 2.0 (our earlier deviation) back to **5.0** to match the
  paper, since we're now explicitly replicating their scheme rather than deviating from it.
- Verified with a synthetic two-intervention episode reproducing Figure 17's exact shape.
- **Not implemented**: the paper's advantage/value-based weighting (a separate, sim-round-only
  mechanism requiring a trained reward/value model — explicitly out of scope, not "easy").

### 4b. Action-relabel fix — action[t] = follower's own state at t+5, not the leader's raw position
**This is the fix for the "drifts right during driver-grab" symptom.** Root cause (found by an
independent investigation on the user's other machine, cross-verified here against the actual
`_collect_tasks`/topic-mapping code): `action` was recorded from the **leader** topic, `state`
from the **follower**. Every time an arm is deliberately disengaged (e.g. holding one arm static
for camera framing) the leader free-wheels while the follower holds still, and the two never
resync — measured 0.3mm gap before the first disengage, 157mm by the driver grasp on episode 32.
The model faithfully learned to predict leader-frame targets; deployment commands the follower.

- `ConversionConfig.relabel_action_to_follower_lead: bool = False` (opt-in — this redefines what
  `action` means for every dataset this shared converter produces, so it's gated, not a silent
  default flip) + `action_lead_frames: int = 5` (validated independently: median leader→follower
  tracking lag on frames before any disengage, consistent across 35 demo episodes).
- `_relabel_action_from_follower_lead()`: `action[i] = observation_state[min(i+5, last)]`,
  tail-clamped to hold the final state rather than dropping frames.
- **A real bug was found and fixed during 2-episode trial validation**: `to_lerobot_v30.py`'s
  `ProcessPoolExecutor` parallel-conversion path (used for every real multi-episode run) builds a
  `config_dict` for worker processes by hand-listing every field name. The two new fields were
  missing from that list, so every worker silently reconstructed a config with the flag back at
  its default (off) — meaning a real 35-episode conversion would have silently done nothing,
  despite the fix being correctly wired everywhere else. Fixed by adding both fields to that dict.

**Validated on a real 2-episode trial conversion** (`Task_3_Screwing_MCAP` episodes 0-1, via
`RosbagToLerobotV30Converter` called directly, bypassing the ROS service layer):

| check | before | trial (broken, config_dict bug) | trial (fixed) |
|---|---|---|---|
| action[t] vs follower[t+5] gap | 3.42° | 4.00° | **0.000°** |
| frozen-but-commanded frames | 1.55% | 2.12% | 1.19% (see caveat below) |
| label consistency across episodes | 4.04° | 2.72° | **0.08°** |

Freeze-frame check still doesn't fully clear the gate's 0.5% threshold — this is the known,
already-flagged edge case: the last ~5 frames of each freeze, where `follower[t+5]` reaches past
the release and briefly commands real motion again. Not a new problem.

### 4c. `tasks.parquet` — task language was in a COLUMN, not the index; model silently trained on
"0"/"1"/"2" as its language conditioning
Confirmed against `Isaac-GR00T/scripts/lerobot_conversion/convert_v3_to_v2.py`
(`for task, row in tasks.iterrows(): ... "task": task` — uses the iterrows() index as the task
string) and, more importantly for what we actually train with, against the collaborator's report
that `robot_omy/lerobot`'s `dataset_reader.py:351-352` does
`item["task"] = self._meta.tasks.iloc[task_idx].name` — a **positional** lookup, meaning row
order must equal `task_index` order, not just "task is the index somewhere."
- `_write_tasks_parquet()` now does `pd.DataFrame(tasks_data).sort_values("task_index").set_index("task")`
  before writing (the sort is defensive — `_collect_tasks` already assigns indices sequentially
  for a single conversion, so it's already correct by construction there, but a merged/re-indexed
  source could violate it, and the sort is free).
- Verified both the schema fix and the positional invariant with direct round-trip tests
  (write → read back exactly the way each consumer reads it → assert correctness). Confirmed
  fixed on the real 2-episode trial too (`tasks.parquet` index now reads back as the real
  instruction string).
- **Unconditional fix** (not gated behind a flag) — there's no scenario where the broken schema
  is desirable.

### Not yet done
- **Full 35-episode reconversion** with both fixes has not been run — only the 2-episode trial.
  Per-episode cost from the cold-cache trial: ~26s/episode → estimate ~15 min for all 35. The
  trial was run by calling the converter directly (bypassing the ROS service layer, which doesn't
  currently expose `relabel_action_to_follower_lead`/`action_lead_frames`) and skipping the
  separate video-rotation transcode stage — that's why the trial's camera shapes came out
  240x424 instead of the correct 424x240 portrait; that's a test-harness artifact (rotation is
  applied by an earlier `Mp4ConversionWorker` stage, not by the LeRobot v3.0 converter itself),
  not a defect in either fix. Run the full 35-episode conversion through the *real* pipeline
  (`pipeline_worker.py`'s normal multi-stage flow, not the direct-call trial harness) to avoid
  that.
- `verify_dataset.py` (a gate script, author: Dawit Chun) should be run against the final
  reconverted dataset before any retrain. It's saved at
  `/tmp/claude-1000/-home-robotis-ai-worker/031ba59f-cb0b-44d1-a5ee-5b9ac9f35884/scratchpad/verify_dataset.py`
  on this machine (session-scoped scratch dir — copy it somewhere durable if you want to keep it).
- If the ROS service (`/data/convert`) should support the new relabel flags for a UI-triggered
  conversion (rather than only the direct-call path used for the trial), that plumbing doesn't
  exist yet.

## 5. Training config audit (from the collaborator's review, cross-referenced against NVIDIA's
   FinetuneConfig — not independently re-verified by this session, reported here for the record)
- Almost everything matched NVIDIA's official recipe exactly (`tune_llm=False`,
  `tune_visual=False`, `tune_projector=True`, `tune_diffusion_model=True`, lr=1e-4, etc.).
  `state_dropout_prob`: our `policy_preprocessor.json` says 0.0 but the actual applied value is
  `GR00T_N1_7_DEFAULTS['state_dropout_prob'] = 0.2` — a near-miss false defect, not a real one.
- **Real gap**: zero image augmentation (`image_transforms.enable=false`), vs. NVIDIA's default
  color-jitter-on recipe. Fix: `--dataset.image_transforms.enable=true`.
- **Real gap**: `steps=26110` configured, training stopped at 14400 — cosine LR never annealed
  (stopped at 43% of peak). Set `steps` to what you'll actually run
  (`55,701 train frames ÷ 48 global batch ≈ 1,160 steps/epoch`; NVIDIA's own recipe is
  ~10,000 steps at batch 64 ≈ 11.5 equivalent epochs → `steps=13000` is the matched target).
- **Both of these are second-order.** The last model landed within 1.1× of its own labels' noise
  floor (4.79° vs 4.45°) — it learned the broken data almost perfectly. Section 4b's action-relabel
  fix is the dominant lever; augmentation/schedule won't fix drift on their own.
- **Judging criterion for the next model**: don't accept "MAE went down" — MAE is unsigned and
  hides exactly this failure. Watch the **systematic Cartesian offset at the driver grab**
  (currently ~48mm at `|bias|/|error| ≈ 1.00`) collapsing toward the ~27mm natural
  demonstration-to-demonstration spread.

## 6. GR00T inference-pipeline latency investigation — real numbers, one confirmed correction to
   an earlier wrong conclusion, no root cause fix yet

Files: `cyclo_brain/policy/groot/runtime/inference_engine.py`,
`cyclo_brain/policy/common/runtime/main_runtime/control_loop.py`,
`cyclo_brain/sdk/action_chunk_processing/action_chunk_processing/action_chunk_processor.py`.

- Model chunk horizon: **T=40 steps @ 15Hz = 2.67s** real-time span per chunk (confirmed from the
  deployed `processor_config.json`).
- Pure model forward pass (TensorRT-accelerated DiT, confirmed active via `groot_server` logs):
  **~0.5-0.7s**, measured directly via existing/added timing instrumentation.
- **I initially claimed ~0.8-1.5s of "unaccounted overhead" between model time and observed
  chunk-to-chunk cadence, and that claim was likely wrong** — the cadence gap mostly reflects the
  buffer-refill design (a new chunk isn't requested until the buffer drains back down to a
  threshold sized around the latency EMA), not round-trip latency. Corrected this explicitly
  rather than let it stand.
- Added logging instrumentation (all currently live in the running containers, restarted
  cleanly): per-stage timing in `get_action_chunk` (obs/preprocess/inference/postprocess/total),
  the real client-side round-trip in `control_loop`, and — most importantly —
  `l2_align` in `action_chunk_processor.py` now logs `scheduled_start_delay`, `expected_idx`,
  `late_fallback`, and `best_dist` on every chunk splice. This is what will actually answer
  whether the "sudden lunge on teleop→policy handoff" and "stuck in a loop mid-task" symptoms are
  a timing/alignment problem (the `late_fallback` re-anchor-to-chunk-start path) or a model
  distribution-shift problem (policy poorly calibrated on human-corrected/recovery states) — **no
  live inference run has happened since this instrumentation went in**, so this is unresolved.
- Found and fixed a real bug along the way: `control_loop`'s logger resolved under
  `zenoh_ros2_sdk`'s logger tree, which is capped at WARNING globally — its `.info()`/`.debug()`
  calls were being silently dropped before ever reaching a handler. Gave it (and
  `action_chunk_processor`'s logger) their own dedicated INFO-level handlers instead of raising
  the shared SDK logger's level.
- Checked and ruled out: the zenoh RPC transport (event-driven callback + `threading.Event`, not
  a polling loop) and the server-side dispatch (`EngineWorker.handle`, direct synchronous call,
  no queueing). Checked but not conclusively resolved: the TensorRT "default stream" warning
  (`Using default stream in enqueueV3()...`) — confirmed real in `trt_torch.py`, but that code
  already calls `stream.synchronize()` immediately after submission, so it's already
  correctness-safe and likely a smaller/different-shaped cost than originally guessed. Vendored
  NVIDIA code, not edited.

### Next step when back at the robot
Run one inference session, then read the new `l2_align`/`get_action round-trip` logs from
`docker logs groot_server` to get a real, direct answer on `late_fallback` frequency — that's the
one piece of evidence needed to settle "is the lunge/loop a timing bug or a model problem" instead
of continuing to reason about it from static code alone.
