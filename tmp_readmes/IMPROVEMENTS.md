# IMPROVEMENTS — where the intervention label is lost, and how to fix it

**For: the agent working on `cyclo_intelligence` / `ai_worker`.**
**Written 2026-08-30, from reading the code in `cyclo_intelligence_backup@main` and
`aiw_backup@feature-a2`. Every claim below cites the file and line it came from.**

This document is about **two defects**. Neither is a missing feature — the machinery exists and is
well written. One is never reached on the export path the training tooling uses; the other applies
a rule written for a control mode this robot does not run.

---

## 0. The one-paragraph version

`/task/inference_status` **is** recorded, and `base_converter.py` **does** turn it into a correct
per-frame intervention label. `to_lerobot_v30.py` writes that label out. **`to_lerobot_v21.py`
does not.** Isaac-GR00T requires v2.1. So every dataset exported for GR00T training silently
loses the human/policy label, and with it the entire distinction between HG-DAgger and ordinary
behaviour cloning.

---

## 1. Proof, in four facts

**1. The topic is recorded.** `shared/robot_configs/ffw_sg2_rev1_config.yaml`, under
`recording.extra_topics`, with a comment that names its purpose exactly:

```yaml
recording:
  extra_topics:
    # Source of the per-frame `intervention` label for online-RL runs.
    # InferenceStatus.inference_phase: READY=0 LOADING=1 INFERENCING=2
    # PAUSED=3. Published one-shot on transitions only, so the converter
    # must forward-fill and seed the value at episode start.
    - /task/inference_status
```

**2. The label is computed, and computed well.**
`cyclo_data/converter/base_converter.py:1949` — `_assign_intervention_flags()`:

```
INFERENCING                              -> 0   (policy drove)
PAUSED + teleop engaged + not handoff    -> 1   (human drove)
PAUSED + leader NOT engaged              -> -1  (nobody driving)
handoff / leader slow-start retarget     -> -1  (machine motion, not a correction)
before the first status message          -> -1  (phase genuinely unobserved)
```

The causal fill and the "nobody driving" case are right. The **handoff exclusion is not** --- see
Defect 2 in §2b: it assumes a mode-1 slow-start sweep, and this robot runs mode 2.

**3. v3.0 writes it.** `to_lerobot_v30.py:1484-1489`:

```python
has_hil_feature = any(ep.intervention_flags for ep in episodes)
if has_hil_feature:
    schema_fields.append(pa.field("intervention", pa.int64()))
    schema_fields.append(pa.field("reward", pa.int64()))
    schema_fields.append(pa.field("done", pa.bool_()))
    schema_fields.append(pa.field("sample_weight", pa.float32()))
```

**4. v2.1 does not.** `to_lerobot_v21.py:2089-2103` — the complete schema:

```python
schema_fields = [
    pa.field("index", pa.int64()),
    pa.field("episode_index", pa.int64()),
    pa.field("task_index", pa.int64()),
    pa.field("timestamp", pa.float64()),
]
if action_dim > 0:  schema_fields.append(pa.field("action", ...))
if state_dim > 0:   schema_fields.append(pa.field("observation.state", ...))
if has_subtask_feature: schema_fields.append(pa.field("subtask_index", pa.int64()))
```

`grep -c intervention to_lerobot_v21.py` returns **0**.

---

## 2. What this cost, concretely

The 13-episode online-RL dataset exported on 2026-08-28 reached training with no intervention
column. Downstream, `hil_dagger/eval/split_dagger.py` refuses to run without it, so the correction
frames could not be weighted at all — they were indistinguishable from the policy's own behaviour
and would have been diluted roughly 6:1 against it.

The labels had to be reconstructed after the fact from joint-motion pauses, then validated by
running the base policy over the frames and measuring where it disagreed with the recording
(1.59° on policy frames vs 3.13° on human frames, p ≈ 1e-20). That reconstruction recovered 39
interventions and the operator confirmed they matched what he actually did — but it is inference,
its span boundaries are soft, and none of it should have been necessary.

`reward`, `done` and `sample_weight` are lost the same way, which also rules out HIL-SERL-style
training from a v2.1 export.

---

## 2b. DEFECT 2 — the handoff exclusion is written for a control mode this robot does not use

`_assign_intervention_flags` excludes the first stretch after PAUSE, with this reasoning:

```
# PAUSED means the human has control -- but the first stretch
# after PAUSE is the leader slow-start retargeting onto the
# follower, which is machine motion. Exclude it, or every
# takeover trains on a sweep toward the follower's pose.
```

That is **mode 1** behaviour. `ffw_a2_teleoperation.yaml` sets `default_control_mode: 2`, and mode 2
is explicit about it:

```yaml
"2":
  name: elbow_up_leader
  plugin: cyclo_teleoperation/AiWorkerElbowUpLeaderMode
  # RELATIVE-DELTA teleop. On engage this mode captures a leader anchor
  # and a follower anchor, then commands
  #     goal = follower_anchor + (leader_now - leader_anchor)
  # so the follower moves BY the leader's change rather than TO its pose.
  # Nothing jumps at engage even if leader and follower are far apart --
  # unlike mode 1, which relays absolute leader positions.
  slow_start:
    enabled: false          # mode 1 has this true
```

Under mode 2 **nothing sweeps on engage** — the delta anchor is captured at the tact press and
motion is relative from there. So `_is_handoff_frame` is discarding the opening frames of every
takeover, and those frames are real human correction, not machine motion. They are also the most
informative part of the correction: they are where the operator reacts to whatever the policy got
wrong.

**Fix:** make the exclusion conditional on the active control mode rather than unconditional.
Mode 1 keeps it; mode 2 drops it. The mode is already known --- it is on T4 / T17
(`ControlModeStatus.active_control_mode`). If plumbing the mode through is awkward, gate it on the
`slow_start.enabled` flag for the active mode, which is the thing the exclusion actually depends on.

**Do not simply delete it.** On a mode-1 robot the sweep is real and training on it teaches the
policy to drift toward wherever the arm already is.

## 3. The fix

### 3.1 Primary — emit the columns from the v2.1 writer

In `to_lerobot_v21.py`, mirror the v3.0 block. The flags are already on `EpisodeData`; nothing new
needs computing and no new parameter has to be threaded anywhere.

```python
# Online-RL columns, present only when /task/inference_status was in the bag.
# Mirrors to_lerobot_v30.py:1484-1489. Isaac-GR00T requires v2.1, so without
# this every HG-DAgger export loses the human/policy label and degrades to
# ordinary behaviour cloning.
has_hil_feature = any(ep.intervention_flags for ep in episodes)
if has_hil_feature:
    schema_fields.append(pa.field("intervention", pa.int64()))
    schema_fields.append(pa.field("reward", pa.int64()))
    schema_fields.append(pa.field("done", pa.bool_()))
    schema_fields.append(pa.field("sample_weight", pa.float32()))
```

Then the matching array fill (v3.0 lines 1504-1557) and the `hf_features` entry (v3.0 line 1639)
so `meta/info.json` declares it. **Declaring it in `info.json` is not optional** — tooling that
reads the feature list will not see a column that exists only in the parquet.

### 3.2 Keep the convention, and write it down

```
intervention:  1 = human drove
               0 = policy drove
              -1 = excluded (handoff, slow-start, nobody driving, phase unobserved)
```

**This is the inverse of `task_is_policy`**, the column name some downstream tooling expects
(`split_dagger.py`, where 1 = policy). Both names are in use. If a converter ever emits
`task_is_policy` while a consumer reads `intervention`, or vice versa, **nothing errors** — the
run trains on the exact inverse of the intended data. Emit one name, state the polarity in
`info.json`, and make any consumer assert on the name rather than the position.

### 3.3 Do not paper over an absent topic

If `/task/inference_status` is missing from a bag, `intervention_flags` is left empty and no
column is emitted. Keep that. An all-zero column would read as "the policy drove the entire
episode" and is strictly worse than an absent one, because it looks valid.

### 3.4 While in there: record `active_arms_str`

The same robot config already documents this and it is still not done:

```yaml
# /leader/teleoperation/control_status intentionally NOT here.
# robotis_interfaces isn't installed where service_bag_recorder runs ...
# Fix path: record leader_bridge.py's plain std_msgs/String
# /leader/teleoperation/active_arms_str instead (same information,
# a type every environment can load) once that's wired up.
```

`_assign_intervention_flags` consults `teleop_active_arms_messages` to decide whether the human
was *actually engaged* during a PAUSE. Without it that check degrades, and "policy paused but
nobody driving" gets labelled as a human correction. Recording the `std_msgs/String` version
sidesteps the missing-typesupport failure that killed the launch group before.

---

## 4. Verify the fix on real data, not on a unit test

After converting one online-RL session to v2.1:

```python
import pandas as pd, glob, json
df = pd.concat([pd.read_parquet(f) for f in glob.glob("<ds>/data/**/*.parquet", recursive=True)])
assert "intervention" in df.columns, "column missing"
assert "intervention" in json.load(open("<ds>/meta/info.json"))["features"], "not declared in info.json"
print(df.intervention.value_counts())          # expect all three of 1 / 0 / -1
```

Sanity bounds from the 2026-08-28 session, for comparison:

| | |
|---|---|
| human (`1`) | ~15% of frames |
| excluded (`-1`) | ~11% |
| policy (`0`) | ~74% |
| interventions per episode | 3 |
| median duration | 3.8 s (place-in-hole) → 8.0 s (screw-in) |

**A run of human frames shorter than 40 (one GR00T action chunk) yields zero training samples**,
so a session of short nudges can convert cleanly and still be worth nothing. The UI already shows
the live 40-frame counter; the converter should log the same thing per episode so it is visible
without watching the screen.

---

## 5. What is already right — do not "fix" these

- **The action topic is shared between leader and policy** — both publish to
  `/leader/joint_trajectory_command_broadcaster_{left,right}/joint_trajectory`, which is why
  `action` is correct in *both* modes and why `/task/policy_active` exists to stop them
  contending. An earlier design note worried that policy frames would record a stale leader pose;
  that is not what this system does, and the recorded data confirms it (action tracks
  `follower[t+5]` to 0.159° across all frames, 0.10% freeze frames).
- **`_assign_intervention_flags`'s causal fill.** Correct as written. (Its *handoff exclusion* is
  the subject of Defect 2 above --- the causal fill itself is fine.)
- **The gripper holds its teleoped value on disengage.** An operator fix — it did not retain
  the value originally. Implemented in `cyclo_teleoperation` (external to this repo, so not
  read from source here). Distinct from `robot_client.py`'s `GRIPPER_HOLD_EFFORT_THRESHOLD`,
  which freezes the *policy's* commanded gripper on measured effort and defaults to `0.0`.
- **`record_triggers_enabled: false`.** The tact press both engages a leader arm and fires a
  record trigger; during INFERENCING that engages an arm onto a topic the policy is driving.
  Leaving it disabled is right.
