# Recording HG-DAgger / gate-as-potential / HIL-SERL data from Cyclo Intelligence

**For: the agent on `ffw-snpr48a1106`. From: the agent on Dawit's workstation.**
**Date: 2026-08-21.**

Plan change: we are **not** building a separate recorder. Cyclo Intelligence's UI already does
freeze, teleop, segmented recording, and inference pause/resume, and it is what Dawit actually uses.
We collect data through Cyclo. Training stays outside Cyclo, on the workstation.

**Do not rebuild what already exists.** Read section 2 before writing any code — most of what these
three methods need is already implemented and exposed over ROS services. The real work is section 3:
**one per-frame column** that records who was driving. That column is the entire difference between a
teleop dataset and a HIL dataset.

---

## 1. Scope

Three methods, in this order. All three need the same recorded signal, so build once.

| method | needs from the data |
|---|---|
| **HG-DAgger** (first) | frames where the **human** drove, in runs of ≥40 |
| **gate-as-potential** (ours) | **every** frame, labelled by who drove |
| **HIL-SERL** (if time) | the above plus a per-episode success/failure label |

One `int8` per frame satisfies all three:

```
 0  policy drove and the operator let it
 1  the operator drove          <- the expert action
-1  neither: frozen, or the follower ramping up on release
```

`-1` is not a detail. ROBOTIS's teleop slow-start ramps the follower up to the leader after a
freeze. That ramp is the follower catching up, **not** a demonstration. Labelling it `1` teaches the
policy that being corrected means "ease in and stop moving", which is the opposite of the correction
being made. Frozen ticks fail the same way. Both must be `-1`, and both are excluded from every
method.

---

## 2. What Cyclo already has — reuse, do not reimplement

From `interfaces/srv` and `interfaces/msg`. These are already wired to the UI.

### `RecordingCommand.srv`
```
START=0  STOP=1  PAUSE=2  RESUME=3  FINISH=4  MOVE_TO_NEXT=5  RERECORD=6
SKIP_TASK=7  CANCEL=8  REFRESH_TOPICS=9
START_SEGMENT=10  STOP_SEGMENT=11  DISCARD_SEGMENT=12
FINISH_EPISODE=13  DISCARD_EPISODE=14  SET_TASK_INFO=15  CANCEL_SEGMENT=16
string[] topics        # <- section 3.1 depends on this field
```
`START_SEGMENT` / `STOP_SEGMENT` / `DISCARD_SEGMENT` already give segment-level control. **Do not
add a parallel "intervention segment" concept.** Segments are for subtask structure; intervention is
per-frame and orthogonal — a takeover can begin and end in the middle of one segment.

### `InferenceCommand.srv`
```
LOAD=0  START=1  PAUSE=2  RESUME=3  STOP=4  UNLOAD=5  UPDATE_INSTRUCTION=6
bool publish_to_robot        # false = dry-run / 3D preview only
```
- `PAUSE` / `RESUME` **is** the HG-DAgger takeover. Pause stops the policy publishing while keeping
  it loaded on the GPU; resume hands control back. No model reload, no reconnect.
- `UPDATE_INSTRUCTION` re-conditions language mid-run without a reload — this is how the subtask
  stage advances. Our checkpoint is conditioned on five per-frame subtask sentences and will
  otherwise perform stage 1 and sit there.
- `publish_to_robot=false` is the dry-run gate. **Use it for every first test.**

### `RecordingStatus.msg`
Already publishes `record_phase` (READY / RECORDING / SAVING / PAUSED), `current_subtask_index`,
`subtask_instructions`, `saved_subtask_indices`. No new status topic is needed for subtasks.

### `InferenceStatus.msg`
Already publishes `inference_phase` (READY / LOADING / INFERENCING / PAUSED). **This is the primary
source for the label in section 3** — `PAUSED` while recording means the human is driving.

---

## 3. The work: one per-frame `intervention` column

Cyclo records **rosbag first, then converts to LeRobot v3.0**. That is convenient: anything published
as a topic while recording is already in the bag. So this is two small changes — get the signal into
the bag, then turn it into a column.

### 3.1 Get the signal into the bag

The label has two independent sources, and **both must be recorded**; do not derive one from the
other at record time.

| topic | type | meaning |
|---|---|---|
| `/inference/status` (`InferenceStatus`) | existing | `PAUSED` → the human is driving |
| `/arm_freeze/status` | `std_msgs/String` | `mode=… left=… right=…`, carries mode and easing |
| `/arm_freeze/ready` | `std_msgs/Bool` | latched; `false` = frozen, easing, or joint states stale |

`RecordingCommand.srv` carries `string[] topics`, and the orchestrator forwarder aggregates that list
from `Communicator`'s topic inventory. **Add all three there** so they land in the bag. Verify after
one recording:

```bash
ros2 bag info <bag> | grep -E "arm_freeze|inference"
```

If they are not listed, nothing downstream can work and the rest of this section is moot.

### 3.2 Convert it to a per-frame column

`subtask_index` is already an optional per-frame `int64` column. **Copy that pattern exactly** —
every touch point below is where `subtask_index` is handled, so grep for it if a line number has
drifted.

**`cyclo_data/converter/base_converter.py`**

| line | what | change |
|---|---|---|
| 408 | `subtask_indices: List[int]` in `EpisodeData` | add `intervention_flags: List[int] = field(default_factory=list)` |
| 1754 | `_assign_subtask_indices()` | template for a new `_assign_intervention_flags()` |
| 1155 | `self._assign_subtask_indices(episode_data)` | call the new one alongside |
| 1651 | `single.subtask_indices = [0] * length` | single-episode path must fill it too |
| 1703 | `stitched.subtask_indices.extend(...)` | stitched path must fill it too |

`_resample_to_fps()` (line 3561) already documents **"causal sync (previous value only)"**. That is
exactly right for a latched state flag — hold the last value, never interpolate. Interpolating a
state label would invent half-intervened frames that never existed. Reuse it; do not write a new
sampler.

**`cyclo_data/converter/to_lerobot_v30.py`**

| line | what `subtask_index` does | do the same |
|---|---|---|
| 1338 | `has_subtask_feature = any(ep.subtask_indices for ep in episodes_data)` | `has_intervention_feature` |
| 1471 | `schema_fields.append(pa.field("subtask_index", pa.int64()))` | add `intervention`, `pa.int8()` |
| 1484 | allocate the array | same |
| 1508 | fill per episode | same, **but see the warning below** |
| 1543 | `arrays.append(pa.array(...))` | same |
| 1580 | `hf_features["subtask_index"] = {...}` | `hf_features["intervention"] = {"dtype": "int8", "_type": "Value"}` |
| 904, 922 | `_array_cache_signature` / cache key | **must include the new field**, or a cached re-run silently reuses a conversion that had no intervention column |
| 1621 | `has_subtask_feature = any("subtask_index" in frame ...)` | the frames path needs it too |

**One deliberate difference from the `subtask_index` template.** At line 1515 the fallback for a
length mismatch is `0`:

```python
else:
    subtask_index[offset:end] = 0
```

For `subtask_index` that is harmless. For intervention, `0` means **"the policy drove and the
operator let it"** — a length mismatch would silently relabel a whole episode as clean policy data
and feed it straight into training. **Use `-1` as the fallback**, so a mismatch excludes the frames
instead of mislabelling them, and log a warning naming the episode.

### 3.3 Deriving the value

Per output row timestamp, using causal (previous-value) lookup on each topic:

```
if not ready              -> -1     # frozen, easing, or joint states stale
elif inference == PAUSED  ->  1     # human driving
else                      ->  0     # policy driving
```

Take `ready` as a **veto**, not an override: if `/arm_freeze/ready` is false *or* the status string
says frozen/easing, the frame is `-1`. The two errors are not symmetric — dropping a good frame costs
one sample, while keeping a frozen frame teaches the wrong lesson. If the two sources disagree, log
it once with the timestamp; that means one of them is stale and we want to know.

If `/arm_freeze/ready` was never seen in the bag, fall back to parsing the status string, and **say
so in the conversion log**. Silent fallback is how we end up trusting a column that was guessed.

---

## 4. Two traps that have already cost us

### 4.1 `tasks.parquet` — the language lands in the INDEX

`_write_tasks_parquet()` (`to_lerobot_v30.py:4055`) builds a list of dicts:

```python
tasks_data = [{"task_index": idx, "task": task, "task_name": ...} for ...]
```

so `task` becomes a **column**. LeRobot v3.0's `load_tasks` reads the task string from the **index**
and only *renames* it. With a plain `RangeIndex`, the model is conditioned on the literal strings
`"0"`, `"1"`, `"2"` — and it fails silently, because those are valid distinct strings. We have hit
this on every Cyclo conversion so far.

Verify with LeRobot's own loader, never by reading the parquet directly:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="local", root="<converted>")
print(ds.meta.tasks.index[:5].tolist())   # must be sentences, not "0","1","2"
assert not all(str(t).isdigit() for t in ds.meta.tasks.index)
```

### 4.2 Cyclo datasets are readable but not appendable

Cyclo's output diverges from what LeRobot's own writer produces, so a Cyclo dataset cannot simply be
appended to. Do not try to append DAgger rounds onto an existing Cyclo dataset in place. Canonicalise
once through LeRobot's writer, then append to the canonical copy. On the workstation side
`hil_dagger/dagger_aggregate.py` already does this (`canonicalize()`, `_fix_image_stats()`,
`_strip_nonwriter_episode_columns()`); the same rule applies here.

---

## 5. Camera orientation — check before collecting, not after

Our GR00T checkpoint was trained on wrist frames in **portrait 424×240**. The live wrist cameras
deliver **240×424 landscape** — the transpose. `spec_sg2.py` requests portrait and the plugin hands
back the transpose anyway.

If Cyclo records the raw landscape frames, that is **fine and correct** — the original 35-episode
dataset is portrait, so as long as Cyclo's output matches the *training set's* orientation, nothing
needs to change. What must not happen is a dataset with mixed orientations. **Check one converted
episode before collecting a full session:**

```python
import json
info = json.load(open("<converted>/meta/info.json"))
for k, v in info["features"].items():
    if "image" in k:
        print(k, v["shape"])        # wrists must be [3, 424, 240]
```

If they come out `[3, 240, 424]`, the conversion needs a 90° CCW rotation on the wrists only — the
scene camera (376×672) is already correct and rotating it would break the one view that was right.
Getting this wrong is not a crash; the policy just behaves as if badly trained. Measured cost:
commanded motion 0.034 rad wrong-way vs 0.169 correct, against 0.232 in the demos.

---

## 6. What we need back

1. **One converted episode** with the `intervention` column, from a session that includes at least
   one freeze → teleop → release cycle. We will validate the labels against
   `hil_dagger/dagger_aggregate.py` before anyone collects a full set.
2. **Confirmation of the three topics in the bag** (`ros2 bag info` output).
3. **The `tasks.parquet` check from 4.1** run on that episode, output pasted.
4. **Wrist image shape from 5.**
5. Confirmation that `ready` goes false for the **entire** slow-start ramp including its last tick.
   If it clears early, the tail is recorded as human demonstration — the exact failure `-1` exists to
   prevent.

**Do not collect a full session until 1–4 come back clean.** A session recorded with a wrong or
missing intervention column looks completely normal and is unusable, and we would not find out until
training.

---

## 7. Explicitly out of scope for you

Training, aggregation, and the reward/potential design stay on the workstation. You do not need to
touch GR00T, checkpoints, or `dagger_aggregate.py`. If a change seems to require modifying how the
policy is trained, stop and send it back instead — that is our side.
