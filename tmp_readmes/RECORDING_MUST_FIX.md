# CORRECTION — four things that decide whether the recording is usable at all

**For: the agent on `ffw-snpr48a1106`. Read before writing recorder code.**
**Corrects `CYCLO_HIL_INTEGRATION.md` and `CYCLO_ONLINE_RL_UI.md`. Date: 2026-08-21.**

I under-specified four requirements in the earlier documents. Each of them fails **silently** — the
recording completes, the UI looks normal, the dataset converts, and the problem only surfaces at
training. Please treat this document as the acceptance criteria.

---

## FIX 1 — the `action` column is the LEADER topic, and that is wrong for policy frames

This is the important one.

`cyclo_data/converter/base_converter.py:858-871`:

```python
state_topics[key]  = cfg["topic"]     # key = f"follower_{name}"
action_topics[key] = cfg["topic"]     # key = f"leader_{modality}"     <-- LEADER
```

So `observation.state` is the follower and `action` is the **leader**. For ordinary teleop that is
exactly right: the human moves the leader, the follower tracks, the leader command is the action.

For an online-RL run it is only right **half the time**:

| who is driving | leader topic | is `action` correct? |
|---|---|---|
| human (policy PAUSED) | operator is moving it | **yes** |
| policy (INFERENCING) | sitting idle wherever the operator left it | **no — stale pose** |

Every policy-driven frame would record a constant, meaningless leader pose as its action. Nothing
errors. The column is full of plausible float arrays.

### Who this breaks

- **HG-DAgger** — survives. It only consumes `intervention == 1` frames, where the leader *is* the
  action. **If HG-DAgger is all you enable first, this is not yet fatal.**
- **gate-as-potential (ours)** — broken. It consumes every frame.
- **HIL-SERL** — broken. It needs an action on every transition.

Two of the three methods are unusable, so fix it now rather than re-collecting later.

### The fix

Record **both** action sources, and compose the column by mode:

```
action[i] = leader_topic[i]           if intervention[i] == 1     (human drove)
          = policy_command_topic[i]   if intervention[i] == 0     (policy drove)
          = <whatever>                if intervention[i] == -1    (excluded; never trained on)
```

Steps:

1. **Identify the topic the policy container publishes to.** With the gate reverted this is whatever
   the stock `cyclo_brain/policy/<backend>` Main runtime → `RobotClient` publishes when
   `publish_to_robot=true`. **Tell us the exact topic name** — we need it on our side too.
2. Add it to the `string[] topics` list in `RecordingCommand` so it lands in the bag, next to the
   leader topics.
3. In the converter, build `action` per-frame from whichever source matches the intervention label,
   using the same causal (previous-value) sampling as `_resample_to_fps`.
4. **Log the composition counts** per episode: `action from leader: N frames, from policy: M frames`.
   If either is zero on a run that had both modes, the wiring is wrong and this line is how you find
   out.

### Do not substitute the follower's achieved state

It is tempting to use `observation.state` shifted by one as the action for policy frames. Don't:
the follower lags the command, so you would train the policy to output where the arm *got to*, not
what was *asked for*, and the error compounds at every step. Use the actual published command.

---

## FIX 2 — a takeover shorter than 40 frames produces ZERO training samples

`hil_dagger/dagger_aggregate.py`:

```python
CHUNK = 40          #GR00T N1.7 action horizon; a training sample is this many consecutive frames

def select(flags, method, min_run=CHUNK):
    if method == "hg_dagger":
        return [(a, b) for a, b in runs_of(flags == 1) if b - a >= min_run]
```

A training sample is a **40-frame window**, not a frame. A run of intervened frames shorter than 40
yields nothing at all — it is not partially useful, it is discarded. This is RLinf's rule and we are
matching it deliberately.

What that means for the operator, by control rate:

| `control_hz` | 40 frames is | shorter takeovers are **worthless** |
|---|---|---|
| 30 Hz | **1.33 s** | under 1.33 s |
| 20 Hz | **2.00 s** | under 2.00 s |
| 15 Hz | **2.67 s** | under 2.67 s |

A run of 45 frames yields only `45 - 40 + 1 = 6` windows. A run of 120 yields 81. **Sample count is
strongly nonlinear in takeover length**, so a few long corrections are worth far more than many short
ones.

### What the UI must show

In the Online-RL page, during a takeover, display a **live frame counter for the current run**, with
the 40-frame threshold marked:

```
HUMAN   ██████████░░░░  28 / 40 frames   (12 more before this correction counts)
```

and once past it, keep counting so the operator can see the value accumulating:

```
HUMAN   ████████████████  87 frames  ->  48 training windows
```

Without this the operator has no way to know a correction was too short, and a whole session can
consist of 30-frame nudges that aggregate to zero. That is the single most likely way this data
collection fails.

Read `min_run` from a config field rather than hard-coding 40 — we may retune it.

---

## FIX 3 — HG-DAgger uses SUCCESSFUL episodes only, so the outcome prompt is not optional

`dagger_aggregate.py:139`:

```python
ap.add_argument("--only-success", action="store_true", default=True, ...)
```

**Default is True.** A failed episode's human frames are corrections toward an outcome that never
worked, so they are excluded.

In `CYCLO_ONLINE_RL_UI.md` I presented the outcome prompt as a HIL-SERL requirement. That was wrong
— it is required for **HG-DAgger too**, which is the method we run first. An episode saved without an
outcome is not merely missing a reward signal; it is **dropped entirely** by the default aggregation
path.

So: the outcome prompt is mandatory for every episode, from the first session. `SUCCESS` / `FAILURE`
/ `DISCARD`, blocking, no default selection — a pre-selected default gets click-through and we would
rather have a pause than a wrong label.

---

## FIX 4 — the two pages are separate, so the Online-RL page must ATTACH, not start

Confirmed by Dawit: **start/stop of the policy stays on the existing Inference page.** The Online-RL
Data page is where recording happens. So the operator's flow is:

```
InferencePage    LOAD the model, START the policy      <- unchanged, existing page
      |
      v
OnlineRLPage     attach to the running session, record, take over, label outcome
```

This constrains the new page in four ways. Each of these is a way to lose a session.

### 4.1 Attach to the running session — never re-LOAD

The Online-RL page must **not** issue `InferenceCommand.LOAD`, and must not `START` a second session.
`LOAD` frees and reloads 12.58 GB of GPU weights; doing it because the operator navigated to another
page would drop the session and take a minute. The page reads the existing session state and issues
only `PAUSE` / `RESUME` / `UPDATE_INSTRUCTION`.

Practically: the inference session state belongs in a **shared Redux slice** that both pages read,
not in `OnlineRLPage`'s local component state. Follow how `features/tasks/taskSlice.js` is shared
today.

### 4.2 Refuse to record when no policy is running

The intervention label is derived from `InferenceStatus.inference_phase`. If inference is `READY`
(never started), every frame is ambiguous — there is no policy to be paused from.

**If `inference_phase` is not `INFERENCING` or `PAUSED`, disable the record button** and say why:

```
Cannot record: no policy running.
Start the policy on the Inference page first, then return here.
```

Silently recording in that state produces a dataset that looks like a normal teleop session and is
worthless for all three methods.

### 4.3 Takeover controls must exist on the recording page

`PAUSE` / `RESUME` is the takeover. The operator performs it **while recording**, with both hands
occupied — so those controls must be on the Online-RL page, even though `START`/`STOP` live on the
Inference page. Do not make the operator navigate away mid-episode to take over; navigation during a
run is how a session gets lost.

Ideally the takeover is also bound to a physical control on the leader, since the operator's hands
are on it. If the stock ROBOTIS teleop exposes a button or gesture that can be read as a topic, wire
it to `PAUSE`/`RESUME` and tell us which one — that removes the last reason to touch the screen
mid-episode.

### 4.4 Ordering, and what happens if the session dies

- Recording must **start after** the policy is running, and **stop before** the policy is stopped.
- If `inference_phase` drops to `READY` or `error` mid-recording, **stop the recording and mark the
  episode `DISCARD`**. Do not save a partial episode whose tail has no policy behind it: the
  intervention labels for those frames are meaningless and there is no way to tell afterwards.
- If the operator navigates back to the Inference page mid-run, keep recording — the session lives in
  the shared slice, not the page. Losing an episode to a page change would be an unforced error.

---

## Acceptance test — run this before collecting a full session

Record **one** episode containing, in order: policy driving → a takeover of **at least 3 seconds** →
hand back to policy → mark it `SUCCESS`. Convert it, then run:

```python
import json, numpy as np, pandas as pd, glob
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = "<converted episode>"
ds = LeRobotDataset(repo_id="local", root=ROOT)

# 1. language is in the INDEX, not "0"/"1"/"2"
tasks = ds.meta.tasks.index[:5].tolist()
print("tasks:", tasks)
assert not all(str(t).isdigit() for t in tasks), "tasks.parquet index trap -- see FIX in doc 1"

# 2. the intervention column exists and has all three states
df = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True)])
iv = df["intervention"].to_numpy()
print("counts:", {int(v): int((iv == v).sum()) for v in (-1, 0, 1)})
assert (iv == 1).any() and (iv == 0).any(), "no takeover or no policy frames recorded"

# 3. the takeover is long enough to yield samples
runs, s = [], None
for i, v in enumerate(iv):
    if v == 1 and s is None: s = i
    elif v != 1 and s is not None: runs.append(i - s); s = None
if s is not None: runs.append(len(iv) - s)
print("intervened runs (frames):", runs)
print("training windows:", sum(max(0, r - 40 + 1) for r in runs))
assert sum(max(0, r - 40 + 1) for r in runs) > 0, "every takeover was under 40 frames -- FIX 2"

# 4. THE ONE THAT CATCHES FIX 1
#    the action must CHANGE during policy-driven frames. A stale leader pose is constant.
act = np.stack(df["action"].to_numpy())
pol = act[iv == 0]
moved = np.abs(np.diff(pol, axis=0)).sum()
print(f"action variation over {len(pol)} policy frames: {moved:.4f}")
assert moved > 1e-3, "action is CONSTANT while the policy drove -- still recording the idle leader"

# 5. wrist images match the training set
info = json.load(open(f"{ROOT}/meta/info.json"))
for k, v in info["features"].items():
    if "image" in k: print(k, v["shape"])      # wrists must be [3, 424, 240]

# 6. outcome recorded
print("outcome:", ds.meta.episodes[0].get("outcome", "MISSING"))
```

**Send us the full output of this script**, not a summary. Check 4 is the one that catches the
action-source bug, and it is the check that would have caught it silently passing everything else.

---

## Summary

| # | what | breaks | visible? |
|---|---|---|---|
| 1 | `action` is the leader topic, stale on policy frames | ours + HIL-SERL | **no** |
| 2 | takeovers under 40 frames yield zero samples | HG-DAgger | **no** |
| 3 | episodes without `SUCCESS` are dropped by default | HG-DAgger | **no** |
| 4 | recording with no policy running, or re-LOADing on page change | all three | **no** |

None of the three raises an error, and all three are cheap to fix now and expensive to discover after
a collection session.
