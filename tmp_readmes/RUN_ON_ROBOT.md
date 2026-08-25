# Moving GR00T inference onto the robot

> **STATUS 2026-08-21: SUPERSEDED as a plan. Keep as reference.**
>
> Inference *is* moving onto the robot — but through **Cyclo Intelligence**, not by porting
> `online_rl.py`. Sections 1–3, 7 and 11 (copying our stack, our CLI, our topics) no longer apply.
>
> Still accurate and worth reading: **section 4** (loop rate — now more relevant, not less),
> **section 5** (camera orientation), **section 8** (measured control constants), **section 9**
> (known-bad things), **section 10** (the unexplained right-arm dive).
>
> Current plan: `CYCLO_ONLINE_RL_UI.md`.

**For: the agent running on `ffw-snpr48a1106`. From: the agent on Dawit's workstation.**

Dawit is moving the whole inference + HG-DAgger recording stack onto the robot. Round-tripping
observations and commands to the workstation over zenoh is too awkward to iterate on. He accepts
that per-step inference latency will be worse on the robot — **that trade is deliberate, do not
try to undo it.** What you must not accept is the *control loop* silently inheriting that latency.
Section 4 is the part that matters most; everything before it is setup.

---

## 1. What this stack is

A GR00T N1.7-3B policy fine-tuned on 35 teleop episodes of a screwing task, driving both arms of an
AI Worker at ~30 Hz. Two entry points share one control path:

| file | role |
|---|---|
| `aiworker_deploy/infer.py` | plain inference. The reference implementation. |
| `hil_dagger/online_rl.py` | the same rollout **plus** HG-DAgger recording of human takeovers. |

`online_rl.py` is the one Dawit runs now. It must behave identically to `infer.py` while the policy
drives — corrections recorded against a different controller than the deployed one are worthless.

Supporting modules: `robot_session.py` (connect + pre-flight gates), `groot_policy.py` (model
wrapper), `control_math.py` (rate limiting), `trajectory.py` (publishing), `spec_sg2.py` (all
constants), `takeover.py` (keyboard/joystick state machine), `safety.py`.

---

## 2. What to copy

| what | from | size | notes |
|---|---|---|---|
| checkpoint | `runs_screwing/groot35/checkpoints/014400/pretrained_model` | **12 GB** | the whole dir; `config.json` + `model.safetensors` + both pre/post-processors are all required |
| dataset | `datasets/screwing35_subtask` | 199 MB | copy all of it |
| deploy code | `aiworker_deploy/*.py` | ~400 KB | |
| dagger code | `hil_dagger/{online_rl,takeover,dagger_aggregate,preview}.py` | ~150 KB | |
| lerobot fork | `/home/robotis/robot_omy/lerobot/src` | 7 MB | the **fork**, not pip `lerobot` |

The dataset's `videos/` (194 MB of the 199 MB) is not read during inference, but copy it anyway —
`LeRobotDataset` construction is what reads the tree, and a partial copy fails in a confusing way
for no meaningful disk saving.

**Why the dataset is needed at inference at all** (it is not for weights):
- `meta/info.json` — joint dimension names, and the camera shapes that GATE 0 checks against
- `data/*.parquet` — the median demo start pose used for homing

---

## 3. Paths and environment

Absolute paths that used to be hardcoded are now environment-overridable. Set these before running:

```bash
export LEROBOT_SRC=/path/on/robot/lerobot/src      # the fork's src dir
export AIW_DEPLOY=/path/on/robot/aiworker_deploy   # where the deploy .py files live
```

Both fall back to the workstation paths if unset, so **if you skip this you get an import error, not
a wrong result.** That is intentional.

Two more are plain CLI flags, not env vars:

```
--dataset  /path/on/robot/screwing35_subtask
--out      /path/on/robot/episodes
```

Dependencies: `torch`, `transformers`, `numpy`, `pandas`, `scipy`, `opencv-python`, `safetensors`,
plus the zenoh stack (`zenoh_ros2_sdk`, `rosbags`, `lerobot_robot_ros2_zenoh`). On the workstation
these live in a venv at `/home/robotis/lerobot_venv`. Verify with:

```bash
python3 -c "import torch, cv2, scipy, pandas, rosbags, zenoh_ros2_sdk; print('deps ok')"
python3 -c "import lerobot, sys; print('lerobot from', lerobot.__file__)"   # must be the fork
```

---

## 4. The loop-rate problem — read this before the first rollout

This is the one thing that moving to the robot genuinely breaks, and it will not announce itself.

`online_rl.py` computes `dt = 1.0 / args.hz` with `--hz` defaulting to **30**. That `dt` is used for
two different things:

1. how long the loop *tries* to sleep between ticks
2. **the deadline stamped on every `send_point` command** — "reach this pose in `dt` seconds"

On the workstation (RTX 4090) inference keeps up with 30 Hz, so those agree. On the robot, if a tick
actually takes 60 ms but `dt` still says 33 ms, every command tells the controller to arrive in
33 ms while the next one does not arrive for 60 ms. The arm sprints to each target, arrives, sits
idle, then sprints again. **That reads as jitter and looks exactly like a bad policy.**

### Measure the real rate first

Do a dry run — it publishes nothing:

```bash
python3 online_rl.py --dataset <ds> --out <out> \
  --policy-path <ckpt>/pretrained_model --episodes 1 --max-steps 300
```

The status line prints elapsed time as `len(q_cmd)/args.hz`, which **assumes** the rate rather than
measuring it. So compare against a real clock: if 300 ticks are reported as 10.0 s but your wall
clock says 20 s, you are running at 15 Hz, not 30.

### Then set `--hz` to what you actually achieve

```bash
--hz 15        # if you measured ~15 Hz
```

Round **down**, never up. `--hz` below the true rate is harmless (commands get a generous deadline);
above it produces the sprint-and-wait stutter.

Also raise `--execute-steps` if inference is slow — it controls how many waypoints run per plan, so
a higher value means fewer inference calls per second:

```bash
--execute-steps 25    # the spec default; do not go below this on the robot
```

If you can, the *right* fix is `infer.py`'s approach: it runs a **background observer thread**
because `robot.get_observation()` blocks waiting for camera frames over zenoh. `online_rl.py` calls
it synchronously inside the loop, so camera latency adds directly to tick time. Porting that thread
across is the single highest-value change available to you — but **measure the rate first**, so you
know whether the bottleneck is the cameras or the model.

---

## 5. Camera orientation — do not "fix" this

The wrist cameras deliver frames **transposed** from the training set.

| | scene | wrists |
|---|---|---|
| training set (`meta/info.json`) | 376×672 | **424×240** portrait |
| what the camera delivers | 376×672 | **240×424** landscape |
| after `--wrist-rot 90` | 376×672 ✓ | **424×240** ✓ |

`spec_sg2.py` *requests* 240×424 for the wrists; the plugin hands back the transpose anyway. Both
`infer.py` and `online_rl.py` default to `--wrist-rot 90` to correct it. The scene camera is
**already correct and is never rotated** — rotating all three breaks the one view that was right.

This was a live bug on the workstation until 2026-08-21: `online_rl.py` had no rotation at all, so
the policy was fed sideways wrist views and behaved erratically. Measured cost of getting it wrong:

```
as-is     commands 0.034 rad, R/L ratio 1.00
rot90CCW  commands 0.169 rad, R/L ratio 2.78     (demos: 0.232, 2.90)
```

**GATE 0** in `robot_session.connect()` now verifies frame shapes against `meta/info.json` before
the arm may move, and names the offending camera:

```
camera frames do not match what the policy was trained on (--wrist-rot 0):
    cam_left_wrist       got (240, 424, 3)   expected (424, 240, 3)  <- transposed
    cam_right_wrist      got (240, 424, 3)   expected (424, 240, 3)  <- transposed
  every mismatch is a transpose -- try --wrist-rot 90
```

If you see this, **change the flag, do not edit the shapes or disable the gate.** If a camera is
ever physically remounted, the gate fires and the correct response is a new `--wrist-rot` value.

---

## 6. Topics — nothing here changes

Running on the robot does **not** change the wiring. Same topics, same arbitration.

| writer | topic |
|---|---|
| leader device (100 Hz, continuous) | `/leader/...` |
| this stack | `/policy/joint_trajectory_command_broadcaster_{left,right}/joint_trajectory` |
| `arm_freeze_gate` → the arms | `/gated/arm_{l,r}_controller/joint_trajectory` |

`arm_freeze_gate` is the **only** writer to the controllers. It arbitrates leader vs policy; we
never publish to `/gated/` or `/leader/` ourselves.

- LEFT joystick — POLICY ↔ HUMAN
- RIGHT joystick — freeze / unfreeze
- status on `/arm_freeze/status`, latched at 1 Hz: `mode=POLICY left=free right=FROZEN`

One advantage of running locally: the single-writer check can finally *run*. On the workstation
there is no ROS CLI, so it printed "proceeding by contract" — an assumption. On the robot, verify:

```bash
ros2 topic info /policy/joint_trajectory_command_broadcaster_right/joint_trajectory --verbose
```

Exactly one publisher. More than one means two inference processes are live; kill one before
running.

**~~Outstanding request: allow the LEFT joystick toggle while frozen.~~ DONE 2026-08-20.**
Freeze → switch → release is now the supported sequence; the refusal was removed and the log reads
`MODE -> HUMAN (while FROZEN; applied on release)`. On release, `toggle_freeze` reads whichever mode
is then active and applies the matching transition — clutch offset into HUMAN, ease-in into POLICY.
No mid-motion switch is required.

**Single-writer check, answered.** `Publisher count: 0` when idle, 1 when running; `Subscription
count: 1` (the gate). Note the gate subscribes only to the **active** source — in POLICY mode it does
not subscribe to `/leader/...` at all, so a `--verbose` listing on the leader topic will not show it.

**Gate-side jitter cause, found and fixed by the robot agent.** The gate was a single-threaded rclpy
relay deserialising ~400 msg/s it then discarded — 200 of them `/leader/...` JointTrajectory ignored
in POLICY mode. **Our 18–21 Hz command stream was queueing behind that.** Fixed by subscribing to the
active source only. `/leader/dynamic_joint_states` at 100 Hz is still subscribed — the heaviest
message in the graph, read for two joystick numbers, unavoidable because it carries the freeze/mode
gesture. If jitter persists, that is the next suspect and the honest fix is moving the gate to C++.

**A latching bug they found in the three-state labelling:** easing latched true forever when the
source went silent mid-ease, because syncing only cleared on the next relayed message — every
subsequent tick would have been marked eased and **excluded from both methods**. Now expires on
`sync_timeout` via the 1 Hz status timer regardless of traffic.

---

## 7. Running it

Always dry-run new code first. Dawit's standing rule: **never hand him a `--live` command until that
exact code path has been dry-run.** It has nearly broken his robot before.

```bash
# 0. offline, no robot contact, 11 checks
python3 online_rl.py --self-test

# 1. dry run: connects, gates, plans, publishes NOTHING
python3 online_rl.py --dataset <ds> --out <out> \
  --policy-path <ckpt>/pretrained_model --episodes 1 --max-steps 300

# 2. live, only after 1 is clean and --hz reflects the measured rate
python3 online_rl.py --dataset <ds> --out <out> \
  --policy-path <ckpt>/pretrained_model --live --episodes 10 \
  --hz <measured> --execute-steps 25
```

Gates fire in order, all before the arm moves: **GATE 0** camera shapes → **GATE 1** arm at rest →
**GATE 2** start pose within `--start-tol` (0.30 rad; do not tighten, 89% of demo starts fail at
0.15) → **GATE 3** single writer.

Recording: `--auto-start` begins with the policy driving. Takeovers are detected from
`/arm_freeze/status`, not keystrokes, so Dawit can teleop with both hands. ENTER advances the
subtask stage. Frames are stored per tick with a three-state flag: `0` policy, `1` human,
`-1` frozen (excluded from both methods).

---

## 8. Control constants — measured, not guessed

In `spec_sg2.py`. Each was measured against the demonstrations; **changing them without re-measuring
regresses the robot.**

| constant | value | why |
|---|---|---|
| `MAX_ACC` | **2.0** | the jerk lever. 6.0 → 1.74× human jerk, 3.0 → 0.86×, 2.0 → 0.55×. Best on both jerk *and* tracking. |
| `MAX_VEL` | 0.6 | binds only 0.5% of the time; raising it does nothing |
| `EXECUTE_STEPS` | 25 | waypoints per plan. Every replan is a seam. |
| `--seam-blend` | 20 | must be **<** `--execute-steps` or fresh plans never take over |
| `--start-tol` | 0.30 | 0.15 rejects 89% of demo starts |

Several analysis tools defaulted to `MAX_ACC 6.0` while the robot has always used 2.0, which made
every jerk figure ~3× overstated for a while. If you write a new tool, read the value from `spec`.

`control_math.clamp_step()` is the per-tick rate limiter: it clamps against the **last command**
(not the measured pose — chaining off the measurement lets tracking lag inflate the step and turns a
lagging arm into a lurching one) and carries velocity **across replans**. `online_rl.py` used only
`clamp_chunk`, which restarts from `v_prev = 0` every plan; measured peak acceleration was
**18.0 rad/s² against the 2.0 limit** before this was fixed on 2026-08-21.

---

## 9. Known-bad things to not repeat

- **Rolling out on episode 0.** It is training data *and* an outlier — up to 0.3275 rad from the
  median start. Held-out tracking is 0.0641 rad vs 0.0275 on training episodes. Use **episode 32**.
- **Slicing 22-D dataset vectors positionally to 16-D.** The dataset interleaves grippers (idx 7,
  15); the robot's `arms16` layout puts both grippers **last**. Positional slicing silently compares
  `arm_r_joint1` against `gripper_l_joint1`. **Map by name, always.**
- **Squashing camera frames to 224×224.** Costs a measured 8.9× in action accuracy
  (0.0093 → 0.0833 rad) and fails silently. Feed native resolution.
- **`pkill -f <pattern>`** — the pattern matches your own command line. It has killed the ssh
  session twice.
- **Writing task strings to a `tasks.parquet` column.** LeRobot v3.0 keeps them in the **index**.
  Get it wrong and the model reads the literal string `"0"` as its instruction.

---

## 10. Open issue, not yet solved

**The right arm drives downward at rollout start**, as if pushing into the table. Not reproduced on
the workstation and not explained.

What is *verified*: the 22→16 mapping is correct joint-by-joint by name; `demo_start_pose` is the
median first frame across all 35 episodes.

What was **wrongly** claimed and is retracted: that the pose "cannot collide with the table because
MuJoCo reports zero contacts." The MuJoCo model at `robotis_mujoco_menagerie/robotis_ffw/ffw_sg2.xml`
contains **no table, floor, or ground plane at all** — 64 geoms, zero planes. Zero contacts there
proves nothing about a real table. Every "SAFE TO RUN" from that tool means *no self-collision,
within joint limits* and has never meant "will not hit the table."

`trim_incoherent_head()` mitigates part of it: the first plan of a run has no predecessor to
cross-fade with, and its head does not advance monotonically (waypoint-direction cosine −0.55 at
waypoint 2, settling to +0.84…+0.96 only from waypoint 20). Executed from rest that is a visible
lunge. It is a mitigation, not a diagnosis.

**Robot-side evidence, 2026-08-21 — this shifts the diagnosis.** When the right arm dove, the gate
logged near-zero effort (**−1.0, −0.9, −0.5**) on `arm_r_joint1..3` while joints 4–7 tracked exactly.
A joint pushing into a table shows *high* effort — the lift read **460** when it stalled. So this is
not contact: those three joints read as **not being driven**, and the arm is sagging rather than
pushing. Note our publisher is not dropping them: `spec_sg2.py` maps every `arm_r*` joint to
`CMD_RIGHT` by prefix, so all seven are in the message.

The `--trace` capture is what separates the two remaining explanations: if **commanded** positions
for joints 1–3 descend, the policy is driving the dive (our side); if commanded stays put while
**achieved** descends, the joints are limp (robot side).

**What would actually settle it:**

```bash
python3 infer.py ... --trace /tmp/push.csv
```

That logs commanded vs achieved per joint per tick. A commanded pose the arm cannot reach shows up
immediately as a persistent commanded-minus-achieved gap on the downward joints. Also send the
**real table height** relative to the robot base — the pose has never been checked against it,
because the only model available lacks a table.

---

## 11. What to send back

1. **Measured loop rate** on the robot, and the `--hz` you settled on. This determines whether the
   background-observer port is needed.
2. Whether GATE 0 passes on first connect, and at what `--wrist-rot`.
3. Publisher count on `/policy/...` — confirming the single-writer assumption we could never check.
4. `--trace` CSV from a run where the right arm dives, plus the table height.
5. ~~Whether the LEFT-joystick-while-frozen toggle can be enabled.~~ **Answered: enabled 2026-08-20.**
