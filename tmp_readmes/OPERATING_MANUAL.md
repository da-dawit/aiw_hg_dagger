# FFW SG2 + A2 Leader — Operating Manual

Plain-language, exact-commands guide to bringing this robot up, teleoperating it, running the policy, and collecting HG-DAgger data. Written for someone who has never touched this system before. No ROS/robotics jargon assumed beyond what's explained inline.

---

## 1. What this system actually is

Three separate pieces of software have to be running at the same time:

1. **The robot's own ROS2 stack** — runs directly on this machine (not in Docker). Two halves:
   - **Follower** (`ffw_sg2_rev1`) — the physical arms/head/lift/base you're actually controlling.
   - **Leader** (`ffw_a2_leader`) — the two hand controllers ("leader arms") a human moves; their motion is what drives the follower.
2. **`cyclo_intelligence`** — a Docker container. Runs the web UI (the page you look at in the browser), the recording/data pipeline, and the orchestration logic that ties leader input, follower control, and the AI policy together.
3. **`groot_server`** — a separate Docker container. This is the AI model itself (GR00T policy). It only matters when you're running inference (the robot moving on its own), not for plain teleoperation.

None of these three auto-repairs the others. If one is down, the other two look "connected" but nothing useful happens.

---

## 2. Bringing everything up, in order

**You do this yourself — never ask an AI assistant to run these for you.** These are the exact commands, run on the host machine (not inside a container):

```bash
# 1. Follower (the physical arms/head/lift you're controlling)
ros2 launch ffw_bringup ffw_sg2_follower_ai.launch.py ros2_control_type:=ffw_sg2_follower_smooth

# 2. Leader (the two hand controllers)
ros2 launch ffw_bringup ffw_a2_leader_ai.launch.py

# 3. The bridge script that makes the leader's pause/resume gesture work
python3 /home/robotis/ai_worker/docker/workspace/leader_bridge.py
```

Run them in **separate terminals**, follower first, then leader, then the bridge script. Each one keeps running in its terminal — don't close it.

`cyclo_intelligence` and `groot_server` are Docker containers and are usually already running in the background (check with `docker ps`). If either is down:

```bash
docker start cyclo_intelligence
docker start groot_server
```

### 2.1 How to know it actually worked (don't just assume)

- **Follower up correctly**: the head should move to its home position on its own within a few seconds of the follower launch finishing. If it doesn't move at all, or moves to a position that looks wrong (head touching the body, or not lifting up), something is broken — see Troubleshooting.
- **Leader up correctly**: doing the freeze-gate gesture (see §4) should show a status change; the leader arms should feel like they can drive the follower when you pick them up.
- **`cyclo_intelligence` up correctly**: the web UI loads, camera feeds are visibly moving (not frozen), and CPU/RAM gauges at the top show real numbers.
- **`groot_server` up correctly**: in the UI's Task Information panel, "GROOT Docker" shows "Running" (not "Stopped"), and eventually "TRT Engine: Ready".

### 2.2 If you have to kill and restart something mid-session

- **Killing the follower or leader**: just Ctrl+C the terminal it's running in, or `kill` the `ros2 launch` process. Before relaunching, make sure nothing from the old launch is still alive — check with `ps aux | grep ros2_control_node`. If an old `ros2_control_node` is still running when you launch a new one, they'll fight over the same USB serial port and the new one can **segfault**. Kill the old one first (`kill <pid>`, or `kill -9` if it won't die), confirm it's actually gone, then launch.
- **Restarting `groot_server`**: `docker restart groot_server` is safe any time nothing is actively mid-inference. It takes ~20-40 seconds to reload the model — watch the UI's "GROOT Docker" status.
- **Restarting `cyclo_intelligence`**: this is riskier — see §8.3 (Troubleshooting: duplicate processes) before doing this.

---

## 3. Teleoperation — normal vs. this robot's actual setup

There are two different control modes built into the software. **This robot is configured to use mode 2 (relative-delta), not mode 1.**

### Mode 1 — absolute (the "normal" way most systems work)
The follower's joints go directly to wherever the leader's joints currently are. Simple, but if the leader and follower start out in very different poses, the follower **snaps/jumps** the instant you engage.

### Mode 2 — relative-delta (what this robot actually runs)
On engage, the software remembers where the leader is *and* where the follower is, then moves the follower **by the same amount the leader moves**, not **to** the leader's absolute position. No jump on engage, at the cost of a small amount of lag (the follower "chases" the leader rather than mirroring it instantly). This is controlled by two tunable numbers in `ffw_bringup/config/ffw_a2_leader/ffw_a2_teleoperation.yaml`:
  - `kp_position` / `kp_orientation` — how tightly the follower chases the leader. Higher = snappier but more overshoot risk. Currently `70.0` (raised from a default of `50.0` for responsiveness).
  - `max_joint_velocity` — hard speed cap, currently `1.5` rad/s.

### What's deliberately disabled on this robot, and why

- **Head jog via the leader's left joystick**: disabled (`sensorxel_l_joy_jog_scale: 0.0` and the joystick's own trajectory publisher is pointed at a dead-end topic). This is because the joystick control was found to continuously re-broadcast a stale head position at 100Hz onto the same topic the follower's head controller listens to, which **permanently overrides any other head command** (including the automatic home-on-startup). Since the leader hardware doesn't even have head joints of its own, this joystick control was never functional anyway — disabling it costs nothing.
- **Lift jog via the leader's right joystick**: still enabled (`sensorxel_r_joy_controlled_joints: [lift_joint]`), this one does work.
- **Head/lift/mobile base during AI inference**: the policy is configured with `FROZEN_ACTION_MODALITIES=head,lift,mobile` — meaning even though the AI model technically predicts values for these joints, they are never actually sent to the robot. Only the two arms move under AI control. This is because training data was recorded with head/lift/base held at a fixed pose the whole time, so commanding them from the policy would just fight the holding controllers.

### The actual home / initial pose

The head's home pose is defined in **two places that must be kept in sync manually** — there's no code that derives one from the other:
1. `ffw_bringup/config/ffw_sg2_rev1_follower/ffw_sg2_follower_initial_positions.yaml` — used automatically at follower startup.
2. `cyclo_intelligence/orchestrator/ui/src/constants/homePose.js` — used by the UI's "Home" button.

Current confirmed-correct value: `head_joint1 = 0.782330201821470`, `head_joint2 = 0.069029135454647`. Positive `head_joint1` = pitch **down**; negative = pitch up. The head's hardware ceiling (`Max Position Limit` on the Dynamixel, `dxl61`) is `2560` ticks ≈ `0.785` rad — don't set a home pose above that or it'll be physically unreachable regardless of what any software command says.

---

## 4. HG-DAgger gate operation (freeze / pause-resume)

### The takeover loop, in the order you actually do it

Assume the policy is running a rollout.

1. **Joystick drag-down, hold ~1 s** → policy stops. INFERENCING → **PAUSED**.
2. **Press tact** on the arm you want → that arm engages. You are now driving, and these are
   the frames HG-DAgger trains on.
3. **Press tact again** → that arm disengages. The gripper holds its teleoped value.
4. **Joystick drag-down, hold ~1 s** → policy resumes. PAUSED → **INFERENCING**.

The tact is a *toggle* — same button engages and stops. The drag-down gesture is also one
topic with two meanings (`/leader/teleoperation/toggle_pause_request`, `std_msgs/Empty`); the
orchestrator decides pause-vs-resume from the phase it is currently in.

The gaps between steps 1→2 and 3→4 — policy paused, no arm engaged, nobody driving — are
labelled `-1` and excluded from training. That is correct and expected; you do not need to rush.

**Engage does not sweep.** SG2 runs control mode 2 (relative delta): on engage it captures a
leader anchor and a follower anchor and commands
`goal = follower_anchor + (leader_now - leader_anchor)`. Nothing jumps even if the leader and
follower are far apart. Mode 1 (absolute MoveJ) *does* sweep, which is why `slow_start` is
enabled there and disabled in mode 2.


The `arm_freeze_gate` process (auto-started by the follower launch) is the single point that decides whether the arms follow the human (teleop) or the AI policy, and can freeze both arms instantly.

**Gestures** (push the leader joystick down and hold ~0.8s, then **release** — don't hold it down, there's a 2-second cooldown so each push is one toggle):

| Gesture | Effect |
|---|---|
| RIGHT joystick down | Freeze / release **both arms** |
| LEFT joystick down | Toggle mode: **POLICY** ↔ **HUMAN** |

**To run the AI policy**: hold LEFT joystick down 0.8s → status should flip to `mode=POLICY`. Keep your hand near the physical e-stop and the table clear. If anything looks wrong, RIGHT joystick 0.8s freezes instantly.

**Status format** (shown in the UI, republished once a second and immediately on any change):
```
mode=HUMAN easing=false left=free right=false
```
- `easing=true` means the gate is currently smoothing a transition (e.g. just came out of a freeze) — the published action during this window is a blend, not the real source's output.
- Releasing a freeze in **HUMAN** mode does *not* move the arm — it just re-zeros the leader's reference point, so a relaxed leader against a folded-up follower is safe.
- Releasing a freeze in **POLICY** mode eases back in gradually (never jumps straight to the policy's raw target).

---

## 5. Collecting HG-DAgger data — exact workflow

This is the process for recording an episode with a running policy, human corrections, and correctly per-phase-labeled instructions — the actual workflow this robot's data collection is built around.

1. Make sure the **Online-RL** page in the UI is open (not the plain Record page — the plain Record page's subtask feature is for a different, discrete recording mode this robot doesn't use for AI training).
2. In **Task Information**, set the top **Task Instruction** field to the literal, stable overall-task description — e.g. `Screw the orange bolt into the hole using the driver.` — **before** you press Start.
3. Set the **Preset** dropdown to the first phase (e.g. "Grab the orange bolt with the left arm"), confirm it's in the Task Instruction box, then click **Update Task Instruction**. *(Note: this specific click is what actually matters — it's the only thing that both sends the instruction to the policy **and** gets logged with a real timestamp for later labeling. Just picking something from the dropdown without clicking this button does nothing.)*
4. Start recording (or use "Auto-record on inference start" if you're about to hit Start policy anyway).
5. As the physical task moves from phase to phase, switch the Preset dropdown to the next phase's instruction and click **Update Task Instruction** again, every time. Do this for **every** phase change — this is what produces correctly per-phase-labeled training data instead of one blob of "whatever was set last."
6. When the whole task is done, **Stop & save**, then label the outcome (SUCCESS / FAILURE / DISCARD) in the modal that appears. Labeling is required — an episode saved without an outcome gets dropped entirely later, not just missing a reward.
7. Repeat for each episode. The session folder (`Task_<timestamp>_inference_MCAP`) stays the same across all episodes in one sitting — episode numbers just increment (`0`, `1`, `2`, ...). Starting a completely new session (e.g. after a `cyclo_data` restart) starts a fresh folder and resets numbering to `0`.

### What NOT to do
- Don't leave the Task Instruction on one preset for the whole episode if the physical task actually has multiple phases — every frame will get labeled with whatever was set last, and the earlier phases end up mislabeled or unlabeled.
- Don't skip clicking "Update Task Instruction" and assume the dropdown alone did something — it didn't.
- Don't assume a recording is safely saved just because you clicked something — always confirm via the outcome-labeling modal actually completing, and ideally verify the resulting episode on disk (§6).

---

## 6. Verifying a recording is actually good (don't just trust the UI)

The UI can say "Recording started!" and still have produced nothing usable. Real verification, every time, looks like this:

```bash
# Find the current session folder
docker exec cyclo_intelligence bash -lc 'ls -td /workspace/rosbag2/Task_*inference_MCAP | head -1'

# Check one episode's metadata (replace <session> and <N>)
docker exec cyclo_intelligence bash -lc 'python3 -m json.tool /workspace/rosbag2/<session>/<N>/episode_info.json'
```

In the output, check:
- `"outcome"` — should be `SUCCESS`, `FAILURE`, or `DISCARD`, never missing.
- `"video_remux_status": "done"` and `"transcoding_cameras_failed": {}` — if this isn't empty, a camera failed to record.
- `"segments"` — should have **one entry per phase you actually switched through**, each with a distinct `sub_task_instruction` and a non-overlapping `frame_duration` range. If there's only one entry covering the whole episode, the per-phase instruction switching didn't happen (or wasn't logged) for that episode.
- `"instruction_change_log"` — the raw timestamped log this all gets built from; useful for spot-checking timing.

Then verify the actual video files exist and are real (not zero-byte or corrupted):

```bash
docker exec cyclo_intelligence bash -lc '
d=/workspace/rosbag2/<session>/<N>/videos/<N>_0
for f in "$d"/*.mp4; do
  echo "=== $(basename $f) ==="
  ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,duration -of default=noprint_wrappers=1 "$f"
done
'
```

All 4 cameras should show real dimensions, matching durations, and a frame count consistent with ~30fps (duration × 30 ≈ nb_frames).

---

## 7. Converting and pushing to Hugging Face

Conversion is triggered via a ROS service call (there's a UI path too, but this is the exact underlying call):

```bash
docker exec cyclo_intelligence bash -lc 'source /opt/ros/jazzy/setup.bash; source /root/ros2_ws/install/setup.bash 2>/dev/null; ros2 service call /data/convert interfaces/srv/StartConversion "{dataset_path: \"/workspace/rosbag2/<session>\", robot_type: \"ffw_sg2_rev1\", robot_config_path: \"/root/ros2_ws/install/shared/share/shared/robot_configs/ffw_sg2_rev1_config.yaml\", convert_v21: false, convert_v30: false, fps: 30}"'
```

Notes:
- `fps: 30` should always be passed explicitly (matches this robot's actual camera/inference rate). Leaving it `0`/default now correctly resolves to `30` too (this was a bug — used to silently default to `15` — now fixed).
> **The v2.1 export drops the online-RL labels.** `to_lerobot_v30.py` writes `intervention`,
> `reward`, `done` and `sample_weight`; `to_lerobot_v21.py` does not. Isaac-GR00T requires v2.1,
> so a v2.1-only export of an online-RL session arrives at training with **no record of which
> frames were yours**. Until that is fixed (`tmp_readmes/IMPROVEMENTS.md` §3), **always convert
> v3.0 as well and keep it** — it is the only copy that has the labels.

- `convert_v21: false, convert_v30: false` — both false means "run both formats." Isaac-based tooling needs **v2.1** specifically (v3.0's file layout is incompatible with it), so if you only need one, request `convert_v21: true` explicitly.
- Poll progress via `/data/convert/status` with the `job_id` the start call returns.

Pushing to HF (after registering your token once via the UI's "HF Token" button, or the `/register_hf_user` service):

```bash
docker exec cyclo_intelligence bash -lc 'source /opt/ros/jazzy/setup.bash; source /root/ros2_ws/install/setup.bash 2>/dev/null; ros2 service call /data/hub interfaces/srv/HfOperation "{operation: 0, repo_type: 0, repo_id: \"<your-username>/<dataset-name>\", local_dir: \"/workspace/lerobot/<session>_lerobot_v30\"}"'
```
(`operation: 0` = UPLOAD, `repo_type: 0` = DATASET.)

**Always verify the upload actually landed** — don't just trust a success message:
```bash
docker exec cyclo_intelligence bash -lc '
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files(\"<your-username>/<dataset-name>\", repo_type=\"dataset\")
print(f\"Total files: {len(files)}\")
"
'
```

---

## 8. Troubleshooting — things that actually went wrong tonight, and the fix

### 8.1 Robot won't relaunch / segfaults on startup
An old `ros2_control_node` from a previous launch is still alive and holding the USB serial port. Find it (`ps aux | grep ros2_control_node`), kill it, confirm it's actually gone, then relaunch.

### 8.2 UI shows stale/wrong status, buttons don't respond, "Clear" doesn't work
Almost always a duplicate-process problem: two copies of `cyclo_data_node` and/or `orchestrator_node` running at once, fighting over the same ROS topics/services. Check:
```bash
docker exec cyclo_intelligence bash -lc 'ps aux | grep -E "cyclo_data_node|orchestrator_node" | grep -v grep'
```
If you see more than one of either, something is duplicating the stack. This container runs its whole backend (orchestrator + cyclo_data + rosbridge + video server) as **one single s6-supervised service called `cyclo_intelligence`** — don't try to separately manage an `orchestrator` or `cyclo_data` s6 service alongside it, that's exactly what causes the duplication. If it's already duplicated, kill everything and restart just the `cyclo_intelligence` service cleanly:
```bash
docker exec cyclo_intelligence bash -lc '/command/s6-svc -r /run/service/cyclo_intelligence'
```
then re-check the process list — sometimes one stray process (usually `rosbridge_websocket`) doesn't die on the first try and needs a manual `kill -9`.

### 8.3 A container's model/service looks "stuck loading" forever
Check whether the container is actually running first (`docker ps`) before assuming it's a frontend bug — a genuinely stopped container will show "Loading..." forever in the UI because there's nothing on the other end to respond.

### 8.4 Camera video files come out empty / "missing video file"
This was a real bug (not a timing fluke) tonight: `video_recorder.py`'s camera subscription was accidentally commented out in a local, uncommitted edit, and a separate `_on_frame` callback had a stray `return` at the top making it dead code. Both are fixed now (verified against git's committed baseline). If this ever comes back, check `cyclo_data/cyclo_data/recorder/video_recorder.py` for exactly these two things before assuming it's a hardware/camera problem — the raw camera ROS topics can be streaming perfectly fine while the recorder itself silently receives nothing.

### 8.5 Every frame of a recording has the same subtask label
Means the per-phase "Update Task Instruction" clicks either didn't happen or aren't being logged. Check `episode_info.json`'s `instruction_change_log` — if it has 0 or 1 entries for a multi-phase episode, the operator didn't click Update Task Instruction at each phase transition during recording.

### 8.6 The whole machine reboots unexpectedly
Happened once tonight, likely from sustained heavy load (repeated model reloads + TensorRT builds + container restarts). After a reboot: all Docker containers with a proper restart policy come back automatically **except** `groot_server`, which needs a manual `docker start groot_server`. The robot's own ROS bringup (follower/leader) is **not** containerized and does **not** survive a reboot — you always have to relaunch it yourself from scratch.

---

## 9. Hardware facts worth remembering

- **Head joint sign convention**: positive `head_joint1` = pitch down, negative = pitch up. Confirmed `~50°` down ≈ `0.87` rad.
- **Dynamixel position units**: raw ticks, center = 2048, scale = `2π/4096` rad/tick. `Max Position Limit` / `Min Position Limit` are firmware-level hard clamps, completely separate from and not automatically synced with the URDF's software joint limits — raising one does not raise the other.
- **XML comments**: a bare `--` (double hyphen) anywhere inside a `<!-- ... -->` comment body is invalid XML and will break `xacro` parsing at launch time. Use periods or semicolons instead of dashes in xacro file comments.
- **The follower's "smooth" ros2_control profile is a genuine tradeoff, not a bug**: it deliberately slows the servo's `Profile Acceleration Time`/`Profile Time` registers for cleaner AI-policy motion, at the direct cost of felt teleoperation lag, because both teleop and policy commands go through the exact same hardware register — there is no way to make the servo behave differently depending on who's commanding it.
