# Deploying checkpoint-30000 on the robot

**For: Dawit, on `ffw-snpr48a1106`. Date: 2026-08-26.**

The model to deploy is **`checkpoint-30000`** from the fixed-label run. Measured on held-out
episode 32, at the driver grasp — the phase that was failing:

| | Cartesian error | systematic (drift) |
|---|---|---|
| old model 014400 | 54.0 mm | **1.00** — every prediction off the same way |
| **checkpoint-30000** | **21.4 mm** | **0.48** |
| natural spread between your own demos | 27 mm | — |

---

## 1. Get the checkpoint onto the robot

It is on HF (private) and on this workstation.

```bash
# on the robot, from HF (fastest -- ~40 MB/s, 6.4 GB)
hf download dawity/groot_screwing35 --include 'follower/checkpoint-30000/*' \
   --local-dir /tmp/ckpt30k
mkdir -p ~/cyclo_intelligence/docker/workspace/model/groot/screwing35_follower_30k
cp /tmp/ckpt30k/follower/checkpoint-30000/* \
   ~/cyclo_intelligence/docker/workspace/model/groot/screwing35_follower_30k/
```

`workspace/` is bind-mounted into the container as `/workspace`, so inside the groot container the
path is:

```
/workspace/model/groot/screwing35_follower_30k
```

**Verify all 15 files landed**, especially both shards — a truncated download gives a confusing
load error rather than a missing-file error:

```bash
ls ~/cyclo_intelligence/docker/workspace/model/groot/screwing35_follower_30k
# must include: model-00001-of-00002.safetensors (4.7G), model-00002-of-00002.safetensors (1.8G),
#               model.safetensors.index.json, config.json, processor_config.json,
#               statistics.json, dataset_statistics.json, final_processor_config.json, ...
du -sh ~/cyclo_intelligence/docker/workspace/model/groot/screwing35_follower_30k   # ~6.4G
```

---

## 2. Smoke-test it inside the container BEFORE touching the robot

The container ships its own smoke script. Run it with `publish_to_robot` off — nothing moves.

```bash
docker exec -it groot_server python3 /opt/groot/scripts/smoke_groot_n17.py \
  --model-path /workspace/model/groot/screwing35_follower_30k
```

If that path is wrong inside the container, find it with:
`docker exec groot_server find / -name smoke_groot_n17.py -maxdepth 6 2>/dev/null`

**This is the step that catches a bad copy.** It costs a minute and it is the difference between a
load error at the terminal and a load error with the arm powered.

---

## 3. Load it — dry run first

Through the UI (Inference page) or the service directly. Fill `InferenceCommand` LOAD as:

| field | value |
|---|---|
| `command` | `0` (LOAD) |
| `model_path` | `/workspace/model/groot/screwing35_follower_30k` |
| `embodiment_tag` | `new_embodiment` |
| `robot_type` | `ffw_sg2_rev1` |
| `task_instruction` | `Grab the orange bolt` (stage 0 — see §5) |
| **`publish_to_robot`** | **`false`** for the first run |
| `action_request_mode` | `async` |
| `acceleration_mode` | `pytorch` for now (see §6) |

`publish_to_robot=false` / `inference_mode="simulation"` gives preview-only trajectories in the 3D
viewer. Watch one full task there before allowing it to publish.

---

## 4. Control constants MUST change — this is the one that bites

This model has a **16-step action horizon**, not 40. `seam_blend()` cross-fades
`min(seam, horizon, horizon - execute_steps)` waypoints, so with the current `EXECUTE_STEPS = 25`
the third term is **zero** and the cross-fade **silently disappears** — every re-plan seam executes
raw. That is exactly the jerk we removed earlier this month.

Verified numerically:

| horizon | execute_steps | seam_blend | waypoints actually blended |
|---|---|---|---|
| 40 | 25 | 20 | 15 — the old, working setup |
| **16** | **25 (today's spec)** | 20 | **0 — no blending at all** |
| **16** | **10** | **6** | **6 — correct** |

So set, for this model:

```
spec.EXECUTE_STEPS = 10
--seam-blend 6
```

`control_math.check_horizon()` prints the required values on the first re-plan of every run, so if
you forget it will tell you rather than degrade quietly.

Unchanged and still correct: `MAX_VEL = 0.6`, `MAX_ACC = 2.0`. Both were measured, and `MAX_ACC 2.0`
was best on **both** jerk and tracking.

---

## 5. The five stage sentences — byte-for-byte

The policy is conditioned per frame. These must match training exactly; a missing full stop is a
different string to the tokenizer.

```
0  Grab the orange bolt
1  Place the orange bolt into the hole
2  Grab the driver.
3  Screw in the bolt by pushing down.
4  Go back to home after done.
```

Note 0 and 1 have **no** trailing period; 2, 3 and 4 **do**. Advance stages with
`InferenceCommand.UPDATE_INSTRUCTION` (command `6`) — it re-conditions without reloading the model.

Without advancing, the policy performs stage 1 and then sits there. It was trained to be told.

---

## 6. TensorRT — optional, and after you have a baseline

The container ships the builder. Run it **outside** the inference start path:

```bash
docker exec -it groot_server python3 /opt/groot/runtime/prepare_trt_engine.py \
  --model-path /workspace/model/groot/screwing35_follower_30k \
  --robot-type ffw_sg2_rev1
# writes <model_path>/dit_model_bf16.trt
```

Then LOAD again with `acceleration_mode = tensorrt_dit`.

**Get a working pytorch baseline first.** TensorRT accelerates the DiT only, and if behaviour
changes you want to know whether it was the model or the engine.

---

## 7. What to expect, and what would mean it worked

The measured improvement is **agreement with your demonstrations**, open-loop. That is not the same
as task success, and I want to be straight about the gap:

- **Should be visibly better:** the right arm no longer drifting to one side when approaching the
  driver. That was a 54 mm systematic offset; it is now 21 mm and about half as directional.
- **Not measured at all:** whether it completes the task. Open-loop evaluation feeds the policy the
  *human's* observation at every step, so it cannot show error accumulation — the failure mode
  closed-loop control actually suffers from.
- **Unchanged:** the right-arm-dives-at-start issue from before is a separate, still-unexplained
  problem. Your effort logs (−1.0/−0.9/−0.5 on `arm_r_joint1..3` while 4–7 tracked, against 460 on
  the stalled lift) pointed at those joints not being driven, not at the policy. If it recurs,
  capture `--trace` and we settle it.

If the drift is gone but it still fumbles the grasp, that is the **HG-DAgger** case: the corrections
you record there are worth more than more demonstrations, because they land exactly where the policy
is wrong.

---

## 8. Order of operations

1. copy checkpoint → `workspace/model/groot/screwing35_follower_30k`, verify 15 files / 6.4 GB
2. smoke test in the container
3. LOAD with `publish_to_robot=false`, watch a full task in the 3D viewer
4. set `EXECUTE_STEPS=10`, `--seam-blend 6`
5. LOAD with `publish_to_robot=true`, hand on the freeze
6. only then consider `tensorrt_dit`
