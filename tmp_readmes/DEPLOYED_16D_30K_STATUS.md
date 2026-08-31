# Deployed: `16d_30k/checkpoint-30000` on `groot_server` — status and configuration

**2026-08-28. First live test result: best performance seen so far.** Screwing-down stage not yet
reliable; everything else (grab bolt, place bolt, grab driver) working well. This document records
exactly what's live on this robot right now, so it can be reproduced or rolled back precisely.

This is a companion to `DEPLOY_16D_30K.md`, not a replacement — that doc describes a standalone
`aiworker_deploy/infer.py` pipeline that does **not** run on this robot. Everything below is what
was actually integrated into the real, running pipeline: the `groot_server` container
(`main_runtime`/`engine-process`/`ActionChunkProcessor`/`ControlLoop`).

---

## What's deployed

- Model: `dawity/groot_screwing35` → `16d_30k/checkpoint-30000`
- Location: `/workspace/model/groot/screwing35_base_30k` (inside `groot_server`)
- Also downloaded, untested: `checkpoint-20000` → `/workspace/model/groot/screwing35_base_20k`
- TensorRT engine: built successfully — `dit_model_bf16.trt`, 2.19GB, default 4096MB workspace.
  (Two earlier attempts on this checkpoint OOM'd the host — see "TRT build memory" below.)
- Kept deliberately separate from the old run's checkpoints (`screwing35_follower_20k`/`_30k`,
  22-dim, letterboxed) so both remain usable without collision.

## What changed from the previous model

- **16-dimensional action space**, not 22 — the frozen/near-constant `head`, `lift`, `mobile` dims
  from the old run are gone. Confirmed via `processor_config.json`: `modality_configs.new_embodiment`
  has no head/lift/mobile keys.
- **Action horizon: 16 steps** — confirmed via `processor_config.json`'s
  `new_embodiment.action.delta_indices`, length 16. Same horizon as the old checkpoints, so the
  `BLEND_DURATION_S`/`CHUNK_ALIGN_WINDOW_S` tuning done earlier for horizon-16 models already
  applies correctly here with no extra change needed.
- **Square crop instead of letterboxing** for all three cameras (see below).
- **Arm-named per-stage instructions**, and per-stage conditioning is real for this checkpoint
  (unlike the old one, where subtask strings were off-distribution — see below).

---

## Square crop integration

The checkpoint was trained with a per-camera square crop, not GR00T's stock letterbox padding.
Skipping this is the "fails silently" failure mode both deploy docs warn about — no error, just a
worse policy.

- **Patch**: `crop_to_square.patch` applied to
  `/gr00t/gr00t/model/gr00t_n1d7/image_augmentations.py` inside `groot_server`. Adds
  `class CropToSquare`, and swaps it in for `LetterBoxPad()` in
  `build_image_transformations_albumentations()`, gated by `GR00T_SQUARE_CROP`.
- **Made durable.** `/gr00t` is baked into the image, not bind-mounted, so a live `docker exec`
  patch does not survive a container recreate. Fixed by adding an idempotent apply-step to the
  *existing*, already-bind-mounted `engine-process` startup script:
  `cyclo_intelligence/cyclo_brain/policy/groot/s6-services/engine-process/run` — checks for
  `class CropToSquare` in the target file on every container start, applies
  `crop_to_square.patch` (copied alongside the `run` script, so it's bind-mounted too) if missing.
  Verified: survives a full `docker compose down`/`up` recreate.
- **ROI config**: `/workspace/config/roi_hybrid.json` (bind-mounted `/workspace`, persists on its
  own). Head keeps its full 672×376 field of view (padded to square, not cropped); both wrists crop
  to their bottom 240×240 (the ceiling is gone — it was 88% of the wrist frame at the grasp
  instant).
- **Enabled via env vars** on `groot_server`, set in `cyclo_intelligence/docker/docker-compose.yml`:
  ```
  GR00T_SQUARE_CROP=1
  GR00T_ROI_JSON=/workspace/config/roi_hybrid.json
  ```
- **Verified this actually applies to the real pipeline, not just the patched library in
  isolation**: read `processing_gr00t_n1d7.py` directly — `letter_box_transform` is an unused
  backward-compat parameter (stored, never read), and `build_image_transformations_albumentations()`
  is called unconditionally whenever the processor initializes (i.e. on every LOAD), regardless of
  which checkpoint or acceleration mode (`pytorch`/`tensorrt_dit`) is selected. The crop is a
  property of `groot_server`'s current runtime state, not something baked into a specific
  checkpoint or TRT engine.

## Instruction strings

Added as their own group in `orchestrator/ui/src/constants/taskInstructionPresets.js` —
`SUBTASK_INSTRUCTIONS_16D` — rather than overwriting the old checkpoint's strings, since the old
22-dim checkpoint is still usable as a rollback and uses different (off-distribution) subtask text.

Byte-exact, reproduced from `DEPLOY_16D_30K.md`:
```
0  Grab the orange bolt with the left arm
1  Place the orange bolt into the hole with the left arm
2  Grab the driver with the right arm
3  Screw in the bolt by pushing down with the right arm
4  Return both arms to home
```
Shows in the UI dropdown as "Subtasks — 16d_30k trained (arm-named)". Do not use these against the
old `follower/checkpoint-30000`, and do not use the old off-distribution subtask strings against
this checkpoint.

## TRT build memory

The DiT build (ONNX export → TensorRT optimization) competes for memory with everything else on
this box — it's a Jetson (Orin, confirmed via `tegrastats`), so there is no separate GPU VRAM; CUDA
and host allocations share one ~30GB pool. Two build attempts failed with a kernel-level, host-wide
OOM kill (confirmed via `journalctl -k`, exit code 137) — the second one got much further (into
real TensorRT layer optimization, ~5.9GB resident) than the first (ONNX export only, ~1.8GB), with
no concurrent inference session either time. The actual cause was accumulated memory/swap pressure
from a very long session (many container recreates, repeated model loads), not a structural
incompatibility — confirmed by checking `free -h` immediately before each attempt: swap was
partially in use both times. **The successful build ran with a clean, freshly-recreated
`groot_server` and 23GB+ available, 0 swap in use, default 4096MB workspace.** If a future build
fails, check `free -h`/swap usage before assuming the model or script is at fault.

## UI rebuild

`orchestrator/ui` source has no build tooling on the host; `node`/`npm` (v22.23.2) exist inside the
`cyclo_intelligence` container. Copied `src/`, `public/`, `package.json`, `package-lock.json`,
`postcss.config.js`, `tailwind.config.js` in, ran `npm install && npm run build` there, deployed the
output to `/usr/share/nginx/html`. This also picked up an unrelated pending fix (the "Online-RL
Data" tab, present in source since 2026-08-25 but never deployed — the running image predated it by
three weeks).

## Known gaps / next steps

- **Screwing-down stage is the current weak point** — exactly the target of the sample-weight
  change already made to `cyclo_data`'s converter this session: human-correction frames during the
  driver-grasp and screw-in stages now get `_HARD_STAGE_HUMAN_WEIGHT = 6.0` instead of the flat
  `2.0`, matched by keyword (`"driver"`, `"screw"`) against the subtask instruction text. Ready for
  HG-DAgger data collection against this checkpoint as the base policy.
- `spec_sg2.py`'s measured `MAX_VEL=0.6`/`MAX_ACC=2.0` clamps (from the reference `infer.py`
  pipeline, tuned against demonstration velocity statistics) have **not** been compared against
  what `groot_server`'s actual `ActionChunkProcessor`/`ControlLoop` enforces. Still open — worth
  checking before pushing this checkpoint harder.
- `checkpoint-20000` (`screwing35_base_20k`) is downloaded but neither TRT-built nor tested.

## Reference material (not the deployed pipeline)

`infer.py`, `spec_sg2.py`, `groot_policy.py`, `robot_session.py`, `control_math.py` in
`tmp_readmes/` are a separate, standalone deployment path (its own dry-run/live gating, its own
`EXECUTE_STEPS`/`seam_blend` chunk handling) that was useful for understanding the crop/ROI
mechanism and confirming the checkpoint's real action horizon, but none of it is wired into
`groot_server`. Don't run `infer.py` expecting it to reach the actual robot control path used above.
