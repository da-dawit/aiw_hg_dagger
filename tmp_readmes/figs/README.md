# Figures for HIL_MANUAL.tex

The 18 files here are generated from recorded data; each has its script beside it, so any of them
can be rebuilt after a new session or checkpoint.

| script | produces |
|---|---|
| `mkfigs.py` | `camera_views.png` |
| `mkfigs5.py` | `crop_methods.png`, `crop_evidence.png`, `crop_applied.png` |
| `mkattn3.py` | `attn_map.png` (needs two `attn_map.py --dump` runs, one per crop) |
| `mkfigs4.py` | `parquet_columns`, `launcher_preflight`, `verify_output`, `mixture_proportions`, `checkpoint_sweep` |
| `mkhw.py` | `hardware_rollout.png` and the four `ui_state_*` / `ui_instruction` frames |
| `replot3d.py` | `rollout3d.png` from `rollout3d.py --dump` |

`intervention_timeline.pdf` is drawn in TikZ; its source is in the session scratchpad.

## Seven files are not here

They are photographs, screen captures or diagrams that were not produced by these scripts, and the
manual will not compile without them:

    session overview.png     ai_worker.png        ai_worker record.png
    leader_controls.png      overall_ui.png       trainingloss.png
    hg-dagger.png

`ai_worker.png` and `ai_worker record.png` are the two architecture diagrams; the rest are photos,
a training-loss plot and the HG-DAgger loop diagram.

Two of the names contain a space, which some TeX distributions handle and others do not. Renaming
them to `session_overview.png` and `ai_worker_record.png`, and amending the two
`\includegraphics` calls, removes a class of build failure that only appears on another machine.
