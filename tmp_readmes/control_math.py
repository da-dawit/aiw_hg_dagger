"""Pure control math shared by the deployment loop and the offline analysis tools.

Author: Dawit Chun

WHY THIS FILE. `clamp_chunk` is arithmetic on numpy arrays, but it lived in infer.py, which imports
`trajectory` -> `zenoh_ros2_sdk`. An offline sweep that only wanted to rate-limit a trajectory
therefore had to import the entire deployment stack, and failed with a missing zenoh module on any
machine that was not set up to talk to the robot.

Nothing here connects, publishes, or imports anything beyond numpy. infer.py imports these so the
deployed path and the offline analysis apply the SAME limits -- a sweep that clamps differently from
the robot measures a controller you are not going to run.
"""
from __future__ import annotations

import os
import numpy as np


def clamp_chunk(chunk, q_now, dt, max_vel, max_acc, arm_dims):
    """Rate-limit a whole chunk before it reaches the controller.

    Two separate clamps, and the first is the one that matters: the step from where the arm ACTUALLY
    is into the policy's first waypoint. Nothing else guards that, and a mid-training checkpoint can
    put chunk[0] a long way from the current pose.
    """
    out = np.asarray(chunk, dtype=float).copy()
    prev = np.asarray(q_now, dtype=float)
    v_prev = np.zeros_like(prev)
    for i in range(len(out)):
        dq = out[i] - prev
        dq = np.clip(dq, -max_vel * dt, max_vel * dt)
        v = dq / dt
        v = v_prev + np.clip(v - v_prev, -max_acc * dt, max_acc * dt)
        dq = v * dt
        v_prev = v
        out[i] = prev + dq
        prev = out[i]
    #Gripper passes through RAW: it is a class decision with its own hysteresis, and rate-limiting a
    #class produces a value between "open" and "closed" that means neither.
    out[:, arm_dims:] = np.asarray(chunk, dtype=float)[:, arm_dims:]
    return out


def clamp_step(target, q_prev, v_prev, dt, max_vel, max_acc, arm_dims):
    """Rate-limit ONE command against the previous one, carrying velocity. Returns (q, v).

    `clamp_chunk` restarts from v_prev = 0 on every call, which is right for a fresh chunk and
    WRONG for a streaming command: applied one waypoint at a time it constrains |v - 0| <= a*dt,
    i.e. it silently becomes a velocity cap of max_acc*dt (0.2 rad/s at 6.0 and 30 Hz) while
    leaving real acceleration unbounded. Measured exactly that before this existed.

    A streaming controller has to remember how fast it was already going.
    """
    q_prev = np.asarray(q_prev, dtype=float)
    v_prev = np.asarray(v_prev, dtype=float)
    out = np.asarray(target, dtype=float).copy()
    dq = np.clip(out - q_prev, -max_vel * dt, max_vel * dt)
    v = dq / dt
    v = v_prev + np.clip(v - v_prev, -max_acc * dt, max_acc * dt)
    q = q_prev + v * dt
    #gripper passes through raw, as everywhere else
    q[arm_dims:] = out[arm_dims:]
    v[arm_dims:] = 0.0
    return q, v


def check_horizon(chunk_len, execute_steps, seam_blend, warned=[]):
    """Refuse to let a short action horizon silently disable the seam cross-fade.

    seam_blend() blends min(seam, horizon, horizon - execute_steps) waypoints. When
    execute_steps >= horizon the third term is 0, so the blend disappears WITHOUT ANY ERROR and
    every re-plan seam is executed raw -- which is the jerk this cross-fade exists to remove.

    It matters now because ROBOTIS's CYCLO modality config for the FFW SG2 trains with a 16-step
    action horizon (examples/CYCLO/ffw_sg2_rev1_config.py, delta_indices=range(16)) while
    spec.EXECUTE_STEPS is 25, tuned for the 40-step horizon the LeRobot port uses. Loading a
    16-step model with the old constant blends nothing.
    """
    if warned:
        return
    warned.append(True)
    overlap = chunk_len - execute_steps
    blended = min(seam_blend, chunk_len, max(0, overlap))
    if seam_blend <= 0:
        #the operator turned the cross-fade off; that is a choice, not a structural break
        print(f"  [horizon] chunk {chunk_len}, execute {execute_steps}, seam blending OFF",
              flush=True)
    elif overlap <= 0:
        print(f"\n  [horizon] the policy returns {chunk_len} waypoints but --execute-steps is "
              f"{execute_steps}.\n"
              f"            Nothing is left over to cross-fade, so EVERY re-plan seam runs raw.\n"
              f"            For a {chunk_len}-step horizon use --execute-steps "
              f"{max(1, int(chunk_len * 0.625))} --seam-blend {max(1, int(chunk_len * 0.375))} "
              f"(the ratio tuned at 40/25).", flush=True)
    elif blended < seam_blend:
        print(f"  [horizon] chunk {chunk_len}, execute {execute_steps} -> only {blended} of "
              f"{seam_blend} seam waypoints blend.", flush=True)
    else:
        print(f"  [horizon] chunk {chunk_len}, execute {execute_steps}, seam {blended} -- ok",
              flush=True)

def seam_blend(new_plan, prev_plan, prev_i, n, arm_dims):
    """Cross-fade the first `n` waypoints of a fresh plan with the tail of the previous one.

    The live loop does this and the offline sweep did not, which made the offline numbers show a
    velocity spike at every re-plan that the robot would not actually see. Sharing it removes that
    discrepancy.
    """
    if prev_plan is None or n <= 0:
        return new_plan
    out = np.asarray(new_plan, dtype=float).copy()
    m = min(n, len(out), max(0, len(prev_plan) - prev_i))
    if m <= 0:
        return out
    w = np.linspace(0.0, 1.0, m + 2)[1:-1].reshape(-1, 1)
    tail = np.asarray(prev_plan, dtype=float)[prev_i:prev_i + m, :arm_dims]
    out[:m, :arm_dims] = w * out[:m, :arm_dims] + (1.0 - w) * tail
    return out


def jerk(traj, arm_dims=14):
    """Mean |third difference| over the arm dims -- the quantity an operator reads as shaking."""
    t = np.asarray(traj, dtype=float)
    if len(t) < 4:
        return 0.0
    return float(np.abs(np.diff(t[:, :arm_dims], n=3, axis=0)).mean())


def self_test():
    import sys
    print("control_math self-test\n")
    fails = []

    def check(c, m):
        print(f"  [{'ok ' if c else 'FAIL'}] {m}")
        if not c:
            fails.append(m)

    dt, A = 1 / 30, 14
    q0 = np.zeros(16)

    #the streaming clamp must bound BOTH rates across successive calls
    q, v = q0.copy(), np.zeros(16)
    tgt = np.concatenate([np.full(A, 2.0), np.zeros(2)])
    traj = []
    for _ in range(60):
        q, v = clamp_step(tgt, q, v, dt, 0.8, 6.0, A)
        traj.append(q.copy())
    traj = np.stack(traj)
    vv = np.abs(np.diff(traj[:, :A], axis=0)).max() / dt
    aa = np.abs(np.diff(traj[:, :A], n=2, axis=0)).max() / dt ** 2
    check(vv <= 0.8 + 1e-6, f"streaming velocity bounded ({vv:.3f} <= 0.8)")
    check(aa <= 6.0 + 1e-6, f"streaming ACCELERATION bounded ({aa:.3f} <= 6.0)")
    check(traj[-1, 0] > traj[0, 0] + 0.5, "and it still makes progress toward the target")

    #a step input must come out rate-limited
    step = np.tile(np.concatenate([np.full(A, 1.0), np.zeros(2)]), (40, 1))
    out = clamp_chunk(step, q0, dt, 0.8, 6.0, A)
    v = np.abs(np.diff(out[:, :A], axis=0)).max() / dt
    first = np.abs(out[0, :A] - q0[:A]).max() / dt
    check(v <= 0.8 + 1e-6, f"velocity clamped to 0.8 (got {v:.3f})")
    check(first <= 0.8 + 1e-6, f"the step from the CURRENT pose is clamped too ({first:.3f})")

    #the gripper must not be rate-limited
    g = np.tile(np.concatenate([np.zeros(A), [1.0, 1.0]]), (10, 1))
    og = clamp_chunk(g, q0, dt, 0.8, 6.0, A)
    check(np.allclose(og[:, A:], 1.0), "gripper passes through raw, not rate-limited")

    #seam blending must reduce the discontinuity at a re-plan
    prev = np.tile(np.zeros(16), (40, 1))
    new = np.tile(np.concatenate([np.full(A, 0.3), np.zeros(2)]), (40, 1))
    raw_gap = abs(new[0, 0] - prev[10, 0])
    bl = seam_blend(new, prev, 10, 12, A)
    check(abs(bl[0, 0] - prev[10, 0]) < raw_gap, "seam blend softens the jump at a re-plan")
    check(np.allclose(bl[-1, :A], new[-1, :A]), "the far end of the plan is untouched")
    check(np.allclose(seam_blend(new, None, 0, 12, A), new), "no previous plan: unchanged")

    #jerk responds to smoothness, in the right direction
    smooth = np.cumsum(np.tile(np.full(16, 0.001), (60, 1)), axis=0)
    noisy = smooth + np.random.default_rng(0).normal(0, 0.01, smooth.shape)
    check(jerk(noisy) > jerk(smooth) * 5, "jerk is much larger for a noisy trajectory")

    #the point of the file
    import subprocess
    r = subprocess.run([sys.executable, "-c",
                        "import sys; sys.path.insert(0,'.'); import control_math; "
                        "print('no zenoh needed')"],
                       capture_output=True, text=True, cwd=os.environ.get("AIW_DEPLOY", "/home/robotis/robot_aiworker/aiworker_deploy"))
    check("no zenoh needed" in r.stdout, "imports with no deployment stack present")

    print(f"\n{'all checks passed' if not fails else 'FAILED: ' + str(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
