#!/usr/bin/env python
"""Run a fine-tuned TurboVLA on the real AI Worker SG2. Everything executes on this workstation.

Same shape as `omy_deploy/infer.py`, and for the same reason: the policy, the rate limiting and the
homing all run on the 4090, and the robot is reached over zenoh through the stock
`lerobot_robot_ros2_zenoh` plugin. Nothing is installed on the robot, which matters because it is
shared.

    ROS2Zenoh.get_observation()  ->  TurboVLA  ->  clamp  ->  send_trajectory()

TWO PROCESSES, BOTH ON THIS MACHINE, and the reason is dependency pins rather than distance.
TurboVLA needs transformers 4.57.6 (its BERT wrapper uses `get_head_mask`, removed in v5);
`lerobot_robot_ros2_zenoh` lives in an env on transformers 5.5.4 where TurboVLA cannot import at
all. Neither pin set can absorb the other, so:

    terminal 1  (turbovla-aiworker)  TurboVLA/scripts/aiworker/serve.sh   -> localhost:10091
    terminal 2  (lerobot_venv)       this file                            -> zenoh to the robot

Both run on the 4090, so the websocket is a loopback hop, not a network one. No SSH tunnel, and
nothing is installed on the robot -- which matters because it is shared.

DRY RUN IS THE DEFAULT. This commands a bimanual arm in ABSOLUTE joint positions. A dry run does
everything except publish -- observes, infers, clamps, and prints the plan with its largest joint
step -- so the first thing you see is what the policy WANTS to do. `--live` is the opt-in.

SAFETY, each guarding a failure this setup can produce:

  start pose    refuses to run unless the arms are near where the demonstrations begin. A policy
                asked to act from a pose it has never seen is extrapolating, and chunk 0 is where
                that shows.
  clamp         bounds the step INTO the chunk (from the arm's real pose to the policy's first
                waypoint -- the dangerous one, guarded by nothing else) and every step within it.
  gripper raw   the gripper bypasses rate limiting. It is near-binary; throttling it to arm
                acceleration means the hand never closes before the next re-plan.
  hold on fault a failed inference or a bad observation holds position rather than reusing the
                previous chunk, which was planned for a scene that has since moved.
  budget        --max-steps bounds the episode.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import json
import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / "TurboVLA", HERE.parent / "AIWORKER",
          HERE.parent / "TurboVLA" / "third_party" / "starvla_runtime",
          HERE.parent / "lerobot_robot_ros2_zenoh"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import spec_sg2 as spec  # noqa: E402
from safety import check_observation  # noqa: E402
from trajectory import send_trajectory  # noqa: E402


#PRE-ROLLOUT STILLNESS. The start-pose guard answers "is the arm in the right PLACE"; it cannot
#answer "is the arm MOVING". Those are different failures. An arm being jogged by hand, drifting
#under a slipping brake, or reporting a glitched encoder can sit inside the position tolerance
#while its joints change, and a policy chunk is 100 waypoints predicted for the pose held at the
#instant of the observation. Planning from a moving arm executes a plan for a pose that has
#already gone.
#
#Thresholds are measured, not chosen. Over the 90 demonstrations, taking the quietest 0.5 s window
#of each episode -- the arm holding position under position control, which is exactly the
#pre-rollout condition:
#
#    holding still     p99 0.000005 rad/step = 0.0002 rad/s     (max over all 90: 0.0002 rad/s)
#    task motion       p50 0.007478 rad/step = 0.2243 rad/s
#
#a separation of 1443x. REST_VEL_MAX sits at 0.06 rad/s: 300x above the measured rest ceiling, so
#sensor noise cannot trip it, and 3.7x below median task speed, so anything moving with intent
#does. REST_DRIFT_MAX catches the case velocity misses -- a creep too slow to flag per sample that
#still walks the arm somewhere over the window. At rest the window total is ~1e-4 rad, so 0.010 is
#100x margin while staying far under the 0.15 rad start-pose tolerance.
REST_VEL_MAX = 0.06      # rad/s, per joint
REST_DRIFT_MAX = 0.010   # rad, total excursion of any joint across the window
REST_WINDOW = 15         # samples (0.5 s at 30 Hz)


def arm_motion(samples, arm_dims):
    """Peak joint velocity and total excursion over a window of (timestamp, q) samples.

    Pure function of the samples so it can be tested without a robot -- see --self-test.
    Velocity rather than per-sample delta because the sampling period is not guaranteed: a slow
    observation would inflate a per-step threshold while meaning nothing about how fast the arm
    is actually moving.
    """
    import numpy as _np
    t = _np.asarray([s[0] for s in samples], dtype=_np.float64)
    Q = _np.stack([_np.asarray(s[1], dtype=_np.float64)[:arm_dims] for s in samples])
    dt = _np.diff(t)
    assert (dt > 0).all(), "sample timestamps must strictly increase"
    vel = _np.abs(_np.diff(Q, axis=0)) / dt[:, None]      # (n-1, arm_dims) rad/s
    drift = Q.max(0) - Q.min(0)                            # (arm_dims,) rad
    j_vel = int(vel.max(0).argmax())
    j_drift = int(drift.argmax())
    return float(vel.max()), float(drift.max()), j_vel, j_drift


def require_at_rest(sample_fn, arm_dims, joint_names, n=REST_WINDOW, period=1.0 / 30.0,
                    vel_max=REST_VEL_MAX, drift_max=REST_DRIFT_MAX, label="start", on_fail=None):
    """Sample the arm for n frames and REFUSE unless every joint is holding still.

    `sample_fn` returns the current joint vector. `on_fail` is called before raising so the caller
    can disconnect the robot. Raises SystemExit; never moves anything.
    """
    import time as _time
    import numpy as _np
    samples = []
    for i in range(n):
        samples.append((_time.perf_counter(), _np.asarray(sample_fn(), dtype=_np.float64).copy()))
        if i < n - 1:
            _time.sleep(period)
    span = samples[-1][0] - samples[0][0]
    vel, drift, j_vel, j_drift = arm_motion(samples, arm_dims)
    print(f"[rest] {label}: sampled {n} frames over {span:.2f}s -- peak joint velocity "
          f"{vel:.4f} rad/s ({joint_names[j_vel]}), total excursion {drift:.4f} rad "
          f"({joint_names[j_drift]})")
    bad = []
    if vel > vel_max:
        bad.append(f"peak velocity {vel:.4f} rad/s exceeds {vel_max} on {joint_names[j_vel]}")
    if drift > drift_max:
        bad.append(f"excursion {drift:.4f} rad exceeds {drift_max} on {joint_names[j_drift]}")
    if bad:
        if on_fail is not None:
            on_fail()
        raise SystemExit(
            "REFUSING TO RUN: the arm is not at rest.\n  " + "\n  ".join(bad) +
            "\n  A predicted chunk describes the pose held when it was observed. Planning from a\n"
            "  moving arm executes a plan for a pose that is already gone.\n"
            "  Let the arm settle, check nothing is leaning on it, then rerun. If a joint reads\n"
            "  as moving while the arm is visibly still, that is an encoder or comms fault --\n"
            "  do not raise --rest-vel-max to get past it.")
    return vel, drift


#clamp_chunk moved to control_math.py so offline tools can rate-limit a trajectory without
#importing the deployment stack (trajectory -> zenoh_ros2_sdk). Re-exported here so every existing
#caller and `infer.clamp_chunk` keep working, and so the sweep and the robot apply the SAME limits.
from control_math import clamp_chunk, seam_blend  # noqa: E402,F401


def go_home(robot, q_home, q_now, vel=0.15, fps=30.0):
    """Drive to `q_home` as ONE slow splined trajectory, not a lunge.

    The arm can be anywhere when a run ends or is aborted, so the move home is exactly the kind of
    large unplanned motion that needs its own explicit speed limit.
    """
    q_home, q_now = np.asarray(q_home, float), np.asarray(q_now, float)
    dist = float(np.abs(q_home - q_now).max())
    if dist < 1e-3:
        return
    n = max(2, int(np.ceil(dist / max(vel, 1e-6) * fps)))
    path = np.stack([q_now + (q_home - q_now) * (i / n) for i in range(1, n + 1)])
    print(f"[home] {dist:.3f} rad max joint move over {n/fps:.1f}s at <={vel} rad/s")
    send_trajectory(robot, path, dt=1.0 / fps, speed=1.0)
    time.sleep(n / fps + 0.5)



class _SubtaskKeys:
    """ENTER advances the subtask, BACKSPACE goes back. Non-blocking, safe without a TTY.

    The 35-episode checkpoints are conditioned on FIVE per-frame subtask sentences, so one --task
    makes the policy perform a single stage and then sit there. It was trained to be told when the
    stage changes; this is how the operator tells it.

    cbreak, not raw, so Ctrl-C still raises KeyboardInterrupt -- the operator must never lose the
    interrupt they already know while standing next to a moving arm. Without a TTY (piped, nohup)
    it degrades to "no key ever pressed" rather than failing, because a rollout must not depend on
    a console. The terminal is restored through atexit so a crash mid-rollout cannot leave the
    shell in cbreak.
    """

    def __init__(self, sentences, enabled=True):
        self.sentences = list(sentences)
        self.i = 0
        self.fd = self.saved = None
        if enabled and self.sentences and sys.stdin.isatty():
            import atexit, termios, tty
            self.fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            atexit.register(self.restore)

    def restore(self):
        if self.fd is not None and self.saved is not None:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            self.fd = self.saved = None

    @property
    def instruction(self):
        return self.sentences[self.i] if self.sentences else None

    def poll(self):
        """Consume pending keys; return a message if the stage changed, else None."""
        if self.fd is None:
            return None
        import select
        msg = None
        while True:
            #select on the SAME fd we read from. Selecting sys.stdin while reading self.fd
            #means that if they ever differ, select reports ready and the read blocks
            #forever -- with the arm mid-trajectory.
            r, _, _ = select.select([self.fd], [], [], 0)
            if not r:
                break
            #os.read, NOT sys.stdin.read: the latter goes through Python's buffered reader and
            #can BLOCK even after select() reported the fd readable. A blocking read inside this
            #loop would stall the control tick with the arm in motion. os.read on the raw fd
            #returns exactly what is there and cannot block once select has said ready.
            k = os.read(self.fd, 1).decode(errors="ignore")
            if not k:
                break
            if k in ("\r", "\n"):
                if self.i < len(self.sentences) - 1:
                    self.i += 1
                    msg = f"-> [{self.i}/{len(self.sentences)-1}] {self.instruction!r}"
                else:
                    msg = f"already on the last stage [{self.i}]"
            elif k in ("\x7f", "\b") and self.i > 0:
                self.i -= 1
                msg = f"<- [{self.i}/{len(self.sentences)-1}] {self.instruction!r}"
        return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/robotis/robot_aiworker/TurboVLA/results/Checkpoints/"
                                      "turbovla_aiworker_vanilla_12500/checkpoints/"
                                      "steps_12500_ema_pytorch_model.pt")
    ap.add_argument("--task",
                    default="Move both bottles into the basket, starting from right then left.",
                    help="THE INSTRUCTION THE DEMONSTRATIONS WERE RECORDED UNDER. Deployment had "
                         "been sending a paraphrase, which changes the predicted chunk by only "
                         "0.0127 rad -- the model saw one instruction across all 90 episodes, so "
                         "its text encoder is effectively constant and paraphrases are free. It "
                         "still matters for data collection: the feature width depends on the "
                         "token count, so a paraphrase silently produces features the demo half's "
                         "PCA basis cannot project.")
    ap.add_argument("--router-ip", default="127.0.0.1",
                    help="the ROBOT's zenoh router IP (e.g. ffw-snpr48a1106.local)")
    ap.add_argument("--policy-host", default="127.0.0.1", help="TurboVLA server (serve.sh)")
    ap.add_argument("--policy-port", type=int, default=10091)
    ap.add_argument("--domain-id", type=int, default=30)
    ap.add_argument("--live", action="store_true", help="actually publish; default is a dry run")
    ap.add_argument("--speed", type=float, default=0.7,
                    help="TIME scale in (0,1]. 1.0 = demo speed. Scales time, never the angles.")
    ap.add_argument("--max-vel", type=float, default=spec.MAX_VEL)
    ap.add_argument("--max-acc", type=float, default=spec.MAX_ACC)
    ap.add_argument("--execute-steps", type=int, default=spec.EXECUTE_STEPS,
                    help="waypoints executed from each plan before re-planning. THE central knob "
                         "of receding horizon, and a real trade-off: fewer means the policy sees "
                         "fresh observations more often (more reactive, but a re-plan seam more "
                         "often, and seams are where the start-stop stutter comes from); more "
                         "means smoother motion but longer blind open-loop stretches. TurboVLA's "
                         "own LIBERO protocol uses 12 of its chunk; spec.EXECUTE_STEPS is 25 of "
                         "50 here.")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--start-tol", type=float, default=0.30,
                    help="max rad from the demo start pose before refusing to run")
    ap.add_argument("--no-rest-check", dest="rest_check", action="store_false",
                    help="skip the pre-rollout stillness guard (NOT recommended)")
    ap.add_argument("--rest-window", type=int, default=REST_WINDOW,
                    help="frames sampled to decide the arm is at rest")
    ap.add_argument("--rest-vel-max", type=float, default=REST_VEL_MAX,
                    help="max per-joint velocity (rad/s) tolerated before rollout. Measured rest "
                         "ceiling is 0.0002; demo task median is 0.2243")
    ap.add_argument("--rest-drift-max", type=float, default=REST_DRIFT_MAX,
                    help="max per-joint excursion (rad) across the rest window")
    ap.add_argument("--self-test", action="store_true",
                    help="verify the stillness guard against synthetic traces and exit. Touches "
                         "no robot and needs no network")
    ap.add_argument("--data", default="/home/robotis/robot_aiworker/datasets/aiw_pp_2bttles_lerobot")
    ap.add_argument("--home-after", action="store_true",
                    help="(kept for compatibility; homing on exit is now the default)")
    ap.add_argument("--no-home", dest="home", action="store_false",
                    help="do NOT return to the start pose on exit or Ctrl-C")
    ap.add_argument("--home-first", action="store_true",
                    help="if the arm is outside --start-tol, drive it to the demo start pose "
                         "first, as one slow splined move. Requires --live (it is motion).")
    ap.add_argument("--home-vel", type=float, default=0.15,
                    help="rad/s for the homing move. Deliberately slower than the policy limit.")
    ap.add_argument("--no-cameras", action="store_true", help="joint-only smoke test")
    ap.add_argument("--wrist-rot", type=int, default=90, choices=[0, 90, 180, 270],
                    help="degrees COUNTER-CLOCKWISE to rotate the WRIST frames before inference. "
                         "The live wrist cameras deliver frames rotated 90 degrees clockwise from "
                         "the recorded dataset -- visible by eye (training shows ceiling across "
                         "the top and the table at the bottom; live had that on its side) and "
                         "confirmed by running the model on the REAL saved live frames:\n"
                         "    as-is     commands 0.034 rad, R/L 1.00\n"
                         "    rot90CCW  commands 0.169 rad, R/L 2.78   (demos: 0.232, 2.90)\n"
                         "So the policy was being shown a scene it had never seen, commanded "
                         "almost nothing, and stalled -- and because it still moved a little, it "
                         "looked like a policy problem rather than an input problem. The scene "
                         "camera is unaffected and is NOT rotated. Set 0 to disable if the "
                         "cameras are ever remounted.")
    ap.add_argument("--policy", choices=("turbovla", "act_prior", "groot"), default="turbovla",
                    help="which policy produces the action chunk.\n"
                         "`turbovla` calls the websocket server on --policy-port, because TurboVLA "
                         "needs transformers 4.57.6 while this file runs in an env on 5.5.4 and "
                         "neither pin can absorb the other.\n"
                         "`act_prior` is a LeRobot policy and imports cleanly HERE, so it is loaded "
                         "in-process and no server is involved. Everything downstream is shared: "
                         "receding horizon, seam blending, gripper decode, homing, the safety "
                         "clamps. Only the thing producing the chunk changes.")
    ap.add_argument("--policy-path", default="",
                    help="checkpoint directory for --policy act_prior, e.g. "
                         "/home/robotis/robot_aiworker/act_prior_aiw/005000/pretrained_model")
    ap.add_argument("--residual", default="/tmp/residual_bc_demo.pt",
                    help="the learned residual applied on top of the frozen policy. ON BY DEFAULT: "
                         "measured on libero_10 task success, a ~17K-parameter residual takes the "
                         "frozen policy from 0.35 to 0.72. Pass an empty string to run the frozen "
                         "policy unchanged -- that is the control arm and the fallback, and it is "
                         "the same code path either way.\n"
                         "Trained by UNIFORM-WEIGHT regression on the demonstrations. Both of "
                         "those were measured, not assumed: advantage weighting lost to uniform "
                         "weights in six experiments, and adding base-policy rollouts to the "
                         "buffer COST 13 points (0.72 demo-only against 0.59 mixed) because BC "
                         "imitates them uniformly and they dilute the demonstrations.")
    ap.add_argument("--residual-buffer", default="/tmp/aiworker_demo_half.pt",
                    help="the buffer the residual was trained from, for its PCA basis. Required: "
                         "a residual applied through a different projection than it was fitted in "
                         "produces plausible-looking nonsense rather than an error.")
    ap.add_argument("--residual-scale", type=float, default=1.0,
                    help="multiplies the edit. 0.0 is exactly the frozen policy, so a live run can "
                         "walk the residual in rather than trusting it whole on episode one.")
    ap.add_argument("--save-obs", default="",
                    help="directory to dump the LIVE observation every --save-every steps: the "
                         "three camera frames as PNG plus the joint state as .npy. Every offline "
                         "check says the policy is correct -- it reproduces the demos' right-arm "
                         "lead at ratio 2.93 vs 2.90, commits 1.03x the demonstrated motion, and "
                         "denormalizes to within 1e-4 of training. It is only wrong on the robot. "
                         "That means the live observation differs from the recorded one, and "
                         "guessing at how has already cost several runs. This makes the live "
                         "input a file I can compare against the training frames directly, and "
                         "feed back through the model offline.")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--label", action="store_true",
                    help="ask WHICH BOTTLES were placed when the episode ends, and write "
                         "success.json beside the saved observations.\n"
                         "Graded, not binary, and that is not a nicety. The task is two "
                         "sub-goals -- right arm places, then left -- so a binary label collapses "
                         "'placed one of two' onto 'failed', which is the single most informative "
                         "outcome we can record: it says the reach worked and the second half did "
                         "not. Partial credit gives the critic three distinct return levels to "
                         "separate instead of two, and separating returns is precisely what "
                         "adv_signal measures. Every recorded demonstration succeeded at both, so "
                         "this grading is the only source of variation in the whole buffer.")
    ap.add_argument("--trace", default="",
                    help="CSV of what was COMMANDED against what the arm actually DID, per joint "
                         "per step. Everything upstream of the publish is verified correct -- the "
                         "policy's target is forward of the current pose in 100%% of episode "
                         "starts (cosine +0.796, raw radians), the normalizer matches training to "
                         "1e-4 on the arms, and the sender routes by joint name. So if the arm "
                         "still moves backward, the disagreement is between the command and the "
                         "motion, and only a live run can show it. `q_prev` is a pure integrator "
                         "seeded from the last COMMAND and never reads the arm back, so a joint "
                         "that does not track diverges silently: this is the log that makes it "
                         "visible.")
    ap.add_argument("--offset", type=int, default=0,
                    help="RETIRED. Kept so old commands still parse. Under receding horizon the "
                         "plan is executed in order from waypoint 0, so there is nothing to skip: "
                         "this value is ignored.")
    ap.add_argument("--unused-offset", type=int, default=0,
                    help="which step of each fresh chunk to execute. 0 is the ramp-in (where the "
                         "arm already is) and commanding it every tick keeps the arm still; a few "
                         "steps in is where the chunk commits to real motion. Raise for more "
                         "aggressive tracking, lower if it overshoots.")
    ap.add_argument("--hz", type=float, default=0.0,
                    help="control rate in Hz. 0 keeps the derived spec.FPS*speed. Lowering this "
                         "alone does NOT slow the approach -- max_vel does -- but it does give "
                         "the controller longer to settle between commands.")
    ap.add_argument("--follow-speed", type=float, default=0.0,
                    help="cap each joint's speed at this multiple of the speed the CHUNK ITSELF "
                         "predicts for that joint right now, instead of always allowing max_vel.\n"
                         "Why this exists: with a far --offset the target is ~0.8 s ahead, so the "
                         "velocity clip is saturated on every step and the arm travels at max_vel "
                         "from start to finish. The demonstrations do not -- they decelerate into "
                         "the grasp, and that deceleration is exactly what a chunked policy "
                         "predicts in the spacing of its own waypoints. Reading the local spacing "
                         "back out restores the profile: fast in free space, slow at contact. "
                         "0 disables and restores the flat max_vel cap.")
    ap.add_argument("--mode", choices=("ensemble", "receding"), default="receding",
                    help="`ensemble` re-plans EVERY tick and commands, for the current instant, a "
                         "weighted average of what the last N chunks each predicted for that same "
                         "instant. Inference is ~18 ms against a 33-48 ms tick, so there is no "
                         "reason to run open-loop off a stale plan at all.\n"
                         "This is why it works where re-planning every tick previously deadlocked: "
                         "that version executed chunk[offset] from the NEWEST chunk only, and this "
                         "model's chunk head is both rough and non-monotone (cos -0.55 at waypoint "
                         "2, -0.39 at 16, coherent only from 20 on), so the arm never left the "
                         "ramp-in. Averaging over chunks predicted 1..N ticks ago pairs that noisy "
                         "head with the MIDDLE of older chunks, which is exactly the coherent "
                         "region -- every term describes the present moment, so nothing is stale "
                         "and nothing is skipped.\n"
                         "`receding` keeps the previous behaviour (execute a plan open-loop, then "
                         "re-plan) and with it --execute-steps, --seam-blend and --trim-head, all "
                         "of which exist only to work around the head this mode averages away.")
    ap.add_argument("--plan-start", type=int, default=0, help=argparse.SUPPRESS)  # removed: see below
    ap.add_argument("--plan-start-doc", type=int, default=0,
                    help="waypoint each plan starts executing from. The head of this model's chunk "
                         "does not advance -- per-step direction vs the plan's own overall "
                         "direction is +0.41, -0.55, ... -0.39 through waypoint 17, then +0.84 and "
                         "above from 20 on. That head is why --execute-steps cannot go below ~25: "
                         "a short plan is ALL head, ends where it began, and the arm dwells.\n"
                         "Starting at 18 removes that constraint. Executing waypoints 18..18+K "
                         "keeps only the coherent stretch, so K can be small and the policy can "
                         "re-plan far more often -- at K=8 that is one inference every 0.38 s "
                         "instead of every 1.19 s, roughly 3x more closed-loop correction, and at "
                         "18 ms per inference it costs about 5%% duty cycle.\n"
                         "This is NOT the old --offset. That executed ONE waypoint and discarded "
                         "the other 49; this executes a contiguous run and skips only a measured, "
                         "reproducible artefact of the chunk head. 0 disables.")
    ap.add_argument("--trim-head", type=int, default=24,
                    help="maximum leading waypoints that may be dropped from a plan that has no "
                         "predecessor to cross-fade against -- i.e. the FIRST plan of a run, and "
                         "any plan following a dropped inference. Every later plan gets "
                         "--seam-blend, but the first one has nothing to blend with, so its "
                         "incoherent head executes raw from a standstill, which is the most "
                         "visible reversal of the whole run. The cut point is found from the plan "
                         "itself, not fixed: the first waypoint after which the next 8 steps all "
                         "advance along the plan's overall direction. On this model that lands "
                         "around 18-20, matching the measured profile (cos -0.55 at waypoint 2, "
                         "-0.39 at 16, +0.84 from 20 on). 0 disables.")
    ap.add_argument("--seam-blend", type=int, default=20,
                    help="waypoints over which a fresh plan is cross-faded with the PREVIOUS "
                         "plan's continuation. 0 disables.\n"
                         "The chunk's head does not advance monotonically -- measured against each "
                         "plan's own overall direction, the per-waypoint step has cos -0.55 at "
                         "waypoint 2 and -0.39 at 16, only settling to +0.84..+0.96 from waypoint "
                         "20 on. Executed raw, that is a third of every plan spent wandering, "
                         "which reads as circular or reversing motion rather than steady progress. "
                         "The previous plan's TAIL covers the same instants and is coherent there "
                         "(+0.90), so fading from it into the new head cancels the wander while "
                         "still handing control to the fresher observation. Nothing is discarded: "
                         "both predictions are used, weighted.")
    ap.add_argument("--savgol", type=int, default=0,
                    help="Savitzky-Golay window over the plan's ARM dims, odd, 0 = off. Replaces "
                         "--plan-smooth when set. A boxcar average fits a constant over its window "
                         "and so blunts every turn; SavGol fits a degree-`--savgol-order` "
                         "polynomial, removing per-inference jitter while keeping the curvature "
                         "and the endpoints. Try 11 with order 3.")
    ap.add_argument("--savgol-order", type=int, default=3,
                    help="polynomial order for --savgol. 2-3 is normal; higher preserves more "
                         "detail and therefore more jitter.")
    ap.add_argument("--plan-smooth", type=int, default=9,
                    help="moving-average width, in waypoints, applied ALONG each fresh plan's arm "
                         "trajectory. The shake is in the prediction, not the controller: the "
                         "model's chunk has a 2nd-difference of 0.0039 rad/waypoint against the "
                         "demonstrations' 0.0009 -- it predicts a path 4.1x rougher than the "
                         "humans actually moved, and receding horizon follows it faithfully. "
                         "Filtering the PLAN costs almost nothing (window 5: jerk drops to 0.72x "
                         "the demos, waypoints move 0.007 rad, and 100.8%% of the right-arm motion "
                         "survives) whereas filtering the COMMAND cost ~28%% of the motion, which "
                         "is what made the arm fall behind its plan. 0 or 1 disables. Arms only -- "
                         "averaging the gripper would blur a class decision into a value between "
                         "open and closed.")
    ap.add_argument("--vel-smooth", type=float, default=0.0,
                    help="low-pass on the commanded VELOCITY, 0 = off, 1 = frozen. This is a jerk "
                         "limit: the acceleration clip already bounds how fast velocity may "
                         "change, but it is a hard clip, so the command still corners sharply "
                         "whenever the target jumps between re-plans. Filtering velocity rounds "
                         "those corners. It is applied to velocity rather than to position on "
                         "purpose -- smoothing position would delay the whole trajectory, whereas "
                         "smoothing velocity only softens the transitions between segments.")
    ap.add_argument("--min-vel", type=float, default=0.04,
                    help="floor for the chunk-derived cap, so a momentarily flat patch of the "
                         "horizon cannot stall the arm outright.")
    ap.add_argument("--ensemble", type=int, default=1,
                    help="how many recent chunks to average the arm target over (1 = off). "
                         "TEMPORAL ENSEMBLING, ALIGNED IN ABSOLUTE TIME -- not a low-pass on the "
                         "command. Each re-plan predicts the whole horizon, so a waypoint at a "
                         "given wall-clock instant was predicted by several successive chunks, "
                         "each from a different observation. Averaging THOSE cancels per-inference "
                         "jitter without adding lag, because every term refers to the same moment. "
                         "An EMA on the command would smooth by delaying it, which on a chunked "
                         "policy means tracking a stale target. The control loop runs at "
                         "spec.FPS*speed while the chunk is indexed at spec.FPS, so chunk i steps "
                         "back is read `stride` deeper to land on the same instant.")
    ap.add_argument("--ensemble-decay", type=float, default=0.0,
                    help="exponential weight decay per chunk of age. Higher trusts the newest "
                         "observation more; 0 weights all chunks in the window equally.")
    ap.add_argument("--grip-lead", type=int, default=4,
                    help="how many chunk steps ahead of the executed one may trigger a CLOSE. "
                         "The close test is `any step in the window says closed`, so the window "
                         "length IS how early the hand shuts: sharing the open window (25 steps, "
                         "~0.8 s at 30 Hz) meant the gripper closed most of a second before the "
                         "grasp was due. Opening keeps the long window and the vote, because the "
                         "two errors are not symmetric -- closing early knocks the bottle over, "
                         "releasing early drops it mid-carry.")
    ap.add_argument("--grip-hold", type=int, default=10,
                    help="consecutive ticks the chunk must agree the hand should OPEN before it "
                         "actually opens. Measured on the demonstrations: each gripper closes "
                         "exactly ONCE per episode and holds a median 188 steps (6.3 s), minimum "
                         "127 -- it never reopens mid-carry. So any brief flicker to `open` in the "
                         "prediction is noise, and with no hysteresis it released the bottle "
                         "immediately, which is the short grip. Closing stays instant: the two "
                         "errors are not symmetric.")
    ap.add_argument("--grip-hold-unused", type=int, default=0,
                    help="consecutive chunks that must agree before the gripper CHANGES state. "
                         "The gripper is binary and re-decided every re-plan, so two chunks that "
                         "disagree make the hand open and re-grab -- it grabs, drops, grabs. "
                         "Requiring agreement latches a grasp until the policy is consistently "
                         "sure it is done. 0 disables.")
    ap.add_argument("--no-trim-lag", dest="trim_lag", action="store_false",
                    help="do NOT drop the stale prefix of each chunk. The prefix was planned from "
                         "a pose the arm has already left, so keeping it pulls the arm backward "
                         "at every re-plan seam.")
    ap.add_argument("--carry-vel", action="store_true",
                    help="send per-waypoint velocities so momentum carries across each re-plan "
                         "seam. Without it the controller assumes zero velocity at every point, "
                         "so the arm decelerates into each new chunk and re-accelerates -- a "
                         "visible dip every 25 steps. Opt-in because a jumpy checkpoint can make "
                         "the derived velocities worse than none.")
    ap.add_argument("--subtasks", action="store_true",
                    help="drive the SUBTASK instruction from the keyboard: ENTER = next stage, "
                         "BACKSPACE = previous. Sentences are read from --data. Needed because the "
                         "35-episode checkpoints are conditioned on five per-frame subtasks, so a "
                         "single --task performs one stage only.")

    ap.add_argument("--official", action="store_true",
                    help="run the OFFICIAL GR00T N1.7 inference contract and nothing else. "
                         "LeRobot's GrootPolicy.select_action decodes one chunk, executes "
                         "min(n_action_steps, checkpoint action_horizon, execution_horizon) of it "
                         "one step at a time, then re-plans -- for `new_embodiment` that is all 40. "
                         "It applies NO temporal ensembling, NO Savitzky-Golay and NO seam "
                         "blending; those are ours, inherited from ACT, and they change what the "
                         "policy commands. This flag sets --execute-steps 40, --seam-blend 0, "
                         "--savgol 0, --ensemble 1. The velocity/acceleration clamps are NOT "
                         "disabled: they are safety, not smoothing, and official GR00T simply has "
                         "no equivalent because it is not driving this robot.\n"
                         "MEASURED, AND IT IS WORSE HERE. Same checkpoint (002400), same 1,887-frame "
                         "episode, official contract vs our smoothing (execute 10 / ensemble 4 / "
                         "savgol 15):\n"
                         "    tracking  0.0391 -> 0.0275 rad   (-30%)\n"
                         "    jerk      2.69x  -> 1.85x human  (-31%)\n"
                         "    tip error 22-56  -> 13-43 mm     (better on EVERY subtask, both arms)\n"
                         "Smoothing usually trades tracking away for smoothness; here it improves "
                         "both, because GR00T's raw chunk is noisy -- its commanded acceleration "
                         "averages 5.07 rad/s^2 against the human's 0.81, pinning our clamp. So "
                         "--official is for REPRODUCING the documented contract, not for getting "
                         "the best behaviour out of this robot. Do not make it the default.")

    a = ap.parse_args()

    subtask_list = []
    if a.subtasks:
        import pandas as _pd
        _sp, _tp = Path(a.data) / "meta" / "subtasks.parquet", Path(a.data) / "meta" / "tasks.parquet"
        if _sp.exists():
            _sub = _pd.read_parquet(_sp).sort_values("subtask_index")
            _col = "subtask" if "subtask" in _sub.columns else _sub.columns[-1]
            subtask_list = [str(getattr(r, _col)) for r in _sub.itertuples()]
        elif _tp.exists():
            _t = _pd.read_parquet(_tp)
            if _t.index.dtype == object:
                _pairs = sorted(zip(_t.index, _t.get("task_index", range(len(_t)))),
                                key=lambda kv: kv[1])
                subtask_list = [str(k) for k, _ in _pairs]
        if not subtask_list:
            raise SystemExit(f"--subtasks: no subtask sentences under {a.data}/meta")
        a.task = subtask_list[0]
        print(f"[subtasks] {len(subtask_list)} stages -- ENTER next, BACKSPACE previous")
        for _i, _s in enumerate(subtask_list):
            print(f"    {_i}  {_s}")

    if a.official:
        #Fixed at the documented contract rather than tuned. Deviating from it is a decision that
        #should be made deliberately and named, not inherited from a different policy's defaults.
        a.execute_steps, a.seam_blend, a.savgol, a.ensemble = 40, 0, 0, 1
        a.ensemble_decay = 0.0
        a.mode = "receding"
        print("[official] GR00T N1.7 contract: execute_steps 40, no ensembling, no savgol, "
              "no seam blend.")
        print(f"[official] clamps KEPT for safety: {a.max_vel} rad/s, {a.max_acc} rad/s^2.")

    if a.self_test:
        #Verify the stillness guard WITHOUT a robot. arm_motion is a pure function of the sampled
        #window, so every case below is exercised on the same code the live run uses.
        import numpy as _np
        A_ = 14
        FPS_ = 30.0

        def trace(vel_rad_s, n=REST_WINDOW, joint=3, noise=5e-6, seed=0):
            """Synthetic window: one joint moving at vel_rad_s, all joints at the measured noise."""
            rng = _np.random.default_rng(seed)
            out = []
            for i in range(n):
                q = rng.normal(0.0, noise, A_ + 2)
                q[joint] += vel_rad_s * (i / FPS_)
                out.append((i / FPS_, q))
            return out

        fails = []

        def case(name, samples, want_refuse):
            vel, drift, _, _ = arm_motion(samples, A_)
            refused = vel > REST_VEL_MAX or drift > REST_DRIFT_MAX
            ok = refused == want_refuse
            print(f"  [{'ok ' if ok else 'FAIL'}] {name:<46} vel {vel:8.5f} rad/s  "
                  f"drift {drift:7.5f} rad  -> {'REFUSE' if refused else 'allow'}")
            if not ok:
                fails.append(name)

        print(f"stillness guard self-test  (vel_max {REST_VEL_MAX} rad/s, "
              f"drift_max {REST_DRIFT_MAX} rad, window {REST_WINDOW})\n")
        print(" MUST ALLOW -- a genuinely still arm")
        case("arm at rest, measured noise floor 5e-6", trace(0.0), False)
        case("rest with 10x worse encoder noise", trace(0.0, noise=5e-5, seed=1), False)
        print("\n MUST REFUSE -- motion the guard exists to catch")
        case("demo task median speed 0.2243 rad/s", trace(0.2243), True)
        case("slow hand-jog 0.10 rad/s", trace(0.10), True)
        case("creep below vel limit, caught by drift", trace(0.04), True)
        case("single-sample encoder glitch 0.05 rad", 
             [(i / FPS_, (lambda q: (q.__setitem__(7, 0.05), q)[1])(_np.zeros(A_ + 2)) if i == 7
               else _np.zeros(A_ + 2)) for i in range(REST_WINDOW)], True)
        print("\n NEGATIVE CONTROL -- the guard must not fire on everything")
        case("static offset, arm still but 0.5 rad off pose",
             [(i / FPS_, _np.full(A_ + 2, 0.5)) for i in range(REST_WINDOW)], False)
        print("\n GRIPPER EXCLUSION -- only arm joints are judged")
        case("gripper slamming, arms still",
             [(i / FPS_, _np.concatenate([_np.zeros(A_), [0.9 * (i % 2), 0.0]]))
              for i in range(REST_WINDOW)], False)

        if fails:
            raise SystemExit(f"\nSELF-TEST FAILED: {len(fails)} case(s): {fails}")
        print("\nall cases passed -- guard is sound. Nothing was connected or moved.")
        return 0


    #ZENOH_CONFIG_OVERRIDE, if already set in the shell, WINS over the config object -- and the
    #workstation has one left over from the omy robot pointing at tcp/192.168.0.10:7447. The
    #symptom is a connection failure to an address you never passed, which reads like a network
    #problem rather than a stale variable. Set it from --router-ip so the CLI argument is the
    #single source of truth, and do it BEFORE anything imports zenoh.
    import os
    endpoint = f'tcp/{a.router_ip}:7447'
    prev = os.environ.get("ZENOH_CONFIG_OVERRIDE", "")
    if prev and endpoint not in prev:
        print(f"[zenoh] overriding stale ZENOH_CONFIG_OVERRIDE ({prev[:60]}...)")
    os.environ["ZENOH_CONFIG_OVERRIDE"] = (
        f'transport/shared_memory/enabled=true;mode="client";connect/endpoints=["{endpoint}"]')
    print(f"[zenoh] endpoint {endpoint}")

    #--- the pose the demonstrations begin from ---------------------------------------
    sys.path.insert(0, os.environ.get("LEROBOT_SRC", "/home/robotis/robot_omy/lerobot/src"))
    import torch
    _o = torch.load
    torch.load = lambda *ar, **kw: _o(*ar, **{**kw, "weights_only": False})
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("dawity/aiw_pp_2bttles_lerobot", root=a.data)

    #THE MEDIAN START POSE OVER EVERY EPISODE, NOT EPISODE 0's FIRST FRAME.
    #This used to be `ds.hf_dataset[0]["observation.state"]` -- one sample, from one episode. The
    #90 episodes do not start in the same place, and episode 0 is an outlier: it sits up to
    #0.3275 rad from the median, and the two worst joints are the ELBOWS (arm_l_joint6 off by
    #0.302, arm_r_joint6 by 0.328). So --home-first drove the elbows a third of a radian from
    #where every demonstration begins, the policy spent its whole first move correcting for that,
    #and then stalled off-manifold. Measured on the robot as a ~0.5 rad excursion in arm_l_joint1
    #that decayed to nothing by step 100.
    #
    #The start-pose guard could not catch this: it measures deviation FROM q_home, and --home-first
    #drives TO q_home, so it always reported 0.000 no matter how wrong q_home was. A guard whose
    #reference is the thing being checked is not a guard.
    #
    #The median matches the robot's own arm_*_joint_trajectory_executor `home` parameter to
    #0.003 rad, which is derived the same way from the same 90 episodes -- so this now agrees with
    #what the robot itself homes to.
    _st = np.asarray(ds.hf_dataset["observation.state"], dtype=np.float64)
    _ep = np.asarray(ds.hf_dataset["episode_index"])
    q_home = np.median(np.stack([_st[np.where(_ep == e)[0][0]] for e in np.unique(_ep)]), axis=0)

    #THE DATASET MAY NOT SPEAK THE ROBOT'S LAYOUT. The two-bottle set is 16-D in exactly
    #spec.MODEL_JOINTS order. The screwing set is 22-D with the grippers INTERLEAVED (gripper_l at
    #7, gripper_r at 15) plus head, lift and base. Comparing a 22-D q_home against a 16-D q_now
    #would raise here, which is the good case; silently taking the first 16 would put the start-pose
    #guard on the wrong joints, which is not. Resolve by NAME, from the dataset's own recorded
    #names, and refuse if a joint the robot commands is absent.
    if len(q_home) != len(spec.MODEL_JOINTS):
        import json as _json
        _info = _json.loads((Path(a.data) / "meta" / "info.json").read_text())
        _names = (_info["features"].get("observation.state") or {}).get("names")
        if not _names:
            raise SystemExit(
                f"{a.data} is {len(q_home)}-D but the robot commands {len(spec.MODEL_JOINTS)}, and "
                f"its meta/info.json records no dimension names. Refusing to guess the layout.")
        _missing = [j for j in spec.MODEL_JOINTS if j not in _names]
        if _missing:
            raise SystemExit(f"{a.data} has no dimension(s) named {_missing}, which the robot "
                             f"commands. Wrong dataset for this robot?")
        _idx = [_names.index(j) for j in spec.MODEL_JOINTS]
        print(f"[home] dataset is {len(q_home)}-D; mapping to the robot's "
              f"{len(spec.MODEL_JOINTS)}-D layout by name")
        q_home = q_home[_idx]
    print(f"[spec] {len(spec.MODEL_JOINTS)} joints, arms 0..{spec.ARM_DIMS-1}, "
          f"grippers {spec.ARM_DIMS}..{len(spec.MODEL_JOINTS)-1}")
    print(f"[home] demo start pose (median over {len(np.unique(_ep))} episodes): "
          f"{np.round(q_home, 3).tolist()}")

    #--- robot ------------------------------------------------------------------------
    from lerobot_robot_ros2_zenoh.ros2_zenoh import ROS2Zenoh
    robot = ROS2Zenoh(spec.robot_config(router_ip=a.router_ip, domain_id=a.domain_id,
                                        cameras=not a.no_cameras))
    print(f"[robot] connecting to zenoh {a.router_ip}:7447 domain {a.domain_id} ...")
    robot.connect()
    print(f"[robot] connected")
    #Which topics actually got publishers, and which joints route to each. If an arm never moves,
    #this is the first thing to check: a topic with no publisher, or joints routed to the wrong one,
    #both look like "the policy is not commanding that arm".
    j2t = getattr(robot, "_joint_to_topic", {})
    pubs = getattr(robot, "_joint_trajectory_publishers", {})
    for topic in sorted(set(j2t.values())):
        owned = [j for j in spec.MODEL_JOINTS if j2t.get(j) == topic]
        print(f"[robot] {'PUB ' if topic in pubs else 'NO PUBLISHER '}{topic}")
        print(f"        joints: {owned}")
    unrouted = [j for j in spec.MODEL_JOINTS if j not in j2t]
    if unrouted:
        print(f"[robot] WARNING: no topic for {unrouted} -- those joints can never be commanded")

    _raw_shapes = {}

    def _to_224(im):
        """Resize on the CLIENT, before the wire.

        We were shipping three full-resolution frames -- 672x376 plus two 240x424, about 1.37 MB of
        raw pixels per inference -- msgpack-encoded over a websocket, and the server then resized
        them to 224x224 and discarded ~90% of what we sent. Resizing here cuts the payload to
        0.30 MB (4.5x) and takes the resize off the inference path entirely.
        It is also MORE faithful, not less: the training loader fed the model 224x224, so this is
        the size the weights were fitted on. Compression would shrink the wire further but adds
        encode/decode on both ends and a lossy step the training data never had; dropping the
        pixels we were about to throw away is strictly better than compressing them.
        """
        import cv2
        a_ = np.asarray(im)
        #RAW SHAPE MATTERS AND WAS NEVER CHECKED. Training squashed 240w x 424h portrait wrists
        #into 224x224 -- a 0.53 vertical compression. Our rotation is applied AFTER the squash, so
        #it only lands correctly if the live camera hands us the TRANSPOSE (424w x 240h). If it is
        #already portrait, we compress the wrong axis and every wrist feature is distorted, which
        #degrades exactly the fine spatial discrimination that off-centre objects need.
        if a_.shape[:2] not in _raw_shapes:
            _raw_shapes[a_.shape[:2]] = True
            print(f"[cam] raw frame {a_.shape[1]}w x {a_.shape[0]}h "
                  f"(training: scene 672x376, wrists 240x424 portrait)")
        if a_.shape[:2] == (224, 224):
            return a_
        return cv2.resize(a_, (224, 224), interpolation=cv2.INTER_AREA)

    #Rotation is applied AFTER the 224x224 resize, which is where it was verified. For a square
    #target this is equivalent to rotating the raw frame first: a 90-degree rotation swaps the two
    #scale factors, and both axes land on 224 either way.
    _ROT = {0: None, 90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_CLOCKWISE}

    def observe():
        obs = robot.get_observation()
        cams = [] if a.no_cameras else spec.CAMERA_ORDER
        check_observation(robot, obs, cams)
        #ROS2Zenoh keys joints as "{name}.pos", not the bare name. Read BY NAME in MODEL order
        #(grippers last) rather than trusting the dict's iteration order -- the whole point of the
        #name-keyed lookup is that a reordering upstream cannot silently permute the vector.
        missing = [j for j in spec.MODEL_JOINTS if f"{j}.pos" not in obs]
        if missing:
            raise KeyError(f"observation is missing {missing}; it has: "
                           f"{[k for k in obs if k.endswith('.pos')]}")
        q = np.array([float(obs[f"{j}.pos"]) for j in spec.MODEL_JOINTS], dtype=np.float64)
        #RESIZE ONLY FOR TURBOVLA. `_to_224` exists because TurboVLA's training loader fed it
        #224x224; ACT / act_prior are LeRobot policies trained on the dataset's NATIVE frames
        #(scene 376x672, wrists 424x240 portrait). Squashing those to a square costs a MEASURED
        #8.9x in action accuracy -- 0.0093 -> 0.0833 rad open-loop against the demonstrations --
        #and it fails silently: the arm just behaves as if the policy were badly trained.
        #The wrist ROTATION below is still required for both; only the resize is policy-specific.
        imgs = ([_to_224(obs[c]) for c in cams] if a.policy == "turbovla"
                else [np.asarray(obs[c]) for c in cams])
        #WRISTS ONLY. The scene camera already matches the recorded orientation; rotating it too
        #would break the one view that was right.
        r = _ROT.get(a.wrist_rot)
        if r is not None:
            imgs = [im if i == 0 else cv2.rotate(im, r) for i, im in enumerate(imgs)]
        return q, imgs

    #CHECK 1 of 2, before the policy is loaded. Cheap to fail here: nothing is committed yet.
    if a.rest_check:
        require_at_rest(lambda: observe()[0], spec.ARM_DIMS, spec.MODEL_JOINTS,
                        n=a.rest_window, vel_max=a.rest_vel_max, drift_max=a.rest_drift_max,
                        label="before policy load", on_fail=robot.disconnect)

    q_now, imgs = observe()
    #Compare the ARMS in radians and the GRIPPERS as booleans. The grippers use `binary`
    #normalization (threshold 0.49), so the policy never sees their raw value -- 0.318 and 0.025
    #are the SAME input to it. Judging them in radians refuses valid start poses over a difference
    #the model cannot perceive.
    err = np.abs(q_now - q_home)
    arm_err = err[:spec.ARM_DIMS]
    grip_now = (q_now[spec.ARM_DIMS:] > 0.49)
    grip_home = (q_home[spec.ARM_DIMS:] > 0.49)
    grip_mismatch = (grip_now != grip_home)
    print(f"[start] current pose: {np.round(q_now, 3).tolist()}")
    print(f"[start] max ARM deviation {arm_err.max():.3f} rad "
          f"({spec.MODEL_JOINTS[int(arm_err.argmax())]})")
    print(f"[start] grippers (binary): now {grip_now.astype(int).tolist()} "
          f"vs demo {grip_home.astype(int).tolist()}"
          f"{'  MISMATCH' if grip_mismatch.any() else '  match'}")
    err = arm_err
    if err.max() > a.start_tol and a.home_first:
        if not a.live:
            robot.disconnect()
            raise SystemExit("--home-first moves the robot, so it requires --live. Without it "
                             "nothing can be published and the pose cannot change.")
        print(f"[home-first] {err.max():.3f} rad out of tolerance -- driving to the demo start "
              f"pose at {a.home_vel} rad/s")
        go_home(robot, q_home, q_now, vel=a.home_vel, fps=spec.FPS)
        q_now, imgs = observe()
        err = np.abs(q_now - q_home)[:spec.ARM_DIMS]
        print(f"[home-first] arrived: max ARM deviation now {err.max():.3f} rad")
        #go_home just drove the arm. Confirm it actually STOPPED before anything plans from it --
        #a splined move that is still settling looks identical, to the position guard, to one that
        #has come to rest.
        if a.rest_check:
            require_at_rest(lambda: observe()[0], spec.ARM_DIMS, spec.MODEL_JOINTS,
                            n=a.rest_window, vel_max=a.rest_vel_max, drift_max=a.rest_drift_max,
                            label="after homing", on_fail=robot.disconnect)

    if err.max() > a.start_tol:
        #Name the joints. "0.340 rad from the start pose" tells the operator they are off but not
        #which way to move, so the next step is a guess; the per-joint delta makes it one move.
        _order = np.argsort(err)[::-1]
        _worst = [f"{spec.MODEL_JOINTS[i]:<18}{q_now[i]:+.3f} -> {q_home[i]:+.3f}"
                  f"  (move {q_home[i]-q_now[i]:+.3f})"
                  for i in _order[:4] if err[i] > a.start_tol * 0.5]
        robot.disconnect()
        raise SystemExit(
            f"REFUSING TO RUN: {err.max():.3f} rad from the demonstrated start pose "
            f"(tolerance {a.start_tol}).\n\n"
            f"  joint             now      -> demo start   delta\n  "
            + "\n  ".join(_worst) +
            f"\n\n  The policy has only seen this task begin near that pose.\n"
            f"  --home-first  moves there for you (needs --live; it is a real move).\n"
            f"  --start-tol   raise it only to test generalisation deliberately -- on a DRY RUN\n"
            f"                that is harmless, since nothing is published.")

    #--- policy: the TurboVLA server, over loopback ------------------------------------
    #The ChunkNormalizer belongs to TurboVLA, which normalises outside the policy. LeRobot
    #checkpoints (act_prior, groot) carry their own pre/post-processors, so this is only needed for
    #the gripper LEVELS. Failing to build it must not stop a LeRobot run: --ckpt points at a
    #TurboVLA file that has nothing to do with either.
    from normalizer import ChunkNormalizer
    try:
        norm = ChunkNormalizer.from_run_dir(a.ckpt)
    except Exception as _exc:
        if a.policy == "turbovla":
            raise
        print(f"[norm] ChunkNormalizer unavailable ({type(_exc).__name__}); "
              f"falling back to gripper levels from the dataset")
        norm = None
    if a.policy == "act_prior":
        #No server, no ChunkNormalizer on the chunk: a LeRobot checkpoint carries its own
        #pre/post-processors, so normalizing again here would double-apply. `norm` is still built
        #because the start-pose guard and the gripper levels read from it.
        if not a.policy_path:
            raise SystemExit("--policy act_prior needs --policy-path <checkpoint dir>")
        from act_prior_policy import ActPriorPolicy
        cam_keys = ["observation.images.scene", "observation.images.wrist_left",
                    "observation.images.wrist_right"]
        policy = ActPriorPolicy(a.policy_path, camera_keys=cam_keys,
                                device="cuda" if torch.cuda.is_available() else "cpu")
    elif a.policy == "groot":
        #GR00T N1.7 is a LeRobot policy and imports cleanly here, so it runs in-process like
        #act_prior rather than behind the TurboVLA websocket.
        #
        #THE DIMENSION MAP IS THE DANGEROUS PART, not the loading. The screwing checkpoint speaks
        #22 dims with the grippers INTERLEAVED (gripper_l at 7, gripper_r at 15) while this file
        #speaks spec.MODEL_JOINTS, 16 dims with both grippers LAST. Feeding one where the other is
        #expected raises nothing -- it commands arm_r_joint1 with a gripper value. groot_policy.py
        #resolves every index BY NAME from the checkpoint's own recorded names and refuses to run
        #if a commanded joint is missing. Run `python3 groot_policy.py` to see that verified.
        if not a.policy_path:
            raise SystemExit("--policy groot needs --policy-path <checkpoint dir>")
        from groot_policy import GrootPolicy
        cam_keys = ["observation.images.rgb.cam_left_head",
                    "observation.images.rgb.cam_left_wrist",
                    "observation.images.rgb.cam_right_wrist"]
        policy = GrootPolicy(a.policy_path, camera_keys=cam_keys,
                             robot_joints=spec.MODEL_JOINTS,
                             #the checkpoint records shapes but not dimension NAMES, so the map
                             #comes from the dataset it was trained on -- the same --data that
                             #supplies q_home, which keeps one stated authority for the layout
                             names_from=a.data,
                             device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        from aiworker.remote_policy import RemoteTurboVLAPolicy, RemoteConfig
        policy = RemoteTurboVLAPolicy(
            normalizer=norm,
            config=RemoteConfig(host=a.policy_host, port=a.policy_port))
    resid = None
    if a.residual and a.policy in ("act_prior", "groot"):
        #The residual was fitted on TurboVLA's conditioning features and its PCA basis. Those do
        #not exist here, so silently applying it would edit with a projection from another model.
        print(f"[residual] disabled: it is fitted to TurboVLA features, not {a.policy}'s")
        a.residual = ""
    if a.residual:
        from residual import ResidualPolicy
        resid = ResidualPolicy(a.residual, arm_dims=list(range(spec.ARM_DIMS)),
                               device="cuda" if torch.cuda.is_available() else "cpu",
                               scale=a.residual_scale)
        resid.load_pca(a.residual_buffer)
        policy.want_feature = True          #makes the server ship the conditioning feature
    info = policy.connect()
    print(f"[policy] {a.policy} -> {info}"
          + (f"  (server {a.policy_host}:{a.policy_port})" if a.policy == "turbovla" else ""))
    print(f"[mode] {'LIVE -- THE ROBOT WILL MOVE' if a.live else 'DRY RUN -- nothing published'}")
    print(f"[safety] vel<={a.max_vel} rad/s acc<={a.max_acc} rad/s^2 speed={a.speed} "
          f"budget={a.max_steps}")

    #==================================================================================
    #CLOSED-LOOP CONTROL. Observe -> infer -> execute ONE step -> repeat.
    #
    #Measured on this setup: observe 1 ms, policy round trip 19 ms. That is a 20 ms loop against a
    #33 ms budget at 30 Hz, so there is no reason to run open-loop off a buffered plan at all --
    #and the buffer is what produced every failure so far. Splicing a fresh plan behind queued
    #actions commands the arm back to a pose it has already left; publishing trajectories makes the
    #controller reset its time base mid-motion. Both are structural, and both disappear if every
    #command is computed from the pose the arm is ACTUALLY in.
    #
    #THE OFFSET IS NOT OPTIONAL. A chunked policy's step 0 is a ramp-in from the current pose -- it
    #is where the arm already is. Executing step 0 every tick therefore asks the arm to stay put,
    #which is the mirror image of the buffer bug. Reading a few steps in skips the ramp and
    #commands where the arm should be SHORTLY, which is what actually produces motion.
    #==================================================================================
    from trajectory import send_point
    import threading

    #BACKGROUND OBSERVER. `robot.get_observation()` blocks waiting for camera frames over zenoh and
    #was measured at 53-476 ms -- against a 19 ms policy round trip and a 111 ms control period.
    #It was both the bottleneck AND the jitter: a loop whose period swings between 70 ms and 500 ms
    #commands the arm at a wildly varying rate, which is what the arm sounds like.
    #A thread keeps the newest observation cached; the control loop takes whatever is current and
    #never blocks. A frame one tick stale is worth far more than a loop that stalls half a second.
    _latest = {"q": q_now.copy(), "imgs": imgs, "t": time.perf_counter(), "n": 0}
    _obs_stop = threading.Event()

    def _observer():
        while not _obs_stop.is_set():
            try:
                q_, i_ = observe()
                _latest.update(q=q_, imgs=i_, t=time.perf_counter(), n=_latest["n"] + 1)
            except Exception as exc:
                print(f"  [obs] {type(exc).__name__}: {exc}")
                time.sleep(0.02)

    _obs_thread = threading.Thread(target=_observer, daemon=True)
    _obs_thread.start()

    hz = a.hz if a.hz > 0 else spec.FPS * a.speed
    dt_ctl = 1.0 / hz
    #How many chunk waypoints of trajectory-time pass per control tick.
    _stride = max(1, int(round(spec.FPS * dt_ctl)))
    grip_held = None
    grip_votes = np.zeros(len(spec.MODEL_JOINTS) - spec.ARM_DIMS, dtype=int)
    if resid is not None:
        print(f"[mode] RESIDUAL ACTIVE (scale {a.residual_scale})")
    if a.mode == "ensemble":
        print(f"[mode] ENSEMBLE {hz:.1f} Hz  re-plan EVERY tick, command the time-aligned mean of "
              f"the last {20 if a.ensemble <= 1 else a.ensemble} chunks (stride {_stride})")
    else:
        print(f"[mode] RECEDING HORIZON {hz:.1f} Hz  execute {a.execute_steps}/{spec.CHUNK_SIZE} "
              f"waypoints per plan  (one waypoint per tick, then re-plan)")

    q_prev = q_now.copy()
    v_prev = np.zeros_like(q_prev)
    plan, plan_i, replans = None, 0, 0
    n_edit = 0
    ens_target = None
    steps, holds, sent = 0, 0, 0
    interrupted = False
    obs_ms = rt_ms = 0.0
    t_next = time.perf_counter()
    t_start = t_next
    A = spec.ARM_DIMS
    #Per-gripper class levels in radians, and the midpoint that separates them. Falls back to the
    #old 0/1 decode when the checkpoint has no gripper_levels.json, so behaviour is unchanged for
    #a run that lacks it rather than silently half-converted.
    _lv = getattr(norm, "grip_levels", None) if norm is not None else None
    #THE LEVELS MUST COME FROM THIS TASK. The ChunkNormalizer is a TurboVLA artefact built on the
    #two-bottle set, where "open" is 0.320 rad. The screwing set opens at 0.035. Using the wrong
    #levels commands the jaws ~0.285 rad more closed than intended -- about 37 mm of lost opening,
    #which is the difference between clearing a screw and knocking it over. For a LeRobot policy the
    #dataset given by --data is the authority; the normalizer is only right for turbovla.
    if a.policy != "turbovla":
        _lv = None
    if _lv is None and len(q_home):
        #Derive the open/closed levels from the ACTION column of the same dataset q_home came from,
        #by name, so a LeRobot run does not depend on a TurboVLA artefact.
        try:
            _ac = np.asarray(ds.hf_dataset["action"], dtype=np.float64)
            if _ac.shape[1] != len(spec.MODEL_JOINTS):
                _ac = _ac[:, _idx]
            _lv = {}
            for _g in range(spec.ARM_DIMS, len(spec.MODEL_JOINTS)):
                _col = _ac[:, _g]
                _lv[_g] = (float(np.percentile(_col, 5)), float(np.percentile(_col, 95)))
            print(f"[grip] levels from the dataset action column: "
                  f"{ {spec.MODEL_JOINTS[k]: (round(v[0],3), round(v[1],3)) for k, v in _lv.items()} }")
        except Exception as _exc:
            print(f"[grip] could not derive levels ({type(_exc).__name__}); using defaults")
            _lv = None
    if _lv:
        grip_lo = np.array([_lv[int(d)][0] for d in norm.binary], dtype=np.float64)
        grip_hi = np.array([_lv[int(d)][1] for d in norm.binary], dtype=np.float64)
    else:
        grip_lo = np.zeros(len(spec.MODEL_JOINTS) - A)
        grip_hi = np.ones(len(spec.MODEL_JOINTS) - A)
    grip_mid = (grip_lo + grip_hi) / 2.0
    #PER-CAMERA LIVENESS. check_observation() rejects a camera that is missing or all black, but a
    #FROZEN one -- still delivering the last good frame, forever -- passes every check it makes and
    #is invisible in the logs. It matters because the views are per-arm: a stale left wrist leaves
    #the policy acting on a left arm it cannot see, which is what a large spurious arm_l motion
    #alongside a roughly-correct arm_r looks like. `dcam` is the mean absolute change since the
    #previous frame; a live camera is never 0.000 twice in a row.
    #Recent chunks, newest first, for temporal ensembling. Only arm dims are averaged: the
    #grippers are a class decision with their own hysteresis, and averaging a class would produce
    #a value between "open" and "closed" that means neither.
    _chunks = []
    if a.ensemble > 1:
        print(f"[ensemble] averaging arm targets over {a.ensemble} chunks, decay "
              f"{a.ensemble_decay}, stride {_stride} chunk-steps per control step")
    _prev_imgs = None
    _obs_dir = None
    if a.save_obs:
        import os
        _obs_dir = a.save_obs
        os.makedirs(_obs_dir, exist_ok=True)
        print(f"[save-obs] dumping live observations every {a.save_every} steps -> {_obs_dir}")
    print(f"[grip] open {np.round(grip_lo,3).tolist()}  closed {np.round(grip_hi,3).tolist()}  "
          f"switch at {np.round(grip_mid,3).tolist()} rad")
    #CHECK 2 of 2, immediately before the first publish. Check 1 ran BEFORE the policy was loaded,
    #and loading weights takes seconds -- a window in which the arm can be bumped, jogged, or
    #released by an operator who reasonably assumed the run had not started. Re-verify both
    #stillness AND position here, against the same q_home, so nothing that happened during the
    #load can carry into the first commanded chunk.
    if a.rest_check:
        require_at_rest(lambda: _latest["q"], spec.ARM_DIMS, spec.MODEL_JOINTS,
                        n=a.rest_window, vel_max=a.rest_vel_max, drift_max=a.rest_drift_max,
                        label="before first publish", on_fail=robot.disconnect)
        _drift = np.abs(_latest["q"] - q_home)[:A]
        if _drift.max() > a.start_tol:
            robot.disconnect()
            raise SystemExit(
                f"REFUSING TO RUN: the arm moved during policy load -- now {_drift.max():.3f} rad "
                f"from the demo start pose (tolerance {a.start_tol}), joint "
                f"{spec.MODEL_JOINTS[int(_drift.argmax())]}.")

    trace_fh = None
    q_last_meas = _latest["q"].copy()
    if a.trace:
        trace_fh = open(a.trace, "w")
        trace_fh.write(",".join(["step"] + [f"cmd_{j}" for j in spec.MODEL_JOINTS[:A]] +
                                [f"act_{j}" for j in spec.MODEL_JOINTS[:A]]) + "\n")
        print(f"[trace] per-joint commanded vs achieved -> {a.trace}")
    try:
        _keys = _SubtaskKeys(subtask_list, enabled=a.subtasks)
        while steps < a.max_steps:
            #Read stage changes BEFORE predicting, so a chunk is never planned under
            #the old instruction and then executed under the new one.
            if a.subtasks:
                _m = _keys.poll()
                if _m:
                    a.task = _keys.instruction
                    print(f"\n[subtask] {_m}", flush=True)
            try:
                t0 = time.perf_counter()
                #Take the cached observation. Never blocks; `age` says how stale it is.
                q_now, imgs = _latest["q"], _latest["imgs"]
                age_ms = (t0 - _latest["t"]) * 1000.0
                t1 = time.perf_counter()
                #RECEDING HORIZON. Re-plan only when the current plan is used up, then execute its
                #waypoints IN ORDER, one per control tick.
                #
                #This replaces "re-plan every tick and execute waypoint[offset]", which discarded
                #49 of every 50 predicted waypoints and turned `offset` into a tuned knob that was
                #doing much of the work. It also never followed a predicted path -- the arm traced
                #the envelope of successive lookahead points, which is not the trajectory the
                #policy predicted and not what the model was trained or published under.
                #
                #Executing in order also removes the reason `offset` existed. The chunk's first
                #waypoints are a ramp-in that barely advances; under the old scheme we re-entered
                #that ramp-in every tick and deadlocked, so the cure was to skip past it. Here the
                #ramp-in is executed once and the plan carries on into the part that does the work.
                if a.mode == "ensemble":
                    chunk = np.asarray(policy.predict_chunk(imgs, q_now, a.task), dtype=np.float64)
                    _chunks.insert(0, chunk)
                    del _chunks[max(a.ensemble, 1) if a.ensemble > 1 else 20:]
                    #Chunk predicted i ticks ago describes NOW at index i*stride.
                    num = np.zeros(A); den = 0.0
                    for i, ch in enumerate(_chunks):
                        k = i * _stride
                        if k >= len(ch):
                            break
                        w = float(np.exp(-a.ensemble_decay * i))
                        num += w * ch[k][:A]
                        den += w
                    plan, plan_i, replans = chunk, 0, replans + 1
                    ens_target = (num / den) if den > 0 else chunk[0][:A]
                elif plan is None or plan_i >= min(a.execute_steps, len(plan)):
                    chunk = np.asarray(policy.predict_chunk(imgs, q_now, a.task),
                                       dtype=np.float64)
                    import control_math as _cmh
                    _cmh.check_horizon(len(chunk), a.execute_steps,
                                       getattr(a, "seam_blend", 0))
                    if resid is not None and getattr(policy, "last_feature", None) is not None:
                        #Edit in NORMALIZED space, where the residual was trained, then denormalize
                        #once -- exactly the path the unedited chunk already takes.
                        base_n = policy.last_normalized
                        edited = resid.edit(policy.last_feature,
                                            norm.to_normalized(
                                                torch.from_numpy(q_now)).numpy(),
                                            base_n)
                        chunk = norm.to_raw(
                            torch.as_tensor(edited, dtype=torch.float32)).numpy().astype(
                                np.float64)
                        n_edit += 1
                    if a.ensemble > 1:
                        _chunks.insert(0, chunk)
                        del _chunks[a.ensemble:]
                        #TEMPORAL ENSEMBLING ACROSS PLANS, aligned in absolute time. Because we
                        #re-plan exactly every `execute_steps` ticks and execute one waypoint per
                        #tick, the plan from i re-plans ago describes this same instant starting
                        #`execute_steps * i` waypoints into itself. Averaging those terms cancels
                        #per-inference jitter without lag: every term refers to the same moment.
                        num = np.zeros_like(chunk[:, :A])
                        #Per-index weight, NOT a scalar. Older plans only reach partway into the
                        #new horizon, so a single denominator would divide late waypoints by
                        #weight they never received and shrink the far end of the trajectory
                        #toward zero.
                        den = np.zeros((len(chunk), 1))
                        for i, ch in enumerate(_chunks):
                            sh = a.execute_steps * i
                            if sh >= len(ch):
                                break
                            w = float(np.exp(-a.ensemble_decay * i))
                            seg = ch[sh:sh + len(chunk)][:, :A]
                            num[:len(seg)] += w * seg
                            den[:len(seg)] += w
                        chunk = chunk.copy()
                        m = (den[:, 0] > 0)
                        chunk[m, :A] = num[m] / den[m]
                    #NO PREDECESSOR -> trim the incoherent head instead of blending it away.
                    if plan is None and a.trim_head > 0:
                        d = chunk[-1, :A] - chunk[0, :A]
                        nd = np.linalg.norm(d)
                        if nd > 1e-9:
                            d = d / nd
                            #W must span BOTH dips in the head. At 8 the rule stops at the first
                            #locally-coherent stretch (waypoint 4) and the reversal at 16 survives;
                            #at 12 or more it holds out for waypoint 18, after which nothing in the
                            #executed span moves backward at all (worst cos +0.40 vs -0.55 raw).
                            cut, W = 0, 12
                            for k0 in range(min(a.trim_head, len(chunk) - W - 1)):
                                ok_ = True
                                for k1 in range(k0, k0 + W):
                                    st_ = chunk[k1 + 1, :A] - chunk[k1, :A]
                                    n_ = np.linalg.norm(st_)
                                    if n_ > 1e-9 and float(st_ @ d) / n_ < 0.3:
                                        ok_ = False
                                        break
                                if ok_:
                                    cut = k0
                                    break
                            if cut > 0:
                                chunk = chunk[cut:]
                                print(f"  [plan] trimmed {cut} incoherent lead waypoints "
                                      f"from the first plan")
                    #SEAM CROSS-FADE. The old plan's waypoint (execute_steps + k) and the new
                    #plan's waypoint k describe the same instant, because exactly execute_steps
                    #waypoints of time elapsed while the old plan ran.
                    if a.seam_blend > 0 and plan is not None:
                        W = int(a.seam_blend)
                        lim = min(W, len(plan) - a.execute_steps - 1, len(chunk) - 1)
                        if lim > 1:
                            #CUBIC HERMITE, matching POSITION AND VELOCITY at both ends.
                            #The linear cross-fade this replaces matched position only, so the
                            #command arrived at the seam with the wrong slope: measured on the
                            #robot, the commanded step at a re-plan boundary was 1.88x the step
                            #elsewhere -- not a reversal (seams reverse 0.0% against 2.4%
                            #elsewhere) but a velocity discontinuity, felt as a lurch every
                            #execute_steps ticks. Simulation of the same geometry reproduces it
                            #at 2.00x and puts C1 blending at 1.20x.
                            #
                            #This is the idea behind Physical Intelligence's Real-Time Chunking
                            #adapted to a deterministic head. RTC inpaints the committed prefix
                            #INSIDE a flow-matching sampler; TurboVLA's ACTDecoder is a single
                            #forward pass with no sampler to condition, so the constraint is
                            #imposed on the emitted chunk instead. Weaker than RTC proper, and it
                            #should be described that way -- but it targets the same discontinuity.
                            t = (np.arange(lim, dtype=np.float64) + 1.0) / (lim + 1.0)
                            t = t[:, None]
                            h00 = 2 * t**3 - 3 * t**2 + 1
                            h10 = t**3 - 2 * t**2 + t
                            h01 = -2 * t**3 + 3 * t**2
                            h11 = t**3 - t**2
                            o0 = a.execute_steps
                            p0 = plan[o0, :A]
                            p1 = chunk[lim, :A]
                            #endpoint velocities, scaled to the blend window
                            m0 = (plan[min(o0 + 1, len(plan) - 1), :A] - p0) * lim
                            m1 = (chunk[lim, :A] - chunk[max(lim - 1, 0), :A]) * lim
                            chunk[:lim, :A] = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
                    if a.savgol and a.savgol >= 5:
                        #SAVITZKY-GOLAY instead of the boxcar. A moving average is a low-pass that
                        #flattens curvature: it fits a CONSTANT over the window, so a trajectory
                        #that is turning gets pulled toward the chord and the turn is blunted --
                        #which on this arm reads as sluggish, rounded-off motion. SavGol fits a
                        #polynomial of degree `savgol_order` instead, so it removes per-inference
                        #jitter while preserving the curvature and the endpoints. Same window, less
                        #distortion. Arm dims only: the grippers are a class decision and
                        #polynomial-smoothing a class produces a value between open and closed.
                        from scipy.signal import savgol_filter
                        w = int(a.savgol) | 1               #odd, centred
                        w = min(w, (len(chunk) // 2) * 2 - 1)
                        if w >= a.savgol_order + 2:
                            chunk[:, :A] = savgol_filter(
                                chunk[:, :A], window_length=w, polyorder=a.savgol_order,
                                axis=0, mode="nearest")
                    elif a.plan_smooth and a.plan_smooth >= 3:
                        w = int(a.plan_smooth) | 1          #odd, so the window is centred
                        k = np.ones(w) / w
                        pad = w // 2
                        sm = chunk.copy()
                        for c in range(A):
                            e = np.pad(chunk[:, c], (pad, pad), mode="edge")
                            sm[:, c] = np.convolve(e, k, mode="valid")[:len(chunk)]
                        chunk = sm
                    #ALWAYS FROM WAYPOINT 0. Starting deeper was tried and reverted: --seam-blend
                    #indexes the previous plan at execute_steps + k, which only lines up in time
                    #when execution begins at 0, so a non-zero start blended misaligned waypoints
                    #and took the reversal rate from 2.2% (matching the demos' 2.1%) to 10.9%.
                    plan, plan_i = chunk, 0
                    replans += 1
                t2 = time.perf_counter()
                obs_ms, rt_ms = age_ms, (t2 - t1) * 1000.0
            except Exception as exc:
                holds += 1
                print(f"  [hold] {type(exc).__name__}: {exc}")
                time.sleep(dt_ctl)
                continue

            #Read `offset` steps in, past the ramp-in. Clamped so a short chunk cannot index past
            #its end.
            chunk = plan
            if a.mode == "ensemble":
                #Gripper comes from the FRESHEST chunk at index 0 -- it is a class decision and
                #averaging it would give a value between open and closed that means neither.
                j = 0
                target = plan[0].copy()
                target[:A] = ens_target
            else:
                j = min(plan_i, len(plan) - 1)
                target = plan[j].copy()
                plan_i += 1

            #Gripper: fast to close, slow to open. Decided over the whole chunk, not one step --
            #the grasp lives deep in the horizon and reading a single step never sees it.
            #`chunk` arrives in RADIANS, and since gripper_levels.json the two classes decode to
            #the radians they actually occupy (~+0.32 rest, ~+0.86 closed) rather than 0.0/1.0.
            #So the class test is against the MIDPOINT between those levels, not 0.5, and what gets
            #commanded is the level itself. Testing against 0.5 here would call the rest pose
            #"open" and then re-emit 0.0 -- exactly the decode this replaced.
            gw = (chunk[:, A:] > grip_mid).astype(np.float64)
            if grip_held is None:
                grip_held = gw[j].copy()
            if a.grip_hold > 0:
                #THE WINDOW FOLLOWS THE OFFSET. It used to be `gw[:execute_steps]` -- anchored at
                #the START of the chunk -- which was already wrong and became badly wrong once
                #--offset moved execution deep into the horizon. The open vote requires that NO
                #step in the window says "closed", so a window still covering the chunk's early
                #ramp-in keeps seeing the grasp that is about to happen (or just happened) and
                #never votes to release. Measured on the robot as a hand that grips, places, and
                #then refuses to let go. Anchoring at `j` asks about the part of the chunk we are
                #actually executing and the part just after it, which is what "should I be holding
                #this NOW" means.
                w1 = min(j + a.execute_steps, len(gw))
                win = gw[j:w1] if w1 > j else gw[j:j + 1]
                #Separate lookahead for closing: short, so the hand shuts when the grasp is due
                #rather than as soon as one is anywhere on the horizon.
                c1 = min(j + max(a.grip_lead, 1), len(gw))
                cwin = gw[j:c1] if c1 > j else gw[j:j + 1]
                wants_open = (win.max(axis=0) < 0.5)
                #A CLOSED HAND MAY ONLY OPEN THROUGH THE VOTE. The previous form was
                #`if closed and wants_open: vote ... else: grip_held = cwin.max()`, and that `else`
                #also caught "closed, but wants_open is False" -- which then reassigned grip_held
                #straight from the short close-window and could OPEN the hand instantly,
                #bypassing --grip-hold entirely. That happens exactly at a grasp: the long window
                #still sees the close that just fired (so wants_open is False) while the short
                #window has moved past it. Measured on the robot as the gripper stuttering as it
                #tries to grab.
                #Closing stays instant, which is the asymmetry the vote was built around.
                for g in range(gw.shape[1]):
                    if grip_held[g] > 0.5:
                        if wants_open[g]:
                            grip_votes[g] += 1
                            if grip_votes[g] >= a.grip_hold:
                                grip_held[g] = 0.0
                                grip_votes[g] = 0
                        else:
                            grip_votes[g] = 0
                    else:
                        grip_votes[g] = 0
                        grip_held[g] = float(cwin[:, g].max() > 0.5)
                target[A:] = np.where(grip_held > 0.5, grip_hi, grip_lo)
            else:
                target[A:] = np.where(gw[j] > 0.5, grip_hi, grip_lo)

            #Rate limit against the LAST COMMAND, not the measured pose: chaining off the
            #measurement lets tracking lag inflate the allowed step and turns a lagging arm into a
            #lurching one.
            #Per-joint speed cap from the chunk's OWN local spacing, so the commanded profile
            #decelerates where the demonstrations decelerate rather than saturating throughout.
            vcap = np.full(A, a.max_vel)
            if a.follow_speed > 0 and j + _stride < len(chunk):
                vloc = np.abs(chunk[j + _stride][:A] - chunk[j][:A]) / dt_ctl
                vcap = np.minimum(a.max_vel, np.maximum(a.follow_speed * vloc, a.min_vel))
            lim = np.concatenate([vcap, np.full(len(target) - A, a.max_vel)]) * dt_ctl
            dq = np.clip(target - q_prev, -lim, lim)
            v = dq / dt_ctl
            v = v_prev + np.clip(v - v_prev, -a.max_acc * dt_ctl, a.max_acc * dt_ctl)
            if a.vel_smooth > 0:
                v = a.vel_smooth * v_prev + (1.0 - a.vel_smooth) * v
            dq = v * dt_ctl
            cmd = q_prev + dq
            cmd[A:] = target[A:]              #gripper raw: binary, never rate limited
            v[A:] = 0.0
            v_prev = v

            if a.live:
                send_point(robot, {f"{jn}.pos": float(x)
                                   for jn, x in zip(spec.MODEL_JOINTS, cmd)}, dt_ctl)
            if trace_fh is not None:
                #cmd_d  what we asked the joint to change by, since the last command
                #act_d  what the joint actually changed by, measured, over the same interval
                #A joint tracking correctly has act_d following cmd_d with a small lag. A joint
                #with an inverted sign has act_d ~ -cmd_d, which no amount of upstream checking
                #can reveal.
                cmd_d = cmd[:A] - q_prev[:A]
                act_d = q_now[:A] - q_last_meas[:A]
                trace_fh.write(",".join([str(steps)] +
                                        [f"{x:.6f}" for x in cmd_d] +
                                        [f"{x:.6f}" for x in act_d]) + "\n")
                if steps % 40 == 0:
                    trace_fh.flush()
                q_last_meas = q_now.copy()
            q_prev = cmd
            steps += 1
            if _obs_dir is not None and steps % a.save_every == 0:
                import cv2 as _cv2
                for _ci, _cn in enumerate(spec.CAMERA_ORDER if not a.no_cameras else []):
                    if _ci < len(imgs):
                        #PNG is lossless, and BGR here only because imwrite expects it -- the
                        #array itself is exactly what went to the model.
                        _cv2.imwrite(f"{_obs_dir}/s{steps:04d}_{_ci}_{_cn}.png",
                                     _cv2.cvtColor(np.asarray(imgs[_ci]), _cv2.COLOR_RGB2BGR))
                np.save(f"{_obs_dir}/s{steps:04d}_state.npy", q_now)
            if steps % 20 == 0:
                dcam = []
                if _prev_imgs is not None and len(_prev_imgs) == len(imgs):
                    for a_, b_ in zip(_prev_imgs, imgs):
                        dcam.append(float(np.abs(np.asarray(a_, np.float32) -
                                                 np.asarray(b_, np.float32)).mean()))
                _prev_imgs = [np.asarray(i).copy() for i in imgs]
                cam_s = ("  dcam " + "/".join(f"{x:.2f}" for x in dcam)) if dcam else ""
                track = float(np.abs(q_now[:A] - q_prev[:A]).max())
                print(f"  step {steps:4d}  obsage {obs_ms:.0f} rt {rt_ms:.0f}ms  "
                      f"fps {_latest['n']/max(time.perf_counter()-t_start,1e-6):.1f}  "
                      f"track-err {track:.4f}  "
                      f"L{np.abs(cmd[:7]-q_home[:7]).max():.3f} "
                      f"R{np.abs(cmd[7:14]-q_home[7:14]).max():.3f}  "
                      f"grip {np.round(cmd[A:],2).tolist()}  plan {plan_i}/{a.execute_steps} "
                      f"re {replans}{cam_s}", flush=True)
            t_next += dt_ctl
            time.sleep(max(0.0, t_next - time.perf_counter()))
    except KeyboardInterrupt:
        interrupted = True
        _obs_stop.set()
        print("\n[stop] interrupted -- returning to the start pose")
    finally:
        if trace_fh is not None:
            trace_fh.close()
        if a.label and _obs_dir is not None:
            #Asked AFTER the run so the operator answers having just watched it, and written next
            #to the frames so the pair can never drift apart.
            try:
                ans = input("\n[label] which bottles ended up in the basket? "
                            "[n]one / [r]ight / [l]eft / [b]oth: ").strip().lower()
            except EOFError:
                ans = ""
            key = ans[:1] if ans[:1] in ("n", "r", "l", "b") else "n"
            right = key in ("r", "b")
            left = key in ("l", "b")
            n_placed = int(right) + int(left)
            Path(_obs_dir).joinpath("success.json").write_text(json.dumps(
                {"n_placed": n_placed, "right": right, "left": left,
                 #kept so anything reading the old schema still works
                 "success": n_placed == 2,
                 "steps": steps, "task": a.task, "ckpt": str(a.ckpt),
                 "holds": holds}, indent=2))
            print(f"[label] {n_placed}/2 placed (right={right} left={left}) "
                  f"-> {_obs_dir}/success.json")
        #Home on ANY exit, not just a clean one. Ctrl-C is the case that matters most: the arm is
        #mid-trajectory somewhere unplanned, and leaving it there is exactly the state you do not
        #want a shared robot in. --no-home opts out.
        if a.live and a.home:
            try:
                q_now, _ = observe()
                print(f"[home] {'after interrupt' if interrupted else 'run complete'} -- "
                      f"returning to start pose at {a.home_vel} rad/s")
                go_home(robot, q_home, q_now, vel=a.home_vel, fps=spec.FPS)
            except KeyboardInterrupt:
                #A second Ctrl-C during homing means "stop now". Respect it rather than fighting
                #the operator for control of the arm.
                print("[home] aborted by second interrupt -- arm left where it is")
            except Exception as exc:
                print(f"[home] skipped: {exc}")
        robot.disconnect()
    _obs_stop.set()
    print(f"\n[done] {steps} steps, {holds} holds. "
          f"{'Robot was commanded.' if a.live else 'DRY RUN -- nothing was sent.'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
