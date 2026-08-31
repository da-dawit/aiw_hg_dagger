"""Run a LeRobot GR00T N1.7 checkpoint in-process, behind the same interface as act_prior.

Author: Dawit Chun

THE PROBLEM THIS FILE EXISTS TO SOLVE, AND IT IS NOT LOADING THE MODEL.

The screwing checkpoint was trained on a 22-dimensional vector whose grippers are INTERLEAVED:

    0-6  arm_l_joint1..7    7  gripper_l_joint1
    8-14 arm_r_joint1..7   15  gripper_r_joint1
    16-17 head             18  lift          19-21 base (linear_x, linear_y, angular_z)

The robot, through spec_sg2.MODEL_JOINTS, speaks 16 dimensions with both grippers LAST:

    0-6 arm_l_joint1..7   7-13 arm_r_joint1..7   14 gripper_l   15 gripper_r

Both vectors are float arrays of plausible magnitude, so feeding one where the other is expected
does not raise. It silently commands `arm_r_joint7` with a gripper value and vice versa, which on
this hardware means driving a wrist joint to a gripper's angle. Every index here is therefore
resolved BY NAME from the checkpoint's own recorded names and spec_sg2.MODEL_JOINTS. Nothing is
hardcoded, and `--self-test` asserts the round trip.

THE SIX EXTRA DIMENSIONS. head, lift and base are in the policy's input but not in the robot's
action space. Measured across all 40,632 training frames they barely move -- head_joint1 spans
0.006 rad, lift 0.033, the base essentially zero -- so they are filled from the dataset MEDIAN
rather than zeros. Zeros would be out of distribution for lift (median -0.2355) and head_joint1
(median +0.8682), and an out-of-distribution input to a 3B model produces confident nonsense.

WHAT THIS DELIBERATELY DOES NOT TOUCH. Everything downstream of "produce a chunk" is shared with
the act_prior and TurboVLA paths: receding horizon, seam blending, plan smoothing, the gripper class
decode, homing, the start-pose guard, the stillness check and both safety clamps. Each was fixed
against a measured failure on this robot and none are policy-specific. Only the chunk source
changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

#Filled from the training set's own medians; see the module docstring. Overridable per deployment
#because a different task may hold the head somewhere else.
STATIC_FILL = {
    "head_joint1": 0.8682,
    "head_joint2": 0.0506,
    "lift_joint": -0.2355,
    "linear_x": 0.0,
    "linear_y": 0.0,
    "angular_z": 0.0,
}


def _resolve(ckpt_names, robot_names):
    """Index maps between the checkpoint's vector and the robot's, resolved by joint name.

    Returns (to_policy, from_policy, static):
      to_policy[i]   = robot index feeding policy dim i, or -1 if it must be filled
      from_policy[j] = policy index that supplies robot dim j
      static         = {policy index: constant} for dims the robot does not provide
    """
    r_of = {n: i for i, n in enumerate(robot_names)}
    to_policy, static = [], {}
    for i, n in enumerate(ckpt_names):
        if n in r_of:
            to_policy.append(r_of[n])
        else:
            if n not in STATIC_FILL:
                raise KeyError(f"checkpoint dim {i} is {n!r}, which the robot does not provide "
                               f"and STATIC_FILL has no value for")
            to_policy.append(-1)
            static[i] = float(STATIC_FILL[n])
    p_of = {n: i for i, n in enumerate(ckpt_names)}
    from_policy = []
    for n in robot_names:
        if n not in p_of:
            raise KeyError(f"robot commands {n!r} but the checkpoint has no such dimension")
        from_policy.append(p_of[n])
    return np.asarray(to_policy), np.asarray(from_policy), static


class GrootPolicy:
    """`predict_chunk(images, state, instruction) -> (H, 16) chunk in RAW robot action units."""

    def __init__(self, ckpt_dir: str, camera_keys, robot_joints, state_key="observation.state",
                 device: str = "cuda", names_from: str | None = None):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.dev = device if torch.cuda.is_available() else "cpu"
        p = Path(ckpt_dir)
        if (p / "pretrained_model").is_dir():
            p = p / "pretrained_model"
        raw = json.loads((p / "config.json").read_text())

        #The checkpoint records the names of its own dimensions. Trusting those rather than an
        #assumption about layout is the whole point of this file.
        feats = raw.get("input_features", {}) or {}
        names = (feats.get(state_key) or {}).get("names")
        if not names:
            out = raw.get("output_features", {}) or {}
            names = (out.get("action") or {}).get("names")
        declared = ((feats.get(state_key) or {}).get("shape") or [None])[0]

        if not names:
            #A LeRobot checkpoint records the SHAPE of its state vector but not the NAMES of the
            #dimensions. Without names the 22-to-16 map cannot be resolved, and guessing is how you
            #command a shoulder with a gripper value. So the names must be supplied from a stated
            #authority -- the dataset the checkpoint was trained on -- and the length is checked
            #against the shape the checkpoint itself declares.
            if not names_from:
                raise SystemExit(
                    f"{p}/config.json records no dimension names (only shape {declared}).\n"
                    f"  Pass names_from=<dataset root> so the mapping comes from the dataset this\n"
                    f"  checkpoint was trained on. Refusing to guess the layout.")
            src = Path(names_from) / "meta" / "info.json"
            info = json.loads(src.read_text())
            names = (info["features"].get(state_key) or {}).get("names")
            if not names:
                raise SystemExit(f"{src} records no names for {state_key} either")
            print(f"[groot] dimension names taken from {src}")
        if declared is not None and len(names) != declared:
            raise SystemExit(
                f"the checkpoint declares a {declared}-D state but the names source gives "
                f"{len(names)} names. These must match or the mapping is meaningless.")
        self.ckpt_names = list(names)
        self.robot_names = list(robot_joints)
        self.to_policy, self.from_policy, self.static = _resolve(self.ckpt_names, self.robot_names)

        self.policy = get_policy_class(raw["type"]).from_pretrained(str(p))
        self.policy.to(self.dev).eval()
        self.policy.reset()
        cfg = PreTrainedConfig.from_pretrained(str(p))
        #The checkpoint's OWN statistics. A rebuild from dataset stats can differ, and a normalizer
        #that disagrees with training is indistinguishable from a broken policy at the joint level.
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=str(p))

        self.cams = list(camera_keys)
        self.state_key = state_key
        n = sum(x.numel() for x in self.policy.parameters())
        filled = [self.ckpt_names[i] for i in sorted(self.static)]
        print(f"[groot] {p.parent.name}/{p.name} | {n/1e9:.2f}B params | {self.dev}")
        print(f"[groot] policy {len(self.ckpt_names)}-D <-> robot {len(self.robot_names)}-D, "
              f"mapped by name")
        print(f"[groot] filled from training medians: {filled}")
        print(f"[groot] cameras {self.cams}")

    def connect(self):
        return {"policy": "groot", "device": self.dev, "processors": True,
                "policy_dims": len(self.ckpt_names), "robot_dims": len(self.robot_names)}

    def _expand(self, robot_state):
        """robot 16-D -> policy 22-D, by name, with the static dims filled."""
        s = np.asarray(robot_state, np.float32).reshape(-1)
        if len(s) != len(self.robot_names):
            raise ValueError(f"expected {len(self.robot_names)} robot dims, got {len(s)}")
        out = np.empty(len(self.ckpt_names), np.float32)
        for i, r in enumerate(self.to_policy):
            out[i] = self.static[i] if r < 0 else s[r]
        return out

    def _shrink(self, policy_chunk):
        """policy (H,22) -> robot (H,16), by name."""
        return np.asarray(policy_chunk, np.float32)[:, self.from_policy]

    @torch.no_grad()
    def predict_chunk(self, images, state, instruction):
        obs = {self.state_key: torch.from_numpy(self._expand(state))}
        for key, im in zip(self.cams, images):
            a = np.asarray(im)
            t = torch.from_numpy(a).permute(2, 0, 1).float()
            obs[key] = t / 255.0 if float(t.max()) > 1.5 else t
        obs["task"] = str(instruction)

        batch = self.pre(obs)
        if hasattr(self.policy, "predict_action_chunk"):
            chunk = self.policy.predict_action_chunk(batch)
        else:
            chunk = self.policy.select_action(batch)
        c = chunk if chunk.ndim == 3 else chunk.unsqueeze(0)
        #Unnormalize one waypoint at a time: the postprocessor is written for a single action.
        out = torch.stack([self.post(c[:, i])[0].float().cpu() for i in range(c.shape[1])])
        return self._shrink(out.numpy())


def self_test():
    """Verify the mapping without a checkpoint, a GPU or a robot."""
    print("groot_policy mapping self-test\n")
    ckpt = ([f"arm_l_joint{i}" for i in range(1, 8)] + ["gripper_l_joint1"]
            + [f"arm_r_joint{i}" for i in range(1, 8)] + ["gripper_r_joint1"]
            + ["head_joint1", "head_joint2", "lift_joint", "linear_x", "linear_y", "angular_z"])
    robot = ([f"arm_l_joint{i}" for i in range(1, 8)]
             + [f"arm_r_joint{i}" for i in range(1, 8)]
             + ["gripper_l_joint1", "gripper_r_joint1"])
    to_p, from_p, static = _resolve(ckpt, robot)
    fails = []

    def check(c, m):
        print(f"  [{'ok ' if c else 'FAIL'}] {m}")
        if not c:
            fails.append(m)

    check(len(ckpt) == 22 and len(robot) == 16, "22-D checkpoint, 16-D robot")
    check(sorted(static) == [16, 17, 18, 19, 20, 21], "head/lift/base are the filled dims")

    #a robot state whose value encodes its own index, so any permutation error is visible
    s = np.arange(16, dtype=np.float32)
    exp = np.empty(22, np.float32)
    for i, r in enumerate(to_p):
        exp[i] = static[i] if r < 0 else s[r]
    #the two that would silently swap if indices were hardcoded
    check(exp[7] == s[robot.index("gripper_l_joint1")], "policy dim 7 gets gripper_l, not arm_r1")
    check(exp[8] == s[robot.index("arm_r_joint1")], "policy dim 8 gets arm_r_joint1")
    check(exp[15] == s[robot.index("gripper_r_joint1")], "policy dim 15 gets gripper_r")
    check(abs(exp[18] - STATIC_FILL["lift_joint"]) < 1e-6,
          f"lift filled with the training median {STATIC_FILL['lift_joint']}, not zero")

    #round trip: expand then shrink must be the identity on the robot's own dims
    back = exp[from_p]
    check(np.array_equal(back, s), "expand -> shrink is the identity on all 16 robot dims")

    #a chunk round trip, since that is what actually reaches the arm
    H = 30
    chunk = np.tile(np.arange(22, dtype=np.float32), (H, 1))
    got = chunk[:, from_p]
    want = np.tile(np.array([ckpt.index(n) for n in robot], np.float32), (H, 1))
    check(np.array_equal(got, want), f"({H},22) chunk -> ({H},16) preserves every joint by name")

    #a checkpoint missing a joint the robot commands must refuse, not silently truncate
    try:
        _resolve([n for n in ckpt if n != "gripper_r_joint1"], robot)
        check(False, "a checkpoint missing gripper_r must raise")
    except KeyError:
        check(True, "a checkpoint missing a commanded joint raises instead of guessing")

    #an unknown extra dim with no fill value must refuse too
    try:
        _resolve(ckpt + ["mystery_joint"], robot)
        check(False, "an unfillable extra dim must raise")
    except KeyError:
        check(True, "an extra dim with no STATIC_FILL entry raises")

    print(f"\n{'all mapping checks passed' if not fails else 'FAILED: ' + str(fails)}")
    print("no checkpoint was loaded, no GPU used, no robot contacted.")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(self_test())
