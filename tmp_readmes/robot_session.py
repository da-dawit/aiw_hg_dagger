"""Connect to the AI Worker and clear every safety gate, once, for any caller.

Author: Dawit Chun

WHY THIS EXISTS. infer.py's connect path is not boilerplate -- it is a sequence of guards, each
added after a specific failure on this robot:

    the zenoh override      a stale ZENOH_CONFIG_OVERRIDE from the omy robot silently wins over the
                            config object, and the symptom is a connection failure to an address
                            you never passed
    q_home from the MEDIAN  episode 0 is an outlier by up to 0.33 rad; homing to it drove the elbows
                            a third of a radian from where every demonstration begins
    the start-pose guard    a chunk is 100 waypoints predicted for the pose held when it was
                            observed; planning from somewhere else executes a plan for a situation
                            that does not exist
    the stillness check     the start-pose guard measures WHERE the arm is and cannot see that it is
                            MOVING; those are different failures

online_rl.py needs all of them. Copying them would produce a second version that drifts from the
one that has actually been run against the hardware, and a safety guard that differs between two
files is worse than one that exists in neither. So both import this.

Nothing here commands the robot. `connect()` reads, checks, and refuses; only the caller publishes.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from pathlib import Path
import numpy as np


@dataclass
class Session:
    robot: object
    observe: object                 # () -> (q 16-D, [images])
    q_home: np.ndarray
    spec: object
    joint_to_topic: dict = field(default_factory=dict)

    def sample(self):
        return self.observe()[0]


def set_zenoh_endpoint(router_ip):
    """Must run BEFORE anything imports zenoh; a shell-set override otherwise wins."""
    endpoint = f"tcp/{router_ip}:7447"
    prev = os.environ.get("ZENOH_CONFIG_OVERRIDE", "")
    if prev and endpoint not in prev:
        print(f"[zenoh] overriding stale ZENOH_CONFIG_OVERRIDE ({prev[:60]}...)")
    os.environ["ZENOH_CONFIG_OVERRIDE"] = (
        f'transport/shared_memory/enabled=true;mode="client";'
        f'connect/endpoints=["{endpoint}"]')
    print(f"[zenoh] endpoint {endpoint}")


def demo_start_pose(data_root):
    """The pose the demonstrations begin from: the MEDIAN first frame, never episode 0.

    Episode 0 sits up to 0.3275 rad from the median and its worst joints are the elbows. Homing to
    it was measured as a ~0.5 rad excursion in arm_l_joint1 that took 100 steps to decay. The median
    matches the robot's own trajectory-executor `home` parameter to 0.003 rad.
    """
    sys.path.insert(0, os.environ.get("LEROBOT_SRC", "/home/robotis/robot_omy/lerobot/src"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id="local", root=data_root)
    st = np.asarray(ds.hf_dataset["observation.state"], dtype=np.float64)
    ep = np.asarray(ds.hf_dataset["episode_index"])
    return np.median(np.stack([st[np.where(ep == e)[0][0]] for e in np.unique(ep)]), axis=0)



def check_camera_shapes(imgs, data_root, cam_order, wrist_rot, on_fail=None):
    """Refuse to run if a frame is not the shape meta/info.json says the policy was trained on.

    Reports the offending camera by name and, when a different --wrist-rot would have matched,
    names it -- the failure is almost always a transpose, and guessing at it costs a rollout.
    """
    import json as _json
    info = _json.loads((Path(data_root) / "meta" / "info.json").read_text())
    feats = info["features"]
    want = {}
    for c in cam_order:
        f = feats.get(f"observation.images.rgb.{c}")
        if f and f.get("shape"):
            c_, h, w = f["shape"]               #info.json is (channels, height, width)
            want[c] = (h, w, c_)
    if not want:
        print(f"[cams] {data_root}/meta/info.json records no image shapes -- cannot verify")
        return

    bad = [(c, tuple(im.shape), want[c]) for c, im in zip(cam_order, imgs)
           if c in want and tuple(im.shape) != want[c]]
    if not bad:
        print(f"[cams] {len(want)} frames match the training shapes "
              f"{[want[c] for c in cam_order if c in want]} at --wrist-rot {wrist_rot}")
        return

    lines = [f"camera frames do not match what the policy was trained on "
             f"(--wrist-rot {wrist_rot}):"]
    for c, got, exp in bad:
        hint = "  <- transposed" if (got[1], got[0], got[2]) == exp else ""
        lines.append(f"    {c:<20} got {got}   expected {exp}{hint}")
    if all((g[1], g[0], g[2]) == e for _, g, e in bad):
        alt = 90 if wrist_rot == 0 else 0
        lines.append(f"  every mismatch is a transpose -- try --wrist-rot {alt}")
    lines.append(f"  expected shapes come from {data_root}/meta/info.json")
    if on_fail:
        on_fail()
    raise SystemExit("\n".join(lines))

def connect(router_ip, spec, data_root=None, domain_id=30, cameras=True, wrist_rot=0,
            start_tol=0.30, rest_check=True, home_first=False, home_vel=0.10, live=False):
    """Connect, then clear every gate in order. Returns a Session, or raises SystemExit.

    `live` only gates the OPTIONAL homing move; this function publishes nothing else ever.
    """
    import infer  #the guards live there and are imported, never reimplemented

    set_zenoh_endpoint(router_ip)
    import cv2
    from lerobot_robot_ros2_zenoh.ros2_zenoh import ROS2Zenoh
    from safety import check_observation

    robot = ROS2Zenoh(spec.robot_config(router_ip=router_ip, domain_id=domain_id, cameras=cameras))
    print(f"[robot] connecting to zenoh {router_ip}:7447 domain {domain_id} ...")
    robot.connect()
    print("[robot] connected")
    j2t = dict(getattr(robot, "_joint_to_topic", {}) or {})

    _ROT = {0: None, 90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_CLOCKWISE}

    def observe():
        obs = robot.get_observation()
        cams = spec.CAMERA_ORDER if cameras else []
        check_observation(robot, obs, cams)
        missing = [j for j in spec.MODEL_JOINTS if f"{j}.pos" not in obs]
        if missing:
            raise KeyError(f"observation is missing {missing}")
        q = np.array([float(obs[f"{j}.pos"]) for j in spec.MODEL_JOINTS], dtype=np.float64)
        #NATIVE frames. Squashing to 224 cost a measured 8.9x in action accuracy and fails silently.
        imgs = [np.asarray(obs[c]) for c in cams]
        r = _ROT.get(wrist_rot)
        if r is not None:
            imgs = [im if i == 0 else cv2.rotate(im, r) for i, im in enumerate(imgs)]
        return q, imgs

    #GATE 0: the CAMERAS must hand the policy the shape it was trained on.
    #
    #Nothing downstream can detect this. A transposed wrist view is a valid array of the right
    #dtype; the policy consumes it, commands almost nothing (measured: 0.034 rad as-is against
    #0.169 rotated, demos 0.232), and the arm looks badly trained rather than badly fed. It also
    #silently poisons any frames recorded for DAgger, which are retrained ALONGSIDE the original
    #episodes in whatever orientation they were stored.
    #
    #`wrist_rot` defaults to 0 here while infer.py defaults to 90, so a caller that simply omits it
    #gets the broken orientation -- which is exactly what online_rl.py did. Checking the resulting
    #shape against meta/info.json catches that no matter which caller, default, or camera remount
    #causes it.
    if cameras and data_root:
        check_camera_shapes(observe()[1], data_root, spec.CAMERA_ORDER,
                            wrist_rot, on_fail=robot.disconnect)

    #GATE 1: the arm must be holding still. Thresholds derived from the demonstrations, where a
    #resting arm and a working one differ by 1443x.
    if rest_check:
        infer.require_at_rest(lambda: observe()[0], spec.ARM_DIMS, spec.MODEL_JOINTS,
                              label="before policy load", on_fail=robot.disconnect)

    q_home = None
    if data_root:
        q_home = demo_start_pose(data_root)
        #MAP 22 -> 16 BY NAME. The dataset stores 22 dims (grippers interleaved per arm, plus
        #head, lift and the base velocities); the robot is commanded in the 16-D arms16 layout
        #(both grippers LAST). Comparing them raw raises "operands could not be broadcast
        #(16,) (22,)", and -- far worse -- slicing the first 16 positionally would silently
        #compare arm_r_joint1 against gripper_l_joint1. infer.py already does this; robot_session
        #did not, so online_rl.py could never reach its first observation.
        if len(q_home) != len(spec.MODEL_JOINTS):
            import json as _json
            _info = _json.loads((Path(data_root) / "meta" / "info.json").read_text())
            _names = (_info["features"].get("observation.state") or {}).get("names")
            if not _names:
                robot.disconnect()
                raise SystemExit(
                    f"{data_root} is {len(q_home)}-D but the robot commands "
                    f"{len(spec.MODEL_JOINTS)}, and its meta/info.json records no dimension "
                    f"names. Refusing to guess the layout.")
            _missing = [j for j in spec.MODEL_JOINTS if j not in _names]
            if _missing:
                robot.disconnect()
                raise SystemExit(f"{data_root} has no dimension(s) named {_missing}, which the "
                                 f"robot commands. Wrong dataset for this robot?")
            q_home = q_home[[_names.index(j) for j in spec.MODEL_JOINTS]]
            print(f"[home] dataset is {len(_names)}-D; mapped to the robot's "
                  f"{len(spec.MODEL_JOINTS)}-D layout by name")
        print(f"[home] demo start pose (median over episodes): {np.round(q_home, 3).tolist()}")
        q_now, _ = observe()
        err = np.abs(q_now - q_home)[:spec.ARM_DIMS]
        print(f"[start] max ARM deviation {err.max():.3f} rad "
              f"({spec.MODEL_JOINTS[int(err.argmax())]})")

        if err.max() > start_tol and home_first:
            if not live:
                robot.disconnect()
                raise SystemExit("--home-first moves the robot, so it requires --live.")
            print(f"[home-first] {err.max():.3f} rad out of tolerance -- driving to the demo start "
                  f"pose at {home_vel} rad/s")
            infer.go_home(robot, q_home, q_now, vel=home_vel, fps=spec.FPS)
            q_now, _ = observe()
            err = np.abs(q_now - q_home)[:spec.ARM_DIMS]
            print(f"[home-first] arrived: max ARM deviation now {err.max():.3f} rad")
            if rest_check:
                infer.require_at_rest(lambda: observe()[0], spec.ARM_DIMS, spec.MODEL_JOINTS,
                                      label="after homing", on_fail=robot.disconnect)

        #GATE 2: the policy has only seen this task begin near that pose.
        if err.max() > start_tol:
            robot.disconnect()
            raise SystemExit(
                f"REFUSING TO RUN: {err.max():.3f} rad from the demonstrated start pose "
                f"(tolerance {start_tol}).\n"
                f"  Move the arms there, or raise --start-tol if you are deliberately testing "
                f"generalization.")

    return Session(robot=robot, observe=observe, q_home=q_home, spec=spec, joint_to_topic=j2t)


def self_test():
    """Verify what can be checked without hardware: the zenoh override and the module's shape."""
    print("robot_session self-test\n")
    fails = []

    def check(c, m):
        print(f"  [{'ok ' if c else 'FAIL'}] {m}")
        if not c:
            fails.append(m)

    os.environ["ZENOH_CONFIG_OVERRIDE"] = 'connect/endpoints=["tcp/192.168.0.10:7447"]'
    set_zenoh_endpoint("10.42.0.22")
    got = os.environ["ZENOH_CONFIG_OVERRIDE"]
    check("10.42.0.22:7447" in got, "a stale override is replaced by --router-ip")
    check("192.168.0.10" not in got, "the stale omy endpoint is gone")
    check('mode="client"' in got, "client mode set")

    import inspect
    src = inspect.getsource(connect)
    check("require_at_rest" in src, "the stillness gate is present")
    check("REFUSING TO RUN" in src, "the start-pose gate is present")
    check("infer.go_home" in src, "homing uses infer.py's implementation, not a copy")
    check(src.count("send_trajectory") == 0,
          "this module never publishes a trajectory -- only the caller does")

    import infer
    for fn in ("require_at_rest", "clamp_chunk", "go_home", "arm_motion"):
        check(hasattr(infer, fn), f"infer.{fn} is importable, so it can be shared not copied")

    print(f"\n{'all checks passed' if not fails else 'FAILED: ' + str(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(self_test())
