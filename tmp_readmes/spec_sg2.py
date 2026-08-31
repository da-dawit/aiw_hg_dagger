"""The AI Worker FFW-SG2 robot interface, as observed in Dawit's own recordings.

NOT taken from `lerobot_robot_ros2_zenoh/config_ffw_bg2.py`. That file is for the BG2; this robot
is an SG2 and the two are not interchangeable. Everything below was read out of the rosbags the
demonstrations were recorded from, by `IGEN_ACT/scripts/convert_aiworker_to_lerobot.py`, so it
matches the data the policy was actually trained on rather than a config for a sibling robot.

TWO ORDERINGS, AND THEY ARE DIFFERENT. This is the trap that has cost this project twice.

  MODEL order      arm_l 1-7, arm_r 1-7, gripper_l, gripper_r      -- both grippers LAST
                   What the policy emits and what the dataset stores. Matches
                   `aiworker.embodiment.get_profile("arms16").joint_names` and RoboTwin's released
                   action head, which is why the checkpoint's weights land on the right columns.

  CONTROLLER order arm_l 1-7, gripper_l   /   arm_r 1-7, gripper_r -- per-arm, gripper INTERLEAVED
                   What each arm's JointTrajectoryController expects. The leader publishes exactly
                   this during teleop.

`joint_trajectory_topic` is a DICT here for that reason: the stock ROS2Zenoh config supports
mapping each joint name to its own topic, so the split happens by name and never by position.
Indexing positionally is how you command the wrong joint while every array shape still matches.
"""

from __future__ import annotations

#Model order -- what the policy emits. Grippers last.
MODEL_JOINTS = [
    "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
    "arm_l_joint5", "arm_l_joint6", "arm_l_joint7",
    "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
    "arm_r_joint5", "arm_r_joint6", "arm_r_joint7",
    "gripper_l_joint1", "gripper_r_joint1",
]
ARM_DIMS = 14 #indices 0..13 are arm joints; 14,15 are grippers

#The same topics the leader publishes to during teleop, so the follower tracks a policy chunk
#exactly as it tracked a human. Verified in the recordings.
#WE PUBLISH TO /policy/, NOT /leader/. Changed 2026-08-20 when the robot side added arbitration.
#
#`cyclo_teleoperation` publishes to the /leader/ topics CONTINUOUSLY at 100 Hz -- even with nobody
#touching the leader -- so our 21-30 Hz was being lost among its messages, and the single-writer
#guard was right to refuse. `arm_freeze_gate` now selects between /leader/ and /policy/ by mode
#and relays the winner to /gated/arm_{l,r}_controller/joint_trajectory, which is what the follower
#controllers actually subscribe to. Nothing else publishes to /policy/, so we are the sole writer.
CMD_LEFT = "/policy/joint_trajectory_command_broadcaster_left/joint_trajectory"
CMD_RIGHT = "/policy/joint_trajectory_command_broadcaster_right/joint_trajectory"
#The gate's status, latched TRANSIENT_LOCAL depth 1 and republished at 1 Hz:
#    "mode=POLICY left=free right=free"   mode is POLICY|HUMAN; freeze overrides and restores it
FREEZE_STATUS_TOPIC = "/arm_freeze/status"
STATE_TOPIC = "/joint_states"

#Per-joint topic map. Built by NAME so a reordering upstream cannot silently mis-route a joint.
TRAJECTORY_TOPICS = {
    **{j: CMD_LEFT for j in MODEL_JOINTS if j.startswith("arm_l")},
    "gripper_l_joint1": CMD_LEFT,
    **{j: CMD_RIGHT for j in MODEL_JOINTS if j.startswith("arm_r")},
    "gripper_r_joint1": CMD_RIGHT,
}

#Contract camera order. The policy concatenates views POSITIONALLY and applies a per-view
#embedding, so swapping a pair is a silent accuracy loss rather than an error.
#
#Topic names follow the pattern documented in ROS2CameraConfig's own examples. The WIDTH/HEIGHT
#are the resolutions actually present in the recorded dataset (meta/info.json), so the plugin
#delivers frames the same size the policy was trained on -- note the wrists are PORTRAIT
#(240x424), which is easy to transpose by accident.
CAMERAS = {
    "cam_left_head": dict(topic="/zed/zed_node/left/image_rect_color/compressed",
                          width=672, height=376),      #ZED left eye = the scene view
    "cam_left_wrist": dict(topic="/camera_left/camera_left/color/image_rect_raw",
                           width=240, height=424),
    "cam_right_wrist": dict(topic="/camera_right/camera_right/color/image_rect_raw",
                            width=240, height=424),
}
CAMERA_ORDER = ["cam_left_head", "cam_left_wrist", "cam_right_wrist"]

FPS = 30.0
CHUNK_SIZE = 50
#Of a 50-step plan, before re-planning. Lower is closer to the old re-plan-every-tick behaviour:
#more reactive to fresh observations, at the cost of a re-plan seam more often. It also decides how
#many ensemble members actually overlap the executed span, since consecutive plans are offset by
#exactly this many waypoints:
#THERE IS A HARD FLOOR AT 25, and it is measured, not a matter of taste. Cutting the plan at N
#waypoints and asking how far each arm actually moves, on live frames:
#
#    N     12     16     20     25     30     35
#    L   0.071  0.071  0.071  0.071  0.073  0.077     <- flat: this is the ramp-in
#    R   0.049  0.052  0.068  0.117  0.214  0.334
#    R/L  0.69   0.73   0.95   1.65   2.95   4.36     <- demos are ~2.9
#
#The left arm's contribution is constant wherever you cut -- below 25 the ramp-in is ALL you get,
#the plan ends about where it started, and the next plan re-enters the same ramp-in. The arm dwells
#at its start pose. Every value under 25 reproduces that, so this is a floor and not a preference.
#30 puts R/L at 2.95 against the demonstrations' 2.90, which is why it is the default: it is the
#shortest plan that reproduces the demonstrated two-arm sequence.
EXECUTE_STEPS = 25

#Conservative defaults for a checkpoint that is still training. rad/s and rad/s^2 per joint.
#The gripper is deliberately NOT rate limited (see trajectory/clamp): throttling a near-binary
#actuator to arm acceleration means the hand never fully closes before the next re-plan.
#Measured against the demonstrations rather than guessed: over 40 episodes the humans move the arms
#at 0.144 rad/s in free space and slow to 0.071 rad/s at the grasp. The old 0.5 was 7x the speed
#they actually grasp at, and because a far --offset keeps the velocity clip saturated the arm ran
#at that ceiling from start to finish instead of decelerating into contact.
#A SAFETY CEILING, NOT A SHAPER. Under receding horizon the plan's own waypoint spacing carries the
#demonstrated speed profile -- including the slow-down into contact -- so clamping below that just
#fights the trajectory. Measured over all 90 episodes the arms run at mean 0.260 rad/s, p95 0.644,
#p99 0.874 (the 9.99 outlier is a single-frame glitch). The previous 0.2 was taken from a window
#around grasps only and then applied everywhere, so it sat BELOW the mean and clipped most of the
#run. 0.6 sits just under p95: it never binds in normal motion and still catches a wild jump.
MAX_VEL = 0.6
#MAX_ACC was 2.0 before 2026-08-20 and REMAINS 2.0 -- measured, not inherited. On held-out episode
#32 with checkpoint 002400 the acceleration clamp turned out to be the dominant smoothing lever,
#far more than ensembling or Savitzky-Golay:
#
#    max_acc 6.0   jerk 1.74x human   tracking 0.0612
#    max_acc 3.0   jerk 0.86x human   tracking 0.0654
#    max_acc 2.0   jerk 0.55x human   tracking 0.0648   <- this value, best on BOTH axes
#
#Why it matters so much: GR00T's raw chunk commands 5.07 rad/s^2 mean acceleration against the
#human's 0.81 (p95 2.76), so a 6.0 clamp barely engages and the policy's own noise reaches the arm.
#At 2.0 the limiter binds ~64% of the time, which is the point -- it is what pulls commanded
#acceleration back into the range the demonstrations actually contain.
MAX_ACC = 2.0


def robot_config(router_ip: str = "127.0.0.1", domain_id: int = 30, cameras: bool = True):
    """Build a ROS2ZenohConfig for this robot. Import is local so the spec stays importable
    without lerobot installed (e.g. for a dry-run that only prints the plan)."""
    from lerobot_robot_ros2_zenoh.config_ros2_zenoh import ROS2ZenohConfig
    from lerobot_robot_ros2_zenoh.config_ros2_camera import ROS2CameraConfig

    cams = ({name: ROS2CameraConfig(fps=int(FPS), domain_id=domain_id, router_ip=router_ip,
                                    **spec_)
             for name, spec_ in CAMERAS.items()} if cameras else {})
    return ROS2ZenohConfig(
        id="ffw_sg2",
        joint_names=list(MODEL_JOINTS),
        joint_states_topic=STATE_TOPIC,
        joint_trajectory_topic=dict(TRAJECTORY_TOPICS),
        domain_id=domain_id,
        router_ip=router_ip,
        cameras=cams,
    )
