# g1_control

ROS 2 package for the two-process G1 RENet deployment controller:

- `renet_policy`: Python, snapshot-triggered at 50 Hz. Builds the exact
  `10 x 78` proprioceptive history, evaluates both RENet branches, selects VP
  or LP, blends branch changes in action space, and publishes one atomic
  `PolicyJointTarget`.
- `lowcmd_bridge`: C++, 500 Hz. Maps the frozen 23 policy joints to Unitree
  motor indices, applies the training PD gains, computes Unitree CRC, and
  publishes `LowCmd` only after explicit remote arming.
- `remote_inspector`: read-only remote logger. It creates no publisher and is
  the required first real-robot check for button bits, axes, signs, and dead
  zone.
- `voice_announcer`: subscribes to accepted control events and uses the native
  G1 `/api/voice/request` TTS service. It announces `策略运行`, `视觉模式`,
  `雷达模式`, and `策略退出`.

The shipped configuration has `dry_run: true`,
`remote_buttons_enabled: false`, `remote_commands_enabled: false`, and
`remote_mode_switch_enabled: false`. Therefore building or starting these
nodes does not by itself arm motor output.

## Agreed deployment contract

The implementation follows these deployment decisions:

- Use a C++ 500 Hz LowCmd bridge. That bridge downsamples `/lowstate` into an
  atomic 50 Hz `ProprioPolicyInput`; each snapshot triggers one Python policy
  inference, so Python never subscribes to the 1 kHz topic directly.
- Start in VP. Manual switching is allowed only after reading and logging the
  real G1 remote controls; automatic selection is deferred to the selector.
- At the current development stage, only a human SELECT press enters the
  damping state. Sensor age, validity, NaN/Inf, policy timeout, and inactive
  inputs remain observable diagnostics and do not trigger a fallback, a mode
  switch, or damping automatically.
- All 23 controlled motors use `motor_cmd[i].mode=1` during normal PD control
  and controlled damping. Damping keeps the motors enabled with zero target
  velocity/torque, zero `kp`, and a damping `kd`; `mode=0` is not the e-stop.
- Use the training joint PD gains initially. Recheck them on the suspended
  robot before any supported or walking test; this is a mandatory bring-up
  item, not a tuning suggestion.
- The runtime computes both action candidates every cycle. The TorchScript
  backend returns both in one call; the TensorRT backend evaluates the frozen
  masked ONNX graph twice (`mask=1` for LP, `mask=0` for VP). Runtime estimator
  value 0 selects VP and value 1 selects LP. This is independent of Unitree
  motor `mode`, `mode_pr`, and `mode_machine` fields.
- A manual VP/LP transition computes both candidate actions and blends the
  final joint targets over the agreed transition interval. It does not
  interpolate depth images, elevation maps, masks, or latent vectors.

## Stamped sensor inputs

The controller will consume the atomic custom messages from `g1_policy_msgs`:

- `/g1/depth/policy_input` (`DepthPolicyInput`)
- `/g1/elevation/policy_input` (`ElevationPolicyInput`)
- `/g1/proprio/policy_input` (`ProprioPolicyInput`, C++ bridge output)

The older array and diagnostic topics remain available for inspection, but
must not be joined in the controller as if separately received messages were
one coherent sample.

The Python process publishes `/g1/policy/joint_target`
(`PolicyJointTarget`). The C++ process is the only component that maps that
message to Unitree motor slots.

## Frozen 23-DOF mapping

The policy order and Unitree motor index are:

| Policy | Joint | Motor |
|---:|---|---:|
| 0 | left hip pitch | 0 |
| 1 | right hip pitch | 6 |
| 2 | waist yaw | 12 |
| 3 | left hip roll | 1 |
| 4 | right hip roll | 7 |
| 5 | left shoulder pitch | 15 |
| 6 | right shoulder pitch | 22 |
| 7 | left hip yaw | 2 |
| 8 | right hip yaw | 8 |
| 9 | left shoulder roll | 16 |
| 10 | right shoulder roll | 23 |
| 11 | left knee | 3 |
| 12 | right knee | 9 |
| 13 | left shoulder yaw | 17 |
| 14 | right shoulder yaw | 24 |
| 15 | left ankle pitch | 4 |
| 16 | right ankle pitch | 10 |
| 17 | left elbow | 18 |
| 18 | right elbow | 25 |
| 19 | left ankle roll | 5 |
| 20 | right ankle roll | 11 |
| 21 | left wrist roll | 19 |
| 22 | right wrist roll | 26 |

The bridge leaves nonexistent 23-DOF slots disabled. All 23 entries in this
table use motor `mode=1` in position PD and in SELECT damping.

## Explicit startup states

After remote calibration and deliberate enabling, the real bridge state
sequence is
`WAITING --START--> ACQUIRING --verified MotionSwitcher release--> STANDING
--3 s--> HOLDING --A--> POLICY`. The bridge publishes no `LowCmd` while it is
WAITING or ACQUIRING. On the first 500 Hz cycle after verified release it
seeds every commanded joint from the latest measured position, then begins
the three-second interpolation to the training default posture. SELECT from
an armed state enters latched `DAMPING`; restarting the bridge is required to
re-arm. No sensor age, validity, NaN/Inf, policy timeout, or inactive branch
automatically changes the estimator or enters damping at this stage.

Accepted events are published on `/g1/control/event`. `A` emits
`policy_running` only when the bridge really enters POLICY; `R1` requests VP
and emits `visual_mode`; `L1` requests LP and emits `lidar_mode`; SELECT emits
`policy_exit` when it latches DAMPING. START deliberately has no speech event
because it only begins the stand interpolation.

## Full dry-run launch

On the robot, source all four workspaces before using the integrated launch:

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/cyclonedds_ws/install/setup.bash
source /home/unitree/Livox-SDK2-ros2/install/setup.bash
source /home/unitree/fastlio2_ws/install/setup.bash
source /home/unitree/cjh/install/setup.bash
ros2 launch g1_control full_stack_dry_run.launch.py
```

This starts D435i depth, the depth preprocessor, MID-360, FAST-LIO, both
validated static transforms, persistent elevation map, TensorRT policy,
LowCmd bridge state machine, and native TTS announcer. The transforms are:

- `torso_link -> livox_frame`: translation
  `[0.0002835, 0.00003, 0.428434]`, quaternion xyzw
  `[0.999673352, 0, -0.025557564, 0]`;
- `g1_world -> camera_init`: zero translation, quaternion xyzw
  `[0.9996734, 0, -0.0255576, 0]`.

The first transform is the previously validated head-mounted MID-360
extrinsic from the lidar bring-up, not a newly estimated value. It has passed
direction and approximate-position checks but is not a precision calibration.
The second transform makes the FAST-LIO world frame Z-up for the elevation
map. The launch loads `control_dry_run.yaml`, which
uses exact Foxy node names and enforces all of the following:

- `dry_run=true`;
- joystick velocity commands remain disabled;
- remote state/mode buttons are enabled for validation;
- the bridge endpoint is redirected to `/g1/control/dry_run_lowcmd` rather
  than the real `/lowcmd` topic;
- the dry-run bridge executes its state machine but publishes no `LowCmd`
  messages.

Do not replace `control_dry_run.yaml` with `control.yaml` in this launch. ROS 2
Foxy gives exact node entries in the first YAML file precedence over later
`/**` launch overrides, which can silently defeat the topic redirection.

## First robot-side check

Build and run only the read-only inspector first:

```bash
cd /home/unitree/cjh
colcon build --packages-select g1_policy_msgs g1_control --symlink-install
source install/setup.bash
ros2 run g1_control remote_inspector --ros-args \
  --params-file /home/unitree/cjh/src/g1_control/config/control.yaml
```

Record the neutral values and move one stick direction or press one button at
a time. Do not turn off `dry_run` until those logs have been reviewed.

## Frozen policy artifact

The deployment policy is
`logs/g1_renet/2026-09-04_14-11-25/exported/policy.pt`. Its 87 state-dict
tensors were compared directly with all checkpoints in that run: it is an
exact match for `model_60000.pt` (maximum absolute tensor difference 0), not
`model_50000.pt` or `model_54000.pt`. Any later dual-action export must be
created from `model_60000.pt` and must pass branch-by-branch parity against
this frozen `policy.pt` before use.

Artifact SHA-256 values at verification time:

- `policy.pt`: `7a34a61b4b6f391f628b46b4da106d99b72f9034dab5f62386bac7db87a351b1`
- `model_60000.pt`: `d588c113dfa797afb486eef69e15889e4787e60257e66c36d6e64f00976f1566`

The deployment will use an additional dual-action TorchScript artifact; it
must not overwrite the frozen `policy.pt`. For this AME-LP checkpoint the two
branches do not share one encoded proprio-history tensor:

- VP embeds all ten 78-value proprio frames with `proprio_embedding`, then
  processes the depth-conditioned ten-step sequence.
- LP slices the last five raw 78-value proprio frames and encodes them with its
  separate `lp_query_mlp` before attending to the 187 elevation tokens.
- Both actions use the same latest 78-value proprio frame and the same actor
  weights. The dual artifact evaluates each branch once and invokes that actor
  once per branch, returning both 23-value actions from one model call.

The verified artifact is
`logs/g1_renet/2026-09-04_14-11-25/exported/renet_dual_branch.pt` with SHA-256
`c512197e3c5e9951c3f183d5be4317777a29bba33be3535370b02d8282b2fb4e`.
Its output tuple is ordered as LP latent, VP latent, LP velocity, VP velocity,
LP action, VP action. The final two tensors are the two 23-value action
candidates used by the transition logic.

## Jetson TensorRT artifact

The CPU-only PyTorch installation on the robot cannot sustain the required
50 Hz rate. The robot therefore uses the original masked ONNX export with a
platform-local TensorRT engine. Build the numerically verified FP32 engine on
the robot with TF32 explicitly disabled:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/unitree/cjh/models/policy.onnx \
  --saveEngine=/home/unitree/cjh/models/renet_policy_fp32_notf32.engine \
  --noTF32 --workspace=2048 --buildOnly
```

Verified artifact hashes:

- `policy.onnx`:
  `be807a35d87cc780ae861ac874ecbff67eb68aa3988736347d4e873d1202a049`
- `renet_policy_fp32_notf32.engine`:
  `6761e83826331db123fb9a4ef4bbaf8b6cfded928077c21cd6da5713d452118a`

Do not copy this TensorRT engine to another Jetson/TensorRT platform; rebuild
it there from the verified ONNX file. FP16 and default TF32 builds were
rejected because their action error was too large. Across nine LP/VP parity
cases, the no-TF32 engine differed from the dual TorchScript artifact by at
most `7.153e-06`.

## Read-only integrated bring-up result

The full perception and policy chain was run on the robot without starting
`lowcmd_bridge` or publishing any new `LowCmd`. Measured rates were:

| Topic | Rate |
|---|---:|
| `/camera/depth/image_rect_raw` | 30.080 Hz |
| `/g1/depth/policy_input` | 10.090 Hz |
| `/cloud_registered` | 10.084 Hz |
| `/Odometry` | 10.009 Hz |
| `/g1/elevation/policy_input` | 49.606 Hz |
| `/g1/policy/joint_target` | 50.038 Hz |

A separate 500-message policy timing check measured 50.010 Hz, contiguous
sequence numbers, 19.996 ms mean period, 23.622 ms P99 period, and 1.133 ms
mean message age. The full two-branch TensorRT call averaged 6.901 ms in the
periodic logs.

This test was performed while the robot was folded and pitched, not in the
training default posture: hip pitch was about `-2.53 rad`, knee position about
`2.90 rad`, and the largest default-position error was `2.48 rad`. The large
policy actions observed in that state are outside the intended standing
bring-up condition. Before enabling the bridge, repeat the read-only target
test with the robot securely suspended in the training default posture and
review every mapped joint target and limit.

The subsequent standing dry-run exercised the complete remote sequence. The
bridge reached `STANDING`, `HOLDING`, `POLICY`, and latched `DAMPING`; all four
TTS requests were accepted by the robot voice API. The real `/lowcmd` graph
remained at its original two publishers, and independent four-second and
three-second probes received zero messages on the isolated dry-run LowCmd
topic. At the time of the test, the standing pose differed from the training
default by at most `0.121 rad` and the IMU was nearly level.

The dry-run is not authorization to enable actuation. In the latest
zero-command VP dry-run, actions were approximately `[-5.45, 8.32]` and the
unmodified position targets approximately `[-1.58, 2.48] rad`. The next real
trial must therefore remain suspended/closely spotted while takeover and
zero-command behavior are inspected. The real configuration starts in VP and
keeps remote VP/LP switching disabled so this comparison changes only one
branch. LP coverage does not select a mode or trigger a fallback.

## Unmodified policy targets

The deployment path intentionally contains no joint-target protection layer:
the Python node does not apply a soft joint clamp or a target-rate limiter,
and the C++ bridge does not apply a mechanical-position clamp or a per-cycle
slew limiter. The selected/blended policy action is converted directly by
`default_q + 0.25 * action` and mapped to Unitree motor order. The START/A
state gates, three-second initial stand interpolation, manual VP/LP blend, and
SELECT damping are control-state behavior, not target clipping.

## First real-control launch

`first_real_control.launch.py` starts the same perception chain and static
transforms as the full dry-run, but loads `control_first_real.yaml`. After the
zero-command takeover trial succeeded, this file is configured for the first
walking trial:

- initial estimator is VP;
- remote velocity is enabled with a 0.10 stick dead zone;
- full-stick limits are -0.40/0.80 m/s longitudinal, -0.40/0.40 m/s
  lateral, and -1.57/1.57 rad/s yaw; the asymmetric longitudinal mapping
  uses the full negative and positive halves of the stick travel;
- remote VP/LP switching remains disabled;
- only START, A, and SELECT affect the control state;
- real `/lowcmd` output is START-gated and requires a verified
  MotionSwitcher release.

Both integrated launches pass `g1_depth/config/realsense_d435i.yaml` to the
official `realsense2_camera` node. The native spatial and temporal filters use
the SDK's default magnitude/alpha/delta values. Temporal persistency is
explicitly disabled. Decimation stays off so the camera model does not change,
and all hole-filling modes stay off so moving stair edges are not synthesized
from stale neighbouring pixels. `initial_reset` is enabled to recover a D4xx
left in a stale streaming/recovery state by an interrupted prior process.

Policy inference is idle before A. Pressing A now performs a handshake:

1. the C++ bridge discards every cached target and remains in HOLDING;
2. the Python policy clears its 10-frame proprioception history,
   `previous_action`, transition state, and output sequence;
3. the first target is blended from zero action for five 50 Hz frames;
4. only that fresh post-A target can move the bridge into POLICY and trigger
   the `策略运行` announcement.

Start the stack in the first terminal. Merely launching leaves the bridge in
WAITING, does not call MotionSwitcher, and does not publish `LowCmd`:

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/cyclonedds_ws/install/setup.bash
source /home/unitree/Livox-SDK2-ros2/install/setup.bash
source /home/unitree/fastlio2_ws/install/setup.bash
source /home/unitree/cjh/install/setup.bash
ros2 launch g1_control first_real_control.launch.py
```

After all nodes and topic types exist, start the lightweight control recording
in a second terminal. Raw images and registered point clouds are deliberately
omitted to avoid adding heavy I/O load to the first actuation test:

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/cyclonedds_ws/install/setup.bash
source /home/unitree/cjh/install/setup.bash
mkdir -p /home/unitree/cjh/bags
ros2 bag record -o /home/unitree/cjh/bags/first_real_$(date +%Y%m%d_%H%M%S) \
  /lowstate /lowcmd \
  /g1/proprio/policy_input \
  /g1/control/status /g1/control/event \
  /g1/policy/joint_target \
  /g1/depth/policy_input /g1/depth/policy_frame \
  /g1/depth/policy_valid /g1/depth/source_age /g1/depth/valid_ratio \
  /g1/elevation/policy_input /g1/elevation/policy_valid \
  /g1/elevation/valid_ratio /g1/elevation/pose_sync_error \
  /camera/depth/camera_info /Odometry /tf /tf_static
```

Expected operator sequence:

1. Do not press any button until all perception topics are running and
   `/g1/control/status` shows `state: 0`, `dry_run: false`, and
   `motion_service_released: false`.
2. Secure the robot with the suspension/spotters, keep SELECT immediately
   reachable, then press START once. Wait for state 2 (STANDING), then state 3
   (HOLDING). Do not press A during the three-second transition.
3. At HOLDING, inspect posture. Center both sticks, press A once to enter VP
   policy (state 4), then introduce a small forward command gradually instead
   of moving immediately to full stick.
4. Press SELECT immediately if the sound, posture, vibration, or motion is
   abnormal. State 5 is latched damping and announces `策略退出`.
5. Once START has been pressed, never use Ctrl-C as the normal control exit.
   Press SELECT first, verify state 5 (DAMPING), and wait until the robot is
   physically supported/safely settled. Only then stop the launch and rosbag
   with Ctrl-C. A new bridge process is required for another arm attempt.

For the first run, do not press L1/R1 expecting a mode change: logging still
sees the buttons, but `remote_mode_switch_enabled=false`. The selector and LP
switching are intentionally outside this trial.
