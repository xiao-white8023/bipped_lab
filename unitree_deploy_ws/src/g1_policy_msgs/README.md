# g1_policy_msgs

Stamped, fixed-size ROS 2 interfaces between the G1 sensor preprocessors and
the real-robot RENet controller.

- `DepthPolicyInput`: two `18 x 32` normalized VP frames, ordered oldest to
  newest. Its header carries the newest D435i source timestamp.
- `ElevationPolicyInput`: one `11 x 17` LP height grid. Its header carries the
  pose/sampling timestamp, while `source_stamp` identifies the newest LiDAR
  cloud present in the persistent map.
- `ProprioPolicyInput`: one atomic 50 Hz bridge snapshot containing IMU,
  23 policy-order joint positions/velocities, and the 40 remote bytes. Its
  arrival triggers exactly one Python policy inference.
- `PolicyJointTarget`: one atomic 50 Hz controller output containing the
  selected estimator mode, blended normalized action, and 23 position targets.
  Both arrays remain in the frozen Isaac Lab policy joint order; only the
  500 Hz LowCmd bridge is allowed to map them to Unitree motor indices.
- `ControlStatus`: lightweight 50 Hz bridge state and target-age diagnostics.

The fixed array sizes deliberately make the exported policy contract part of
the ROS type. A publisher or controller built for a different policy shape
must use a different interface revision instead of silently reshaping data.
