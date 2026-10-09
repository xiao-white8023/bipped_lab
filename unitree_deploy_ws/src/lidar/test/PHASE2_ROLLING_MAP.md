# G1 LiDAR phase 2: motion-compensated rolling map

## Status

The mapper now keeps a configurable source-timestamp history in `g1_world`
and reprojects it into the current gravity-aligned `capture_torso_yaw` frame.
Policy control must remain disabled until the map is visually checked on the
robot and stale-observation behaviour is accepted.

## Processing path

1. Synchronize `/cloud_registered` and `/Odometry` by source timestamp.
2. Transform the current registered scan through `g1_world` into the current
   `capture_torso_yaw` frame.
3. Remove only the fixed head-return box, before storing any point in history.
4. Keep the highest actual point in each 0.1 m cell for the current packet.
5. Store those representative points in `g1_world` for up to 10 seconds.
6. Reproject all retained points into the current `capture_torso_yaw` frame.
7. Crop and rasterize the current 3.2 x 3.2 m internal map, then publish the
   unchanged row-major 11 x 17 policy contract.

The history is additionally bounded to 200 packets.  A ROS timestamp rewind
(for example looped bag playback) clears the complete history.

## Head-return filter

The original bag contains 1,562 policy-cloud points with raw capture-frame
`z >= 0.15 m`.  Of those, 1,505 are concentrated at grid centres
`x = 0.0/0.1 m`, `y = -0.2...0.2 m`; the remaining points around
`x = 0.7/0.8 m`, `y = 0.3...0.5 m` are external surfaces and must not be
removed globally.

The phase-2 filter therefore uses this fixed box in `capture_torso_yaw`:

```text
x: [-0.05, 0.15] m
y: [-0.25, 0.25] m
z: [ 0.15, 0.50] m
```

The filter is applied before world-frame history insertion.  The final replay
reported zero remaining policy-cloud points inside this box.

## Offline replay result

Input bag:

```text
g1_lidar_phase1_20260928_211955_0.db3
154.56 seconds, 1546 registered-cloud/odometry pairs
```

Final phase-2 report:

```text
unitree_deploy_ws/bags/phase2_results/bag_20260928_head_filtered/
```

| Metric | Single-frame baseline | Phase-2 history |
|---|---:|---:|
| Mean valid cells/frame | 4.09 / 187 | 48.11 / 187 |
| Mean valid ratio | 2.19% | 25.72% |
| Median valid cells/frame | 2 | 18 |
| P95 valid cells/frame | 11 | 187 |
| Maximum valid cells/frame | 19 | 187 |
| Cells never valid | 143 / 187 | 0 / 187 |
| Exact head-box candidates | 1505 | 0 |
| Array/encoding errors | 0 | 0 |
| Cloud/odometry source-time error | 0 ms | 0 ms |

The formerly empty front columns are now observed during motion:

```text
x=+0.2 m: 20.96%
x=+0.3 m: 21.30%
x=+0.4 m: 21.89%
x=+0.5 m: 27.56%
x=+0.6 m: 36.51%
```

The 10-second history cannot create terrain that was never visible.  There
are still 475 zero-valid frames, mainly in stationary intervals before a
ground patch has entered the LiDAR field of view.  Conversely, moving
intervals can reach 100% policy-grid coverage.

## New diagnostics

- `/g1/elevation/policy_age` publishes the freshest source-observation age for
  every policy cell in seconds; invalid cells contain `-1.0`.
- `/g1/elevation/history_span` publishes the current retained source-time
  span.
- Point-cloud debug `intensity` is a confidence value that reaches 1 after
  `confidence_observations` contributions.

Ten seconds is intentionally conservative for coverage testing.  Some cells
in the replay reach 8-10 seconds of age.  Such cells are more sensitive to
FAST-LIO drift and can retain moved objects.  Do not interpret full coverage
as fresh coverage; inspect `policy_age` during the next robot experiment.

## Build and run

```bash
cd /home/unitree/cjh
source /opt/ros/foxy/setup.bash
colcon build --packages-select lidar --symlink-install
source install/setup.bash

ros2 run lidar rolling_elevation_map --ros-args \
  --params-file \
  /home/unitree/cjh/install/lidar/share/lidar/config/phase2_rolling.yaml
```

`phase1_baseline.yaml` explicitly selects one frame and disables the head
filter, so the original single-frame result remains reproducible for A/B
testing.
