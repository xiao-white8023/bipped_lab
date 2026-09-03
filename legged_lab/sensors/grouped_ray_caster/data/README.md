# MID-360 angle sequence

`mid360_angles.npy` is derived from `scan_mode/mid360.csv` in the
[Livox laser simulation](https://github.com/Livox-SDK/livox_laser_simulation)
repository at commit `1cce1073633a062b92e30243a4c2920e45551bb5`.

- Original CSV SHA-256: `aa1fc08b6a4400608dbd6ee832b7ea3a9c3c37197e734f60f58fe5abf762269a`
- Derived NPY SHA-256: `0b10e6481d95fd415fc5412123db8515a6f2f47ab6283a55b2c73cec08513d10`
- Shape and type: `(800000, 2)`, `uint16`
- Columns: azimuth and zenith, quantized to `0.01 degree`
- Maximum quantization error relative to the source: `0.005 degree`

The source repository is distributed under the MIT License. Its copyright
notice and license text are reproduced in `LICENSE.livox_laser_simulation`.

At runtime, `Mid360PatternCfg` advances through 20,000 source points per
0.1-second scan. The default temporal stride of five casts 4,000 rays while
preserving their original order and time offsets. The whole packet currently
uses one sensor pose; intra-scan motion compensation belongs to the later
rolling-map stage.
