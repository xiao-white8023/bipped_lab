# Fixed deployment contract for the LP branch of g1_renet.
# header.stamp is the pose/sampling timestamp of this policy grid.
# source_stamp is the newest LiDAR cloud fused into the persistent map.
# Values use row-major y/x order in capture_torso_yaw.

uint32 ROWS=11
uint32 COLS=17
uint32 VALUE_COUNT=187

std_msgs/Header header
builtin_interfaces/Time source_stamp
float32[187] height
uint8[187] valid
float32 valid_ratio
