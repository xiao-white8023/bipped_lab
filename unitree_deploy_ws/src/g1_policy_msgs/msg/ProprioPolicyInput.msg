# Atomic 50 Hz proprioceptive snapshot produced by the C++ LowState bridge.
# Joint arrays use the frozen 23-DOF Isaac Lab policy order. The header stamp
# is the ROS receive time of the newest Unitree LowState represented here.

uint32 JOINT_COUNT=23
uint32 REMOTE_BYTE_COUNT=40

std_msgs/Header header
uint64 sequence
float32[3] angular_velocity
float32[4] quaternion_wxyz
float32[23] joint_position
float32[23] joint_velocity
uint8[40] wireless_remote
