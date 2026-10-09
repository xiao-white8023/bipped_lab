# Atomic output of the 50 Hz RENet policy process.
# action and q_target use the frozen 23-DOF Isaac Lab policy order documented
# in g1_control/contracts.py. The 500 Hz bridge owns the policy-to-motor map.

uint8 ESTIMATOR_VP=0
uint8 ESTIMATOR_LP=1
uint32 JOINT_COUNT=23

std_msgs/Header header
uint64 sequence
uint8 estimator_mode
float32 transition_progress
float32[23] action
float32[23] q_target
