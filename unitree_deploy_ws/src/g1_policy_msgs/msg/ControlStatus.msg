# Lightweight 500 Hz bridge diagnostics for the first real-robot trials.
# Motor command and measured joint arrays remain in /lowcmd and /lowstate.

uint8 STATE_WAITING=0
uint8 STATE_ACQUIRING=1
uint8 STATE_STANDING=2
uint8 STATE_HOLDING=3
uint8 STATE_POLICY=4
uint8 STATE_DAMPING=5

std_msgs/Header header
uint8 state
bool dry_run
bool motion_service_released
uint8 mode_pr
uint8 mode_machine
uint64 target_sequence
float32 target_age
