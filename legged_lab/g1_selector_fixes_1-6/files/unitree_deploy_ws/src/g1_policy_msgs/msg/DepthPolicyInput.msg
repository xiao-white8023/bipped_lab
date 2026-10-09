# Fixed deployment contract for the VP branch of g1_renet.
# header.stamp is the source timestamp of the newest depth frame.
# Values are normalized depth in oldest-to-newest/frame/y/x order.

uint32 HISTORY_FRAMES=2
uint32 HEIGHT=18
uint32 WIDTH=32
uint32 VALUE_COUNT=1152

std_msgs/Header header
float32[1152] history
# Raw valid support before invalid fill/blur/clip, in the SAME history order.
float32[1152] valid_support
bool valid
float32 source_age
float32 valid_ratio
