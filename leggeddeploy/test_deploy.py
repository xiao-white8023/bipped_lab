import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize,ChannelSubscriber

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG

class Statereader:
    def __init__(self,id,net,topic):
        self.low_state = None
        self.count = 0
        ChannelFactoryInitialize(id,net)
        self.Subscriber=ChannelSubscriber(topic,LowStateHG)
        self.Subscriber.Init(self.low_state_callback, 10)
        print(f"[INFO] 接受状态的话题是: {topic}")
        print(f"[INFO] 网络名称是: {net}")
    
    def low_state_callback(self, msg: LowStateHG):
        self.low_state = msg
        self.count += 1
    def spin(self):
        last_count = 0
        last_time = time.time()

        while True:
            time.sleep(1.0)

            if self.low_state is None:
                print("[WAIT] No low_state received yet...")
                continue

            now = time.time()
            hz = (self.count - last_count) / (now - last_time + 1e-6)

            last_count = self.count
            last_time = now

            print(
                f"[OK] tick={self.low_state.tick}, "
                f"mode_machine={self.low_state.mode_machine}, "
                f"recv_hz={hz:.1f}"
            )
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="enp4s0")
    parser.add_argument("--id", type=int, default=0)

    parser.add_argument("--topic", type=str, default="rt/lowstate")
    args = parser.parse_args()

    reader = Statereader(id=args.id,net=args.net, topic=args.topic)
    reader.spin()
