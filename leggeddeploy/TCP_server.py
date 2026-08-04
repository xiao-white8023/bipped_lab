#!/usr/bin/env python3

# ========================================
# 本文件是TCP的发送端口。订阅depth 并通过TCP发送出去。
# ========================================

import rospy
from sensor_msgs.msg import Image

import socket
import struct
import threading
import time

HOST = "0.0.0.0" # 表示监听机器人的机载电脑的所有网卡
PORT = 5555 # 这是 TCP 端口，和电脑端 client 保持一致即可
TOPIC = "/camera/depth/image_rect_raw" # 就是 RealSense ROS 发布的原始深度图 topic

CMD_INTERRUPT = 0 # 终止
CMD_RAW = 1  # 接收到client端的命令和这个一样的话 就发送原生深度图

# ============================================
# 这个类中要包含ros的订阅、TCP server以及发送线程
# ============================================
class DepthRawSender:
    def __init__(self):
        self.lastest_msg=None # 用来保存最新收到的一帧ROS深度图的消息。
        self.lock=threading.Lock() # 线程锁

        # 建立TCP server
        # ================================================================
        """
        socket.AF_INET 表示使用IPv4    socket.SOCK_STREAM 表示使用TCP
        """
        self.server = socket.socket(socket.AF_INET,socket.SOCK_STREAM) 
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((HOST, PORT)) # 表示机器人端开始监听
        self.server.listen(1) # 表示等待一个客户端连接
        
        # 订阅的深度相机
        # ================================================================
        self.sub = rospy.Subscriber(
                TOPIC,
                Image,
                self.depth_callback,  # 回调函数
                queue_size=1, # 表示只保留最新的一帧
                buff_size=2 ** 24, # 缓冲区的大小
        )
        # ================================================================
        print(f"Waiting client on {HOST}:{PORT} ...")
        self.conn, self.addr = self.server.accept() # 机器人端程序运行到这里，会停住，直到电脑端 client 连上来
        print("Client connected:", self.addr)
        # ================================================================


        
        # 创建一个发送的线程
        # ================================================================
        self.sender_thread=threading.Thread(target=self.send_loop)
        self.sender_thread.daemon = True # 当主程序退出的时候，这个线程也会退出
        self.sender_thread.start()
        # ================================================================


    # 回调函数 这里只接受最新的深度图。不对深度图进行处理
    def depth_callback(self,msg):
        with self.lock:
            self.lastest_msg=msg
        
    # 回调函数 这里只发送最新的深度图
    def send_loop(self):
        while not rospy.is_shutdown():  # 只要ros循环没有关闭，就一直发送
            cmd_bytes = recvall(self.conn, 4)
            if cmd_bytes is None:
                print("Client disconnected.")
                break
            # 2. 解包命令
            cmd = struct.unpack("!I", cmd_bytes)[0]
            # 3. 如果 client 要断开
            if cmd == CMD_INTERRUPT:
                print("Client requested interrupt.")
                break
                # 等到有图
            if cmd == CMD_RAW:
                msg = None
                while not rospy.is_shutdown():
                    with self.lock:
                        msg = self.lastest_msg

                    if msg is not None:
                        break

                    time.sleep(0.001)

                if msg is None:
                        break
                # 其中，这个ros中的msg是有这些字段的：
                    # ======================================
                                # msg.width        图像宽度，例如 848
                                # msg.height       图像高度，例如 480
                                # msg.encoding     图像格式，例如 16UC1
                                # msg.is_bigendian 大小端信息
                                # msg.step         每一行多少字节
                                # msg.data         真正的图像 每一个像素是16个二进制表示的整数 占两个字节。如果选择的原始分辨率是848 × 480   则总共有848 × 480 × 2 = 814080 字节
                    # ======================================
                encoding_bytes = msg.encoding.encode("utf-8") 
                payload = bytes(msg.data)  # 将ROS中的字节转化成 python专用的字节

                # 电脑端接收到这些信号之后，必须可以解码出来。所以要固定字段.每一个都使用4字节无符号整数，则一共有8x4=32字节
                    # ======================================            
                                # width
                                # height
                                # step
                                # is_bigendian
                                # stamp_sec  时间戳的秒部分  
                                # stamp_nsec  时间戳的纳秒部分 电脑端也可以用它统计延迟、频率、丢帧
                                # encoding_len
                                # payload_len
                    # =======================================
                # 将定义的字段打包    
                header = struct.pack(
                        "!IIIIIIII",                # ！表示大端序 “IIIIIIII”表示8个无符号整数  每一个“I” 代表四个字节  则总共有32个字节
                        msg.width,   
                        msg.height,
                        msg.step,
                        msg.is_bigendian,
                        msg.header.stamp.secs,
                        msg.header.stamp.nsecs,
                        len(encoding_bytes),
                        len(payload),
                    )
                try:
                    self.conn.sendall(header)
                    self.conn.sendall(encoding_bytes)
                    self.conn.sendall(payload)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    print("Client disconnected while sending.")
                    break

def recvall(sock, n):
    data = b""

    while len(data) < n:
        packet = sock.recv(n - len(data))

        if not packet:
            return None

        data += packet

    return data

if __name__ == "__main__":
    rospy.init_node("depth_raw_sender")
    sender = DepthRawSender()
    rospy.spin()