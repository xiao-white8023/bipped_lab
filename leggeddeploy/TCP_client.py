import socket
import struct
import time

import numpy as np
import cv2


ROBOT_IP = "192.168.123.164"
PORT = 5555

REQUEST_HZ = 10.0
REQUEST_DT = 1.0 / REQUEST_HZ

CMD_RAW = 1   # 请求机器人端发送原生的深度图
CMD_INTERRUPT = 0 # 终止

DEPTH_MAX = 2.5
DEPTH_MIN = 0.1
RAW_H = 36
RAW_W = 64
CROP_UP = 18
CROP_DOWN = 0
CROP_LEFT = 16
CROP_RIGHT = 16

def get_policy_invalid_mask(depth_uint16):
    """
    返回经过同样 resize + crop 后的 invalid mask。
    白色表示原始深度为 0，也就是无效深度点。
    黑色表示原始深度有效。
    """

    # raw depth 里，0 表示无效点
    invalid = (depth_uint16 == 0).astype(np.uint8)

    # 和 policy depth 使用同样的 resize
    invalid_small = cv2.resize(
        invalid,
        (RAW_W, RAW_H),
        interpolation=cv2.INTER_NEAREST,
    )

    # 和 policy depth 使用同样的 crop
    invalid_crop = invalid_small[
        CROP_UP : RAW_H - CROP_DOWN,
        CROP_LEFT : RAW_W - CROP_RIGHT,
    ]

    # 为了显示，把 18×32 放大
    invalid_show = cv2.resize(
        invalid_crop * 255,
        (640, 360),
        interpolation=cv2.INTER_NEAREST,
    )

    return invalid_crop, invalid_show

def visualize_raw_depth(depth_uint16):
    depth_m = depth_uint16.astype(np.float32) * 0.001

    invalid = depth_uint16 == 0
    depth_m[invalid] = DEPTH_MAX

    depth_vis = np.clip(depth_m / DEPTH_MAX, 0.0, 1.0)
    depth_u8 = (depth_vis * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

    return depth_color

def visualize_policy_depth(depth_norm):
    """
    depth_norm: float32, shape = (18, 32), range roughly [0, 1]
    0 表示近，1 表示远。
    """

    # 为了显示清楚，先放大
    depth_show = cv2.resize(
        depth_norm,
        (640, 360),
        interpolation=cv2.INTER_NEAREST,
    )

    # 直接显示训练输入的数值分布：0近，1远
    depth_u8 = (depth_show * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

    return depth_color

def preprocess_depth_for_policy(depth_uint16):
    # 1. uint16 mm -> float32 meter
    depth_m = depth_uint16.astype(np.float32) * 0.001

    # 2. 真实相机无效值处理
    invalid = (depth_uint16 == 0) | np.isnan(depth_m) | np.isinf(depth_m)
    depth_m[invalid] = DEPTH_MAX

    # 3. resize 到训练相机原始分辨率: 64×36
    depth_small = cv2.resize(
        depth_m,
        (RAW_W, RAW_H),
        interpolation=cv2.INTER_NEAREST,
    )

    # 4. 按训练 crop
    depth_crop = depth_small[
        CROP_UP : RAW_H - CROP_DOWN,
        CROP_LEFT : RAW_W - CROP_RIGHT,
    ]

    # 5. clip
    depth_clip = np.clip(depth_crop, DEPTH_MIN, DEPTH_MAX)

    # 6. normalize 到 [0, 1]
    depth_norm = depth_clip / DEPTH_MAX

    return depth_norm.astype(np.float32)

def recvall(sock, n):
    data = b""

    while len(data) < n:
        packet = sock.recv(n - len(data))

        if not packet:
            return None

        data += packet

    return data

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((ROBOT_IP, PORT))

print("Connected to robot.")

frame_count = 0
last_time = time.time()

while True:
    loop_start = time.time()
    client_header = struct.pack(
                    "!I",                # ！表示大端序 “IIIIIIII”表示8个无符号整数  每一个“I” 代表四个字节  则总共有4个字节
                    CMD_RAW
                )
    try:
        client.sendall(client_header)
    except (BrokenPipeError, ConnectionResetError, OSError):
        print("在发送请求的时候 连接失败了")
        break

    # 1. 接收固定 32 字节 header
    header = recvall(client, 32)

    if header is None:
        print("Connection closed while receiving header.")
        break

    # 2. 解包 header
    width, height, step, is_bigendian, stamp_sec, stamp_nsec, encoding_len, payload_len = struct.unpack(
        "!IIIIIIII",
        header
    )

    # 3. 接收 encoding
    encoding_bytes = recvall(client, encoding_len)

    if encoding_bytes is None:
        print("Connection closed while receiving encoding.")
        break

    encoding = encoding_bytes.decode("utf-8")

    # 4. 接收真正的图像数据
    payload = recvall(client, payload_len)

    if payload is None:
        print("Connection closed while receiving payload.")
        break

    # 5. 检查图像格式
    if encoding != "16UC1":
        print("Unexpected encoding:", encoding)
        continue

    # 6. 根据大小端恢复 uint16 深度图
    if is_bigendian:
        dtype = np.dtype(">u2")
    else:
        dtype = np.dtype("<u2")

    depth = np.frombuffer(payload, dtype=dtype).reshape(height, width)

    depth_norm = preprocess_depth_for_policy(depth)

    raw_color = visualize_raw_depth(depth)
    policy_color = visualize_policy_depth(depth_norm)

    policy_invalid_crop, policy_invalid_show = get_policy_invalid_mask(depth)

    cv2.imshow("raw depth from robot", raw_color)
    cv2.imshow("policy depth 18x32", policy_color)
    cv2.imshow("policy invalid mask 18x32", policy_invalid_show)

    # 10. 显示基本信息
    frame_count += 1
    now = time.time()

    if now - last_time >= 1.0:
        fps = frame_count / (now - last_time)
        frame_count = 0
        last_time = now

        valid = (depth > 0) & (depth < 10000)

        if np.any(valid):
            valid_depth = depth[valid]
            policy_invalid_ratio = np.mean(policy_invalid_crop > 0)
            far_ratio = np.mean(depth_norm >= 0.999)
            print(
                f"fps: {fps:.1f}, "
                f"raw_shape: {depth.shape}, "
                f"policy_shape: {depth_norm.shape}, "
                f"encoding: {encoding}, "
                f"raw_min: {valid_depth.min()}, "
                f"raw_max: {valid_depth.max()}, "
                f"raw_mean: {valid_depth.mean():.1f}, "
                f"policy_min: {depth_norm.min():.3f}, "
                f"policy_max: {depth_norm.max():.3f}, "
                f"policy_mean: {depth_norm.mean():.3f}, "
                f"policy_invalid: {policy_invalid_ratio:.3f}, "
                f"far_ratio: {far_ratio:.3f}"
            )
        else:
            print(f"fps: {fps:.1f}, no valid depth")

    key = cv2.waitKey(1)


    # 按 ESC 退出
    if key == 27:
        client_header = struct.pack(
                    "!I",                # ！表示大端序 “IIIIIIII”表示8个无符号整数  每一个“I” 代表四个字节  则总共有4个字节
                    CMD_INTERRUPT
                )
        client.sendall(client_header)
        break

    # ====================================
    # 确保是按照前面设置的频率进行发送的 
    elapsed = time.time() - loop_start
    sleep_time = REQUEST_DT - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)

client.close()
cv2.destroyAllWindows()