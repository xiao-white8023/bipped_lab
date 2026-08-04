import socket
import struct
import time
from threading import Lock

import numpy as np
import cv2

class DepthClient:
    def __init__(self,config):
        # ==============================
        # 1. 从 config 读取 TCP 参数
        # ==============================
        self.robot_ip = config.robot_ip
        self.TCP_port = config.TCP_port

        # ==============================
        # 2. 从 config 读取深度图预处理参数
        # ==============================
        self.depth_min = config.depth_min
        self.depth_max = config.depth_max

        self.raw_h = config.raw_h
        self.raw_w = config.raw_w

        self.crop_up = config.crop_up
        self.crop_down = config.crop_down
        self.crop_left = config.crop_left
        self.crop_right = config.crop_right

        self.history_frames = config.history_frames

        # ==============================
        # 3. TCP 命令
        # ==============================
        self.cmd_raw = config.CMD_RAW
        self.cmd_interrupt = config.CMD_INTERRUPT

        # ==============================
        # 4. socket 先创建，但不要一上来就死循环
        # ==============================
        self.client = None
        self.connected = False

        # ==============================
        # 多线程以后会用到
        # ==============================
        self.lock = Lock()

        # ==============================
        # 计算单帧 policy depth 的尺寸
        # ==============================
        self.policy_depth_h = self.raw_h - self.crop_up - self.crop_down
        self.policy_depth_w = self.raw_w - self.crop_left - self.crop_right

        if self.policy_depth_h <= 0 or self.policy_depth_w <= 0:
            raise ValueError(
                f"[DepthClient] Invalid crop setting: "
                f"raw=({self.raw_h}, {self.raw_w}), "
                f"crop_up={self.crop_up}, crop_down={self.crop_down}, "
                f"crop_left={self.crop_left}, crop_right={self.crop_right}"
            )
        

        # ==============================
        # 深度图显示配置
        # ==============================
        self.show_raw_depth = getattr(config, "show_raw_depth", False)
        self.show_pre_norm_depth = getattr(config, "show_pre_norm_depth", False)
        self.show_policy_depth = getattr(config, "show_policy_depth", False)

        self.raw_depth_window_name = getattr(config, "raw_depth_window_name", "robot_raw_depth")
        self.pre_norm_depth_window_name = getattr(config, "pre_norm_depth_window_name", "pre_norm_depth_m")
        self.policy_depth_window_name = getattr(config, "policy_depth_window_name", "policy_depth_norm")

        self.raw_depth_display_scale = getattr(config, "raw_depth_display_scale", 1)
        self.pre_norm_depth_display_scale = getattr(config, "pre_norm_depth_display_scale", 10)
        self.policy_depth_display_scale = getattr(config, "policy_depth_display_scale", 10)



    def connect(self):
        """
        连接机器人端 TCP server。
        机器人端需要先运行 TCP_server.py。
        """
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((self.robot_ip, self.TCP_port))

        self.connected = True
        print(f"[DepthClient] Connected to robot depth server: {self.robot_ip}:{self.TCP_port}")
    
    def close(self):
        """
        关闭 TCP 连接。
        如果可以的话，先通知机器人端停止发送。
        """

        if self.client is not None:
            try:
                interrupt_header = struct.pack("!I", self.cmd_interrupt)
                self.client.sendall(interrupt_header)
            except OSError:
                pass

            try:
                self.client.close()
            except OSError:
                pass

        self.connected = False
        if self.show_raw_depth:
            try:
                cv2.destroyWindow(self.raw_depth_window_name)
            except cv2.error:
                pass

        if self.show_pre_norm_depth:
            try:
                cv2.destroyWindow(self.pre_norm_depth_window_name)
            except cv2.error:
                pass

        if self.show_policy_depth:
            try:
                cv2.destroyWindow(self.policy_depth_window_name)
            except cv2.error:
                pass
    

    def recvall(self, n):
        data = b""

        while len(data) < n:
            packet = self.client.recv(n - len(data))

            if not packet:
                return None

            data += packet

        return data

    def read_one_depth(self):
        if not self.connected:
            raise RuntimeError("[DepthClient] TCP is not connected. Call connect() first.")
        
        # ==============================
        # 发送请求命令 CMD_RAW
        # ==============================
        request_header = struct.pack("!I", self.cmd_raw)

        try:
            self.client.sendall(request_header)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            self.connected = False
            raise RuntimeError(f"[DepthClient] Failed to send depth request: {e}")

        # ==============================
        # 接收sever发送的固定 32 字节 header
        # ==============================
        header = self.recvall(32)

        if header is None:
            self.connected = False
            raise RuntimeError("[DepthClient] Connection closed while receiving header.")

        width, height, step, is_bigendian, stamp_sec, stamp_nsec, encoding_len, payload_len = struct.unpack(
            "!IIIIIIII",
            header,
        )

        # ==============================
        # 接收 encoding
        # ==============================
        encoding_bytes = self.recvall(encoding_len)

        if encoding_bytes is None:
            self.connected = False
            raise RuntimeError("[DepthClient] Connection closed while receiving encoding.")

        encoding = encoding_bytes.decode("utf-8")

        # ==============================
        # 接收真正的图像数据 payload
        # ==============================
        payload = self.recvall(payload_len)

        if payload is None:
            self.connected = False
            raise RuntimeError("[DepthClient] Connection closed while receiving payload.")
        
        # ==============================
        # 检查图像格式
        # ==============================
        if encoding != "16UC1":
            raise RuntimeError(f"[DepthClient] Unexpected depth encoding: {encoding}")

        # ==============================
        # 根据大小端恢复 uint16 深度图
        # ==============================
        if is_bigendian:
            dtype = np.dtype(">u2")
        else:
            dtype = np.dtype("<u2")

        depth_uint16 = np.frombuffer(payload, dtype=dtype).reshape(height, width)

        return depth_uint16
    
    def preprocess_depth_for_policy(self, depth_uint16):
        """
        把 RealSense 原始深度图处理成 policy 输入。
        这里将无效深度映射为 0，使部署端遮挡状态与训练时 dropout 噪声一致。
        """

        if depth_uint16 is None:
            raise RuntimeError("[DepthClient] depth_uint16 is None.")

        if depth_uint16.ndim != 2:
            raise RuntimeError(
                f"[DepthClient] Expected 2D depth image, got shape {depth_uint16.shape}."
            )

        # uint16 mm -> float32 meter
        depth_m = depth_uint16.astype(np.float32) * 0.001

        # 原始无效深度 mask
        invalid = (
            (depth_uint16 == 0)
            | np.isnan(depth_m)
            | np.isinf(depth_m)
        )

        # 这里先临时置 0，主要是为了 resize 时有数值
        depth_m[invalid] = 0.0

        # 同步 resize 深度图和 invalid mask
        depth_small = cv2.resize(
            depth_m,
            (self.raw_w, self.raw_h),
            interpolation=cv2.INTER_NEAREST,
        )

        invalid_small = cv2.resize(
            invalid.astype(np.uint8),
            (self.raw_w, self.raw_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        # crop
        depth_crop = depth_small[
            self.crop_up : self.raw_h - self.crop_down,
            self.crop_left : self.raw_w - self.crop_right,
        ]

        invalid_crop = invalid_small[
            self.crop_up : self.raw_h - self.crop_down,
            self.crop_left : self.raw_w - self.crop_right,
        ]

        # 对有效深度做正常 clip
        depth_clip = np.clip(
            depth_crop,
            self.depth_min,
            self.depth_max,
        )

        # 关键：无效区域在 clip 之后强制设为 0
        depth_clip[invalid_crop] = 0.0

        if self.show_pre_norm_depth:
            self.show_meter_depth_gray(
                depth_clip,
                window_name=self.pre_norm_depth_window_name,
                scale=self.pre_norm_depth_display_scale,
            )

        # normalize 到 [0, 1]
        depth_norm = depth_clip / self.depth_max
        depth_norm = np.clip(depth_norm, 0.0, 1.0)

        return depth_norm.astype(np.float32)

    def update_once(self):
        """
        请求一帧深度图，并处理成单帧 policy depth。

        显示三张灰度图：
            1. robot_raw_depth：机器人端传来的原始 uint16 深度图
            2. pre_norm_depth_m：resize/crop/clip 后、归一化前的 18x32 米制深度图
            3. policy_depth_norm：最终送进 policy 的归一化 18x32 深度图
        """

        raw_depth = self.read_one_depth()

        # 1. 机器人端传来的原始深度图
        self.show_raw_depth_gray(raw_depth)

        # 2. preprocess 内部会显示归一化之前的 18x32 米制图
        policy_depth = self.preprocess_depth_for_policy(raw_depth)

        # 3. 最终送给 policy 的归一化图
        if self.show_policy_depth:
            self.show_norm_depth_gray(
                policy_depth,
                window_name=self.policy_depth_window_name,
                scale=self.policy_depth_display_scale,
            )

        return policy_depth

    def show_raw_depth_gray(self, depth_uint16: np.ndarray):
        """
        显示机器人端传过来的原始 uint16 深度图。
        单位：mm。
        只做灰度显示，不做伪彩色。
        """

        if not self.show_raw_depth:
            return

        try:
            if depth_uint16 is None:
                return

            depth_m = depth_uint16.astype(np.float32) * 0.001

            valid = np.isfinite(depth_m) & (depth_uint16 > 0)

            depth_vis = np.zeros_like(depth_m, dtype=np.float32)
            depth_vis[valid] = np.clip(
                depth_m[valid],
                self.depth_min,
                self.depth_max,
            ) / self.depth_max

            # 原始图中 0 / invalid 显示为黑色，方便看空洞
            depth_vis[~valid] = 0.0

            depth_u8 = (depth_vis * 255.0).astype(np.uint8)

            scale = int(self.raw_depth_display_scale)
            if scale > 1:
                depth_u8 = cv2.resize(
                    depth_u8,
                    (depth_u8.shape[1] * scale, depth_u8.shape[0] * scale),
                    interpolation=cv2.INTER_NEAREST,
                )

            cv2.imshow(self.raw_depth_window_name, depth_u8)
            cv2.waitKey(1)

        except cv2.error as e:
            print(f"[DepthClient] show_raw_depth_gray failed: {e}")
            self.show_raw_depth = False

    def show_meter_depth_gray(
        self,
        depth_m: np.ndarray,
        window_name: str,
        scale: int,
    ):
        """
        显示单位为 meter 的深度图。
        用于显示归一化之前的 depth_clip。

        灰度含义：
            depth_min 附近 -> 黑
            depth_max 附近 -> 白
        """

        try:
            if depth_m is None:
                return

            depth_vis = np.nan_to_num(
                depth_m.astype(np.float32),
                nan=self.depth_max,
                posinf=self.depth_max,
                neginf=self.depth_min,
            )

            depth_vis = np.clip(depth_vis, self.depth_min, self.depth_max)

            denom = self.depth_max - self.depth_min
            if denom <= 1e-6:
                depth_vis = np.zeros_like(depth_vis, dtype=np.float32)
            else:
                # 显示用映射：[depth_min, depth_max] -> [0, 1]
                depth_vis = (depth_vis - self.depth_min) / denom

            depth_u8 = (depth_vis * 255.0).astype(np.uint8)

            scale = int(scale)
            if scale > 1:
                depth_u8 = cv2.resize(
                    depth_u8,
                    (depth_u8.shape[1] * scale, depth_u8.shape[0] * scale),
                    interpolation=cv2.INTER_NEAREST,
                )

            cv2.imshow(window_name, depth_u8)
            cv2.waitKey(1)

        except cv2.error as e:
            print(f"[DepthClient] show_meter_depth_gray failed for {window_name}: {e}")
            self.show_pre_norm_depth = False

    def show_norm_depth_gray(
        self,
        depth_norm: np.ndarray,
        window_name: str,
        scale: int,
    ):
        """
        显示已经归一化的 policy depth。
        输入范围约 [0, 1]。
        """

        try:
            if depth_norm is None:
                return

            depth_vis = np.nan_to_num(
                depth_norm.astype(np.float32),
                nan=1.0,
                posinf=1.0,
                neginf=0.0,
            )

            depth_vis = np.clip(depth_vis, 0.0, 1.0)
            depth_u8 = (depth_vis * 255.0).astype(np.uint8)

            scale = int(scale)
            if scale > 1:
                depth_u8 = cv2.resize(
                    depth_u8,
                    (depth_u8.shape[1] * scale, depth_u8.shape[0] * scale),
                    interpolation=cv2.INTER_NEAREST,
                )

            cv2.imshow(window_name, depth_u8)
            cv2.waitKey(1)

        except cv2.error as e:
            print(f"[DepthClient] show_norm_depth_gray failed for {window_name}: {e}")
            self.show_policy_depth = False