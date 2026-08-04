# Copyright (c) 2022-2025, The unitree_rl_gym Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from unitree_rl_gym Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

import struct


class KeyMap:
    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15


class RemoteController:
    def __init__(self):
        self.lx = 0
        self.ly = 0
        self.rx = 0
        self.ry = 0
        self.button = [0] * 16

    # 把机器人发来的 40 字节遥控器原始数据 data，解码成按键状态和摇杆数值。
    def set(self, data):
        # wireless_remote
        # 取第3个 第4个字节。 使其成为16位的无符号整数，每一个数都代表一个按键
        '''
            bit 0  -> R1
            bit 1  -> L1
            bit 2  -> start
            bit 3  -> select
            bit 8  -> A
            bit 9  -> B
            ...
            bit 15 -> left
        '''
        keys = struct.unpack("H", data[2:4])[0] 
        for i in range(16):
            # 1<<i 把二进制的 1 左移 i 位。
            """
                1 << 0 = 0000 0000 0000 0001
                1 << 1 = 0000 0000 0000 0010
                1 << 2 = 0000 0000 0000 0100
                1 << 8 = 0000 0001 0000 0000
            """
            # keys & (1 << i) 只检查 keys 的第 i 位是不是 1
            '''
            如果A键没有按下,则keys就是0
            keys:      xxxx xxx0 xxxx xxxx
            mask:      0000 0001 0000 0000
            & result:  0000 0000 0000 0000
            >> 8:      0
            '''
            """
            如果A键按下,则keys就是0
            keys:      xxxx xxx1 xxxx xxxx
            mask:      0000 0001 0000 0000
            & result:  0000 0001 0000 0000
            >> 8:      0000 0000 0000 0001
            结果就是    1
            """
            self.button[i] = (keys & (1 << i)) >> i
        self.lx = struct.unpack("f", data[4:8])[0]  # 取对应的4个字节，总共32位。转换成浮点数。表示 左摇杆左右方向 
        self.rx = struct.unpack("f", data[8:12])[0] # 取对应的4个字节，总共32位。转换成浮点数。表示 右摇杆左右方向
        self.ry = struct.unpack("f", data[12:16])[0] # 取对应的4个字节，总共32位。转换成浮点数。表示 右摇杆上下方向
        self.ly = struct.unpack("f", data[20:24])[0] # 取对应的4个字节，总共32位。转换成浮点数。表示 左摇杆上下方向
