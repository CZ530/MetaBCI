import serial
import time
import socket
import threading

# ===========================
#     JDY-16 主机模块
# ===========================
class JDY16Master:
    def __init__(self, port, baudrate=9600, timeout=0.1):
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
        )

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def send_at(self, cmd, wait=0.1):
        if not cmd.endswith("\r\n"):
            cmd += "\r\n"
        self.ser.write(cmd.encode())
        self.ser.flush()
        time.sleep(wait)

    def enter_master_mode(self):
        self.send_at("AT+HOSTEN1", wait=0.5)
        time.sleep(1.5)

    def connect_by_mac(self, mac):
        mac = mac.replace(":", "").upper()
        self.send_at("AT+CONN" + mac, wait=1)

    def send_raw(self, data: bytes):
        self.ser.write(data)
        self.ser.flush()


# ================================================================
#                    轮椅控制核心
# ================================================================
class WheelchairController:
    JOY_CENTER = 0x0844
    X_LEFT     = 0x0FFF
    X_RIGHT    = 0x0000
    Y_FRONT    = 0x0C00
    Y_BACK     = 0x0400

    FRAME_INTERVAL = 0.02

    FORWARD_TIME = 2
    BACK_TIME    = 2

    # ================= Y轴缓启动/缓停止参数 =================
    Y_RAMP_STEP = 0x0040   # 越小越慢，越大越快

    # ================= 速度档 =================
    SPEED_LOW  = 1
    SPEED_HIGH = 2

    def __init__(self, jdy: JDY16Master):
        self.jdy = jdy

        self.current_x = self.JOY_CENTER
        self.current_y = self.JOY_CENTER

        # Y轴目标值，用于缓慢加速/缓慢停止
        self.target_y = self.JOY_CENTER

        # 默认低速（安全）
        self.speed = self.SPEED_LOW

        self.x_end_time = 0
        self.y_end_time = 0

        self.override_stop = False

        self.tx_thread = threading.Thread(
            target=self._tx_loop, daemon=True
        )
        self.tx_thread.start()

    # ================= 速度接口 =================
    def set_low_speed(self):
        print("[Speed] LOW")
        self.speed = self.SPEED_LOW

    def set_high_speed(self):
        print("[Speed] HIGH")
        self.speed = self.SPEED_HIGH

    # ================= 帧构建 =================
    def _build_frame(self, x, y):
        frame = bytearray(15)
        frame[0:5] = b'\xEB\x90\x0F\xA2\xAA'

        frame[5] = x & 0xFF
        frame[6] = (x >> 8) & 0xFF
        frame[7] = y & 0xFF
        frame[8] = (y >> 8) & 0xFF

        frame[9]  = 0x00
        frame[10] = self.speed
        frame[11:15] = b'\xCC\x33\xC3\x3C'
        return frame

    # ================= 数值缓慢逼近 =================
    def _ramp_to_target(self, current, target, step):
        if current < target:
            return min(current + step, target)
        elif current > target:
            return max(current - step, target)
        else:
            return current

    # ================= 强制停止 =================
    def _force_stop(self):
        frame = self._build_frame(self.JOY_CENTER, self.JOY_CENTER)
        for _ in range(3):
            self.jdy.send_raw(frame)
            time.sleep(0.01)

    # ================= 后台发帧 =================
    def _tx_loop(self):
        while True:
            now = time.time()

            # X轴超时 → 回中
            if self.x_end_time and now >= self.x_end_time:
                self.current_x = self.JOY_CENTER
                self.x_end_time = 0

            # Y轴超时 → 缓慢停止
            if self.y_end_time and now >= self.y_end_time:
                self.target_y = self.JOY_CENTER
                self.y_end_time = 0

            if self.override_stop:
                self._force_stop()
                time.sleep(self.FRAME_INTERVAL)
                continue

            # Y轴缓慢逼近目标值，实现缓慢加速/缓慢停止
            self.current_y = self._ramp_to_target(
                self.current_y,
                self.target_y,
                self.Y_RAMP_STEP
            )

            frame = self._build_frame(self.current_x, self.current_y)
            self.jdy.send_raw(frame)
            time.sleep(self.FRAME_INTERVAL)

    # ================= 停止 =================
    def stop(self):
        print("[CMD] STOP")
        self.override_stop = True

        self.current_x = self.JOY_CENTER
        self.current_y = self.JOY_CENTER
        self.target_y = self.JOY_CENTER
        self.x_end_time = 0
        self.y_end_time = 0

        self._force_stop()
        time.sleep(0.05)

        self.override_stop = False

    # ================= 缓慢停止 =================
    def smooth_stop(self):
        print("[CMD] SMOOTH STOP")
        self.current_x = self.JOY_CENTER
        self.target_y = self.JOY_CENTER
        self.x_end_time = 0
        self.y_end_time = 0

    # ================= 前后 =================
    def forward(self, duration=None):
        print("[CMD] FORWARD")
        self.target_y = self.Y_FRONT
        self.y_end_time = time.time() + (duration or self.FORWARD_TIME)

    def backward(self, duration=None):
        print("[CMD] BACKWARD")
        self.target_y = self.Y_BACK
        self.y_end_time = time.time() + (duration or self.BACK_TIME)

    # ================= 左右（原定时版） =================
    def left(self, duration=1):
        print("[CMD] LEFT")
        self.current_x = self.X_LEFT
        self.x_end_time = time.time() + duration

    def right(self, duration=1):
        print("[CMD] RIGHT")
        self.current_x = self.X_RIGHT
        self.x_end_time = time.time() + duration

    # ================= 眼动连续控制专用 =================
    def eye_forward_hold(self):
        self.target_y = self.Y_FRONT
        self.y_end_time = 0

    # ================= 持续后退 =================
    def eye_backward_hold(self):
        self.target_y = self.Y_BACK
        self.y_end_time = 0

    # ================= 释放前后 =================
    def eye_move_release(self):
        self.target_y = self.JOY_CENTER
        self.y_end_time = 0
    def eye_left_hold(self):
        """Continuous left turn for eye-control."""
        print("[EYE] LEFT HOLD")
        self.current_x = self.X_LEFT
        self.x_end_time = 0   # Disable timeout auto-center

    def eye_right_hold(self):
        """Continuous right turn for eye-control."""
        print("[EYE] RIGHT HOLD")
        self.current_x = self.X_RIGHT
        self.x_end_time = 0   # Disable timeout auto-center

    def eye_turn_release(self):
        """Release eye-control turning and center X axis only."""
        self.current_x = self.JOY_CENTER
        self.x_end_time = 0


# ================================================================
#                     UDP 指令处理
# ================================================================
def handle_udp(cmd: str, wc: WheelchairController):

    s = cmd.strip().lower()

    # =====================================================
    # 脑控数字控制
    # =====================================================
    if s == "1":
        wc.forward()

    elif s == "2":
        wc.backward()

    elif s == "3":
        wc.left()

    elif s == "4":
        wc.right()

    elif s == "5":
        wc.stop()

    elif s == "6":
        wc.set_low_speed()

    elif s == "7":
        wc.set_high_speed()

    # =====================================================
    # 头动持续控制
    # =====================================================
    elif s == "eye_forward":
        wc.eye_forward_hold()

    elif s == "eye_backward":
        wc.eye_backward_hold()

    # =====================================================
    # 眼动持续转向
    # =====================================================
    elif s == "eye_left":

        # 持续左转
        wc.eye_left_hold()

    elif s == "eye_right":

        # 持续右转
        wc.eye_right_hold()

    # =====================================================
    # 眼睛回中
    # =====================================================
    elif s in ("eye_center", "eye_release", "center"):

        wc.eye_turn_release()
        wc.eye_move_release()

        # 如果你不想停车，只想回正：
        # wc.eye_turn_release()

    # =====================================================
    # 普通字符串兼容
    # =====================================================
    elif s == "forward":
        wc.forward()

    elif s in ("back", "backward"):
        wc.backward()

    elif s == "left":
        wc.left()

    elif s == "right":
        wc.right()

    elif s == "stop":
        wc.stop()

    else:
        print("[WARN] Unknown:", s)
# def handle_udp(cmd: str, wc: WheelchairController):
#     s = cmd.strip()
#
#     if s == "1":
#         wc.forward()      # 1 前进，缓慢加速
#
#     elif s == "2":
#         wc.backward()     # 2 后退，缓慢加速
#
#     elif s == "3":
#         wc.left()         # 3 左转
#
#     elif s == "4":
#         wc.right()        # 4 右转
#
#     elif s == "5":
#         # wc.smooth_stop()  # 5 缓慢停止
#         wc.forward()
#     elif s == "6":
#     # wc.smooth_stop()  # 5 缓慢停止
#         wc.forward()
#     else:
#         print("[WARN] Unknown label:", s)


# def handle_udp(cmd: str, wc: WheelchairController):
#     s = cmd.strip().lower()
#
#     if s == "stop":
#         wc.stop()
#
#     elif s == "forward":
#         wc.forward()
#
#     elif s in ("back", "backward"):
#         wc.backward()
#
#     # 保留原来的按键/脉冲控制
#     elif s == "left":
#         wc.left()
#
#     elif s == "right":
#         wc.right()
#
#     # 新增眼动持续控制
#     elif s == "eye_left":
#         wc.eye_left_hold()
#
#     elif s == "eye_right":
#         wc.eye_right_hold()
#
#     elif s in ("eye_center", "eye_release", "center"):
#         wc.eye_turn_release()
#
#     elif s == "speed_low":
#         wc.set_low_speed()
#
#     elif s == "speed_high":
#         wc.set_high_speed()
#
#     else:
#         print("[WARN] Unknown:", s)

# ================================================================
#                        主程序
# ================================================================
if __name__ == "__main__":
    COM_PORT = "COM3"
    WHEEL_MAC = "11:89:88:12:97:C2"

    UDP_IP = "0.0.0.0"
    UDP_PORT = 8000

    print("[Init] Serial...")
    jdy = JDY16Master(COM_PORT)

    print("[Init] Enter master mode...")
    jdy.enter_master_mode()

    print("[Init] Connecting...")
    jdy.connect_by_mac(WHEEL_MAC)

    wc = WheelchairController(jdy)
    wc.stop()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"[UDP] Listening on {UDP_PORT}")

    try:
        while True:
            data, addr = sock.recvfrom(1024)
            cmd = data.decode()
            print("[UDP RX]", cmd)

            handle_udp(cmd, wc)

            sock.sendto(b"OK", addr)

    except KeyboardInterrupt:
        print("\n[Exit]")

    finally:
        wc.stop()
        jdy.close()