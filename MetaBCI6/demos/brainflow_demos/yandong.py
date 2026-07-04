import cv2
import mediapipe as mp
import time
import socket
from collections import deque
import numpy as np


# ================= 配置参数 =================
class Config:
    # UDP配置
    UDP_TARGET_IP = "127.0.0.1"
    UDP_TARGET_PORT = 8000

    # 摄像头配置
    CAM_WIDTH = 640
    CAM_HEIGHT = 480
    CAM_FPS = 30

    # 视线检测参数
    CENTER_MIN = 0.40
    CENTER_MAX = 0.60

    # 发送参数
    SEND_INTERVAL = 0.1
    FRAME_SKIP = 1

    # 平滑参数
    SMOOTH_BUFFER_SIZE = 3

    # 防抖参数
    DEBOUNCE_THRESHOLD = 2

    # 新增：校准完成后延时5秒再允许控制
    CONTROL_DELAY_AFTER_CALIBRATION = 5.0


# ================= 关键点定义 =================
mp_face_mesh = mp.solutions.face_mesh

RIGHT_PUPIL = 468
LEFT_PUPIL = 473

RIGHT_EYE_FULL = [33, 133, 157, 158, 159, 160, 161, 173]
LEFT_EYE_FULL = [362, 263, 387, 386, 385, 384, 398]

RIGHT_EYE_POINTS = [362, 263]
LEFT_EYE_POINTS = [33, 133]


# ================= UDP发送类 =================
class UDPClient:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_send_time = 0
        self.last_cmd = None
        self.cmd_stable_count = 0

    def reset_state(self):
        """Reset debounce state after calibration."""
        self.last_send_time = 0
        self.last_cmd = None
        self.cmd_stable_count = 0

    def send(self, cmd):
        now = time.time()

        # eye_center: send immediately
        if cmd == "eye_center":
            self.sock.sendto(cmd.encode("utf-8"), (self.ip, self.port))
            print(f"[UDP] {cmd} at {now:.2f}")
            self.last_send_time = now
            self.last_cmd = cmd
            self.cmd_stable_count = 0
            return True

        # normal debounce for eye_left / eye_right
        if cmd == self.last_cmd:
            self.cmd_stable_count += 1
        else:
            self.cmd_stable_count = 1
            self.last_cmd = cmd
            return False

        if self.cmd_stable_count >= Config.DEBOUNCE_THRESHOLD:
            if now - self.last_send_time >= Config.SEND_INTERVAL:
                self.sock.sendto(cmd.encode("utf-8"), (self.ip, self.port))
                print(f"[UDP] {cmd} at {now:.2f}")
                self.last_send_time = now
                self.cmd_stable_count = 0
                return True

        return False


# ================= 视线平滑器 =================
class GazeSmoother:
    def __init__(self, buffer_size=5):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, value):
        self.buffer.append(value)
        return sum(self.buffer) / len(self.buffer)

    def reset(self):
        """Clear smoothing buffer after calibration."""
        self.buffer.clear()


# ================= 校准器 =================
class Calibration:
    def __init__(self):
        self.left_threshold = Config.CENTER_MIN
        self.right_threshold = Config.CENTER_MAX
        self.calibrated = False

    def collect_gaze_values(self, face_mesh, cap, duration_seconds=2):
        values = []
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0]
                r = self.get_gaze_ratio_advanced(RIGHT_EYE_FULL, RIGHT_PUPIL, lm, w, h)
                l = self.get_gaze_ratio_advanced(LEFT_EYE_FULL, LEFT_PUPIL, lm, w, h)
                ratio = (r + l) / 2
                values.append(ratio)

            cv2.putText(frame, f"Calibrating... {duration_seconds - (time.time() - start_time):.1f}s",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Calibration", frame)
            cv2.waitKey(1)

        if values:
            return np.mean(values)
        return 0.5

    def get_gaze_ratio_advanced(self, eye_points, pupil_idx, lm, w, h):
        x_coords = [int(lm.landmark[pt].x * w) for pt in eye_points]
        min_x, max_x = min(x_coords), max(x_coords)
        px = int(lm.landmark[pupil_idx].x * w)

        width = max_x - min_x
        if width == 0:
            return 0.5
        return (px - min_x) / width

    def calibrate(self, face_mesh, cap):
        print("\n=== 开始校准 ===")

        input("请看向左侧，然后按 Enter 继续...")
        left_value = self.collect_gaze_values(face_mesh, cap)
        print(f"左侧基准值: {left_value:.3f}")

        input("请看向正中间，然后按 Enter 继续...")
        center_value = self.collect_gaze_values(face_mesh, cap)
        print(f"中间基准值: {center_value:.3f}")

        input("请看向右侧，然后按 Enter 继续...")
        right_value = self.collect_gaze_values(face_mesh, cap)
        print(f"右侧基准值: {right_value:.3f}")

        self.left_threshold = center_value - (center_value - left_value) * 0.5
        self.right_threshold = center_value + (right_value - center_value) * 0.5
        self.calibrated = True

        print(f"\n校准完成！")
        print(f"左阈值: {self.left_threshold:.3f}")
        print(f"右阈值: {self.right_threshold:.3f}")
        print("=== 校准结束 ===\n")

        return self.left_threshold, self.right_threshold


# ================= 视线检测器 =================
class GazeDetector:
    def __init__(self):
        self.smoother = GazeSmoother(Config.SMOOTH_BUFFER_SIZE)
        self.calibration = Calibration()
        self.use_advanced = True

    def get_gaze_ratio_basic(self, eye_pts, pupil_idx, lm, w, h):
        x1 = int(lm.landmark[eye_pts[0]].x * w)
        x2 = int(lm.landmark[eye_pts[1]].x * w)
        px = int(lm.landmark[pupil_idx].x * w)

        width = abs(x1 - x2)
        if width == 0:
            return 0.5
        return (px - min(x1, x2)) / width

    def get_gaze_ratio_advanced(self, eye_points, pupil_idx, lm, w, h):
        x_coords = [int(lm.landmark[pt].x * w) for pt in eye_points]
        min_x, max_x = min(x_coords), max(x_coords)
        px = int(lm.landmark[pupil_idx].x * w)

        width = max_x - min_x
        if width == 0:
            return 0.5
        return (px - min_x) / width

    def detect(self, lm, w, h):
        if self.use_advanced:
            r = self.get_gaze_ratio_advanced(RIGHT_EYE_FULL, RIGHT_PUPIL, lm, w, h)
            l = self.get_gaze_ratio_advanced(LEFT_EYE_FULL, LEFT_PUPIL, lm, w, h)
        else:
            r = self.get_gaze_ratio_basic(RIGHT_EYE_POINTS, RIGHT_PUPIL, lm, w, h)
            l = self.get_gaze_ratio_basic(LEFT_EYE_POINTS, LEFT_PUPIL, lm, w, h)

        raw_ratio = (r + l) / 2
        smooth_ratio = self.smoother.update(raw_ratio)

        if smooth_ratio < self.calibration.left_threshold:
            return "LEFT", smooth_ratio
        elif smooth_ratio > self.calibration.right_threshold:
            return "RIGHT", smooth_ratio
        else:
            return "CENTER", smooth_ratio


# ================= 可视化工具 =================
class Visualizer:
    @staticmethod
    def draw_eye_points(frame, lm, w, h):
        right_pupil = (int(lm.landmark[RIGHT_PUPIL].x * w),
                       int(lm.landmark[RIGHT_PUPIL].y * h))
        left_pupil = (int(lm.landmark[LEFT_PUPIL].x * w),
                      int(lm.landmark[LEFT_PUPIL].y * h))

        cv2.circle(frame, right_pupil, 3, (0, 0, 255), -1)
        cv2.circle(frame, left_pupil, 3, (0, 0, 255), -1)

        for point in RIGHT_EYE_FULL + LEFT_EYE_FULL:
            x = int(lm.landmark[point].x * w)
            y = int(lm.landmark[point].y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    @staticmethod
    def draw_gaze_bar(frame, ratio, width=300, height=30, x=30, y=150):
        cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), -1)

        current_width = int(ratio * width)
        cv2.rectangle(frame, (x, y), (x + current_width, y + height), (0, 255, 0), -1)

        left_x = x + int(Config.CENTER_MIN * width)
        right_x = x + int(Config.CENTER_MAX * width)
        cv2.line(frame, (left_x, y - 5), (left_x, y + height + 5), (0, 0, 255), 2)
        cv2.line(frame, (right_x, y - 5), (right_x, y + height + 5), (0, 0, 255), 2)

        cv2.putText(frame, f"Gaze Ratio: {ratio:.3f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "LEFT", (left_x - 30, y + height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "CENTER", (x + width // 2 - 30, y + height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "RIGHT", (right_x + 10, y + height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    @staticmethod
    def draw_info(frame, gaze, ratio, fps):
        cv2.putText(frame, f"FPS: {fps:.1f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        color = (0, 255, 0) if gaze == "CENTER" else (0, 0, 255)
        cv2.putText(frame, f"Gaze: {gaze}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.putText(frame, f"L:{Config.CENTER_MIN:.2f} C:{Config.CENTER_MAX:.2f}",
                    (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


# ================= 主程序 =================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, Config.CAM_FPS)

    udp_client = UDPClient(Config.UDP_TARGET_IP, Config.UDP_TARGET_PORT)
    gaze_detector = GazeDetector()
    visualizer = Visualizer()

    fps = 0
    fps_start_time = time.time()
    fps_frame_count = 0

    frame_counter = 0

    # 新增：控制解锁时间
    control_enable_time = 0.0

    calibrate_choice = input("是否进行校准？(y/n, 默认n): ").strip().lower()
    if calibrate_choice == 'y':
        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
        ) as face_mesh:
            gaze_detector.calibration.calibrate(face_mesh, cap)

        Config.CENTER_MIN = gaze_detector.calibration.left_threshold
        Config.CENTER_MAX = gaze_detector.calibration.right_threshold

        # 新增：校准后5秒内禁止控制
        control_enable_time = time.time() + Config.CONTROL_DELAY_AFTER_CALIBRATION
        gaze_detector.smoother.reset()
        udp_client.reset_state()
        udp_client.send("eye_center")
        print(f"校准完成，等待 {Config.CONTROL_DELAY_AFTER_CALIBRATION:.1f}s 后开始控制...")

    else:
        Config.CENTER_MIN = gaze_detector.calibration.left_threshold
        Config.CENTER_MAX = gaze_detector.calibration.right_threshold

    print("\n程序启动！按 ESC 退出\n")

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("摄像头读取失败")
                break

            frame_counter += 1
            if frame_counter % Config.FRAME_SKIP != 0:
                cv2.imshow("Eye Control - Gaze Tracker", frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            gaze = "CENTER"
            ratio = 0.5

            locked = time.time() < control_enable_time
            remain = max(0.0, control_enable_time - time.time())

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0]

                # 正常检测仍然可以跑，用于显示
                gaze, ratio = gaze_detector.detect(lm, w, h)

                # 只有解锁后才允许发送左右控制
                if not locked:
                    if gaze == "LEFT":
                        udp_client.send("eye_left")
                    elif gaze == "RIGHT":
                        udp_client.send("eye_right")
                    else:
                        udp_client.send("eye_center")

                visualizer.draw_eye_points(frame, lm, w, h)

            fps_frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_frame_count
                fps_frame_count = 0
                fps_start_time = time.time()

            visualizer.draw_info(frame, gaze, ratio, fps)
            visualizer.draw_gaze_bar(frame, ratio)

            # 新增：显示锁定倒计时
            if locked:
                cv2.putText(frame, f"CONTROL LOCK: {remain:.1f}s",
                            (30, 190), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2)

            if udp_client.last_cmd and time.time() - udp_client.last_send_time < 0.5:
                cv2.putText(frame, f"Sending: {udp_client.last_cmd}",
                            (w - 220, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

            cv2.imshow("Eye Control - Gaze Tracker", frame)

            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                break
            elif key == ord('c'):
                print("\n重新校准...")
                with mp_face_mesh.FaceMesh(
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                ) as calib_mesh:
                    gaze_detector.calibration.calibrate(calib_mesh, cap)

                Config.CENTER_MIN = gaze_detector.calibration.left_threshold
                Config.CENTER_MAX = gaze_detector.calibration.right_threshold

                # 重新校准后同样锁5秒
                control_enable_time = time.time() + Config.CONTROL_DELAY_AFTER_CALIBRATION
                gaze_detector.smoother.reset()
                udp_client.reset_state()
                udp_client.send("eye_center")
                print(f"重新校准完成，等待 {Config.CONTROL_DELAY_AFTER_CALIBRATION:.1f}s 后开始控制...")

            elif key == ord('a'):
                gaze_detector.use_advanced = not gaze_detector.use_advanced
                mode = "高级" if gaze_detector.use_advanced else "基础"
                print(f"切换到{mode}检测模式")

    cap.release()
    cv2.destroyAllWindows()
    print("\n程序已退出")


if __name__ == "__main__":
    main()