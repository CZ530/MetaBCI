import queue
import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer
import socket
import time

# ================== UDP ================
UDP_TARGET_IP = "127.0.0.1"
UDP_TARGET_PORT = 8000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_udp(cmd, t_start=None):
    sock.sendto(cmd.encode("utf-8"), (UDP_TARGET_IP, UDP_TARGET_PORT))

    if t_start is not None:
        latency = (time.time() - t_start) * 1000
        print(f"[UDP] 已发送 -> {cmd}   ⏱ 延迟: {latency:.2f} ms")
    else:
        print(f"[UDP] 已发送 -> {cmd}")


# ================== VOSK ================
model = Model("vosk-model-small-cn-0.22")
rec = KaldiRecognizer(model, 16000)

COMMANDS = {
    "前进": "forward", "向前": "forward", "往前": "forward", "走": "forward", "前": "forward",
    "左转": "left", "向左": "left", "左边": "left", "左": "left",
    "右转": "right", "向右": "right", "右边": "right", "右": "right",
    "后退": "back", "向后": "back", "往后": "back", "退": "back",
}

STOP_WORDS = [
    "停", "停止", "停下", "别动",
    "听", "挺", "庭",
    "停停", "听听",
    "停下停", "听下",
    "停止停", "听止",
]

COOLDOWN = 0.15
last_time = 0
last_cmd = None
last_partial_cmd = None


# ================== 音频队列 ================
q = queue.Queue()

def callback(indata, frames, time_info, status):
    # 每个音频块都打上时间戳
    q.put((bytes(indata), time.time()))


print("🔥 极速语音识别（延迟 40~80ms）")
print("⏱ 将显示：音频进入系统 → 指令发出 的真实延迟")


with sd.RawInputStream(samplerate=16000, blocksize=1200,
                       dtype='int16', channels=1, callback=callback):

    while True:
        audio_data, audio_timestamp = q.get()  # ← 关键：带时间戳的输入

        if rec.AcceptWaveform(audio_data):
            final = json.loads(rec.Result()).get("text", "")
            last_partial_cmd = None
            continue

        partial = json.loads(rec.PartialResult()).get("partial", "")
        if not partial:
            continue

        # ================= STOP 优先 =================
        for sw in STOP_WORDS:
            if partial.endswith(sw):
                cmd = "stop"
                now = time.time()

                if cmd != last_cmd or (now - last_time > COOLDOWN):
                    print(f"⚡ 识别：{sw} -> stop")
                    send_udp("stop", audio_timestamp)  # ← 用音频时间戳计算延迟

                    last_cmd = "stop"
                    last_time = now
                    last_partial_cmd = "stop"
                break
        else:
            pass

        if last_partial_cmd == "stop":
            continue

        # ================= 匹配其他动作 =================
        for word in sorted(COMMANDS.keys(), key=len, reverse=True):
            if partial.endswith(word):
                cmd = COMMANDS[word]
                now = time.time()

                if cmd == last_partial_cmd:
                    break
                if cmd == last_cmd and (now - last_time < COOLDOWN):
                    break

                print(f"⚡ 识别：{word} -> {cmd}")
                send_udp(cmd, audio_timestamp)  # ← 延迟从音频进入开始算

                last_cmd = cmd
                last_time = now
                last_partial_cmd = cmd
                break
