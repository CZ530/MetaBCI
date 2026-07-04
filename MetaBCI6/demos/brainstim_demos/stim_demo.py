import json
import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from psychopy import monitors
import numpy as np
from metabci.brainstim.paradigm import (
    SSVEP,
    paradigm,

)
from metabci.brainstim.framework import Experiment

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] in ["offline", "online"]:
        run_mode = sys.argv[1]
    else:
        run_mode = "offline"

        # 第2个参数：打标端口号，例如 COM11、COM15、None
    if len(sys.argv) >= 3:
        selected_port = sys.argv[2]
    else:
        selected_port = "COM7"

    if selected_port.strip().lower() in ["", "none", "null"]:
        selected_port = None

    print("当前运行模式:", run_mode)
    print("当前打标端口:", selected_port)
    mon = monitors.Monitor(
        name="primary_monitor",
        width=59.6,
        distance=60,  # width 显示器尺寸cm; distance 受试者与显示器间的距离
        verbose=False,
    )
    mon.setSizePix([1920, 1080])  # 显示器的分辨率
    mon.save()
    bg_color_warm = np.array([-1, -1, -1])
    win_size = np.array([1920, 1080])
    # esc/q退出开始选择界面

    ex = Experiment(
        monitor=mon,
        bg_color_warm=bg_color_warm,  # 范式选择界面背景颜色[-1~1,-1~1,-1~1]
        screen_id=0,
        win_size=win_size,  # 范式边框大小(像素表示)，默认[1920,1080]
        is_fullscr=True,  # True全窗口,此时win_size参数默认屏幕分辨率
        record_frames=False,
        disable_gc=False,
        process_priority="normal",
        use_fbo=False,
    )
    win = ex.get_window()
    # win = visual.Window(size=(1920, 1080), fullscr=False, color=(0, 0, 0))
    # win.flip()
    # win.close()


    # q退出范式界面
    """
    SSVEP
    """
    n_elements, rows, columns = 7 ,1, 7  # n_elements 指令数量;  rows 行;  columns 列
    stim_length, stim_width = 200, 200  # ssvep单指令的尺寸
    stim_color, tex_color = [1, 1, 1], [1, 1, 1]  # 指令的颜色，文字的颜色
    fps = 60  # 屏幕刷新率
    stim_time = 2  # 刺激时长
    stim_opacities = 1  # 刺激对比度
    freqs = np.arange(10,  17, 1)  # 指令的频率
    phases = np.array([i * 0.35 % 2 for i in range(n_elements)])  # 指令的相位
    stim_pos = np.array([
        [-200, 400],  # 第1个刺激块的 (x, y) 坐标
        [-200, -400],  # 第2个刺激块
        [-600, 000],  # 第3个刺激块
        [200, 000],
        [-200, 000],
        [600, 400],
        [600, -400],

    ])
    basic_ssvep = SSVEP(win=win)

    basic_ssvep.config_pos(
        n_elements=n_elements,
        rows=rows,
        columns=columns,
        stim_pos=stim_pos,
        stim_length=stim_length,
        stim_width=stim_width,
    )
    basic_ssvep.config_text(tex_color=tex_color)
    basic_ssvep.config_color(
        refresh_rate=fps,
        stim_time=stim_time,
        stimtype="sinusoid",
        stim_color=stim_color,
        stim_opacities=stim_opacities,
        freqs=freqs,
        phases=phases,
    )
    basic_ssvep.config_index()
    basic_ssvep.config_response()

    bg_color = np.array([0.3, 0.3, 0.3])  # 背景颜色
    if run_mode == "offline":
        # 离线采集参数：有刺激、有打标、不等待在线反馈
        display_time = 1
        index_time = 1
        rest_time = 2
        response_time = 0
        port_addr = selected_port
        nrep = 5
        lsl_source_id = None
        online = False

    elif run_mode == "online":
        # 在线控制参数：有刺激、有打标、等待在线反馈
        display_time = 1
        index_time = 0
        rest_time = 2
        response_time = 1
        port_addr = selected_port
        nrep = 10
        lsl_source_id = "meta_online_worker"
        online = True

    else:
        raise ValueError(f"未知运行模式: {run_mode}")
    ex.register_paradigm(
        "basic SSVEP",
        paradigm,
        VSObject=basic_ssvep,
        bg_color=bg_color,
        display_time=display_time,
        index_time=index_time,
        rest_time=rest_time,
        response_time=response_time,
        port_addr=port_addr,
        nrep=nrep,
        pdim="ssvep",
        lsl_source_id=lsl_source_id,
        online=online,
    )
    ex.run()
