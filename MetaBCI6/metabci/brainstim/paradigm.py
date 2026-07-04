# -*- coding: utf-8 -*-
import math

# load in basic modules
import os
import os.path as op
import string
import numpy as np
from math import pi
from psychopy import data, visual, event
from psychopy.visual.circle import Circle
from pylsl import StreamInlet, resolve_byprop  ,resolve_streams# type: ignore
from metabci.brainstim.utils import NeuroScanPort, NeuraclePort, _check_array_like
import threading
from copy import copy
import random
from scipy import signal
from PIL import Image


# prefunctions


def sinusoidal_sample(freqs, phases, srate, frames, stim_color):
    """
    Sinusoidal approximate sampling method.

    author: Qiaoyi Wu

    Created on: 2022-06-20

    update log:
        2022-06-26 by Jianhang Wu

        2022-08-10 by Wei Zhao

        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        freqs: list of float             刺激的频率列表，每个元素表示一个刺激的频率（Hz）
            Frequencies of each stimulus.
        phases: list of float            相位列表，与 freqs 对应，表示刺激的相位（π 的倍数）
            Phases of each stimulus.
        srate: int or float              屏幕刷新率（Hz），决定了帧的时间间隔
            Refresh rate of screen.
        frames: int                      总帧数，即刺激持续的时间（帧数 = 采样时间 × 采样率）
            Flashing frames.
        stim_color: list
            Color of stimu.              刺激的颜色（RGB），可以是 [-1, -1, -1] 代表灰度模式

    Returns
    ----------
        color: ndarray
            shape(frames, len(fre), 3)

    """

    time = np.linspace(0, (frames - 1) / srate, frames)         #生成 frames 个时间点，范围从 0 到 (frames-1)/srate（秒）。
    color = np.zeros((frames, len(freqs), 3))                        #形状为 (frames, len(freqs), 3)，表示每一帧，每个刺激，每个颜色通道的值。
    for ne, (freq, phase) in enumerate(zip(freqs, phases)):          #freqs 和 phases 是频率和相位的列表，每个刺激目标对应一个频率和相位。zip(freqs, phases) 让 freq 和 phase 分别从 freqs 和 phases 取值。enumerate() 获取索引 ne，用于存储每个目标的颜色变化。
        sinw = np.sin(2 * pi * freq * time + pi * phase) + 1
        color[:, ne, :] = np.vstack(
            (sinw * stim_color[0], sinw * stim_color[1], sinw * stim_color[2])
        ).T
        if stim_color == [-1, -1, -1]:
            pass
        else:
            if stim_color[0] == -1:
                color[:, ne, 0] = -1
            if stim_color[1] == -1:
                color[:, ne, 1] = -1
            if stim_color[2] == -1:
                color[:, ne, 2] = -1

    return color


def wave_new(stim_num, type):
    """determine the color of each offset dot according to "type".

    author: Jieyu Wu

    Created on: 2022-12-14

    update log:
        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        stim_num: int
            Number of stimuli dots of each target.
        type: int
            avep code.

    Returns
    ----------
        point: ndarray
            (stim_num, 3)

    """
    point = [[-1, -1, -1] for i in range(stim_num)]
    if type == 0:
        pass
    else:
        point[type - 1] = [1, 1, 1]
    point = np.array(point)
    return point


def pix2height(win_size, pix_num):                  #将像素值转换为归一化高度值
    height_num = pix_num / win_size[1]
    return height_num                             #height_num:  归一化高低


def height2pix(win_size, height_num):               #将归一化高度值转换为像素单位
    pix_num = height_num * win_size[1]
    return pix_num


def code_sequence_generate(basic_code, sequences):                #ssavep
    """Quickly generate coding sequences for sub-stimuli using basic endcoding units and encoding sequences.

    author: Jieyu Wu

    Created on: 2023-09-18

    update log:
        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        basic_code: list
            Each basic encoding unit in the encoding sequence.
        sequences: list of array
            Encoding sequences for basic_code.

    Returns
    ----------
        code: ndarray
            coding sequences for sub-stimuli.

    """

    code = []
    for seq_i in range(len(sequences)):
        code_list = []
        seq_length = len(sequences[seq_i])
        for code_i in range(seq_length):
            code_list.append(basic_code[sequences[seq_i][code_i]])
        code.append(code_list)
    code = np.array(code)
    return code


# create interface for VEP-BCI-Speller


class KeyboardInterface(object):
    """Create the interface to the stimulus interface and initialize the window parameters.

    author: Qiaoyi Wu

    Created on: 2022-06-20

    update log:
        2022-06-26 by Jianhang Wu

        2022-08-10 by Wei Zhao

        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        win:
            The window object.
        colorspace: str
            The color space, default to rgb.
        allowGUI: bool
            Defaults to True, which allows frame-by-frame drawing and key-exit.

    Attributes
    ----------
        win:
            The window object.
        win_size: ndarray, shape(width, high)
            The size of the window in pixels.
        stim_length: int
            The length of the stimulus block in pixels.
        stim_width: int
            The width of the stimulus block in pixels.
        n_elements: int
            Number of stimulus blocks.
        stim_pos: ndarray, shape([x, y],...)
            Customize the position of the stimulus blocks with an array length
            that corresponds to the number of stimulus blocks.
        stim_sizes: ndarray, shape([length, width],...)
            The size of the stimulus block, the length of which corresponds to the number of stimulus blocks.
        symbols: str
            Stimulate the text of characters in the block.
        text_stimuli:
            Configuration information required for paradigm characters.
        rect_response:
            Configuration information required for the rectangular feedback box.
        res_text_pos: tuple, shape (x, y)
            The character position of the online response.
        symbol_height: int
            The height of the feedback character.
        symbol_text: str
            The character text of the online response.
        text_response:
            Configuration information for the feedback character.

    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        self.win = win
        win.colorSpace = colorSpace
        win.allowGUI = allowGUI
        win_size = win.size
        self.win_size = np.array(win_size)  # e.g. [1920,1080]

    def config_pos(
        self,
        n_elements=40,
        rows=5,
        columns=8,
        stim_pos=None,
        stim_length=150,
        stim_width=150,
    ):
        """Set the number, position, and size parameters of the stimulus block.

        update log:
            2022-06-26 by Jianhang Wu

            2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

        Parameters
        ----------
            n_elements: int
                Number of stimulus blocks, default is 40.
            rows: int
                Sets the number of stimulus block rows.
            columns: int
                Set the number of stimulus block columns.
           stim_pos: ndarray, shape(x,y)
                自定义刺激块的位置，如果为None，则会以矩形数组排列。
            stim_length: int
                Length of stimulus.
            stim_width: int
                Width of stimulus.

        Raises
        ----------
            Exception: Inconsistent numbers of stimuli and positions

        """

        self.stim_length = stim_length
        self.stim_width = stim_width
        self.n_elements = n_elements
        # highly customizable position matrix
        if (stim_pos is not None) and (self.n_elements == stim_pos.shape[0]):
            # 注意坐标轴的原点应该是屏幕的中心
            #  因此左上角位于第二象限），坐标值越大、
            # 实际位置离中心越远
            self.stim_pos = stim_pos
        # conventional design method
        elif (stim_pos is None) and (rows * columns >= self.n_elements):
            # according to the given rows of columns, coordinates will be
            # automatically converted
            stim_pos = np.zeros((self.n_elements, 2))
            # divide the whole screen into rows*columns' blocks, and pick the
            # center of each block
            first_pos = (                                               #first_pos 计算每个 网格中心点 的坐标。
                np.array([self.win_size[0] / columns, self.win_size[1] / rows]) / 2
            )
            if (first_pos[0] < stim_length /                              #确保刺激块不会重叠，如果单元格太小，抛出异常。
                    2) or (first_pos[1] < stim_width / 2):
                raise Exception(
                    "Too much blocks or too big the stimulus region!")
            for i in range(columns):                                            #按照 columns × rows 的排列方式 计算 刺激块坐标。
                for j in range(rows):
                    stim_pos[i * rows + j] = first_pos + [i, j] * first_pos * 2
            # note that those coordinates are still not the real ones that
            # need to be set on the screen
            stim_pos -= self.win_size / 2  # from Quadrant 1st to 3rd
            stim_pos[:, 1] *= -1  # invert the y-axis        #y 轴翻转，因为 Psychopy 坐标系 y 轴向上，而屏幕 y 轴向下
            self.stim_pos = stim_pos
        else:
            raise Exception("Incorrect number of stimulus!")

        # check size of stimuli
        stim_sizes = np.zeros((self.n_elements, 2))                 #创建 NumPy 数组 记录每个 刺激块的尺寸
        stim_sizes[:] = np.array([stim_length, stim_width])
        self.stim_sizes = stim_sizes
        self.stim_width = stim_width
        self.columns = columns
        self.rows = rows

    def config_text(                                                        #在 Psychopy 视觉刺激界面上添加字符
        self, unit="pix", symbols=None, symbol_height=0, tex_color=[1, 1, 1]
    ):
        """Sets the characters within the stimulus block.

        update log:
            2022-06-26 by Jianhang Wu

            2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

        Parameters
        ----------
            symbols: str
                Edit character text.
            symbol_height: int
                The height of the character in pixels.
            tex_color: list, shape(red, green, blue)
                Set the character color, the value is between -1.0 and 1.0.

        Raises
        ----------
            Exception: Insufficient characters

        """

        # check number of symbols
        # if (symbols is not None) and (len(symbols) >= self.n_elements):
        #     #默认字符集：string.ascii_uppercase（A-Z）1234567890+-*/（数字 & 基本运算符）如果刺激块超过 40 个，但 symbols=None，会报错。
        #     self.symbols = symbols
        # elif self.n_elements <= 40:
        #     self.symbols = "".join([string.ascii_uppercase, "1234567890+-*/"])
        # else:
        #     raise Exception("Please input correct symbol list!")
        print("self.n_elements =", self.n_elements)
        print("symbols =", symbols)
        if (symbols is not None) and (len(symbols) >= self.n_elements):

            self.symbols = symbols

        else:

            default_symbols = [
                "前进",
                "后退",
                "左转",
                "右转",
                "停止",
                "开启",
                "关闭"
            ]

            repeat_times = int(np.ceil(self.n_elements / len(default_symbols)))

            self.symbols = (default_symbols * repeat_times)[:self.n_elements]

        # add text targets onto interface
        if symbol_height == 0:
            symbol_height = self.stim_width / 4  # 汉字建议缩小一点

        self.text_stimuli = []

        for symbol, pos in zip(self.symbols, self.stim_pos):
            self.text_stimuli.append(

                visual.TextStim(
                    win=self.win,
                    text=symbol,

                    # 中文字体（非常重要）
                    font="Microsoft YaHei",

                    pos=pos,
                    color=tex_color,
                    units=unit,

                    height=symbol_height,

                    bold=True,
                    name=symbol,
                )
            )

    def config_response(  # 在线实验
            self,
            symbol_text="Speller:  ",
            symbol_height=0,
            symbol_color=(1, 1, 1),
            bg_color=[-1, -1, -1],
    ):
        """Sets the character of the online response."""

        brige_length = self.win_size[0] / 2 + \
                       self.stim_pos[0][0] - self.stim_length / 2
        brige_width = self.win_size[1] / 2 - \
                      self.stim_pos[0][1] - self.stim_width / 2

        # 保留反馈框对象，但隐藏它，避免后面 draw() 报错
        self.rect_response = visual.Rect(
            win=self.win,
            units="pix",
            width=self.win_size[0] - brige_length,
            height=brige_width * 3 / 3,
            pos=(0, self.win_size[1] / 2 - brige_width / 2),
            fillColor=None,
            lineColor=None,
            opacity=0,
        )

        # 先定义反馈文字位置
        self.res_text_pos = (
            -self.win_size[0] / 2 + brige_length * 3 / 2,
            self.win_size[1] / 2 - brige_width / 2,
        )

        self.reset_res_pos = (
            -self.win_size[0] / 2 + brige_length * 3 / 2,
            self.win_size[1] / 2 - brige_width / 2,
        )

        self.reset_res_text = ">:  "

        # 先定义 symbol_height
        if symbol_height == 0:
            self.symbol_height = brige_width
        else:
            self.symbol_height = symbol_height

        self.symbol_text = symbol_text

        # 只创建一次 text_response，并设置为空和透明
        self.text_response = visual.TextStim(
            win=self.win,
            text="",
            font="Times New Roman",
            pos=self.res_text_pos,
            color=symbol_color,
            units="pix",
            height=self.symbol_height,
            bold=True,
            opacity=0,
        )

        # 绿色反馈倒三角，样式和提示阶段红色倒三角一致
        self.feedback_arrow = visual.TextStim(
            win=self.win,
            text="\u2BC6",
            font="Arial",
            pos=(0, 0),
            color=[-1.0, 1.0, -1.0],
            colorSpace="rgb",
            units="pix",
            height=copy(self.stim_width / 3 * 2),
            bold=True,
            autoLog=False,
        )

# config visual stimuli


class VisualStim(KeyboardInterface):
    """Create various visual stimuli.

    The subclass VisualStim inherits from the parent class KeyboardInterface, duplicate properties are not listed.

    author: Qiaoyi Wu

    Created on: 2022-06-20

    update log:
        2022-06-26 by Jianhang Wu

        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        win:
            The window object.
        colorspace: str
            The color space, default to rgb.
        allowGUI: bool
            Defaults to True, which allows frame-by-frame drawing and key-exit.

    Attributes
    ----------
        index_stimuli:
            Configuration information for the target prompt.

    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        super().__init__(win=win, colorSpace=colorSpace, allowGUI=allowGUI)
        self._exit = threading.Event()

    #win：Psychopy 的 窗口对象，用于显示刺激。

#colorSpace="rgb"：颜色空间（默认 rgb）。

#allowGUI=True：是否允许 GUI 交互（比如键盘退出）。

#self._exit = threading.Event()  用于 线程控制，可能用于 监听键盘输入来终止实验。
    def config_index(self, index_height=0, units="pix"):                #倒三角提示符
        """Config index stimuli: downward triangle (Unicode: \u2BC6)

        Parameters
        ----------
            index_height: int
                The height of the cue symbol, which defaults to half the height of the stimulus block.

        """

        # add index onto interface, with positions to be confirmed.
        if index_height == 0:
            index_height = copy(self.stim_width / 3 * 2)
        self.index_stimuli = visual.TextStim(
            win=self.win,
            text="\u2BC6",
            font="Arial",
            color=[1.0, -1.0, -1.0],
            colorSpace="rgb",
            units=units,
            height=index_height,
            bold=True,
            autoLog=False,
        )



# standard SSVEP paradigm


class SSVEP(VisualStim):
    """Create SSVEP stimuli.

    The subclass SSVEP inherits from the parent class VisualStim, and duplicate properties are not listed.

    author: Qiaoyi Wu

    Created on: 2022-06-20

    update log:
        2022-06-26 by Jianhang Wu

        2022-08-10 by Wei Zhao

        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        win:
            The window object.
        colorspace: str
            The color space, default to rgb.
        allowGUI: bool
            Defaults to True, which allows frame-by-frame drawing and key-exit.

    Attributes
    ----------
        refresh_rate: int
            Screen refresh rate.
        stim_time: float
            Time of stimulus flash
        stim_color: list, shape(red, green, blue)
            The color of the stimulus block, taking values between -1.0 and 1.0.
        stim_opacities: float
            Opacity, default opaque.
        stim_frames: int
            The number of frames contained in a single-trial stimulus.
        stim_oris: ndarray
            Orientation of the stimulus block.
        stim_sfs: ndarray
            Spatial frequency of the stimulus block.
        stim_contrs: ndarray
            Stimulus block contrast.
        freqs: list, shape(fre, …)
            Stimulus block flicker frequency, length consistent with the number of stimulus blocks.
        phases: list, shape(phase, …)
             Stimulus block flicker phase, length consistent with the number of stimulus blocks.
        stim_colors: list, shape(red, green, blue)
            The color configuration required for the stimulus block flashing.
        flash_stimuli:
            The configuration information required for the flashing of the stimulus block.

    Tip
    ----
     .. code-block:: python
        :caption: An example of creating SSVEP stimuli.

        from psychopy import monitors
        import numpy as np
        from brainstim.framework import Experiment
        from brainstim.paradigm import SSVEP,paradigm

        win = ex.get_window()

        # press q to exit paradigm interface
        n_elements, rows, columns = 20, 4, 5
        stim_length, stim_width = 150, 150
        stim_color, tex_color = [1,1,1], [1,1,1]
        fps = 120                                                   # screen refresh rate
        stim_time = 2                                               # stimulus duration
        stim_opacities = 1                                          # stimulus contrast
        freqs = np.arange(8, 16, 0.4)                               # Frequency of instruction
        phases = np.array([i*0.35%2 for i in range(n_elements)])    # Phase of the instruction
        basic_ssvep = SSVEP(win=win)
        basic_ssvep.config_pos(n_elements=n_elements, rows=rows, columns=columns,
            stim_length=stim_length, stim_width=stim_width)
        basic_ssvep.config_text(tex_color=tex_color)
        basic_ssvep.config_color(refresh_rate=fps, stim_time=stim_time, stimtype='sinusoid',
            stim_color=stim_color, stim_opacities=stim_opacities, freqs=freqs, phases=phases)
        basic_ssvep.config_index()
        basic_ssvep.config_response()
        bg_color = np.array([-1, -1, -1])                           # background color
        display_time = 1
        index_time = 0.5
        rest_time = 0.5
        response_time = 1
        port_addr = None 			                                 # Collect host ports
        nrep = 1
        lsl_source_id = None
        online = False
        ex.register_paradigm('basic SSVEP', paradigm, VSObject=basic_ssvep, bg_color=bg_color,
            display_time=display_time,  index_time=index_time, rest_time=rest_time, response_time=response_time,
            port_addr=port_addr, nrep=nrep,  pdim='ssvep', lsl_source_id=lsl_source_id, online=online)

    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        """Item class from VisualStim.

        Args:

        """
        super().__init__(win=win, colorSpace=colorSpace, allowGUI=allowGUI)

    def config_color(
        self,
        refresh_rate,
        stim_time,
        stim_color,
        stimtype="sinusoid",
        stim_opacities=1,
        **kwargs
    ):
        """Config color of stimuli.

        Parameters
        ----------
            refresh_rate: int
                Refresh rate of screen.
            stim_time: float
                Time of each stimulus.
            stim_color: int
                The color of the stimulus block.
            stimtype: str
                Stimulation flicker mode, default to sine sampling flicker.
            stim_opacities: float
                Opacity, default to opaque.
            freqs: list, shape(fre, …)
                Stimulus block flicker frequency, length consistent with the number of stimulus blocks.
            phases: list, shape(phase, …)
                Stimulus block flicker phase, length consistent with the number of stimulus blocks.

        Raises
        ----------
            Exception: Inconsistent frames and color matrices

        """

        # initialize extra inputs
        self.refresh_rate = refresh_rate
        self.stim_time = stim_time
        self.stim_color = stim_color
        self.stim_opacities = stim_opacities
        self.stim_frames = int(stim_time * self.refresh_rate)

        if refresh_rate == 0:
            self.refresh_rate = np.floor(
                self.win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20)
            )

        self.stim_oris = np.zeros((self.n_elements,))  # orientation
        self.stim_sfs = np.zeros((self.n_elements,))  # spatial frequency
        self.stim_contrs = np.ones((self.n_elements,))  # contrast

        # check extra inputs
        if "stim_oris" in kwargs.keys():
            self.stim_oris = kwargs["stim_oris"]
        if "stim_sfs" in kwargs.keys():
            self.stim_sfs = kwargs["stim_sfs"]
        if "stim_contrs" in kwargs.keys():
            self.stim_contrs = kwargs["stim_contrs"]
        if "freqs" in kwargs.keys():
            self.freqs = kwargs["freqs"]
        if "phases" in kwargs.keys():
            self.phases = kwargs["phases"]

        # check consistency
        if stimtype == "sinusoid":
            self.stim_colors = (
                sinusoidal_sample(
                    freqs=self.freqs,
                    phases=self.phases,
                    srate=self.refresh_rate,
                    frames=self.stim_frames,
                    stim_color=stim_color,
                )
                - 1
            )
            if self.stim_colors[0].shape[0] != self.n_elements:
                raise Exception("Please input correct num of stims!")

        incorrect_frame = self.stim_colors.shape[0] != self.stim_frames
        incorrect_number = self.stim_colors.shape[1] != self.n_elements
        if incorrect_frame or incorrect_number:
            raise Exception("Incorrect color matrix or flash frames!")

        # add flashing targets onto interface
        self.flash_stimuli = []
        for sf in range(self.stim_frames):
            self.flash_stimuli.append(
                visual.ElementArrayStim(
                    win=self.win,
                    units="pix",
                    nElements=self.n_elements,
                    sizes=self.stim_sizes,
                    xys=self.stim_pos,
                    colors=self.stim_colors[sf, ...],
                    opacities=self.stim_opacities,
                    oris=self.stim_oris,
                    sfs=self.stim_sfs,
                    contrs=self.stim_contrs,
                    elementTex=np.ones((64, 64)),
                    elementMask=None,
                    texRes=48,
                )
            )




class GetPlabel_MyTherad():
    """
    Open the sub-thread to obtain the online feedback label,
    which is used in the ' con-ssvep ' paradigm. In the traditional
    BCI online experiment, the feedback result is blocked after the
    single trial stimulation, and then the next trial stimulation is started.
    However, the continuous control paradigm does not need to wait for the online
    result, so the sub-thread is opened to receive the online feedback result.

    author: Wei Zhao

    Created on: 2022-07-30

    update log:
        2022-08-10 by Wei Zhao

        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        inlet:
            Stream data online.

    Attributes
    ----------
        inlet:
            pylsl: The data flow realizes the communication between the online数据流实现了在线处理程序和刺激演示程序之间的通信。处理程序和刺激演示程序之间的通信。
            processing program and the stimulus presentation program.
        _exit:
            Make a thread wait for the notification of other threads.  让线程等待其他线程的通知
        online_text_pos: ndarray
            The corresponding position of online prediction results in Speller.在线预测结果在 Speller 中的相应位置。
        online_symbol_text:
            The corresponding letter in Speller for online prediction results.在线预测结果在 Speller 中对应的字母。
        samples: list, shape(label)
            Online processing of predictive labels passed to stimulus programs.   对传递给刺激程序的预测标签进行在线处理。
        predict_id: int
            Online prediction labels.                    在线预测标签。

    Tip
    ----
    .. code-block:: python
       :caption: An example of Opening the sub-thread to receive online feedback results

        MyTherad = GetPlabel_MyTherad(inlet)
        MyTherad.feedbackThread()
        MyTherad.stop_feedbackThread()

    """

    def __init__(self, inlet):
        self.inlet = inlet
        self._exit = threading.Event()          ## 控制线程退出的事件标志

    def feedbackThread(self):
        """Start the thread."""
        self._t_loop = threading.Thread(
            target=self._inner_loop, name="get_predict_id_loop"     #创建一个新线程，并调用 _inner_loop() 处理在线预测 ID。
        )
        self._t_loop.start()

    def _inner_loop(self):
        """The inner loop in the thread."""
        self._exit.clear()
        global online_text_pos, online_symbol_text
        online_text_pos = copy(self.res_text_pos)
        online_symbol_text = copy(self.symbol_text)
        while not self._exit.is_set():                       # 线程循环，直到 `_exit` 触发
            try:
                samples, _ = self.inlet.pull_sample()           # 读取 inlet 数据
                if samples:
                    # online predict id
                    predict_id = int(samples[0]) - 1            # 解析预测 ID
                    online_text_pos = (
                        online_text_pos[0] + self.symbol_height / 3,
                        online_text_pos[1],
                    )                                   # 位置右移
                    online_symbol_text = online_symbol_text + \
                        self.symbols[predict_id]            # 拼接预测字符
            except Exception:
                pass

    def stop_feedbackThread(self):
        """Stop the thread."""
        self._exit.set()
        self._t_loop.join()


# basic experiment control


def paradigm(
    VSObject,
    win,
    bg_color,
    display_time=1.0,
    index_time=1.0,
    rest_time=0.5,
    response_time=2,
    image_time=2,
    port_addr=9045,
    nrep=1,
    pdim="ssvep",
    lsl_source_id=None,
    online=True,
    device_type="NeuroScan",
):
    """
    The classical paradigm is implemented, the task flow is defined, the ' q '
    exit paradigm is clicked, and the start selection interface is returned.

    author: Wei Zhao

    Created on: 2022-07-30

    update log:

        2022-08-10 by Wei Zhao

        2022-08-03 by Shengfu Wen

        2022-12-05 by Jie Mei

        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        VSObject:
            Examples of the three paradigms.
        win:
            window.
        bg_color: ndarray
            Background color.
        fps: int
            Display refresh rate.
        display_time: float
            Keyboard display time before 1st index.
        index_time: float
            Indicator display time.
        rest_time: float, optional
            SSVEP and P300 paradigm: the time interval between the target cue and the start of the stimulus.
            MI paradigm: the time interval between the end of stimulus presentation and the target cue.
        respond_time: float, optional
            Feedback time during online experiment.
        image_time: float, optional,
            MI paradigm: Image time.
        port_addr:
             Computer port , hexadecimal or decimal.
        nrep: int
            Num of blocks.
        pdim: str
            One of the thr
            ee paradigms can be 'ssvep ', ' p300 ', ' mi ' and ' con-ssvep '.
        mi_flag: bool
            Flag of MI paradigm.
        lsl_source_id: str
            The id of communication with the online processing program needs to be consistent between the two parties.
        online: bool
            Flag of online experiment.
        device_type: str
            See support device list in brainstim README file

    """

    if not _check_array_like(bg_color, 3):
        raise ValueError("bg_color should be 3 elements array-like object.")
    win.color = bg_color
    fps = VSObject.refresh_rate

    if device_type == "NeuroScan":
        port = NeuroScanPort(port_addr, use_serial=True) if port_addr else None
    elif device_type == "Neuracle":
        port = NeuraclePort(port_addr) if port_addr else None
    else:
        raise KeyError(
            "Unknown device type: {}, please check your input".format(device_type))
    port_frame = int(0.05 * fps)

    inlet = False
    if online:
        if (
            pdim == "ssvep"
            or pdim == "p300"
            or pdim == "con-ssvep"
            or pdim == "avep"
            or pdim == "ssavep"
        ):
            VSObject.text_response.text = copy(VSObject.reset_res_text)    #符号
            VSObject.text_response.pos = copy(VSObject.reset_res_pos)      #符号位置
            VSObject.res_text_pos = copy(VSObject.reset_res_pos)           #字符位置
            VSObject.symbol_text = copy(VSObject.reset_res_text)           #
            res_text_pos = VSObject.reset_res_pos
        if lsl_source_id:
            inlet = True
            streams = resolve_byprop(
                "source_id", lsl_source_id, timeout=5
            )  # Resolve all streams by source_id

            print(streams)
            if not streams:
                return
            inlet = StreamInlet(streams[0])  # receive stream data
            print(inlet)
            stream_info = inlet.info()  # 关键步骤！
            print("流名称:", stream_info.name())  # 调用 name() 方法
            print("流类型:", stream_info.type())
            print("源ID:", stream_info.source_id())
            print(f"通道数: {stream_info.channel_count()}")
            print(f"数据格式: {stream_info.channel_format()}")

    if pdim == "ssvep":
        # config experiment settings
        conditions = [{"id": i} for i in range(VSObject.n_elements)]
        trials = data.TrialHandler(
            conditions,
            nrep,
            name="experiment",
            method="random")

        # start routine
        # episode 1: display speller interface
        iframe = 0
        while iframe < int(fps * display_time):
            if online:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
            for text_stimulus in VSObject.text_stimuli:
                text_stimulus.draw()
            iframe += 1
            win.flip()

        # episode 2: begin to flash
        if port:
            port.setData(0)
        for trial in trials:
            # quit demo
            keys = event.getKeys(["q"])
            if "q" in keys:
                break

            # initialise index position
            id = int(trial["id"])
            position = VSObject.stim_pos[id] + \
                np.array([0, VSObject.stim_width / 2])
            VSObject.index_stimuli.setPos(position)

            # phase I: speller & index (eye shifting)
            iframe = 0
            while iframe < int(fps * index_time):
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                VSObject.index_stimuli.draw()
                iframe += 1
                win.flip()

            # phase II: rest state
            if rest_time != 0:
                iframe = 0
                while iframe < int(fps * rest_time):
                    if online:
                        VSObject.rect_response.draw()
                        VSObject.text_response.draw()
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    iframe += 1
                    win.flip()

            # phase III: target stimulating
            for sf in range(VSObject.stim_frames):
                if sf == 0 and port and online:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                elif sf == 0 and port:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                if sf == port_frame and port:
                    port.setData(0)
                VSObject.flash_stimuli[sf].draw()
                win.flip()

            # phase IV: respond
            # phase IV: respond
            if inlet:
                # 等待在线预测结果，但最多等 1 秒，防止反馈流异常导致范式卡死
                samples = None
                timestamp = None
                wait_frames = 0
                max_wait_frames = int(fps * 1.0)  # 最多等1秒，可改成2.0

                while samples is None and wait_frames < max_wait_frames:
                    samples, timestamp = inlet.pull_sample(timeout=0.001)

                    # 等待期间显示静止字符界面
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()

                    win.flip()
                    wait_frames += 1

                    # 等待期间也允许按 q 退出
                    keys = event.getKeys(["q"])
                    if "q" in keys:
                        return

                if samples is None:
                    print("本轮未收到在线反馈结果，跳过反馈")
                    continue

                print("收到反馈 samples:", samples)

                try:
                    predict_id = int(samples[0]) - 1
                except Exception as e:
                    print("反馈结果格式错误:", samples, e)
                    continue

                # 防止预测编号越界
                if predict_id < 0 or predict_id >= VSObject.n_elements:
                    print("预测编号越界:", predict_id)
                    continue

                # 绿色箭头放在预测刺激块上方
                arrow_pos = VSObject.stim_pos[predict_id] + np.array(
                    [0, VSObject.stim_width / 2]
                )
                VSObject.feedback_arrow.setPos(arrow_pos)
                # 显示绿色箭头 response_time 秒
                iframe = 0
                while iframe < int(fps * response_time):
                    # 画字符
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()

                    # 画绿色箭头
                    VSObject.feedback_arrow.draw()

                    iframe += 1
                    win.flip()

