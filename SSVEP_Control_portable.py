import os
import sys
import subprocess

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QPlainTextEdit,
)
from PyQt5.QtCore import QProcess, QProcessEnvironment


# ============================================================
# 便携版路径设置
# ============================================================
# 开发运行时：
#     BASE_DIR = 当前 123.py 所在目录
#
# 打包运行时：
#     BASE_DIR = SSVEP_Control.exe 所在目录
#
# 推荐发布目录：
#     SSVEP_Control/
#     ├── SSVEP_Control.exe
#     ├── runtime/
#     │   └── python.exe
#     └── MetaBCI6/
#         ├── metabci/
#         └── demos/
# ============================================================

def get_base_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = get_base_dir()


def run_hidden(args, **kwargs):
    startupinfo = None
    creationflags = 0

    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    return subprocess.run(
        args,
        startupinfo=startupinfo,
        creationflags=creationflags,
        **kwargs
    )


def find_python_exe():
    """
    优先使用便携 runtime/python.exe。
    如果当前还是源码调试，没有 runtime，就使用当前 Python 环境。
    """
    portable_python = os.path.join(BASE_DIR, "runtime", "python.exe")

    if os.path.exists(portable_python):
        return portable_python

    return sys.executable


def find_project_dir():
    """
    兼容两种目录：
    1. 打包发布：
       BASE_DIR/MetaBCI6/metabci
    2. 源码调试：
       BASE_DIR/metabci
    """
    packaged_project = os.path.join(BASE_DIR, "MetaBCI6")
    dev_project = BASE_DIR

    if os.path.exists(os.path.join(packaged_project, "metabci")):
        return packaged_project

    if os.path.exists(os.path.join(dev_project, "metabci")):
        return dev_project

    # 默认按发布目录结构返回，后续路径检查会提示不存在
    return packaged_project


PYTHON_EXE = find_python_exe()
PROJECT_DIR = find_project_dir()

STIM_SCRIPT = os.path.join(
    PROJECT_DIR,
    "demos",
    "brainstim_demos",
    "stim_demo.py"
)

STIM_WORKDIR = os.path.join(
    PROJECT_DIR,
    "demos",
    "brainstim_demos"
)

ONLINE_SCRIPT = os.path.join(
    PROJECT_DIR,
    "demos",
    "brainflow_demos",
    "online nano.py"
)

ONLINE_WORKDIR = os.path.join(
    PROJECT_DIR,
    "demos",
    "brainflow_demos"
)

CHECK_SCRIPT = os.path.join(
    PROJECT_DIR,
    "demos",
    "brainflow_demos",
    "check_validation.py"
)

CHECK_WORKDIR = ONLINE_WORKDIR

DEFAULT_BDF = os.path.join(BASE_DIR, "data", "7.bdf")
if not os.path.exists(DEFAULT_BDF):
    DEFAULT_BDF = r"D:\0425\7.bdf"


class ControlWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SSVEP 离线 / 在线控制界面")

        self.stim_process = QProcess(self)
        self.online_process = QProcess(self)
        self.check_process = QProcess(self)

        layout = QVBoxLayout()

        # =========================
        # 打标端口号
        # =========================
        self.port_edit = QLineEdit("COM11")
        layout.addLayout(self.row("打标端口号", self.port_edit))

        # =========================
        # 离线训练 BDF 文件
        # =========================
        self.train_bdf_edit = QLineEdit(DEFAULT_BDF)
        self.choose_bdf_btn = QPushButton("选择BDF文件")
        self.choose_bdf_btn.clicked.connect(self.choose_train_bdf)

        bdf_row = QHBoxLayout()
        bdf_row.addWidget(QLabel("离线训练BDF"))
        bdf_row.addWidget(self.train_bdf_edit)
        bdf_row.addWidget(self.choose_bdf_btn)
        layout.addLayout(bdf_row)

        # =========================
        # 按钮
        # =========================
        self.offline_btn = QPushButton("启动离线范式")
        self.offline_btn.clicked.connect(self.start_offline)

        self.check_btn = QPushButton("验证训练效果")
        self.check_btn.clicked.connect(self.start_check_validation)

        self.online_worker_btn = QPushButton("启动在线处理")
        self.online_worker_btn.clicked.connect(self.start_online_worker)

        self.online_stim_btn = QPushButton("启动在线范式")
        self.online_stim_btn.clicked.connect(self.start_online_stim)

        self.stop_btn = QPushButton("停止全部")
        self.stop_btn.clicked.connect(self.stop_all)

        layout.addWidget(self.offline_btn)
        layout.addWidget(self.check_btn)
        layout.addWidget(self.online_worker_btn)
        layout.addWidget(self.online_stim_btn)
        layout.addWidget(self.stop_btn)

        # =========================
        # 日志窗口
        # =========================
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)

        layout.addWidget(QLabel("运行日志"))
        layout.addWidget(self.log_box)

        self.setLayout(layout)

        self.log("当前运行目录 BASE_DIR: " + BASE_DIR)
        self.log("当前 Python 解释器: " + PYTHON_EXE)
        self.log("当前 MetaBCI 项目目录: " + PROJECT_DIR)

    def row(self, label, widget):
        box = QHBoxLayout()
        box.addWidget(QLabel(label))
        box.addWidget(widget)
        return box

    def log(self, text):
        self.log_box.appendPlainText(str(text))
        print(text)

    def get_port(self):
        port = self.port_edit.text().strip()
        if port.lower() in ["", "none", "null"]:
            return "None"
        return port

    def choose_train_bdf(self):
        current_path = self.train_bdf_edit.text().strip()
        if os.path.exists(current_path):
            start_dir = os.path.dirname(current_path)
        elif os.path.exists(os.path.join(BASE_DIR, "data")):
            start_dir = os.path.join(BASE_DIR, "data")
        else:
            start_dir = r"D:\0425"

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择离线训练BDF文件",
            start_dir,
            "BDF Files (*.bdf);;All Files (*)"
        )

        if file_path:
            self.train_bdf_edit.setText(file_path)
            self.log(f"已选择离线训练文件: {file_path}")

    def connect_process_output(self, process, name):
        process.readyReadStandardOutput.connect(
            lambda: self.read_process_output(process, name, error=False)
        )
        process.readyReadStandardError.connect(
            lambda: self.read_process_output(process, name, error=True)
        )

    def read_process_output(self, process, name, error=False):
        if error:
            data = process.readAllStandardError()
        else:
            data = process.readAllStandardOutput()

        raw = bytes(data)

        # 优先 utf-8，失败再 gbk
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            text = raw.decode("gbk", errors="ignore")

        text = text.strip()
        if text:
            prefix = f"[{name} ERROR]" if error else f"[{name}]"
            self.log(f"{prefix} {text}")

    def check_python_exe(self):
        if not os.path.exists(PYTHON_EXE):
            self.log(f"Python解释器不存在: {PYTHON_EXE}")
            self.log("请确认 runtime\\python.exe 是否存在，或者当前开发环境是否正确。")
            return False
        return True

    def prepare_process(self, process, workdir):
        """
        设置工作目录和 PYTHONPATH。
        这样即使 MetaBCI 没有 pip install -e，也能从 PROJECT_DIR 导入 metabci。
        """
        process.setWorkingDirectory(workdir)

        env = QProcessEnvironment.systemEnvironment()

        old_pythonpath = env.value("PYTHONPATH")
        if old_pythonpath:
            new_pythonpath = PROJECT_DIR + os.pathsep + old_pythonpath
        else:
            new_pythonpath = PROJECT_DIR

        env.insert("PYTHONPATH", new_pythonpath)

        # 让 Python 子进程输出尽量实时刷新
        env.insert("PYTHONUNBUFFERED", "1")

        process.setProcessEnvironment(env)

    # =========================
    # 启动离线范式
    # =========================
    def start_offline(self):
        if not self.check_python_exe():
            return

        if not os.path.exists(STIM_SCRIPT):
            self.log(f"范式脚本不存在: {STIM_SCRIPT}")
            return

        if self.stim_process.state() != QProcess.NotRunning:
            self.log("范式已经在运行，请先停止")
            return

        port = self.get_port()

        self.stim_process = QProcess(self)
        self.prepare_process(self.stim_process, STIM_WORKDIR)
        self.connect_process_output(self.stim_process, "离线范式")

        self.stim_process.start(
            PYTHON_EXE,
            [STIM_SCRIPT, "offline", port]
        )

        self.log("已启动离线范式")
        self.log(f"执行命令: {PYTHON_EXE} {STIM_SCRIPT} offline {port}")

    # =========================
    # 验证训练效果
    # =========================
    def start_check_validation(self):
        """
        离线验证：
        python check_validation.py 离线训练BDF
        """
        if not self.check_python_exe():
            return

        if not os.path.exists(CHECK_SCRIPT):
            self.log(f"离线验证脚本不存在: {CHECK_SCRIPT}")
            self.log("请先把 check_validation.py 放到 demos\\brainflow_demos 目录")
            return

        train_bdf = self.train_bdf_edit.text().strip()

        if not os.path.exists(train_bdf):
            self.log(f"离线训练BDF文件不存在: {train_bdf}")
            return

        if self.check_process.state() != QProcess.NotRunning:
            self.log("离线验证已经在运行")
            return

        self.check_process = QProcess(self)
        self.prepare_process(self.check_process, CHECK_WORKDIR)
        self.connect_process_output(self.check_process, "训练验证")

        self.check_process.finished.connect(
            lambda exit_code, exit_status: self.log(f"训练验证进程结束，退出码: {exit_code}")
        )

        self.check_process.start(
            PYTHON_EXE,
            [CHECK_SCRIPT, train_bdf]
        )

        self.log("已启动训练效果验证")
        self.log(f"执行命令: {PYTHON_EXE} {CHECK_SCRIPT} {train_bdf}")

    # =========================
    # 启动在线处理
    # =========================
    def start_online_worker(self):
        if not self.check_python_exe():
            return

        if not os.path.exists(ONLINE_SCRIPT):
            self.log(f"在线分类脚本不存在: {ONLINE_SCRIPT}")
            return

        train_bdf = self.train_bdf_edit.text().strip()

        if not os.path.exists(train_bdf):
            self.log(f"离线训练BDF文件不存在: {train_bdf}")
            return

        if self.online_process.state() != QProcess.NotRunning:
            self.log("在线处理已经在运行")
            return

        self.online_process = QProcess(self)
        self.prepare_process(self.online_process, ONLINE_WORKDIR)
        self.connect_process_output(self.online_process, "在线处理")

        self.online_process.start(
            PYTHON_EXE,
            [ONLINE_SCRIPT, train_bdf]
        )

        self.log("已启动在线处理程序")
        self.log(f"执行命令: {PYTHON_EXE} {ONLINE_SCRIPT} {train_bdf}")
        self.log("等待在线处理加载训练文件并完成模型训练后，再点击“启动在线范式”")

    # =========================
    # 启动在线范式
    # =========================
    def start_online_stim(self):
        if not self.check_python_exe():
            return

        if not os.path.exists(STIM_SCRIPT):
            self.log(f"范式脚本不存在: {STIM_SCRIPT}")
            return

        if self.stim_process.state() != QProcess.NotRunning:
            self.log("范式已经在运行，请先停止")
            return

        if self.online_process.state() == QProcess.NotRunning:
            self.log("提示：在线处理程序没有运行，请先点击“启动在线处理”")
            return

        port = self.get_port()

        self.stim_process = QProcess(self)
        self.prepare_process(self.stim_process, STIM_WORKDIR)
        self.connect_process_output(self.stim_process, "在线范式")

        self.stim_process.start(
            PYTHON_EXE,
            [STIM_SCRIPT, "online", port]
        )

        self.log("已启动在线范式")
        self.log(f"执行命令: {PYTHON_EXE} {STIM_SCRIPT} online {port}")

    # =========================
    # 停止当前 QProcess 进程树
    # =========================
    def kill_qprocess_tree(self, process, name):
        if process is None:
            return

        if process.state() == QProcess.NotRunning:
            return

        pid = int(process.processId())
        self.log(f"正在停止{name}，PID={pid}")

        process.terminate()

        if not process.waitForFinished(3000):
            self.log(f"{name}未正常退出，强制结束进程树 PID={pid}")

            run_hidden(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="gbk",
                errors="ignore"
            )

        process.waitForFinished(3000)
        self.log(f"{name}已停止")

    # =========================
    # 按脚本名清理残留 Python 进程
    # =========================
    def kill_python_by_keyword(self, keyword):
        keyword = keyword.replace("'", "''")

        ps_cmd = f"""
        Get-CimInstance Win32_Process |
        Where-Object {{
            ($_.Name -eq 'python.exe' -or $_.Name -eq 'pythonw.exe') -and
            ($_.CommandLine -like '*{keyword}*')
        }} |
        ForEach-Object {{
            Write-Output ('Kill PID=' + $_.ProcessId + ' CMD=' + $_.CommandLine)
            Stop-Process -Id $_.ProcessId -Force
        }}
        """

        result = run_hidden(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="gbk",
            errors="ignore"
        )

        if result.stdout.strip():
            self.log(result.stdout.strip())

        if result.stderr.strip():
            self.log(result.stderr.strip())

    # =========================
    # 清理旧进程
    # =========================
    def cleanup_old_processes(self):
        self.log("开始清理旧 Python 进程...")

        self.kill_qprocess_tree(self.stim_process, "范式程序")
        self.kill_qprocess_tree(self.online_process, "在线处理程序")
        self.kill_qprocess_tree(self.check_process, "训练验证程序")

        self.kill_python_by_keyword("stim_demo.py")
        self.kill_python_by_keyword("online nano.py")
        self.kill_python_by_keyword("check_validation.py")

        self.log("旧进程清理完成")

    # =========================
    # 停止全部：同时执行清理
    # =========================
    def stop_all(self):
        self.cleanup_old_processes()
        self.log("已停止全部进程")

    # =========================
    # 关闭窗口时也自动清理
    # =========================
    def closeEvent(self, event):
        self.cleanup_old_processes()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ControlWindow()
    win.resize(820, 520)
    win.show()
    sys.exit(app.exec_())
