#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DrugScreen AI - 桌面应用入口 (pywebview)

提供原生桌面窗口体验，内嵌Streamlit服务

功能：
1. 启动内嵌Streamlit服务器
2. 使用pywebview创建原生窗口
3. 自动端口管理
4. 优雅关闭机制
5. 系统托盘支持 (可选)

使用方法：
  python desktop_app.py              # 启动桌面应用
  python desktop_app.py --browser    # 使用浏览器模式
  python desktop_app.py --port 8888  # 指定端口
"""

import os
import sys
import time
import signal
import socket
import threading
import subprocess
import webbrowser
import argparse
from pathlib import Path
from datetime import datetime


# ============================================================================
# 配置
# ============================================================================

APP_NAME = "DrugScreen AI"
APP_VERSION = "2.0.0"
DEFAULT_PORT = 8501
HOST = "127.0.0.1"
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
WINDOW_MIN_WIDTH = 800
WINDOW_MIN_HEIGHT = 600


def is_frozen():
    """检测是否为PyInstaller打包环境"""
    return getattr(sys, 'frozen', False)


def get_app_root():
    """获取应用根目录"""
    if is_frozen():
        return Path(sys._MEIPASS)
    return Path(__file__).parent.resolve()


def get_user_data_dir():
    """获取用户数据目录"""
    if is_frozen():
        home = Path.home() / "DrugScreenAI"
    else:
        home = Path(__file__).parent.resolve()
    
    # 创建目录结构
    for subdir in ['data', 'results', 'results/logs', 'temp', 'cache']:
        (home / subdir).mkdir(parents=True, exist_ok=True)
    
    return home


# ============================================================================
# 端口管理
# ============================================================================

def find_free_port(start_port=DEFAULT_PORT, max_tries=30):
    """查找可用端口"""
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, port))
                return port
        except OSError:
            continue
    return start_port


def wait_for_server(port, timeout=60):
    """等待服务器响应"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect((HOST, port))
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(0.3)
    return False


# ============================================================================
# Streamlit 服务器
# ============================================================================

class StreamlitServer:
    """Streamlit服务器管理"""
    
    def __init__(self, port=DEFAULT_PORT):
        self.port = port
        self.process = None
        self.app_root = get_app_root()
        self.user_data_dir = get_user_data_dir()
    
    def start(self):
        """启动Streamlit服务器"""
        self.port = find_free_port()
        app_path = self.app_root / "app.py"
        
        if not app_path.exists():
            print(f"错误: 找不到应用文件 {app_path}")
            return False
        
        # 构建命令
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.headless", "true",
            "--server.port", str(self.port),
            "--server.address", HOST,
            "--browser.gatherUsageStats", "false",
            "--global.developmentMode", "false",
            "--server.maxUploadSize", "500",
        ]
        
        # 环境变量
        env = os.environ.copy()
        env['DRUGSCREEN_ROOT'] = str(self.app_root)
        env['DRUGSCREEN_DATA_DIR'] = str(self.user_data_dir / "data")
        env['DRUGSCREEN_RESULTS_DIR'] = str(self.user_data_dir / "results")
        env['DRUGSCREEN_LOGS_DIR'] = str(self.user_data_dir / "results" / "logs")
        env['STREAMLIT_SERVER_HEADLESS'] = 'true'
        
        try:
            # Windows隐藏控制台窗口
            creation_flags = 0
            if sys.platform == 'win32':
                creation_flags = subprocess.CREATE_NO_WINDOW
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                creationflags=creation_flags,
                cwd=str(self.app_root)
            )
            
            print(f"等待服务器启动 (端口 {self.port})...")
            
            if wait_for_server(self.port, timeout=60):
                print(f"服务器已就绪: http://{HOST}:{self.port}")
                return True
            else:
                # 读取错误输出
                stderr_output = ""
                if self.process.stderr:
                    stderr_output = self.process.stderr.read().decode('utf-8', errors='ignore')
                print(f"服务器启动超时")
                if stderr_output:
                    print(f"错误输出:\n{stderr_output[:500]}")
                return False
                
        except Exception as e:
            print(f"启动服务器失败: {e}")
            return False
    
    def stop(self):
        """停止服务器"""
        if self.process:
            try:
                if sys.platform == 'win32':
                    # Windows: 先尝试正常终止
                    self.process.terminate()
                else:
                    self.process.send_signal(signal.SIGTERM)
                
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                
                print("服务器已停止")
            except Exception as e:
                print(f"停止服务器出错: {e}")
            finally:
                self.process = None
    
    @property
    def url(self):
        return f"http://{HOST}:{self.port}"


# ============================================================================
# 桌面窗口 (pywebview)
# ============================================================================

class DesktopApp:
    """桌面应用窗口管理"""
    
    def __init__(self, url, use_browser=False):
        self.url = url
        self.use_browser = use_browser
        self.window = None
    
    def run(self):
        """运行桌面应用"""
        if self.use_browser:
            return self._run_browser()
        
        try:
            import webview
            return self._run_webview(webview)
        except ImportError:
            print("pywebview未安装，使用浏览器模式")
            return self._run_browser()
        except Exception as e:
            print(f"桌面窗口启动失败: {e}")
            return self._run_browser()
    
    def _run_webview(self, webview):
        """使用pywebview创建原生窗口"""
        
        # 窗口事件回调
        def on_closing():
            print("应用窗口正在关闭...")
        
        def on_loaded():
            print("应用已加载完成")
        
        # 创建窗口
        self.window = webview.create_window(
            title=f"{APP_NAME} v{APP_VERSION}",
            url=self.url,
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
            min_size=(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT),
            text_select=True,
            easy_drag=False,
            on_top=False,
            confirm_close=False,
        )
        
        # 设置事件
        self.window.events.closing += on_closing
        self.window.events.loaded += on_loaded
        
        # 启动webview
        # debug=False 避免打开开发者工具
        webview.start(
            debug=False,
            http_server=False,
            private_mode=False,  # 允许使用本地存储
        )
        
        return True
    
    def _run_browser(self):
        """使用系统浏览器"""
        print(f"\n在浏览器中打开: {self.url}")
        print("按 Ctrl+C 退出应用\n")
        
        # 延迟打开浏览器
        time.sleep(1)
        webbrowser.open(self.url)
        
        # 保持运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n正在退出...")
        
        return True


# ============================================================================
# 应用图标
# ============================================================================

def get_app_icon():
    """获取应用图标路径"""
    icon_paths = [
        get_app_root() / "assets" / "icon.ico",
        get_app_root() / "assets" / "icon.png",
        get_app_root() / "icon.ico",
        get_app_root() / "icon.png",
    ]
    
    for path in icon_paths:
        if path.exists():
            return str(path)
    return None


# ============================================================================
# 单实例检查
# ============================================================================

def check_single_instance():
    """确保只有一个应用实例运行"""
    lock_file = get_user_data_dir() / ".app.lock"
    
    try:
        if lock_file.exists():
            # 检查PID是否还在运行
            old_pid = lock_file.read_text().strip()
            if old_pid:
                try:
                    if sys.platform == 'win32':
                        # Windows: 检查进程是否存在
                        import ctypes
                        kernel32 = ctypes.windll.kernel32
                        PROCESS_QUERY_INFORMATION = 0x0400
                        handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, int(old_pid))
                        if handle:
                            kernel32.CloseHandle(handle)
                            print(f"\n{APP_NAME} 已经在运行中 (PID: {old_pid})")
                            return False
                except:
                    pass
            lock_file.unlink(missing_ok=True)
        
        # 创建锁文件
        lock_file.write_text(str(os.getpid()))
        
        import atexit
        atexit.register(lambda: lock_file.unlink(missing_ok=True))
        
        return True
    except Exception:
        return True


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=f'{APP_NAME} 桌面应用')
    parser.add_argument('--browser', action='store_true',
                       help='使用系统浏览器而非桌面窗口')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                       help=f'服务器端口 (默认: {DEFAULT_PORT})')
    parser.add_argument('--no-single', action='store_true',
                       help='允许多实例运行')
    args = parser.parse_args()
    
    # 打印启动信息
    print(f"\n{'='*60}")
    print(f"  {APP_NAME} v{APP_VERSION}")
    print(f"  智能药物虚拟筛选平台")
    print(f"{'='*60}")
    
    # 用户数据目录
    user_data = get_user_data_dir()
    print(f"  数据目录: {user_data}")
    print(f"  模式: {'浏览器' if args.browser else '桌面窗口'}")
    print(f"{'='*60}\n")
    
    # 单实例检查
    if not args.no_single:
        if not check_single_instance():
            input("按回车键退出...")
            return 1
    
    # 启动服务器
    server = StreamlitServer(port=args.port)
    
    print("正在启动应用服务器...")
    if not server.start():
        print("\n服务器启动失败！")
        input("按回车键退出...")
        return 1
    
    # 启动桌面窗口
    print(f"\n正在打开{APP_NAME}...")
    app = DesktopApp(server.url, use_browser=args.browser)
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n接收到退出信号...")
    finally:
        print("正在关闭服务器...")
        server.stop()
    
    print(f"\n感谢使用 {APP_NAME}！")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n运行时错误: {e}")
        import traceback
        traceback.print_exc()
        input("\n按回车键退出...")
        sys.exit(1)
