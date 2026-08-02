#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DrugScreen AI 桌面启动器

功能：
1. 启动内嵌Streamlit服务器
2. 自动打开桌面应用窗口 (pywebview)
3. 支持系统托盘最小化
4. 优雅关闭机制

打包后可直接双击exe运行，无需安装Python环境
"""

import os
import sys
import time
import signal
import socket
import threading
import subprocess
import webbrowser
from pathlib import Path


# ============================================================================
# 配置
# ============================================================================

APP_NAME = "DrugScreen AI"
APP_VERSION = "2.0.0"
APP_URL = "http://localhost"
DEFAULT_PORT = 8501
HOST = "127.0.0.1"
STARTUP_WAIT = 3  # 等待服务器启动秒数

# 判断是否为打包环境
def is_frozen():
    """检测是否为PyInstaller打包环境"""
    return getattr(sys, 'frozen', False)

def get_app_root():
    """获取应用根目录"""
    if is_frozen():
        return Path(sys._MEIPASS)
    return Path(__file__).parent.resolve()

def get_user_data_dir():
    """获取用户数据目录 (打包后用于存放结果)"""
    if is_frozen():
        # 打包后使用用户目录
        home = Path.home()
        data_dir = home / "DrugScreenAI"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (data_dir / "data").mkdir(exist_ok=True)
        (data_dir / "results").mkdir(exist_ok=True)
        (data_dir / "results" / "logs").mkdir(exist_ok=True)
        (data_dir / "temp").mkdir(exist_ok=True)
        
        return data_dir
    else:
        return Path(__file__).parent.resolve()


# ============================================================================
# 端口管理
# ============================================================================

def find_free_port(start_port=DEFAULT_PORT, max_tries=20):
    """查找可用端口"""
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, port))
                return port
        except OSError:
            continue
    return start_port


def wait_for_server(port, timeout=30):
    """等待服务器启动"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect((HOST, port))
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(0.5)
    return False


# ============================================================================
# Streamlit 服务器管理
# ============================================================================

class StreamlitServer:
    """Streamlit服务器管理器"""
    
    def __init__(self, port=DEFAULT_PORT):
        self.port = port
        self.process = None
        self.app_path = None
        self.user_data_dir = get_user_data_dir()
        
    def _get_streamlit_executable(self):
        """获取streamlit可执行文件路径"""
        if is_frozen():
            # 打包环境: 使用内嵌的streamlit
            return [sys.executable, "-m", "streamlit"]
        else:
            # 开发环境
            return [sys.executable, "-m", "streamlit"]
    
    def _get_app_path(self):
        """获取app.py路径"""
        if is_frozen():
            return str(get_app_root() / "app.py")
        else:
            return str(Path(__file__).parent / "app.py")
    
    def _get_env(self):
        """获取环境变量"""
        env = os.environ.copy()
        
        # 设置用户数据目录作为工作目录
        env['DRUGSCREEN_DATA_DIR'] = str(self.user_data_dir / "data")
        env['DRUGSCREEN_RESULTS_DIR'] = str(self.user_data_dir / "results")
        env['DRUGSCREEN_LOGS_DIR'] = str(self.user_data_dir / "results" / "logs")
        
        # Streamlit配置
        env['STREAMLIT_SERVER_HEADLESS'] = 'true'
        env['STREAMLIT_SERVER_PORT'] = str(self.port)
        env['STREAMLIT_SERVER_ADDRESS'] = HOST
        env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        env['STREAMLIT_BROWSER_SERVER_ADDRESS'] = f'{HOST}:{self.port}'
        
        return env
    
    def start(self):
        """启动Streamlit服务器"""
        self.port = find_free_port()
        self.app_path = self._get_app_path()
        
        if not Path(self.app_path).exists():
            print(f"错误: 找不到应用文件 {self.app_path}")
            return False
        
        cmd = self._get_streamlit_executable()
        cmd.extend([
            "run", self.app_path,
            "--server.headless", "true",
            "--server.port", str(self.port),
            "--server.address", HOST,
            "--browser.gatherUsageStats", "false",
            "--global.developmentMode", "false"
        ])
        
        env = self._get_env()
        
        # 启动进程
        try:
            creation_flags = 0
            if sys.platform == 'win32':
                creation_flags = subprocess.CREATE_NO_WINDOW
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                creationflags=creation_flags
            )
            
            # 等待服务器启动
            if wait_for_server(self.port, timeout=30):
                print(f"服务器已启动: http://{HOST}:{self.port}")
                return True
            else:
                print("服务器启动超时")
                return False
                
        except Exception as e:
            print(f"启动服务器失败: {e}")
            return False
    
    def stop(self):
        """停止Streamlit服务器"""
        if self.process:
            try:
                if sys.platform == 'win32':
                    self.process.terminate()
                else:
                    self.process.send_signal(signal.SIGTERM)
                
                # 等待进程结束
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                
                print("服务器已停止")
            except Exception as e:
                print(f"停止服务器时出错: {e}")
            finally:
                self.process = None
    
    def get_url(self):
        """获取服务器URL"""
        return f"http://{HOST}:{self.port}"


# ============================================================================
# 桌面窗口 (pywebview)
# ============================================================================

def open_desktop_window(url, title=APP_NAME, width=1400, height=900):
    """
    使用pywebview打开桌面窗口
    
    如果pywebview不可用，则回退到系统浏览器
    """
    try:
        import webview
        webview.create_window(
            title=title,
            url=url,
            width=width,
            height=height,
            min_size=(800, 600),
            text_select=True
        )
        webview.start()
        return True
    except ImportError:
        print("pywebview未安装，使用系统浏览器")
        webbrowser.open(url)
        return False
    except Exception as e:
        print(f"桌面窗口启动失败: {e}，回退到浏览器")
        webbrowser.open(url)
        return False


# ============================================================================
# 主启动流程
# ============================================================================

def main():
    """主启动函数"""
    print(f"\n{'='*60}")
    print(f"  {APP_NAME} v{APP_VERSION}")
    print(f"  智能药物虚拟筛选平台")
    print(f"{'='*60}\n")
    
    # 获取用户数据目录
    user_data_dir = get_user_data_dir()
    print(f"用户数据目录: {user_data_dir}")
    
    # 初始化服务器
    server = StreamlitServer()
    
    # 启动服务器
    print("\n正在启动应用服务器...")
    if not server.start():
        print("\n服务器启动失败！")
        input("按回车键退出...")
        return 1
    
    url = server.get_url()
    print(f"\n应用地址: {url}")
    
    # 启动桌面窗口
    print("\n正在打开应用窗口...")
    
    # 在单独的线程中打开浏览器作为备份
    def delayed_browser_open():
        time.sleep(STARTUP_WAIT)
        try:
            webbrowser.open(url)
        except:
            pass
    
    # 尝试使用pywebview
    try:
        import webview
        # pywebview可用，使用桌面窗口
        browser_thread = threading.Thread(target=delayed_browser_open, daemon=True)
        browser_thread.start()
        
        open_desktop_window(url)
        
        # 窗口关闭后停止服务器
        print("\n应用窗口已关闭，正在停止服务器...")
        server.stop()
        
    except ImportError:
        # pywebview不可用，使用系统浏览器
        print("提示: 安装 pywebview 可获得更好的桌面体验")
        print("      pip install pywebview")
        print(f"\n在浏览器中访问: {url}")
        print("\n按 Ctrl+C 停止服务器并退出...")
        
        try:
            # 保持服务器运行
            while server.process and server.process.poll() is None:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n正在停止服务器...")
            server.stop()
    
    print(f"\n感谢使用 {APP_NAME}！")
    return 0


# ============================================================================
# 单实例锁
# ============================================================================

def check_single_instance():
    """检查是否已有实例运行"""
    lock_file = get_user_data_dir() / ".lock"
    
    try:
        # 尝试创建锁文件
        if lock_file.exists():
            # 检查锁文件是否过期 (超过1小时)
            import datetime
            mtime = datetime.datetime.fromtimestamp(lock_file.stat().st_mtime)
            if datetime.datetime.now() - mtime > datetime.timedelta(hours=1):
                lock_file.unlink()
            else:
                print(f"\n{APP_NAME} 已经在运行中！")
                print(f"请关闭已有实例后再启动。")
                input("按回车键退出...")
                return False
        
        # 创建锁文件
        lock_file.write_text(str(os.getpid()))
        
        # 注册退出时删除锁文件
        import atexit
        atexit.register(lambda: lock_file.unlink(missing_ok=True))
        
        return True
        
    except Exception:
        # 锁机制失败时允许启动
        return True


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    # 检查单实例
    if not check_single_instance():
        sys.exit(1)
    
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n运行时错误: {e}")
        import traceback
        traceback.print_exc()
        input("\n按回车键退出...")
        sys.exit(1)
