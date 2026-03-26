#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打包虚拟筛选 GUI 应用程序
使用 PyInstaller 将应用程序打包成.exe 文件
"""

import os
import sys
from pathlib import Path

# 确保使用正确的 Python 解释器
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 打包命令
def build():
    """使用 PyInstaller 打包应用程序"""
    import subprocess
    
    # 打包命令
    # 设置工作目录和输出目录到 E 盘
    workdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
    distdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist")
    
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--name", "虚拟筛选工具",
        "--onefile",
        "--windowed",
        "--workpath", workdir,
        "--distpath", distdir,
        "--hidden-import", "pandas",
        "--hidden-import", "joblib",
        "--hidden-import", "rdkit",
        "--hidden-import", "numpy",
        "--hidden-import", "tqdm",
        "--hidden-import", "sklearn",
        "--hidden-import", "imblearn",
        "--collect-all", "imblearn",
        "screening_gui.py"
    ]
    
    # 执行打包命令
    print("开始打包应用程序...")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode == 0:
        print("打包成功！")
        print(f"可执行文件位置：{os.path.join('dist', '虚拟筛选工具.exe')}")
    else:
        print("打包失败！")
        sys.exit(1)

if __name__ == "__main__":
    # 检查是否安装了 PyInstaller
    try:
        import PyInstaller
    except ImportError:
        print("请先安装 PyInstaller: pip install pyinstaller")
        sys.exit(1)
    
    # 检查是否存在 icon.ico 文件
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
    if not os.path.exists(icon_path):
        print("警告：未找到 icon.ico 文件，将使用默认图标")
    
    # 执行打包
    build()
