#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DrugScreen AI - 一键构建脚本

自动完成以下步骤：
1. 检查/安装依赖
2. 清理旧构建文件
3. 使用PyInstaller打包
4. 复制额外资源文件
5. 创建便携版目录结构

使用方法:
  python build_exe.py              # 完整构建
  python build_exe.py --clean      # 仅清理
  python build_exe.py --no-clean   # 跳过清理
  python build_exe.py --onedir     # 目录模式(启动更快)
  python build_exe.py --onefile    # 单文件模式(默认)
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

# ============================================================================
# 配置
# ============================================================================

APP_NAME = "DrugScreenAI"
APP_VERSION = "2.0.0"
PROJECT_ROOT = Path(__file__).parent.resolve()

# 构建目录
BUILD_DIR = PROJECT_ROOT / "build"
DIST_DIR = PROJECT_ROOT / "dist"
SPEC_FILE = PROJECT_ROOT / "drugscreen.spec"

# 需要复制到dist的额外文件
EXTRA_FILES = [
    "configs/settings.yaml",
    "configs/config.py",
    "configs/config_loader.py",
    "requirements.txt",
]

# 需要复制到dist的目录
EXTRA_DIRS = [
    "data",
]

# 必须的Python包
REQUIRED_PACKAGES = [
    "pyinstaller",
    "streamlit",
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "plotly",
    "pyyaml",
    "joblib",
    "pillow",
    "pywebview",
]


# ============================================================================
# 工具函数
# ============================================================================

def run_command(cmd, cwd=None, check=True):
    """运行命令"""
    print(f"  > {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(
        cmd,
        cwd=cwd or PROJECT_ROOT,
        shell=isinstance(cmd, str),
        capture_output=False
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"命令执行失败: {cmd}")
    return result.returncode


def check_package(package_name):
    """检查包是否已安装"""
    try:
        __import__(package_name.replace('-', '_').split('>=')[0].split('==')[0])
        return True
    except ImportError:
        return False


def install_package(package_name):
    """安装Python包"""
    print(f"  安装: {package_name}")
    run_command([sys.executable, "-m", "pip", "install", package_name])


def ensure_packages():
    """确保所有必需包已安装"""
    print("\n[1/5] 检查依赖包...")
    
    missing = []
    for pkg in REQUIRED_PACKAGES:
        import_name = pkg.replace('-', '_')
        if not check_package(import_name):
            missing.append(pkg)
    
    if missing:
        print(f"  缺少包: {', '.join(missing)}")
        for pkg in missing:
            install_package(pkg)
        print("  依赖安装完成")
    else:
        print("  所有依赖已安装 ✓")


def clean_build():
    """清理旧构建文件"""
    print("\n[2/5] 清理旧构建文件...")
    
    dirs_to_clean = [BUILD_DIR, DIST_DIR, PROJECT_ROOT / "__pycache__"]
    
    for d in dirs_to_clean:
        if d.exists():
            print(f"  删除: {d}")
            shutil.rmtree(d, ignore_errors=True)
    
    # 清理src和configs中的__pycache__
    for pycache in PROJECT_ROOT.rglob("__pycache__"):
        if "dist" not in str(pycache) and "build" not in str(pycache):
            shutil.rmtree(pycache, ignore_errors=True)
    
    # 清理.spec备份
    for spec_backup in PROJECT_ROOT.glob("*.spec.bak"):
        spec_backup.unlink(missing_ok=True)
    
    print("  清理完成 ✓")


def build_exe(mode="onefile"):
    """使用PyInstaller构建exe"""
    print(f"\n[3/5] 构建EXE ({mode}模式)...")
    
    if not SPEC_FILE.exists():
        print(f"  错误: spec文件不存在 {SPEC_FILE}")
        return False
    
    # 使用spec文件构建
    cmd = [
        sys.executable, "-m", "PyInstaller",
        str(SPEC_FILE),
        "--noconfirm",
        "--clean",
    ]
    
    try:
        run_command(cmd)
        print("  构建完成 ✓")
        return True
    except RuntimeError as e:
        print(f"  构建失败: {e}")
        return False


def copy_extra_files():
    """复制额外文件到dist目录"""
    print("\n[4/5] 复制资源文件...")
    
    if not DIST_DIR.exists():
        print(f"  错误: dist目录不存在 {DIST_DIR}")
        return False
    
    # 复制单文件
    for file_path in EXTRA_FILES:
        src = PROJECT_ROOT / file_path
        if src.exists():
            dst = DIST_DIR / file_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  复制: {file_path}")
    
    # 复制目录
    for dir_path in EXTRA_DIRS:
        src = PROJECT_ROOT / dir_path
        if src.exists() and src.is_dir():
            dst = DIST_DIR / dir_path
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  复制目录: {dir_path}/")
    
    # 创建用户目录结构
    user_dirs = ["data", "results", "results/logs", "temp"]
    for d in user_dirs:
        (DIST_DIR / d).mkdir(parents=True, exist_ok=True)
    
    # 创建启动说明
    readme_path = DIST_DIR / "使用说明.txt"
    readme_path.write_text(f"""
{APP_NAME} v{APP_VERSION}
智能药物虚拟筛选平台
{'='*50}

【启动方法】
  双击 {APP_NAME}.exe 即可启动应用

【数据目录】
  应用数据保存在: 用户目录/DrugScreenAI/
    - data/       : 化合物数据
    - results/    : 筛选结果
    - results/logs/: 运行日志

【使用流程】
  1. 启动后自动打开应用窗口
  2. 在左侧面板配置筛选参数
  3. 上传化合物库 (CSV/SMI/SDF)
  4. 点击"开始筛选"按钮
  5. 查看可视化结果和报告

【技术支持】
  如遇问题请检查日志文件或联系技术支持

版本: {APP_VERSION}
构建时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}
""", encoding='utf-8')
    
    print("  资源文件复制完成 ✓")
    return True


def create_portable_package():
    """创建便携版压缩包"""
    print("\n[5/5] 创建便携版压缩包...")
    
    try:
        import zipfile
        zip_path = PROJECT_ROOT / f"{APP_NAME}_v{APP_VERSION}_portable.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(DIST_DIR):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(DIST_DIR.parent)
                    zf.write(file_path, arcname)
        
        print(f"  便携包: {zip_path}")
        print(f"  大小: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
        print("  便携包创建完成 ✓")
    except Exception as e:
        print(f"  创建便携包失败: {e}")
    
    # 显示构建信息
    print(f"\n{'='*60}")
    print(f"  构建完成!")
    print(f"{'='*60}")
    print(f"\n  输出目录: {DIST_DIR}")
    print(f"  可执行文件: {DIST_DIR / (APP_NAME + '.exe')}")
    
    # 计算总大小
    total_size = 0
    for root, dirs, files in os.walk(DIST_DIR):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    print(f"  总大小: {total_size / 1024 / 1024:.1f} MB")
    print(f"\n  双击 {APP_NAME}.exe 启动应用")
    print(f"{'='*60}\n")


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=f'构建 {APP_NAME} 可执行文件',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--clean', action='store_true',
                       help='仅清理构建文件')
    parser.add_argument('--no-clean', action='store_true',
                       help='跳过清理步骤')
    parser.add_argument('--onedir', action='store_true',
                       help='目录模式 (启动更快, 文件分散)')
    parser.add_argument('--onefile', action='store_true', default=True,
                       help='单文件模式 (默认, 便于分发)')
    parser.add_argument('--skip-install', action='store_true',
                       help='跳过依赖检查')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"  {APP_NAME} v{APP_VERSION} 构建工具")
    print(f"{'='*60}")
    print(f"  项目目录: {PROJECT_ROOT}")
    print(f"  Python: {sys.executable}")
    print(f"  Python版本: {sys.version.split()[0]}")
    print(f"{'='*60}")
    
    # 仅清理
    if args.clean:
        clean_build()
        print("\n清理完成!")
        return 0
    
    # 检查依赖
    if not args.skip_install:
        ensure_packages()
    
    # 清理
    if not args.no_clean:
        clean_build()
    else:
        print("\n[2/5] 跳过清理步骤")
    
    # 构建
    mode = "onedir" if args.onedir else "onefile"
    if not build_exe(mode):
        print("\n构建失败！请检查错误信息。")
        return 1
    
    # 复制资源
    if not copy_extra_files():
        print("\n资源文件复制失败！")
        return 1
    
    # 创建便携包
    create_portable_package()
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n构建已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n构建出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
