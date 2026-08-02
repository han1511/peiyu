#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DrugScreen AI 启动脚本

提供多种启动方式：
1. Streamlit Web界面
2. 命令行流程运行
3. 批量处理
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()


def check_dependencies():
    """检查必要的依赖"""
    required = ['streamlit', 'pandas', 'numpy', 'matplotlib', 'sklearn']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"缺少依赖: {', '.join(missing)}")
        print("请安装依赖: pip install -r requirements.txt")
        return False
    
    return True


def run_streamlit():
    """启动Streamlit Web应用"""
    app_path = PROJECT_ROOT / "app.py"
    
    if not app_path.exists():
        print(f"错误: 找不到应用文件 {app_path}")
        return 1
    
    print("正在启动 DrugScreen AI Web界面...")
    print(f"项目目录: {PROJECT_ROOT}")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.headless", "true",
        "--server.port", "8501",
        "--browser.gatherUsageStats", "false"
    ]
    
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT)
        return 0
    except KeyboardInterrupt:
        print("\n应用已停止")
        return 0
    except Exception as e:
        print(f"启动失败: {e}")
        return 1


def run_cli(target: str, library: str, output: str = None, config: str = None):
    """命令行运行筛选流程"""
    from src.pipeline_v2 import run_pipeline
    
    print(f"运行虚拟筛选流程:")
    print(f"  靶点: {target}")
    print(f"  化合物库: {library}")
    print(f"  输出目录: {output or '默认'}")
    
    try:
        results = run_pipeline(
            target_name=target,
            library_path=library,
            output_dir=output,
            config_path=config
        )
        
        if results.get('success'):
            print("\n流程执行成功!")
        else:
            print("\n流程执行失败，请查看日志")
            return 1
            
    except Exception as e:
        print(f"执行出错: {e}")
        return 1
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='DrugScreen AI - 智能药物虚拟筛选平台',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 启动Web界面
  python run_app.py --web
  
  # 命令行运行筛选
  python run_app.py --target NS5 --library data/compounds.csv --output results/run1
  
  # 使用自定义配置
  python run_app.py --target NS5 --library data/compounds.csv --config configs/custom.yaml
        """
    )
    
    parser.add_argument('--web', action='store_true',
                       help='启动Streamlit Web界面')
    parser.add_argument('--target', type=str,
                       help='靶点名称 (如 NS5, NS3, NS2A, Envelope)')
    parser.add_argument('--library', type=str,
                       help='化合物库文件路径 (CSV/SMI/SDF)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径 (YAML)')
    parser.add_argument('--check', action='store_true',
                       help='检查依赖安装状态')
    
    args = parser.parse_args()
    
    # 检查依赖
    if args.check:
        if check_dependencies():
            print("所有依赖已安装")
            return 0
        return 1
    
    # 启动Web界面
    if args.web:
        if not check_dependencies():
            return 1
        return run_streamlit()
    
    # 命令行运行
    if args.target and args.library:
        if not check_dependencies():
            return 1
        return run_cli(args.target, args.library, args.output, args.config)
    
    # 默认启动Web界面
    print("DrugScreen AI v2.0")
    print("=" * 50)
    print("未指定命令，默认启动Web界面...")
    print("使用 --help 查看所有选项")
    print("=" * 50 + "\n")
    
    if not check_dependencies():
        return 1
    return run_streamlit()


if __name__ == "__main__":
    sys.exit(main())
