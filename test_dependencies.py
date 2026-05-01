#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试依赖包是否正常安装
"""

import sys

print(f"Python version: {sys.version}")
print()

# 逐个测试依赖包，单独执行每个测试
dependencies = [
    'numpy',
    'pandas',
    'scikit-learn',
    'xgboost',
    'rdkit'
]

for dep in dependencies:
    print(f"Testing {dep}...")
    try:
        # 使用子进程执行，避免主进程崩溃
        import subprocess
        result = subprocess.run([sys.executable, '-c', f'import {dep}; print("OK")'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✓ {dep} imported successfully")
        else:
            print(f"✗ {dep} import failed: {result.stderr}")
    except Exception as e:
        print(f"✗ {dep} test failed: {e}")
    print()

print("Test completed.")
