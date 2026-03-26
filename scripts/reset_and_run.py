#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重置并运行抗登革病毒药物筛选流程

这个脚本用于：
1. 删除之前的结果文件
2. 重新运行完整的筛选流程
"""

import os
import shutil
import sys
import subprocess

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import RESULTS_DIR, DATA_DIR

def delete_results():
    """删除之前的结果文件"""
    print("正在删除之前的结果文件...")
    
    # 删除models目录中的模型文件
    models_dir = RESULTS_DIR['models']
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            file_path = os.path.join(models_dir, file)
            if os.path.isfile(file_path) and file.endswith(('_model.pkl', '_scaler.pkl', '_results.json', '_test_predictions.pkl')):
                os.remove(file_path)
                print(f"删除文件: {file}")
    
    # 删除figures目录
    figures_dir = RESULTS_DIR['figures']
    if os.path.exists(figures_dir):
        shutil.rmtree(figures_dir)
        print(f"删除目录: {figures_dir}")
    
    # 删除reports目录中的旧报告
    reports_dir = RESULTS_DIR['reports']
    if os.path.exists(reports_dir):
        for folder in os.listdir(reports_dir):
            folder_path = os.path.join(reports_dir, folder)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
                print(f"删除目录: {folder_path}")
    
    # 删除tables目录中的文件
    tables_dir = RESULTS_DIR['tables']
    if os.path.exists(tables_dir):
        for file in os.listdir(tables_dir):
            file_path = os.path.join(tables_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"删除文件: {file}")
    
    print("结果文件删除完成！")

def run_screening():
    """运行完整的筛选流程"""
    print("\n开始运行完整的筛选流程...")
    
    # 运行测试脚本
    test_script = os.path.join(os.path.dirname(__file__), 'run_test.py')
    print(f"运行测试脚本: {test_script}")
    
    try:
        result = subprocess.run([sys.executable, test_script], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.path.dirname(os.path.dirname(__file__)))
        
        print("\n运行输出:")
        print(result.stdout)
        
        if result.stderr:
            print("\n错误输出:")
            print(result.stderr)
        
        print(f"\n运行完成，返回码: {result.returncode}")
        
        if result.returncode == 0:
            print("筛选流程运行成功！")
        else:
            print("筛选流程运行失败！")
            
    except Exception as e:
        print(f"运行脚本时出错: {e}")

def main():
    """主函数"""
    print("=" * 80)
    print("重置并运行抗登革病毒药物筛选流程")
    print("=" * 80)
    
    # 删除之前的结果
    delete_results()
    
    # 运行筛选流程
    run_screening()
    
    print("\n" + "=" * 80)
    print("流程完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重置并运行抗登革病毒药物筛选流程

这个脚本用于：
1. 删除之前的结果文件
2. 重新运行完整的筛选流程
"""

import os
import shutil
import sys
import subprocess

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import RESULTS_DIR, DATA_DIR

def delete_results():
    """删除之前的结果文件"""
    print("正在删除之前的结果文件...")
    
    # 删除models目录
    models_dir = RESULTS_DIR['models']
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            file_path = os.path.join(models_dir, file)
            if os.path.isfile(file_path) and file.endswith(('_model.pkl', '_scaler.pkl', '_results.json', '_test_predictions.pkl')):
                os.remove(file_path)
                print(f"删除文件: {file}")
    
    # 删除figures目录
    figures_dir = RESULTS_DIR['figures']
    if os.path.exists(figures_dir):
        shutil.rmtree(figures_dir)
        print(f"删除目录: {figures_dir}")
    
    # 删除reports目录中的旧报告
    reports_dir = RESULTS_DIR['reports']
    if os.path.exists(reports_dir):
        for folder in os.listdir(reports_dir):
            folder_path = os.path.join(reports_dir, folder)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
                print(f"删除目录: {folder_path}")
    
    # 删除tables目录
    tables_dir = RESULTS_DIR['tables']
    if os.path.exists(tables_dir):
        for file in os.listdir(tables_dir):
            file_path = os.path.join(tables_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"删除文件: {file}")
    
    print("结果文件删除完成！")

def run_screening():
    """运行完整的筛选流程"""
    print("\n开始运行完整的筛选流程...")
    
    # 运行测试脚本
    test_script = os.path.join(os.path.dirname(__file__), 'run_test.py')
    print(f"运行测试脚本: {test_script}")
    
    try:
        result = subprocess.run([sys.executable, test_script], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.path.dirname(os.path.dirname(__file__)))
        
        print("\n运行输出:")
        print(result.stdout)
        
        if result.stderr:
            print("\n错误输出:")
            print(result.stderr)
        
        print(f"\n运行完成，返回码: {result.returncode}")
        
        if result.returncode == 0:
            print("筛选流程运行成功！")
        else:
            print("筛选流程运行失败！")
            
    except Exception as e:
        print(f"运行脚本时出错: {e}")

def main():
    """主函数"""
    print("=" * 80)
    print("重置并运行抗登革病毒药物筛选流程")
    print("=" * 80)
    
    # 删除之前的结果
    delete_results()
    
    # 运行筛选流程
    run_screening()
    
    print("\n" + "=" * 80)
    print("流程完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()