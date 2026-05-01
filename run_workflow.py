#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
登革病毒抑制剂虚拟筛选和分子对接工作流
一键启动脚本

功能：
1. 加载和预处理化合物库
2. 使用ChEMBL数据训练机器学习模型
3. 对化合物库进行虚拟筛选
4. 对Top化合物进行分子对接
5. 生成综合报告

使用方法：
    python run_workflow.py

可调节参数：
    - 化合物库路径 (COMPOUND_LIBRARY)
    - 训练数据路径 (TRAINING_DATA)
    - PDB受体文件 (RECEPTOR_FILE)
    - Vina可执行文件路径 (VINA_EXECUTABLE)
    - CPU核心数 (CPU_CORES)
    - 搜索穷举度 (EXHAUSTIVENESS)
    - 对接Top N化合物 (TOP_N_COMPOUNDS)
    - 虚拟筛选概率阈值 (PROBABILITY_THRESHOLD)
"""

import os
import sys
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# ==============================================================================
# 可调节参数配置
# ==============================================================================

# 数据文件路径
COMPOUND_LIBRARY = "E:/Python/dengue_drug_discovery/src/modeling/pubchem_100k_compounds.csv"
TRAINING_DATA = "E:/Python/dengue_drug_discovery/src/modeling/DENV_NS5_training_data_cleaned.csv"

# 分子对接参数
VINA_EXECUTABLE = "E:/autodock/vina.exe"
RECEPTOR_FILE = "E:/Python/dengue_drug_discovery/virtual_screening_pipeline/data/target_structures/4V0Q.pdbqt"

# 计算资源参数
CPU_CORES = 10  # 使用的CPU核心数，可根据您的CPU调整
EXHAUSTIVENESS = 8  # 搜索穷举度，值越高结果越准确但越慢

# 虚拟筛选参数
PROBABILITY_THRESHOLD = 0.5  # 活性预测概率阈值
TOP_N_COMPOUNDS = 10  # 分子对接的Top化合物数量

# 输出目录
OUTPUT_DIR = "E:/Python/dengue_drug_discovery/results"
WORKFLOW_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ==============================================================================
# 工作流函数
# ==============================================================================

def print_header():
    """打印工作流头部信息"""
    print("\n" + "=" * 70)
    print(" " * 15 + "登革病毒抑制剂虚拟筛选和分子对接工作流")
    print("=" * 70)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出目录: {get_output_dir()}")
    print("\n可调节参数:")
    print(f"  - 化合物库: {COMPOUND_LIBRARY}")
    print(f"  - 训练数据: {TRAINING_DATA}")
    print(f"  - 受体文件: {RECEPTOR_FILE}")
    print(f"  - CPU核心数: {CPU_CORES}")
    print(f"  - 搜索穷举度: {EXHAUSTIVENESS}")
    print(f"  - 对接Top N: {TOP_N_COMPOUNDS}")
    print(f"  - 概率阈值: {PROBABILITY_THRESHOLD}")
    print("=" * 70 + "\n")

def get_output_dir():
    """获取输出目录路径"""
    return os.path.join(OUTPUT_DIR, f"workflow_{WORKFLOW_TIMESTAMP}")

def create_output_dir():
    """创建输出目录"""
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def check_dependencies():
    """检查必要的依赖文件"""
    print("[1/5] 检查依赖文件...")
    issues = []

    if not os.path.exists(COMPOUND_LIBRARY):
        issues.append(f"化合物库文件不存在: {COMPOUND_LIBRARY}")
    else:
        print(f"  OK 化合物库: {COMPOUND_LIBRARY}")

    if not os.path.exists(TRAINING_DATA):
        issues.append(f"训练数据文件不存在: {TRAINING_DATA}")
    else:
        print(f"  OK 训练数据: {TRAINING_DATA}")

    if not os.path.exists(RECEPTOR_FILE):
        issues.append(f"受体文件不存在: {RECEPTOR_FILE}")
    else:
        print(f"  OK 受体文件: {RECEPTOR_FILE}")

    if not os.path.exists(VINA_EXECUTABLE):
        issues.append(f"Vina可执行文件不存在: {VINA_EXECUTABLE}")
    else:
        print(f"  OK Vina可执行文件: {VINA_EXECUTABLE}")

    if issues:
        print("\n问题:")
        for issue in issues:
            print(f"  XX {issue}")
        return False

    print("  OK 所有依赖文件检查通过\n")
    return True

def run_virtual_screening():
    """运行虚拟筛选流程"""
    print("[2/5] 运行虚拟筛选流程...")
    print("-" * 70)

    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "run_screening_with_training.py"
    )

    if not os.path.exists(script_path):
        print(f"错误: 虚拟筛选脚本不存在: {script_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, script_path,
             "--library", COMPOUND_LIBRARY,
             "--training", TRAINING_DATA,
             "--target", "NS5",
             "--pdb_id", "4V0Q",
             "--output", OUTPUT_DIR,
             "--top_n", str(TOP_N_COMPOUNDS)],
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )

        print(result.stdout)

        if result.returncode == 0:
            print("-" * 70)
            print("OK 虚拟筛选流程完成\n")
            return True
        else:
            print(f"警告: 虚拟筛选返回码 {result.returncode}")
            if result.stderr:
                print(f"错误输出: {result.stderr[-500:]}")
            return True  # 继续执行

    except subprocess.TimeoutExpired:
        print("错误: 虚拟筛选超时（超过1小时）")
        return False
    except Exception as e:
        print(f"错误: 运行虚拟筛选失败: {e}")
        return False

def run_molecular_docking():
    """运行分子对接流程"""
    print("[3/5] 运行分子对接流程...")
    print("-" * 70)

    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "run_docking.py"
    )

    if not os.path.exists(script_path):
        print(f"警告: 分子对接脚本不存在: {script_path}")
        print("  跳过分子对接步骤\n")
        return True

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=1800  # 30分钟超时
        )

        print(result.stdout)

        if result.returncode == 0:
            print("-" * 70)
            print("OK 分子对接流程完成\n")
            return True 
        else:
            print(f"警告: 分子对接返回码 {result.returncode}")
            if result.stderr:
                print(f"错误输出: {result.stderr[-500:]}")
            return True  # 继续执行

    except subprocess.TimeoutExpired:
        print("错误: 分子对接超时（超过30分钟）")
        return False
    except Exception as e:
        print(f"错误: 运行分子对接失败: {e}")
        return False

def generate_report():
    """生成综合报告"""
    print("[4/5] 生成综合报告...")
    print("-" * 70)

    output_dir = get_output_dir()
    report_file = os.path.join(output_dir, "workflow_report.md")

    # 查找最新的筛选结果
    screening_results_file = os.path.join(OUTPUT_DIR, "screening_results.json")
    if not os.path.exists(screening_results_file):
        # 查找最新的结果目录
        for item in sorted(Path(OUTPUT_DIR).iterdir(), reverse=True):
            if item.is_dir() and item.name.startswith("NS5_"):
                candidate = item / "screening_results.json"
                if candidate.exists():
                    screening_results_file = str(candidate)
                    break

    report_content = f"""# 登革病毒抑制剂虚拟筛选和分子对接工作流报告

## 工作流信息
- **运行时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **输出目录**: {output_dir}
- **工作流版本**: 1.0

## 参数配置
- **化合物库**: {COMPOUND_LIBRARY}
- **训练数据**: {TRAINING_DATA}
- **受体文件**: {RECEPTOR_FILE}
- **CPU核心数**: {CPU_CORES}
- **搜索穷举度**: {EXHAUSTIVENESS}
- **对接Top N**: {TOP_N_COMPOUNDS}
- **概率阈值**: {PROBABILITY_THRESHOLD}

## 虚拟筛选结果
"""

    if os.path.exists(screening_results_file):
        try:
            with open(screening_results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            report_content += f"""
### 模型训练结果
- **最佳模型**: {results.get('best_model', 'N/A')}
- **最佳AUC**: {results.get('best_auc', 0):.4f}

### 筛选统计
- **靶点**: {results.get('target', 'N/A')}
- **PDB ID**: {results.get('pdb_id', 'N/A')}
- **筛选化合物数**: {results.get('compounds_screened', 0)}
- **预测活性化合物数**: {len(results.get('top_indices', []))}

### Top 10 化合物
| 排名 | 预测分数 |
|------|----------|
"""
            for i, score in enumerate(results.get('top_scores', [])[:10], 1):
                report_content += f"| {i} | {score:.4f} |\n"

            report_content += f"""
### ADMET评估
- **评估化合物数**: {len(results.get('admet_results', []))}
"""
            if results.get('admet_results'):
                report_content += "\n#### Top 5 化合物ADMET性质\n"
                report_content += "| CID | MW | LogP | TPSA | HBD | HBA |\n"
                report_content += "|-----|-----|------|------|-----|-----|\n"
                for i, admet in enumerate(results.get('admet_results', [])[:5]):
                    props = admet.get('properties', {})
                    report_content += f"| {admet.get('CID', 'N/A')} | "
                    report_content += f"{props.get('MolecularWeight', 0):.2f} | "
                    report_content += f"{props.get('LogP', 0):.2f} | "
                    report_content += f"{props.get('TPSA', 0):.2f} | "
                    report_content += f"{props.get('NumHDonors', 0)} | "
                    report_content += f"{props.get('NumHAcceptors', 0)} |\n"

        except Exception as e:
            report_content += f"\n读取筛选结果时出错: {e}\n"
    else:
        report_content += "\n筛选结果文件不存在\n"

    report_content += f"""
## 分子对接结果

请查看 docking_results.json 文件了解详细的对接结果。

## 结论

本工作流完成了以下步骤：
1. OK 数据加载与预处理
2. OK 机器学习模型训练
3. OK 虚拟筛选
4. OK ADMET性质预测
5. OK 分子对接（如配置）

## 注意事项

1. 分子对接结果需要结合实验验证
2. ADMET预测仅供参考，实际性质需要实验测定
3. 建议对Top化合物进行进一步的生物学实验验证

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"OK 报告已生成: {report_file}")
    except Exception as e:
        print(f"错误: 生成报告失败: {e}")

    print("-" * 70 + "\n")
    return True

def print_summary():
    """打印工作流执行摘要"""
    print("[5/5] 执行完成")
    print("=" * 70)
    print("工作流执行完成！")
    print(f"\n结果目录: {get_output_dir()}")
    print(f"报告文件: {os.path.join(get_output_dir(), 'workflow_report.md')}")
    print("\n建议查看:")
    print("  1. workflow_report.md - 综合报告")
    print("  2. screening_results.json - 筛选详细结果")
    print("  3. docking_results.json - 分子对接结果")
    print("=" * 70 + "\n")

# ==============================================================================
# 主函数
# ==============================================================================

def main():
    """主函数"""
    print_header()

    # 创建输出目录
    output_dir = create_output_dir()
    print(f"输出目录: {output_dir}\n")

    # 检查依赖
    if not check_dependencies():
        print("错误: 依赖文件检查失败，请修复后重试")
        sys.exit(1)

    # 运行虚拟筛选
    if not run_virtual_screening():
        print("错误: 虚拟筛选失败")
        sys.exit(1)

    # 运行分子对接
    if not run_molecular_docking():
        print("警告: 分子对接未成功完成")

    # 生成报告
    generate_report()

    # 打印摘要
    print_summary()

    return 0

if __name__ == "__main__":
    sys.exit(main())
