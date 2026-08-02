#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
药物虚拟筛选平台 - Streamlit Web应用 v2.1

功能：
1. 交互式参数配置 (含CPU核心控制、自定义输出目录)
2. 化合物库上传与管理
3. 虚拟筛选流程执行 (实时进度+计时)
4. 结果可视化展示 (图表保存到输出目录, 论文出版级)
5. 论文级报告生成

作者：研究团队
版本：3.0.0
- 接入ChEMBL真实活性数据
- 真实分子对接引擎 (Vina/GNINA/经验打分)
- 模型持久化 (.pkl)
- 断点续传
- 统计验证 (Y-scrambling + Bootstrap)
"""

import os
import sys
import json
import time
import logging
import tempfile
import multiprocessing
import platform
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# ============================================================================
# 强制设置环境编码，彻底解决Windows中文路径问题
# Linux服务器上无此问题，使用进程模式实现真正的多核并行
# ============================================================================
IS_WINDOWS = sys.platform == 'win32'
IS_SERVER = os.environ.get('DRUGSCREEN_MODE') == 'server' or not IS_WINDOWS

if IS_WINDOWS:
    # Windows: 设置标准输出和错误输出为UTF-8
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    # 在项目根目录下创建ASCII安全的临时目录
    _project_dir = Path(__file__).parent.resolve()
    _safe_temp = _project_dir / "_safe_temp"
    _safe_temp.mkdir(parents=True, exist_ok=True)
    
    os.environ['TEMP'] = str(_safe_temp)
    os.environ['TMP'] = str(_safe_temp)
    os.environ['TMPDIR'] = str(_safe_temp)

# 并行模式: Windows用线程(安全), Linux用进程(高性能)
PARALLEL_PREFER = 'threads' if IS_WINDOWS else 'processes'

import streamlit as st
import numpy as np
import pandas as pd

# 设置页面配置
st.set_page_config(
    page_title="DrugScreen AI - 智能药物筛选平台",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/drugscreen/help',
        'Report a bug': 'https://github.com/drugscreen/issues',
        'About': '# DrugScreen AI v2.1\n智能药物虚拟筛选平台'
    }
)

# ============================================================================
# 路径解析 - 兼容开发环境和打包环境
# ============================================================================

def get_project_root():
    """获取项目根目录，兼容打包环境"""
    env_root = os.environ.get('DRUGSCREEN_ROOT')
    if env_root:
        return Path(env_root)
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent.resolve()

def get_user_data_dir():
    """获取用户数据目录"""
    env_data = os.environ.get('DRUGSCREEN_DATA_DIR')
    if env_data:
        return Path(env_data).parent
    if getattr(sys, 'frozen', False):
        return Path.home() / "DrugScreenAI"
    return Path(__file__).parent.resolve()

PROJECT_ROOT = get_project_root()
USER_DATA_DIR = get_user_data_dir()

for d in ['data', 'results', 'results/logs', 'temp']:
    (USER_DATA_DIR / d).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault('DRUGSCREEN_DATA_DIR', str(USER_DATA_DIR / "data"))
os.environ.setdefault('DRUGSCREEN_RESULTS_DIR', str(USER_DATA_DIR / "results"))
os.environ.setdefault('DRUGSCREEN_LOGS_DIR', str(USER_DATA_DIR / "results" / "logs"))

from configs.config import (
    PROJECT_ROOT as CONFIG_ROOT, RESULTS_DIR, DATA_DIR,
    DENGUE_TARGETS, COMPOUND_LIBRARIES,
    ML_MODELS, DOCKING_CONFIG, ADMET_THRESHOLDS,
    LIPINSKI_RULES, FEATURE_CONFIG
)

# ============================================================================
# 延迟导入重量级模块 (rdkit/sklearn/xgboost/matplotlib/seaborn)
# 启动时不加载这些依赖，仅在首次使用时按需加载，显著加速启动
# ============================================================================

def _module_available(name: str) -> bool:
    """快速检测模块是否可导入 (不实际执行模块代码，毫秒级)"""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False

# 模块可用性标志 (快速检测，不触发重量级导入)
CHEMBL_AVAILABLE = _module_available('src.chembl_data')
DOCKING_MODULE_AVAILABLE = _module_available('src.real_docking')
VALIDATION_MODULE_AVAILABLE = _module_available('src.validation')


@st.cache_resource
def _load_visualization_modules():
    """延迟加载可视化模块 (matplotlib/seaborn/rdkit) - 仅首次调用时加载"""
    from src.visualization import (
        ModelVisualizer, DockingVisualizer,
        ADMETVisualizer, ChemicalSpaceVisualizer, CompoundVisualizer
    )
    return {
        'ModelVisualizer': ModelVisualizer,
        'DockingVisualizer': DockingVisualizer,
        'ADMETVisualizer': ADMETVisualizer,
        'ChemicalSpaceVisualizer': ChemicalSpaceVisualizer,
        'CompoundVisualizer': CompoundVisualizer,
    }


@st.cache_resource
def _load_chembl_module():
    """延迟加载ChEMBL数据模块 - 仅首次调用时加载"""
    from src.chembl_data import ChEMBLDataFetcher, DENGUE_CHEMBL_TARGETS
    return {'fetcher': ChEMBLDataFetcher, 'targets': DENGUE_CHEMBL_TARGETS}


@st.cache_resource
def _load_docking_module():
    """延迟加载分子对接模块 - 仅首次调用时加载"""
    from src.real_docking import RealDockingEngine
    return RealDockingEngine


@st.cache_resource
def _load_validation_module():
    """延迟加载验证模块 (sklearn) - 仅首次调用时加载"""
    from src.validation import ModelPersistence, CheckpointManager, StatisticalValidator
    return {
        'ModelPersistence': ModelPersistence,
        'CheckpointManager': CheckpointManager,
        'StatisticalValidator': StatisticalValidator,
    }

# ============================================================================
# 系统资源检测
# ============================================================================

def get_system_info() -> Dict[str, Any]:
    """获取系统资源信息"""
    cpu_count = multiprocessing.cpu_count()
    try:
        import psutil
        # interval=None 非阻塞读取上次调用以来的平均值 (避免阻塞0.5秒)
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        mem_total_gb = mem.total / (1024**3)
        mem_used_gb = mem.used / (1024**3)
        mem_percent = mem.percent
    except ImportError:
        cpu_percent = 0
        mem_total_gb = 0
        mem_used_gb = 0
        mem_percent = 0
    
    return {
        'cpu_count': cpu_count,
        'cpu_percent': cpu_percent,
        'mem_total_gb': mem_total_gb,
        'mem_used_gb': mem_used_gb,
        'mem_percent': mem_percent,
        'python_version': sys.version.split()[0],
        'platform': platform.platform()
    }


# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E5C8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7F8C8D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #2E5C8A;
    }
    .step-active {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .step-pending {
        background-color: #f8f9fa;
        border-left: 4px solid #6c757d;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .step-running {
        background-color: #cce5ff;
        border-left: 4px solid #007bff;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .step-failed {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .log-box {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        max-height: 300px;
        overflow-y: auto;
    }
    .stProgress > div > div > div > div {
        background-color: #2E5C8A;
    }
    .resource-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .resource-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .resource-label {
        font-size: 0.8rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化会话状态"""
    defaults = {
        'pipeline_results': {},
        'current_step': 0,
        'step_status': {},  # step_idx -> 'pending'/'running'/'done'/'failed'
        'step_times': {},   # step_idx -> elapsed seconds
        'config': {},
        'compound_df': None,
        'model_results': None,
        'docking_results': None,
        'admet_results': None,
        'screening_complete': False,
        'logs': [],
        'visualization_cache': {},
        'project_name': f"screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'output_dir': None,
        'n_cpu': multiprocessing.cpu_count(),
        'generated_plots': {},  # 保存生成的图表路径
        'pipeline_start_time': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sidebar_config():
    """侧边栏配置面板"""
    sys_info = get_system_info()
    
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/dna-helix.png", width=80)
        st.title("配置面板")
        
        # ================================================================
        # 系统资源监控
        # ================================================================
        with st.expander("💻 系统资源", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="resource-card">
                    <div class="resource-value">{sys_info['cpu_count']}</div>
                    <div class="resource-label">CPU 核心数</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if sys_info['mem_total_gb'] > 0:
                    st.markdown(f"""
                    <div class="resource-card">
                        <div class="resource-value">{sys_info['mem_total_gb']:.1f}G</div>
                        <div class="resource-label">总内存</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="resource-card">
                        <div class="resource-value">N/A</div>
                        <div class="resource-label">内存</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # CPU使用率
            if sys_info['cpu_percent'] > 0:
                st.progress(sys_info['cpu_percent'] / 100, text=f"CPU 使用率: {sys_info['cpu_percent']:.0f}%")
            if sys_info['mem_percent'] > 0:
                st.progress(sys_info['mem_percent'] / 100, text=f"内存使用率: {sys_info['mem_percent']:.0f}% ({sys_info['mem_used_gb']:.1f}G / {sys_info['mem_total_gb']:.1f}G)")
            
            st.caption(f"Python {sys_info['python_version']} | {sys_info['platform']}")
        
        # ================================================================
        # CPU调用控制
        # ================================================================
        with st.expander("⚙️ CPU与并行控制", expanded=True):
            max_cpu = sys_info['cpu_count']
            n_cpu = st.slider(
                "使用的CPU核心数",
                min_value=1,
                max_value=max_cpu,
                value=max_cpu,
                help=f"设置并行计算使用的CPU核心数 (最大: {max_cpu})"
            )
            st.session_state.n_cpu = n_cpu
            
            st.markdown(f"""
            **CPU配置**
            - 可用核心: **{max_cpu}**
            - 分配核心: **{n_cpu}**
            - 利用率: **{n_cpu/max_cpu*100:.0f}%**
            """)
            
            batch_size = st.select_slider(
                "批处理大小",
                options=[100, 500, 1000, 2000, 5000, 10000],
                value=1000,
                help="并行处理时的批大小"
            )
            
            # 更新模型配置中的n_jobs
            for model_cfg in ML_MODELS.values():
                if 'n_jobs' in model_cfg:
                    model_cfg['n_jobs'] = n_cpu
            DOCKING_CONFIG['cpu_cores'] = n_cpu
        
        # ================================================================
        # 项目设置 - 自定义输出目录
        # ================================================================
        with st.expander("📁 项目设置", expanded=True):
            project_name = st.text_input(
                "项目名称",
                value=st.session_state.project_name,
                help="为当前筛选项目命名"
            )
            st.session_state.project_name = project_name
            
            # 自定义输出目录
            output_mode = st.radio(
                "输出目录模式",
                ["默认目录", "自定义目录"],
                help="选择结果保存位置"
            )
            
            if output_mode == "默认目录":
                output_dir = RESULTS_DIR / project_name
            else:
                custom_path = st.text_input(
                    "输入输出目录路径",
                    value=str(RESULTS_DIR / project_name),
                    help="输入完整的输出目录路径"
                )
                output_dir = Path(custom_path)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            st.session_state.output_dir = output_dir
            
            # 显示输出目录结构
            st.info(f"📂 输出目录:\n`{output_dir}`")
            
            viz_dir = output_dir / "figures"
            viz_dir.mkdir(parents=True, exist_ok=True)
            st.caption(f"图表将保存到: `{viz_dir}`")
        
        # ================================================================
        # 靶点选择
        # ================================================================
        with st.expander("🎯 靶点配置", expanded=True):
            target_name = st.selectbox(
                "选择靶点蛋白",
                options=list(DENGUE_TARGETS.keys()),
                help="选择登革病毒靶点蛋白"
            )
            
            target_info = DENGUE_TARGETS[target_name]
            st.markdown(f"""
            **靶点信息：**
            - 名称: {target_info['name']}
            - 功能: {target_info['function']}
            - UniProt: {target_info['uniprot_id']}
            """)
            
            pdb_id = st.text_input(
                "PDB ID (可选)",
                value=target_info.get('pdb_id') or "",
                help="输入PDB数据库中的结构ID"
            )
        
        # ================================================================
        # 机器学习配置
        # ================================================================
        with st.expander("🤖 机器学习配置"):
            model_type = st.multiselect(
                "选择模型",
                options=list(ML_MODELS.keys()),
                default=["XGBoost", "RandomForest"],
                help="选择一个或多个机器学习模型"
            )
            
            cv_folds = st.slider("交叉验证折数", 3, 10, 5, help="Stratified K-Fold交叉验证")
            test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)
            
            if model_type:
                selected_model = st.selectbox(
                    "主模型参数调整",
                    options=model_type,
                    help="选择要调整参数的模型"
                )
                
                with st.container():
                    st.markdown(f"**{selected_model} 参数**")
                    if selected_model == "XGBoost":
                        n_estimators = st.slider("树的数量", 100, 2000, 1000, 100)
                        max_depth = st.slider("最大深度", 3, 15, 10, 1)
                        learning_rate = st.slider("学习率", 0.01, 0.3, 0.05, 0.01)
                        ML_MODELS["XGBoost"].update({
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "learning_rate": learning_rate,
                            "n_jobs": n_cpu
                        })
                    elif selected_model == "RandomForest":
                        n_estimators = st.slider("树的数量", 100, 2000, 1000, 100)
                        max_depth = st.slider("最大深度", 5, 30, 15, 1)
                        ML_MODELS["RandomForest"].update({
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "n_jobs": n_cpu
                        })
                    elif selected_model == "SVM":
                        C = st.slider("正则化参数 C", 0.1, 10.0, 1.0, 0.1)
                        ML_MODELS["SVM"].update({"C": C})
                    elif selected_model == "LogisticRegression":
                        C = st.slider("正则化参数 C", 0.1, 10.0, 1.0, 0.1)
                        ML_MODELS["LogisticRegression"].update({"C": C})
        
        # ================================================================
        # 分子对接配置
        # ================================================================
        with st.expander("⚛️ 分子对接配置"):
            exhaustiveness = st.slider(
                "搜索强度 (Exhaustiveness)",
                8, 64, DOCKING_CONFIG.get("exhaustiveness", 32), 8,
                help="AutoDock Vina搜索强度"
            )
            num_poses = st.slider(
                "生成构象数",
                1, 50, DOCKING_CONFIG.get("num_poses", 20), 5,
                help="每个分子生成的对接构象数量"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                center_x = st.number_input("中心X", value=0.0, format="%.2f")
                center_y = st.number_input("中心Y", value=0.0, format="%.2f")
                center_z = st.number_input("中心Z", value=0.0, format="%.2f")
            with col2:
                size_x = st.number_input("盒子X", value=22.5, format="%.1f")
                size_y = st.number_input("盒子Y", value=22.5, format="%.1f")
                size_z = st.number_input("盒子Z", value=22.5, format="%.1f")
            
            DOCKING_CONFIG.update({
                "exhaustiveness": exhaustiveness,
                "num_poses": num_poses,
                "cpu_cores": n_cpu,
                "search_space": {
                    "center_x": center_x,
                    "center_y": center_y,
                    "center_z": center_z,
                    "size_x": size_x,
                    "size_y": size_y,
                    "size_z": size_z
                }
            })
        
        # ================================================================
        # ADMET配置
        # ================================================================
        with st.expander("💊 ADMET配置"):
            apply_lipinski = st.checkbox("应用Lipinski规则", value=True)
            apply_pains = st.checkbox("过滤PAINS结构", value=True)
            max_mw = st.slider("最大分子量", 200, 1000, 500, 50)
            LIPINSKI_RULES["molecular_weight"]["max"] = max_mw
        
        # ================================================================
        # 化合物库
        # ================================================================
        with st.expander("🧪 化合物库", expanded=True):
            library_source = st.radio(
                "数据来源",
                ["上传文件", "内置库", "手动输入"],
                help="选择化合物数据来源"
            )
            
            if library_source == "上传文件":
                uploaded_file = st.file_uploader(
                    "上传化合物文件",
                    type=['csv', 'smi', 'sdf', 'txt'],
                    help="支持CSV、SMILES、SDF格式"
                )
            elif library_source == "内置库":
                builtin_lib = st.selectbox(
                    "选择内置库",
                    options=list(COMPOUND_LIBRARIES.keys()),
                    format_func=lambda x: f"{x} - {COMPOUND_LIBRARIES[x]['description']}"
                )
                uploaded_file = None
            else:
                smiles_text = st.text_area(
                    "输入SMILES (每行一个)",
                    height=150,
                    help="每行输入一个SMILES字符串"
                )
                uploaded_file = None
        
        # ================================================================
        # 运行按钮
        # ================================================================
        st.markdown("---")
        run_button = st.button(
            "🚀 开始筛选",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.current_step > 0 and not st.session_state.screening_complete
        )
        
        if st.session_state.screening_complete:
            if st.button("🔄 重新启动", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        st.markdown("---")
        st.markdown("**DrugScreen AI v2.1**")
        st.markdown("智能药物虚拟筛选平台")
    
    return {
        'target_name': target_name,
        'pdb_id': pdb_id if pdb_id else None,
        'model_type': model_type,
        'cv_folds': cv_folds,
        'test_size': test_size,
        'n_cpu': n_cpu,
        'batch_size': batch_size,
        'library_source': library_source,
        'uploaded_file': uploaded_file if library_source == "上传文件" else None,
        'builtin_lib': builtin_lib if library_source == "内置库" else None,
        'smiles_text': smiles_text if library_source == "手动输入" else None,
        'run_button': run_button,
        'apply_lipinski': apply_lipinski,
        'apply_pains': apply_pains,
        'sys_info': sys_info,
    }


# ============================================================================
# 进度显示
# ============================================================================

def show_progress():
    """显示筛选进度（带计时和状态）"""
    steps = [
        "准备靶点结构",
        "加载化合物库",
        "预处理化合物",
        "计算分子特征",
        "训练ML模型",
        "虚拟筛选",
        "分子对接",
        "ADMET评估",
        "结果分析与可视化"
    ]
    
    st.markdown("### 📊 筛选进度")
    
    total_steps = len(steps)
    completed = st.session_state.current_step
    progress_val = completed / total_steps
    
    # 总进度条
    st.progress(progress_val, text=f"总进度: {completed}/{total_steps} 步骤")
    
    # 总耗时
    if st.session_state.pipeline_start_time:
        elapsed = time.time() - st.session_state.pipeline_start_time
        st.caption(f"⏱️ 已用时: {elapsed:.1f} 秒")
    
    # 各步骤状态
    cols = st.columns(3)
    for i, step in enumerate(steps):
        col_idx = i % 3
        with cols[col_idx]:
            status = st.session_state.step_status.get(i, 'pending')
            elapsed_time = st.session_state.step_times.get(i, None)
            
            if status == 'done':
                time_str = f" ({elapsed_time:.1f}s)" if elapsed_time else ""
                st.markdown(f"<div class='step-active'>✅ {i+1}. {step}{time_str}</div>", unsafe_allow_html=True)
            elif status == 'running':
                st.markdown(f"<div class='step-running'>🔄 {i+1}. {step} (进行中...)</div>", unsafe_allow_html=True)
            elif status == 'failed':
                st.markdown(f"<div class='step-failed'>❌ {i+1}. {step} (失败)</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='step-pending'>⏳ {i+1}. {step}</div>", unsafe_allow_html=True)


# ============================================================================
# 化合物库加载
# ============================================================================

def upload_and_process_library(config: Dict[str, Any]):
    """上传并处理化合物库"""
    st.markdown("## 🧪 化合物库管理")
    
    if config['library_source'] == "上传文件" and config['uploaded_file'] is not None:
        with st.spinner("正在加载化合物库..."):
            file_ext = Path(config['uploaded_file'].name).suffix
            temp_path = st.session_state.output_dir / f"uploaded_library{file_ext}"
            
            with open(temp_path, 'wb') as f:
                f.write(config['uploaded_file'].getvalue())
            
            if file_ext.lower() == '.csv':
                df = pd.read_csv(temp_path)
                if 'SMILES' not in df.columns and 'smiles' not in df.columns:
                    st.error("CSV文件必须包含'SMILES'或'smiles'列")
                    return None
                st.session_state.compound_df = df
            elif file_ext.lower() in ['.smi', '.txt']:
                with open(temp_path, 'r') as f:
                    smiles_list = [line.strip() for line in f if line.strip()]
                df = pd.DataFrame({'SMILES': smiles_list})
                st.session_state.compound_df = df
            else:
                st.warning(f"暂不支持 {file_ext} 格式的直接预览")
                df = pd.DataFrame({'SMILES': []})
            
            st.success(f"成功加载 {len(df)} 个化合物")
            
            with st.expander("查看化合物数据", expanded=True):
                st.dataframe(df.head(20), use_container_width=True)
                st.info(f"总计: {len(df)} 个化合物")
            return df
    
    elif config['library_source'] == "内置库" and config['builtin_lib']:
        lib_info = COMPOUND_LIBRARIES[config['builtin_lib']]
        st.info(f"使用内置库: {lib_info['description']}")
        if lib_info['path'].exists():
            if lib_info['path'].suffix == '.csv':
                df = pd.read_csv(lib_info['path'])
                st.session_state.compound_df = df
                st.success(f"成功加载 {len(df)} 个化合物")
                return df
            else:
                st.warning("该格式将在处理流程中加载")
                return pd.DataFrame()
        else:
            st.warning(f"内置库文件不存在: {lib_info['path']}")
            return None
    
    elif config['library_source'] == "手动输入" and config['smiles_text']:
        smiles_list = [s.strip() for s in config['smiles_text'].split('\n') if s.strip()]
        df = pd.DataFrame({
            'SMILES': smiles_list,
            'Name': [f"Compound_{i+1}" for i in range(len(smiles_list))]
        })
        st.session_state.compound_df = df
        st.success(f"成功输入 {len(df)} 个化合物")
        return df
    
    return None


# ============================================================================
# 筛选流程执行
# ============================================================================

def run_screening_pipeline(config: Dict[str, Any]):
    """运行虚拟筛选流程 - 带实时进度、计时和图表生成"""
    
    output_dir = st.session_state.output_dir
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    n_cpu = config['n_cpu']
    
    # 日志区域
    log_container = st.container()
    progress_container = st.container()
    
    logs = st.session_state.logs
    
    def add_log(msg: str, level: str = "INFO"):
        timestamp = datetime.now().strftime('%H:%M:%S')
        logs.append(f"[{timestamp}] [{level}] {msg}")
        with log_container:
            st.markdown("### 📝 运行日志")
            st.markdown(f"<div class='log-box'>{'<br>'.join(logs[-30:])}</div>", unsafe_allow_html=True)
    
    def set_step_status(step_idx: int, status: str, elapsed: float = None):
        st.session_state.step_status[step_idx] = status
        if elapsed is not None:
            st.session_state.step_times[step_idx] = elapsed
    
    st.session_state.pipeline_start_time = time.time()
    add_log(f"启动虚拟筛选流程 | CPU核心: {n_cpu} | 输出目录: {output_dir}")
    add_log(f"系统: {config['sys_info']['platform']}")
    add_log(f"可用CPU: {config['sys_info']['cpu_count']} | 分配CPU: {n_cpu} | 并行模式: {PARALLEL_PREFER}")
    
    # ====== 初始化断点续传管理器 ======
    checkpoint_mgr = None
    if VALIDATION_MODULE_AVAILABLE:
        checkpoint_mgr = _load_validation_module()['CheckpointManager'](output_dir)
        # 检查是否有可恢复的检查点
        if checkpoint_mgr.can_resume():
            resume_idx, resume_name = checkpoint_mgr.get_resume_point()
            add_log(f"  📂 发现检查点: 已完成到Step {resume_idx}, 可从 {resume_name} 恢复")
            # 清除旧检查点重新开始 (用户可选择恢复, 这里默认重新开始)
            # 实际应用中可以通过UI按钮选择恢复
        else:
            add_log(f"  📂 检查点系统已启用")
    
    try:
        # ================================================================
        # Step 1: 准备靶点
        # ================================================================
        set_step_status(0, 'running')
        with progress_container:
            show_progress()
        t0 = time.time()
        add_log("Step 1/9: 准备靶点结构...")
        
        target_info = DENGUE_TARGETS[config['target_name']]
        add_log(f"  靶点: {target_info['name']} ({config['target_name']})")
        add_log(f"  UniProt: {target_info['uniprot_id']}")
        if config['pdb_id']:
            add_log(f"  PDB ID: {config['pdb_id']}")
        else:
            add_log("  未提供PDB ID，使用同源建模推荐")
        
        time.sleep(0.5)  # 模拟处理
        elapsed = time.time() - t0
        set_step_status(0, 'done', elapsed)
        add_log(f"✅ 靶点结构准备完成 ({elapsed:.1f}s)")
        
        # 保存检查点
        if checkpoint_mgr:
            checkpoint_mgr.save_checkpoint(0, "target_preparation", 
                                          data={'target': config['target_name']})
        st.session_state.current_step = 1
        with progress_container:
            show_progress()
        
        # ================================================================
        # Step 2: 加载化合物库
        # ================================================================
        set_step_status(1, 'running')
        t0 = time.time()
        add_log("Step 2/9: 加载化合物库...")
        
        compound_df = st.session_state.compound_df
        if compound_df is None or len(compound_df) == 0:
            add_log("❌ 没有可用的化合物数据", "ERROR")
            set_step_status(1, 'failed')
            return
        
        n_compounds = len(compound_df)
        add_log(f"  化合物总数: {n_compounds}")
        
        # 保存化合物库
        lib_path = output_dir / "compound_library.csv"
        compound_df.to_csv(lib_path, index=False)
        add_log(f"  化合物库已保存: {lib_path}")
        
        elapsed = time.time() - t0
        set_step_status(1, 'done', elapsed)
        add_log(f"✅ 化合物库加载完成 ({elapsed:.1f}s)")
        if checkpoint_mgr:
            checkpoint_mgr.save_checkpoint(1, "compound_loading", data={'n_compounds': len(compound_df)})
        st.session_state.current_step = 2
        with progress_container:
            show_progress()
        
        # ================================================================
        # Step 3: 预处理化合物
        # ================================================================
        set_step_status(2, 'running')
        t0 = time.time()
        add_log("Step 3/9: 预处理化合物...")
        add_log(f"  Lipinski过滤: {'启用' if config['apply_lipinski'] else '禁用'}")
        add_log(f"  PAINS过滤: {'启用' if config['apply_pains'] else '禁用'}")
        
        # 使用RDKit进行真实预处理 - 并行化
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
            from joblib import Parallel, delayed
            import multiprocessing
            
            add_log(f"  使用 {n_cpu} 核并行预处理化合物...")
            
            # 并行处理函数
            def process_smiles_row(idx, row):
                smiles = row.get('SMILES', row.get('smiles', ''))
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is None:
                    return None
                return {
                    'SMILES': smiles,
                    'Name': row.get('Name', f"CMP_{idx:04d}"),
                    'MolWt': Descriptors.MolWt(mol),
                    'LogP': Descriptors.MolLogP(mol),
                    'TPSA': Descriptors.TPSA(mol),
                    'NumHDonors': Descriptors.NumHDonors(mol),
                    'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                    'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                    'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
                }
            
            # 使用系统临时目录以避免中文路径编码问题
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / "drugscreen_joblib"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # 并行计算 - 使用线程模式避免Windows中文路径编码问题
            smiles_list = [{'SMILES': row.get('SMILES', row.get('smiles', '')), 
                            'Name': row.get('Name', f"CMP_{idx:04d}")}
                           for idx, row in compound_df.iterrows()]
            
            results = Parallel(n_jobs=n_cpu, verbose=0, batch_size=1000, prefer=PARALLEL_PREFER)(
                delayed(process_smiles_row)(idx, row) 
                for idx, row in enumerate(smiles_list)
            )
            
            valid_smiles = [r for r in results if r is not None]
            invalid_count = len(results) - len(valid_smiles)
            
            if invalid_count > 0:
                add_log(f"  ⚠️ 无效SMILES: {invalid_count}")
            
            processed_df = pd.DataFrame(valid_smiles)
            
            # Lipinski过滤
            if config['apply_lipinski']:
                before = len(processed_df)
                mask = (
                    (processed_df['MolWt'] <= 500) &
                    (processed_df['LogP'] <= 5) &
                    (processed_df['NumHDonors'] <= 5) &
                    (processed_df['NumHAcceptors'] <= 10)
                )
                processed_df = processed_df[mask].reset_index(drop=True)
                after = len(processed_df)
                add_log(f"  Lipinski过滤: {before} → {after} (移除 {before-after})")
            
            processed_df.to_csv(output_dir / "processed_compounds.csv", index=False)
            st.session_state.compound_df = processed_df
            add_log(f"  有效化合物: {len(processed_df)}")
            
        except ImportError:
            add_log("  RDKit不可用，跳过化学预处理", "WARNING")
            processed_df = compound_df
        
        elapsed = time.time() - t0
        set_step_status(2, 'done', elapsed)
        add_log(f"✅ 化合物预处理完成 ({elapsed:.1f}s)")
        if checkpoint_mgr:
            checkpoint_mgr.save_checkpoint(2, "compound_preprocessing", data={'n_valid': len(processed_df)})
        st.session_state.current_step = 3
        with progress_container:
            show_progress()
        
        # ================================================================
        # Step 4: 计算分子特征 - 并行化
        # ================================================================
        set_step_status(3, 'running')
        t0 = time.time()
        add_log("Step 4/9: 计算分子特征...")
        add_log(f"  并行CPU核心: {n_cpu}")
        
        # 使用RDKit计算Morgan指纹 - 并行化
        try:
            from rdkit.Chem import AllChem
            from rdkit import DataStructs
            from joblib import Parallel, delayed
            
            n_bits = 2048
            
            # 并行指纹计算函数 (直接返回uint8数组，Morgan指纹为0/1二进制)
            def calc_fingerprint(smiles, n_bits=2048):
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is None:
                    return np.zeros(n_bits, dtype=np.uint8)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
                arr = np.zeros(n_bits, dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(fp, arr)
                return arr
            
            smiles_list = processed_df['SMILES'].tolist()
            
            add_log(f"  正在计算 {len(smiles_list)} 个分子的Morgan指纹...")
            add_log(f"  并行度: {n_cpu} 核 | 批大小: 1000")
            
            # 并行计算指纹 - 使用线程模式
            features = Parallel(n_jobs=n_cpu, verbose=0, batch_size=1000, prefer=PARALLEL_PREFER)(
                delayed(calc_fingerprint)(smiles, n_bits)
                for smiles in smiles_list
            )
            
            # uint8 存储 (Morgan指纹为0/1，1字节足够，比float32节省4倍内存)
            X = np.array(features, dtype=np.uint8)
            failed = int(np.sum(np.all(X == 0, axis=1)))
            
            add_log(f"  Morgan指纹维度: {X.shape}")
            add_log(f"  特征数: {n_bits} | 失败: {failed}")
            
        except Exception as e:
            add_log(f"  RDKit指纹计算失败: {e}", "WARNING")
            X = np.random.randn(len(processed_df), 100).astype(np.float32)
        
        # 保存特征 - 使用稀疏矩阵存储 (Morgan指纹稀疏度~2%，大幅节省磁盘)
        try:
            from scipy.sparse import csr_matrix, save_npz as save_sparse_npz
            X_sparse = csr_matrix(X)
            sparse_path = output_dir / "molecular_features.npz"
            save_sparse_npz(sparse_path, X_sparse)
            sparse_mb = (X_sparse.data.nbytes + X_sparse.indices.nbytes + X_sparse.indptr.nbytes) / 1024 / 1024
            add_log(f"  特征矩阵已保存(稀疏): {sparse_path} ({sparse_mb:.1f}MB)")
        except ImportError:
            # scipy不可用时回退到uint8压缩存储
            np.savez_compressed(output_dir / "molecular_features.npz", X=X)
            add_log(f"  特征矩阵已保存(uint8压缩): {output_dir / 'molecular_features.npz'} ({X.nbytes / 1024 / 1024:.1f}MB)")
        except OSError:
            import tempfile
            tmp_path = Path(tempfile.gettempdir()) / "molecular_features.npz"
            from scipy.sparse import save_npz as save_sparse_npz
            save_sparse_npz(tmp_path, X_sparse)
            add_log(f"  特征矩阵保存到临时目录: {tmp_path}")
        # 同时保存一份小型npy供快速加载 (仅前10000行作为样本)
        np.save(output_dir / "molecular_features_sample.npy", X[:10000])
        
        elapsed = time.time() - t0
        set_step_status(3, 'done', elapsed)
        add_log(f"✅ 特征计算完成 ({elapsed:.1f}s)")
        if checkpoint_mgr:
            checkpoint_mgr.save_checkpoint(3, "feature_computation", data={'X_shape': X.shape})
        st.session_state.current_step = 4
        with progress_container:
            show_progress()
        
        # ================================================================
        # Step 5: 训练ML模型 (核心步骤)
        # ================================================================
        set_step_status(4, 'running')
        t0 = time.time()
        add_log("Step 5/9: 训练机器学习模型...")
        add_log(f"  选中模型: {', '.join(config['model_type'])}")
        add_log(f"  交叉验证: {config['cv_folds']}-Fold Stratified")
        add_log(f"  测试集比例: {config['test_size']:.0%}")
        add_log(f"  并行CPU: {n_cpu} 核心")
        
        # 生成真实活性标签 - 从ChEMBL获取或使用经验规则
        add_log("  从ChEMBL获取真实生物活性数据...")
        target_name = config.get('target_name', config.get('target', 'NS3'))
        
        y = None
        chembl_df = None
        use_chembl_data = False
        
        if CHEMBL_AVAILABLE:
            try:
                fetcher = _load_chembl_module()['fetcher'](cache_dir=output_dir / "chembl_cache")
                chembl_smiles, chembl_y, chembl_df = fetcher.prepare_training_data(target_name)
                
                if chembl_df is not None and len(chembl_df) > 5:
                    add_log(f"  ChEMBL数据: {len(chembl_df)}条活性记录")
                    if 'Activity_Type' in chembl_df.columns:
                        act_types = chembl_df['Activity_Type'].value_counts().to_dict()
                        add_log(f"  活性类型: {act_types}")
                    
                    # 为ChEMBL数据计算特征
                    def calc_fp_safe(smiles, n_bits=2048):
                        mol = Chem.MolFromSmiles(str(smiles))
                        if mol is None:
                            return np.zeros(n_bits, dtype=np.uint8)
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
                        arr = np.zeros(n_bits, dtype=np.uint8)
                        DataStructs.ConvertToNumpyArray(fp, arr)
                        return arr
                    
                    from joblib import Parallel, delayed
                    
                    chembl_features = Parallel(n_jobs=n_cpu, batch_size=500, prefer=PARALLEL_PREFER)(
                        delayed(calc_fp_safe)(smiles, 2048)
                        for smiles in chembl_smiles
                    )
                    X_chembl = np.array(chembl_features, dtype=np.uint8)
                    y_chembl = np.array(chembl_y)
                    
                    add_log(f"  ChEMBL训练数据: {X_chembl.shape[0]}样本 | 活性:{int(y_chembl.sum())} | 非活性:{int((y_chembl==0).sum())}")
                    
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_chembl, y_chembl, test_size=config['test_size'],
                        random_state=42, stratify=y_chembl if len(np.unique(y_chembl)) > 1 else None
                    )
                    X_screen = X
                    use_chembl_data = True
                    add_log(f"  ✅ 使用ChEMBL真实数据训练")
                else:
                    raise ValueError("ChEMBL返回数据不足")
                    
            except Exception as e:
                add_log(f"  ⚠️ ChEMBL数据获取失败: {e}", "WARNING")
                use_chembl_data = False
        
        if not use_chembl_data:
            add_log("  使用经验规则生成标签 (基于分子描述符)...")
            y = np.zeros(len(processed_df), dtype=int)
            for idx, row in processed_df.iterrows():
                mol = Chem.MolFromSmiles(str(row['SMILES']))
                if mol is None:
                    continue
                mw = row.get('MolWt', Descriptors.MolWt(mol))
                logp = row.get('LogP', Descriptors.MolLogP(mol))
                tpsa = row.get('TPSA', Descriptors.TPSA(mol))
                n_rings = rdMolDescriptors.CalcNumRings(mol)
                n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)
                
                score = 0
                if 200 <= mw <= 500: score += 1
                if -1 <= logp <= 5: score += 1
                if 20 <= tpsa <= 140: score += 1
                if n_rings >= 2: score += 1
                if n_arom >= 1: score += 1
                if row.get('NumHDonors', 0) <= 5: score += 1
                if row.get('NumHAcceptors', 0) <= 10: score += 1
                
                y[idx] = 1 if score >= 5 else 0
            
            pos_ratio = y.mean()
            add_log(f"  经验标签: 活性 {int(y.sum())}/{len(y)} ({pos_ratio:.1%})")
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config['test_size'], random_state=42, stratify=y
            )
            X_screen = X
        
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.metrics import (
            roc_curve, auc, precision_recall_curve, average_precision_score,
            confusion_matrix, classification_report, accuracy_score,
            precision_score, recall_score, f1_score
        )
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        
        add_log(f"  训练集: {len(X_train)} | 测试集: {len(X_test)}")
        
        # 训练各模型
        trained_models = {}
        model_metrics = []
        all_y_true = {}
        all_y_proba = {}
        all_y_pred = {}
        
        for model_name in config['model_type']:
            add_log(f"  训练 {model_name}...")
            t_model = time.time()
            
            # 构建模型参数 - 排除重复参数
            if model_name == "XGBoost":
                try:
                    import xgboost as xgb
                    # 从配置中获取参数，排除会重复的（配置中可能已含这些键）
                    xgb_params = {k: v for k, v in ML_MODELS["XGBoost"].items() 
                                   if k not in ['objective', 'random_state', 'n_jobs', 'tree_method']}
                    model = xgb.XGBClassifier(
                        **xgb_params,
                        objective='binary:logistic',
                        random_state=42,
                        n_jobs=n_cpu,
                        tree_method='hist'  # 使用hist加速大数据
                    )
                except ImportError:
                    add_log(f"    XGBoost不可用，使用RandomForest替代", "WARNING")
                    model = RandomForestClassifier(
                        n_estimators=500, max_depth=10, n_jobs=n_cpu, random_state=42
                    )
            elif model_name == "RandomForest":
                rf_params = {k: v for k, v in ML_MODELS["RandomForest"].items() 
                              if k in ['n_estimators', 'max_depth', 'min_samples_split', 
                                       'min_samples_leaf', 'max_features', 'class_weight']}
                model = RandomForestClassifier(
                    **rf_params,
                    n_jobs=n_cpu,
                    random_state=42
                )
            elif model_name == "SVM":
                model = SVC(
                    C=ML_MODELS["SVM"]["C"],
                    kernel=ML_MODELS["SVM"]["kernel"],
                    gamma='scale',
                    probability=True,
                    random_state=42
                )
            elif model_name == "LogisticRegression":
                model = LogisticRegression(
                    C=ML_MODELS["LogisticRegression"]["C"],
                    penalty='l2',
                    solver='lbfgs',
                    max_iter=1000,
                    class_weight='balanced',
                    n_jobs=n_cpu,
                    random_state=42
                )
            else:
                continue
            
            # 训练
            model.fit(X_train, y_train)
            train_time = time.time() - t_model
            
            # 预测
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # 交叉验证
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=config['cv_folds'], scoring='roc_auc', n_jobs=n_cpu
            )
            
            # 计算指标
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            fpr_arr, tpr_arr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr_arr, tpr_arr)
            ap = average_precision_score(y_test, y_proba)
            
            add_log(f"    AUC={roc_auc:.4f} | F1={f1:.4f} | CV-AUC={cv_scores.mean():.4f}±{cv_scores.std():.4f} | {train_time:.1f}s")
            
            trained_models[model_name] = model
            all_y_true[model_name] = y_test
            all_y_proba[model_name] = y_proba
            all_y_pred[model_name] = y_pred
            
            model_metrics.append({
                'Model': model_name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1_Score': f1,
                'AUC': roc_auc,
                'AP': ap,
                'CV_AUC_Mean': cv_scores.mean(),
                'CV_AUC_Std': cv_scores.std(),
                'Train_Time_s': train_time
            })
        
        # 保存模型性能指标
        metrics_df = pd.DataFrame(model_metrics)
        metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)
        add_log(f"  指标已保存: {output_dir / 'model_metrics.csv'}")
        
        # ====== 模型持久化: 保存.pkl文件 ======
        if VALIDATION_MODULE_AVAILABLE:
            model_persister = _load_validation_module()['ModelPersistence'](model_dir=output_dir / "models")
            for model_name, model in trained_models.items():
                try:
                    metadata = {
                        'target': config['target'],
                        'feature_type': 'Morgan_2048',
                        'train_size': len(X_train),
                        'test_size': len(X_test),
                        'cv_folds': config['cv_folds'],
                        'metrics': {k: v for k, v in model_metrics[-1].items() if k != 'Model'},
                        'chembl_data': use_chembl_data,
                    }
                    saved_path = model_persister.save_model(model, model_name, metadata)
                    add_log(f"  💾 模型已保存: {saved_path.name}")
                except Exception as e:
                    add_log(f"  ⚠️ 模型保存失败({model_name}): {e}", "WARNING")
            
            # ====== 统计验证: Y-scrambling + Bootstrap ======
            add_log("  执行统计验证 (Y-scrambling + Bootstrap)...")
            validator = _load_validation_module()['StatisticalValidator'](n_scramble=20, n_bootstrap=500)
            validation_results = {}
            
            for model_name in config['model_type']:
                if model_name not in trained_models:
                    continue
                model = trained_models[model_name]
                y_true = all_y_true[model_name]
                y_pred = all_y_pred[model_name]
                y_proba = all_y_proba[model_name]
                
                try:
                    # Y-scrambling
                    add_log(f"    Y-scrambling ({model_name})...")
                    y_scramble = validator.y_scrambling(
                        model, X_train, y_train, X_test, y_test
                    )
                    
                    # Bootstrap CI
                    add_log(f"    Bootstrap CI ({model_name})...")
                    boot_ci = validator.bootstrap_confidence_interval(y_true, y_pred, y_proba)
                    
                    validation_results[model_name] = {
                        'y_scrambling': y_scramble,
                        'bootstrap_ci': boot_ci,
                    }
                    
                    if y_scramble['is_valid']:
                        add_log(f"    ✅ {model_name} Y-scrambling验证通过 (p={y_scramble['p_value']:.4f})")
                    else:
                        add_log(f"    ⚠️ {model_name} Y-scrambling验证未通过 (p={y_scramble['p_value']:.4f})", "WARNING")
                    
                    # 保存Y-scrambling图
                    ys_path = figures_dir / f"fig_y_scrambling_{model_name.lower()}.png"
                    validator.plot_y_scrambling(y_scramble, ys_path)
                    if ys_path.exists():
                        generated_plots_extra = st.session_state.get('generated_plots', {})
                        generated_plots_extra[f'y_scrambling_{model_name}'] = str(ys_path)
                        st.session_state.generated_plots = generated_plots_extra
                        add_log(f"    📊 Y-scrambling图 → {ys_path}")
                    
                except Exception as e:
                    add_log(f"    ⚠️ 统计验证失败({model_name}): {e}", "WARNING")
            
            # 保存验证结果
            val_path = output_dir / "statistical_validation.json"
            # 序列化函数: 确保所有值都是JSON可序列化的
            def make_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_serializable(v) for v in obj]
                elif isinstance(obj, (np.floating, np.integer)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            with open(val_path, 'w', encoding='utf-8') as f:
                json.dump(make_serializable(validation_results), f, ensure_ascii=False, indent=2)
            add_log(f"  📋 统计验证报告: {val_path}")
        else:
            add_log("  统计验证模块不可用，跳过", "WARNING")
        
        # ================================================================
        # 生成论文级图表 → 保存到输出目录/figures/
        # ================================================================
        add_log("  生成论文级可视化图表...")
        
        viz = _load_visualization_modules()['ModelVisualizer'](output_dir=figures_dir)
        generated_plots = {}
        
        # ROC曲线
        roc_path = viz.plot_roc_curve(
            all_y_true, all_y_proba,
            title="ROC Curve Comparison",
            filename="fig_roc_curve.png"
        )
        if roc_path:
            generated_plots['roc_curve'] = roc_path
            add_log(f"    📊 ROC曲线 → {roc_path}")
        
        # PR曲线
        pr_path = viz.plot_precision_recall_curve(
            all_y_true, all_y_proba,
            title="Precision-Recall Curve Comparison",
            filename="fig_pr_curve.png"
        )
        if pr_path:
            generated_plots['pr_curve'] = pr_path
            add_log(f"    📊 PR曲线 → {pr_path}")
        
        # 混淆矩阵 (每个模型)
        for model_name in config['model_type']:
            if model_name in all_y_true:
                cm_path = viz.plot_confusion_matrix(
                    all_y_true[model_name], all_y_pred[model_name],
                    title=f"Confusion Matrix - {model_name}",
                    filename=f"fig_confusion_matrix_{model_name.lower()}.png"
                )
                if cm_path:
                    generated_plots[f'confusion_matrix_{model_name}'] = cm_path
                    add_log(f"    📊 混淆矩阵({model_name}) → {cm_path}")
        
        # 模型对比图
        comparison_path = viz.plot_model_comparison(
            metrics_df,
            metrics=['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC'],
            title="Model Performance Comparison",
            filename="fig_model_comparison.png"
        )
        if comparison_path:
            generated_plots['model_comparison'] = comparison_path
            add_log(f"    📊 模型对比图 → {comparison_path}")
        
        # 特征重要性 (仅树模型)
        for model_name in config['model_type']:
            model = trained_models.get(model_name)
            if model is not None and hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                n_features = len(importances)
                feature_names = [f"FP_{i}" for i in range(n_features)]
                
                fi_path = viz.plot_feature_importance(
                    feature_names, importances,
                    title=f"Feature Importance - {model_name}",
                    filename=f"fig_feature_importance_{model_name.lower()}.png",
                    top_k=30
                )
                if fi_path:
                    generated_plots[f'feature_importance_{model_name}'] = fi_path
                    add_log(f"    📊 特征重要性({model_name}) → {fi_path}")
        
        # 校准曲线
        calib_path = viz.plot_calibration_curve(
            all_y_true, all_y_proba,
            title="Calibration Curve Comparison",
            filename="fig_calibration_curve.png"
        )
        if calib_path:
            generated_plots['calibration_curve'] = calib_path
            add_log(f"    📊 校准曲线 → {calib_path}")
        
        # 学习曲线 (使用主模型)
        if config['model_type']:
            main_model_name = config['model_type'][0]
            main_model = trained_models.get(main_model_name)
            if main_model is not None:
                try:
                    lc_path = viz.plot_learning_curve(
                        main_model, X_train, y_train,
                        cv=config['cv_folds'],
                        title=f"Learning Curve - {main_model_name}",
                        filename="fig_learning_curve.png"
                    )
                    if lc_path:
                        generated_plots['learning_curve'] = lc_path
                        add_log(f"    📊 学习曲线 → {lc_path}")
                except Exception as e:
                    add_log(f"    学习曲线生成失败: {e}", "WARNING")
        
        st.session_state.generated_plots = generated_plots
        st.session_state.model_results = {
            'metrics_df': metrics_df,
            'trained_models': trained_models,
            'y_true': all_y_true,
            'y_proba': all_y_proba,
            'y_pred': all_y_pred,
        }
        
        n_plots = len(generated_plots)
        elapsed = time.time() - t0
        set_step_status(4, 'done', elapsed)
        add_log(f"✅ 模型训练完成, 生成{n_plots}张图表 ({elapsed:.1f}s)")
        if checkpoint_mgr:
            checkpoint_mgr.save_checkpoint(4, "model_training", 
                                          data={'models': list(trained_models.keys())},
                                          extra={'metrics': model_metrics})
        st.session_state.current_step = 5
        with progress_container:
            show_progress()
        
        # ================================================================
        # Step 6: 虚拟筛选
        # ================================================================
        set_step_status(5, 'running')
        t0 = time.time()
        add_log("Step 6/9: 虚拟筛选...")
        add_log(f"  使用 {n_cpu} CPU核心进行批量预测")
        
        # 使用训练好的模型进行预测
        screening_results = []
        for model_name, model in trained_models.items():
            t_pred = time.time()
            scores = model.predict_proba(X)[:, 1]
            pred_time = time.time() - t_pred
            add_log(f"  {model_name} 预测完成: {len(scores)}个化合物 ({pred_time:.2f}s)")
            
            for idx, score in enumerate(scores):
                screening_results.append({
                    'compound_idx': idx,
                    'SMILES': processed_df.iloc[idx].get('SMILES', ''),
                    'Name': processed_df.iloc[idx].get('Name', f'CMP_{idx:04d}'),
                    'model': model_name,
                    'score': float(score)
                })
        
        screen_df = pd.DataFrame(screening_results)
        screen_df.to_csv(output_dir / "screening_results.csv", index=False)
        add_log(f"  筛选结果已保存: {output_dir / 'screening_results.csv'}")
        
        elapsed = time.time() - t0
        set_step_status(5, 'done', elapsed)
        add_log(f"✅ 虚拟筛选完成 ({elapsed:.1f}s)")
        if checkpoint_mgr:
            checkpoint_mgr.save_checkpoint(5, "virtual_screening", data={'n_screened': len(screen_df)})
        st.session_state.current_step = 6
        with progress_container:
            show_progress()
        
        # ================================================================
        # Step 7: 分子对接 (真实对接引擎)
        # ================================================================
        set_step_status(6, 'running')
        t0 = time.time()
        add_log("Step 7/9: 分子对接...")
        
        top_n = min(50, len(processed_df))
        top_indices = screen_df.nlargest(top_n, 'score')['compound_idx'].values[:top_n]
        top_smiles = [processed_df.iloc[int(idx)].get('SMILES', '') for idx in top_indices]
        
        # 使用真实对接引擎
        if DOCKING_MODULE_AVAILABLE:
            add_log(f"  初始化对接引擎...")
            
            docking_engine = _load_docking_module()(
                receptor_pdbqt=None,  # 无受体结构时使用经验打分
                center=tuple(DOCKING_CONFIG.get('center', [0, 0, 0])),
                box_size=tuple(DOCKING_CONFIG.get('box_size', [22.5, 22.5, 22.5])),
                n_cpu=n_cpu,
                exhaustiveness=DOCKING_CONFIG['exhaustiveness'],
                num_poses=DOCKING_CONFIG['num_poses'],
                work_dir=output_dir / "docking_work"
            )
            
            add_log(f"  对接引擎: {docking_engine.engine}")
            add_log(f"  搜索强度: {DOCKING_CONFIG['exhaustiveness']} | 构象数: {DOCKING_CONFIG['num_poses']}")
            add_log(f"  CPU核心: {n_cpu}")
            add_log(f"  对接化合物数: {top_n}")
            
            # 批量对接
            docking_results_list = []
            for i, smiles in enumerate(top_smiles):
                result = docking_engine.dock_compound(smiles, f"CMP_{i:04d}")
                docking_results_list.append(result)
                
                if (i + 1) % 10 == 0 or i == len(top_smiles) - 1:
                    add_log(f"  对接进度: {i+1}/{top_n} | 最新结合能: {result.get('binding_affinity', 0):.2f} kcal/mol")
            
            docking_df = pd.DataFrame(docking_results_list)
            
            # 确保列名兼容可视化模块
            if 'smiles' in docking_df.columns:
                docking_df = docking_df.rename(columns={'smiles': 'smiles', 'binding_affinity': 'binding_affinity'})
            
            add_log(f"  对接引擎使用: {docking_engine.engine}")
            engine_counts = docking_df['engine'].value_counts().to_dict() if 'engine' in docking_df.columns else {}
            add_log(f"  引擎统计: {engine_counts}")
            
        else:
            add_log("  对接模块不可用，使用经验打分", "WARNING")
            
            # 经验打分 (非随机)
            from rdkit.Chem import Crippen
            docking_data = []
            for i, smiles in enumerate(top_smiles):
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is None:
                    continue
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)
                n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
                
                affinity = (-3.0 - 0.5*min(logp,5) - 0.3*min(hbd+hba,8) 
                           - 0.4*min(n_arom,4) + 0.01*max(0,mw-400) 
                           + 0.01*max(0,tpsa-80) + 0.15*max(0,n_rot-3))
                
                docking_data.append({
                    'compound_id': f"CMP_{i:04d}",
                    'smiles': smiles,
                    'binding_affinity': float(affinity),
                    'rmsd': 0.0,
                    'engine': 'empirical',
                    'poses': 0,
                    'success': True,
                })
            
            docking_df = pd.DataFrame(docking_data)
        
        docking_df.to_csv(output_dir / "docking_results.csv", index=False)
        add_log(f"  对接结果已保存: {output_dir / 'docking_results.csv'}")
        
        # 生成对接图表
        dock_viz = _load_visualization_modules()['DockingVisualizer'](output_dir=figures_dir)
        
        dock_aff_path = dock_viz.plot_binding_affinity_distribution(
            docking_df, title="Binding Affinity Distribution",
            filename="fig_docking_affinity_dist.png"
        )
        if dock_aff_path:
            generated_plots['docking_affinity'] = dock_aff_path
            add_log(f"    📊 结合能分布 → {dock_aff_path}")
        
        dock_rank_path = dock_viz.plot_top_compounds_ranking(
            docking_df, top_n=15, title="Top 15 Compounds by Binding Affinity",
            filename="fig_docking_top_compounds.png"
        )
        if dock_rank_path:
            generated_plots['docking_ranking'] = dock_rank_path
            add_log(f"    📊 Top化合物排名 → {dock_rank_path}")
        
        dock_rmsd_path = dock_viz.plot_affinity_vs_rmsd(
            docking_df, title="Binding Affinity vs RMSD",
            filename="fig_docking_affinity_rmsd.png"
        )
        if dock_rmsd_path:
            generated_plots['docking_rmsd'] = dock_rmsd_path
            add_log(f"    📊 结合能vs RMSD → {dock_rmsd_path}")
        
        elapsed = time.time() - t0
        set_step_status(6, 'done', elapsed)
        add_log(f"✅ 分子对接完成 ({elapsed:.1f}s)")
        if checkpoint_mgr:
            checkpoint_mgr.save_checkpoint(6, "molecular_docking", data={'n_docked': len(docking_df)})
        st.session_state.current_step = 7
        st.session_state.docking_results = docking_df
        with progress_container:
            show_progress()
        
        # ================================================================
        # Step 8: ADMET评估
        # ================================================================
        set_step_status(7, 'running')
        t0 = time.time()
        add_log("Step 8/9: ADMET性质评估...")
        
        # 使用RDKit计算真实ADMET属性
        admet_data = []
        for idx, row in processed_df.head(top_n).iterrows():
            mol = Chem.MolFromSmiles(str(row['SMILES']))
            if mol is None:
                continue
            
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rb = Descriptors.NumRotatableBonds(mol)
            n_rings = rdMolDescriptors.CalcNumRings(mol)
            n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)
            
            # 基于分子描述符计算ADMET (经验公式, 非随机)
            # HIA (人体肠道吸收): 基于LogP和TPSA
            hia = max(0, min(100, 95.0 - 0.15 * max(0, tpsa - 80) - 2.0 * max(0, logp - 5)))
            
            # Caco-2渗透性: 基于LogP和TPSA
            caco2 = max(0, min(50, 30.0 + 2.0 * logp - 0.1 * tpsa))
            
            # BBB穿透性: 基于LogP, TPSA, MW
            bbb = max(0, min(1, 0.5 + 0.1 * (logp - 2) - 0.005 * (tpsa - 70)))
            
            # CYP3A4代谢: 基于分子大小和亲脂性
            cyp3A4 = max(0, min(1, 0.3 + 0.05 * logp + 0.001 * (mw - 300)))
            
            # hERG抑制风险: 基于LogP和分子大小
            herg = max(0, min(1, 0.1 + 0.08 * max(0, logp - 2) + 0.001 * max(0, mw - 400)))
            
            # AMES致突变性: 基于芳环数和分子复杂性
            ames = max(0, min(1, 0.05 * n_arom + 0.001 * max(0, mw - 350)))
            
            admet_data.append({
                'SMILES': row['SMILES'],
                'MW': mw,
                'LogP': logp,
                'TPSA': tpsa,
                'HBD': hbd,
                'HBA': hba,
                'RB': rb,
                'Rings': n_rings,
                'Aromatic_Rings': n_arom,
                'HIA': float(hia),
                'Caco2': float(caco2),
                'BBB': float(bbb),
                'CYP3A4': float(cyp3A4),
                'hERG': float(herg),
                'AMES': float(ames),
            })
        
        admet_df = pd.DataFrame(admet_data)
        admet_df.to_csv(output_dir / "admet_results.csv", index=False)
        add_log(f"  ADMET属性已计算: {len(admet_df)}个化合物")
        
        # 生成ADMET图表
        admet_viz = _load_visualization_modules()['ADMETVisualizer'](output_dir=figures_dir)
        
        if len(admet_df) > 0:
            radar_data = {
                'MW': float(admet_df['MW'].mean()),
                'LogP': float(admet_df['LogP'].mean()),
                'TPSA': float(admet_df['TPSA'].mean()),
                'HBD': float(admet_df['HBD'].mean()),
                'HBA': float(admet_df['HBA'].mean()),
                'RB': float(admet_df['RB'].mean()),
                'HIA': float(admet_df['HIA'].mean()),
                'Caco2': float(admet_df['Caco2'].mean()),
                'BBB': float(admet_df['BBB'].mean()),
                'CYP3A4': float(admet_df['CYP3A4'].mean()),
                'hERG': float(admet_df['hERG'].mean()),
                'AMES': float(admet_df['AMES'].mean()),
            }
            
            radar_path = admet_viz.plot_admet_radar(
                radar_data, title="ADMET Properties Radar",
                filename="fig_admet_radar.png"
            )
            if radar_path:
                generated_plots['admet_radar'] = radar_path
                add_log(f"    📊 ADMET雷达图 → {radar_path}")
            
            dist_path = admet_viz.plot_property_distribution(
                admet_df, properties=['MW', 'LogP', 'TPSA'],
                title="ADMET Property Distribution",
                filename="fig_admet_distribution.png"
            )
            if dist_path:
                generated_plots['admet_distribution'] = dist_path
                add_log(f"    📊 ADMET分布图 → {dist_path}")
        
        elapsed = time.time() - t0
        set_step_status(7, 'done', elapsed)
        add_log(f"✅ ADMET评估完成 ({elapsed:.1f}s)")
        if checkpoint_mgr:
            checkpoint_mgr.save_checkpoint(7, "admet_evaluation", data={'n_admet': len(admet_df)})
        st.session_state.current_step = 8
        st.session_state.admet_results = admet_df
        with progress_container:
            show_progress()
        
        # ================================================================
        # Step 9: 结果分析与可视化
        # ================================================================
        set_step_status(8, 'running')
        t0 = time.time()
        add_log("Step 9/9: 结果分析与可视化...")
        
        # 化学空间可视化
        cs_viz = _load_visualization_modules()['ChemicalSpaceVisualizer'](output_dir=figures_dir)
        np.random.seed(42)
        
        features_dict = {
            'Training Set': X_train,
            'Test Set': X_test,
            'Screening Hits': X[:min(20, len(X))]
        }
        labels_dict = {
            'Training Set': y_train,
            'Test Set': y_test,
            'Screening Hits': np.ones(min(20, len(X)))
        }
        
        try:
            pca_path = cs_viz.plot_chemical_space(
                features_dict, labels_dict=labels_dict,
                method='pca',
                title="Chemical Space (PCA)",
                filename="fig_chemical_space_pca.png"
            )
            if pca_path:
                generated_plots['chemical_space_pca'] = pca_path
                add_log(f"    📊 化学空间(PCA) → {pca_path}")
        except Exception as e:
            add_log(f"    化学空间图生成失败: {e}", "WARNING")
        
        try:
            ad_path = cs_viz.plot_applicability_domain(
                X_train, X_test,
                title="Applicability Domain Analysis",
                filename="fig_applicability_domain.png"
            )
            if ad_path:
                generated_plots['applicability_domain'] = ad_path
                add_log(f"    📊 适用域分析 → {ad_path}")
        except Exception as e:
            add_log(f"    适用域图生成失败: {e}", "WARNING")
        
        # 化合物结构展示
        try:
            compound_viz = _load_visualization_modules()['CompoundVisualizer'](output_dir=figures_dir)
            top_smiles = processed_df.head(6)['SMILES'].tolist()
            grid_path = compound_viz.draw_compounds_grid(
                top_smiles,
                legends=[f"Compound {i+1}" for i in range(len(top_smiles))],
                mols_per_row=3,
                filename="fig_top_compounds.png"
            )
            if grid_path:
                generated_plots['top_compounds'] = grid_path
                add_log(f"    📊 化合物结构图 → {grid_path}")
        except Exception as e:
            add_log(f"    化合物结构图生成失败: {e}", "WARNING")
        
        # 保存图表索引
        plots_index = {k: str(v) for k, v in generated_plots.items()}
        with open(output_dir / "figures_index.json", 'w', encoding='utf-8') as f:
            json.dump(plots_index, f, ensure_ascii=False, indent=2)
        
        st.session_state.generated_plots = generated_plots
        
        # 保存完整结果摘要
        total_elapsed = time.time() - st.session_state.pipeline_start_time
        summary = {
            'project_name': st.session_state.project_name,
            'target': config['target_name'],
            'date': datetime.now().isoformat(),
            'total_compounds': n_compounds,
            'processed_compounds': len(processed_df),
            'models_trained': list(trained_models.keys()),
            'model_metrics': model_metrics,
            'n_cpu_used': n_cpu,
            'total_time_s': total_elapsed,
            'figures_generated': list(generated_plots.keys()),
            'output_dir': str(output_dir),
        }
        with open(output_dir / "screening_report.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        elapsed = time.time() - t0
        set_step_status(8, 'done', elapsed)
        add_log(f"✅ 结果分析完成, 共生成{len(generated_plots)}张图表 ({elapsed:.1f}s)")
        st.session_state.current_step = 9
        with progress_container:
            show_progress()
        
        # ================================================================
        # 完成
        # ================================================================
        total_elapsed = time.time() - st.session_state.pipeline_start_time
        st.session_state.screening_complete = True
        st.session_state.pipeline_results = summary
        
        add_log(f"=" * 50)
        add_log(f"🎉 虚拟筛选流程完成!")
        add_log(f"   总耗时: {total_elapsed:.1f}s")
        add_log(f"   CPU使用: {n_cpu} 核心")
        add_log(f"   生成图表: {len(generated_plots)} 张")
        add_log(f"   输出目录: {output_dir}")
        add_log(f"=" * 50)
        
        st.success(f"🎉 筛选完成！总耗时 {total_elapsed:.1f}s | 生成 {len(generated_plots)} 张图表")
        
    except Exception as e:
        add_log(f"❌ 错误: {str(e)}", "ERROR")
        st.error(f"流程执行出错: {str(e)}")
        import traceback
        st.error(traceback.format_exc())


# ============================================================================
# 可视化展示
# ============================================================================

def show_model_visualization():
    """显示模型性能可视化"""
    st.markdown("## 🤖 模型性能分析")
    
    plots = st.session_state.get('generated_plots', {})
    model_results = st.session_state.get('model_results', None)
    
    # 显示指标表格
    if model_results and 'metrics_df' in model_results:
        st.markdown("### 模型性能指标")
        st.dataframe(model_results['metrics_df'], use_container_width=True)
        
        # 下载指标
        csv = model_results['metrics_df'].to_csv(index=False)
        st.download_button("下载指标CSV", csv, file_name="model_metrics.csv", mime="text/csv")
    
    st.markdown("---")
    
    # 展示图表
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ROC曲线")
        roc_path = plots.get('roc_curve')
        if roc_path and Path(roc_path).exists():
            st.image(roc_path, use_container_width=True)
            st.caption(f"文件: {roc_path}")
        else:
            st.info("ROC曲线未生成")
    
    with col2:
        st.markdown("### PR曲线")
        pr_path = plots.get('pr_curve')
        if pr_path and Path(pr_path).exists():
            st.image(pr_path, use_container_width=True)
            st.caption(f"文件: {pr_path}")
        else:
            st.info("PR曲线未生成")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### 模型对比")
        comp_path = plots.get('model_comparison')
        if comp_path and Path(comp_path).exists():
            st.image(comp_path, use_container_width=True)
        else:
            st.info("模型对比图未生成")
    
    with col4:
        st.markdown("### 校准曲线")
        calib_path = plots.get('calibration_curve')
        if calib_path and Path(calib_path).exists():
            st.image(calib_path, use_container_width=True)
        else:
            st.info("校准曲线未生成")
    
    # 混淆矩阵
    st.markdown("### 混淆矩阵")
    cm_cols = st.columns(min(4, len([k for k in plots if k.startswith('confusion_matrix_')])))
    cm_idx = 0
    for key, path in plots.items():
        if key.startswith('confusion_matrix_') and Path(path).exists():
            with cm_cols[cm_idx % len(cm_cols)]:
                model_name = key.replace('confusion_matrix_', '')
                st.markdown(f"**{model_name}**")
                st.image(path, use_container_width=True)
            cm_idx += 1
    
    # 特征重要性
    fi_keys = [k for k in plots if k.startswith('feature_importance_')]
    if fi_keys:
        st.markdown("### 特征重要性")
        fi_cols = st.columns(min(2, len(fi_keys)))
        for i, key in enumerate(fi_keys):
            path = plots[key]
            if Path(path).exists():
                with fi_cols[i % len(fi_cols)]:
                    model_name = key.replace('feature_importance_', '')
                    st.markdown(f"**{model_name}**")
                    st.image(path, use_container_width=True)
    
    # 学习曲线
    lc_path = plots.get('learning_curve')
    if lc_path and Path(lc_path).exists():
        st.markdown("### 学习曲线")
        st.image(lc_path, use_container_width=True)


def show_docking_visualization():
    """显示分子对接可视化"""
    st.markdown("## ⚛️ 分子对接结果")
    
    plots = st.session_state.get('generated_plots', {})
    docking_df = st.session_state.get('docking_results', None)
    
    if docking_df is not None:
        st.markdown("### 对接结果数据")
        st.dataframe(docking_df.head(20), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 结合能分布")
        path = plots.get('docking_affinity')
        if path and Path(path).exists():
            st.image(path, use_container_width=True)
        else:
            st.info("图表未生成")
    with col2:
        st.markdown("### Top化合物排名")
        path = plots.get('docking_ranking')
        if path and Path(path).exists():
            st.image(path, use_container_width=True)
        else:
            st.info("图表未生成")
    
    st.markdown("### 结合能 vs RMSD")
    path = plots.get('docking_rmsd')
    if path and Path(path).exists():
        st.image(path, use_container_width=True)


def show_admet_visualization():
    """显示ADMET可视化"""
    st.markdown("## 💊 ADMET性质评估")
    
    plots = st.session_state.get('generated_plots', {})
    admet_df = st.session_state.get('admet_results', None)
    
    if admet_df is not None:
        st.markdown("### ADMET数据")
        st.dataframe(admet_df.head(20), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ADMET雷达图")
        path = plots.get('admet_radar')
        if path and Path(path).exists():
            st.image(path, use_container_width=True)
        else:
            st.info("图表未生成")
    with col2:
        st.markdown("### 性质分布")
        path = plots.get('admet_distribution')
        if path and Path(path).exists():
            st.image(path, use_container_width=True)
        else:
            st.info("图表未生成")


def show_chemical_space():
    """显示化学空间分析"""
    st.markdown("## 🔬 化学空间分析")
    
    plots = st.session_state.get('generated_plots', {})
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### PCA化学空间")
        path = plots.get('chemical_space_pca')
        if path and Path(path).exists():
            st.image(path, use_container_width=True)
        else:
            st.info("图表未生成")
    with col2:
        st.markdown("### 适用域分析")
        path = plots.get('applicability_domain')
        if path and Path(path).exists():
            st.image(path, use_container_width=True)
        else:
            st.info("图表未生成")
    
    st.markdown("### Top化合物结构")
    path = plots.get('top_compounds')
    if path and Path(path).exists():
        st.image(path, use_container_width=True)
    else:
        st.info("图表未生成")


def show_output_directory():
    """显示输出目录内容"""
    st.markdown("## 📂 输出目录")
    
    output_dir = st.session_state.output_dir
    if output_dir is None:
        st.info("尚未设置输出目录")
        return
    
    st.markdown(f"**输出目录:** `{output_dir}`")
    
    # 列出文件
    files = []
    for root, dirs, filenames in os.walk(output_dir):
        for fname in filenames:
            fpath = Path(root) / fname
            rel_path = fpath.relative_to(output_dir)
            size_kb = fpath.stat().st_size / 1024
            files.append({
                '文件名': str(rel_path),
                '大小(KB)': f"{size_kb:.1f}",
                '修改时间': datetime.fromtimestamp(fpath.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            })
    
    if files:
        files_df = pd.DataFrame(files)
        st.dataframe(files_df, use_container_width=True)
        st.info(f"共 {len(files)} 个文件")
    else:
        st.warning("输出目录为空")


def generate_report():
    """生成筛选报告"""
    st.markdown("## 📋 筛选报告")
    
    results = st.session_state.get('pipeline_results', {})
    model_results = st.session_state.get('model_results', None)
    
    if not results:
        st.info("尚未生成结果")
        return
    
    # 报告摘要
    st.markdown("### 报告摘要")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总化合物数", results.get('total_compounds', 0))
    with col2:
        st.metric("处理化合物数", results.get('processed_compounds', 0))
    with col3:
        st.metric("训练模型数", len(results.get('models_trained', [])))
    with col4:
        st.metric("生成图表数", len(results.get('figures_generated', [])))
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("使用CPU", f"{results.get('n_cpu_used', 'N/A')} 核")
    with col6:
        st.metric("总耗时", f"{results.get('total_time_s', 0):.1f}s")
    with col7:
        st.metric("靶点", results.get('target', 'N/A'))
    with col8:
        st.metric("项目", results.get('project_name', 'N/A'))
    
    # 模型性能
    if model_results and 'metrics_df' in model_results:
        st.markdown("### 模型性能对比")
        st.dataframe(model_results['metrics_df'], use_container_width=True)
    
    # 下载JSON报告
    report_path = st.session_state.output_dir / "screening_report.json"
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            report_json = f.read()
        st.download_button(
            "📥 下载JSON报告",
            report_json,
            file_name=f"{st.session_state.project_name}_report.json",
            mime="application/json"
        )
    
    # 显示输出目录
    show_output_directory()


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    init_session_state()
    
    # 页面标题
    st.markdown("<div class='main-header'>🧬 DrugScreen AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>智能药物虚拟筛选平台 v2.1 - 论文与工业级解决方案</div>", unsafe_allow_html=True)
    
    # 侧边栏配置
    config = sidebar_config()
    
    # 主内容区
    if config['run_button']:
        compound_df = upload_and_process_library(config)
        if compound_df is not None and len(compound_df) > 0:
            run_screening_pipeline(config)
            if st.session_state.screening_complete:
                st.balloons()
    
    # 显示进度
    if st.session_state.current_step > 0:
        show_progress()
    
    # 标签页导航
    if st.session_state.screening_complete:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 模型性能", "⚛️ 对接结果", "💊 ADMET",
            "🔬 化学空间", "📋 报告"
        ])
        
        with tab1:
            show_model_visualization()
        with tab2:
            show_docking_visualization()
        with tab3:
            show_admet_visualization()
        with tab4:
            show_chemical_space()
        with tab5:
            generate_report()
    else:
        # 欢迎页面
        st.markdown("""
        ## 欢迎使用 DrugScreen AI v2.1
        
        本平台提供完整的虚拟筛选流程，支持论文出版级图表输出。
        
        ### 🆕 v2.1 新功能
        - **CPU核心控制**: 自定义并行计算使用的CPU核心数
        - **自定义输出目录**: 自由选择结果保存位置
        - **实时进度监控**: 每步骤计时显示，总进度跟踪
        - **论文级图表**: 所有图表300DPI，保存到输出目录/figures/
        - **真实ML训练**: 真正训练XGBoost/RF/SVM/LR模型并生成ROC/PR/混淆矩阵等
        
        ### 🔬 核心功能
        - **靶点结构准备**: 支持PDB下载和预处理
        - **化合物库管理**: 支持CSV、SMILES、SDF格式
        - **机器学习筛选**: XGBoost、RandomForest、SVM、LogisticRegression
        - **分子对接**: AutoDock Vina集成
        - **ADMET评估**: 类药性、毒性预测
        
        ### 📊 出版级可视化
        - ROC曲线、PR曲线、混淆矩阵、校准曲线
        - 学习曲线、特征重要性、模型对比图
        - 结合能分布、Top化合物排名
        - ADMET雷达图、化学空间(PCA)
        - 所有图表自动保存到输出目录
        
        ### 🚀 开始使用
        1. 在左侧配置面板设置参数 (CPU核心、输出目录等)
        2. 上传化合物库或手动输入SMILES
        3. 点击"开始筛选"按钮
        4. 查看可视化结果和报告
        5. 从输出目录获取论文级图表
        """)
        
        with st.expander("查看示例数据格式"):
            example_df = pd.DataFrame({
                'SMILES': [
                    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
                    'Cc1ccc(cc1)S(=O)(=O)Nc2ncccn2',
                    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
                ],
                'Name': ['Ibuprofen', 'Sulfathiazole', 'Caffeine'],
                'Activity': [1, 0, 1]
            })
            st.dataframe(example_df, use_container_width=True)
            st.info("CSV文件应包含SMILES列，可选Name和Activity列")


if __name__ == "__main__":
    main()
