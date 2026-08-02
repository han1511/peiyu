# -*- mode: python ; coding: utf-8 -*-
"""
DrugScreen AI - PyInstaller 打包配置

使用方法:
  python build_exe.py              # 推荐: 一键构建
  pyinstaller drugscreen.spec      # 手动构建

打包后双击 DrugScreenAI.exe 即可运行
"""

import sys
import os
from pathlib import Path

block_cipher = None

# 项目根目录
PROJECT_ROOT = Path(os.getcwd()).resolve()

# ============================================================================
# 收集数据文件
# ============================================================================

datas = []

# 配置文件
for f in ['configs/settings.yaml', 'configs/config.py', 'configs/config_loader.py']:
    src = str(PROJECT_ROOT / f)
    dst = str(Path(f).parent)
    if os.path.exists(src):
        datas.append((src, dst))

# src目录所有.py文件
src_root = PROJECT_ROOT / 'src'
if src_root.exists():
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if file.endswith('.py'):
                src_file = os.path.join(root, file)
                rel_dir = os.path.relpath(root, PROJECT_ROOT)
                datas.append((src_file, rel_dir))

# data目录
data_root = PROJECT_ROOT / 'data'
if data_root.exists():
    for root, dirs, files in os.walk(data_root):
        for file in files:
            src_file = os.path.join(root, file)
            rel_dir = os.path.relpath(root, PROJECT_ROOT)
            datas.append((src_file, rel_dir))

# app.py
app_path = str(PROJECT_ROOT / 'app.py')
if os.path.exists(app_path):
    datas.append((app_path, '.'))

# settings.yaml (额外副本到根目录)
settings_path = str(PROJECT_ROOT / 'configs' / 'settings.yaml')
if os.path.exists(settings_path):
    datas.append((settings_path, 'configs'))

# ============================================================================
# 隐式导入
# ============================================================================

hiddenimports = [
    # Streamlit
    'streamlit',
    'streamlit.runtime',
    'streamlit.runtime.scriptrunner',
    'streamlit.runtime.state',
    'streamlit.web.bootstrap',
    'streamlit.web.server',
    'streamlit.components',
    
    # 数据处理
    'pandas', 'numpy', 'scipy',
    'sklearn', 'sklearn.ensemble', 'sklearn.svm',
    'sklearn.linear_model', 'sklearn.neighbors',
    'sklearn.preprocessing', 'sklearn.model_selection',
    'sklearn.metrics', 'sklearn.decomposition',
    'sklearn.manifold', 'sklearn.utils',
    
    # 可视化
    'matplotlib', 'matplotlib.pyplot',
    'matplotlib.backends.backend_agg',
    'seaborn', 'plotly', 'plotly.graph_objects',
    'plotly.express', 'plotly.subplots',
    
    # RDKit
    'rdkit', 'rdkit.Chem', 'rdkit.Chem.Draw',
    'rdkit.Chem.Descriptors', 'rdkit.Chem.rdMolDescriptors',
    
    # XGBoost
    'xgboost',
    
    # YAML
    'yaml',
    
    # pywebview
    'webview', 'webview.platforms',
    
    # 项目模块
    'configs', 'configs.config', 'configs.config_loader',
    'src', 'src.visualization',
    'src.visualization.model_plots',
    'src.visualization.docking_plots',
    'src.visualization.admet_plots',
    'src.visualization.chemical_space',
    'src.visualization.compound_plots',
    'src.quality_control', 'src.report_generator',
    'src.pipeline_v2', 'src.target_preparation',
    'src.compound_library', 'src.molecular_features',
    'src.ml_screening', 'src.molecular_docking',
    'src.admet_evaluation', 'src.result_analysis',
]

# Windows pywebview 后端
if sys.platform == 'win32':
    hiddenimports.extend([
        'webview.platforms.winforms',
        'webview.platforms.edgechromium',
    ])

# 可选依赖
for opt_pkg in ['joblib', 'PIL', 'PIL.Image', 'biopython', 'Bio']:
    try:
        __import__(opt_pkg)
        hiddenimports.append(opt_pkg)
    except ImportError:
        pass

# ============================================================================
# 排除模块
# ============================================================================

excludes = [
    'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    'IPython', 'jupyter', 'notebook', 'pytest', 'unittest',
    'deepchem', 'torchvision', 'pubchempy',
    'torch',  # 体积太大，按需启用
]

# ============================================================================
# Analysis
# ============================================================================

a = Analysis(
    ['desktop_app.py'],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(PROJECT_ROOT / 'hooks')],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ============================================================================
# EXE 配置
# ============================================================================

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='DrugScreenAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[
        # 排除不压缩的DLL
        'vcruntime140.dll',
        'python3.dll',
        'mkl_*.dll',
    ],
    runtime_tmpdir=None,
    console=True,  # 保留控制台以便查看错误
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 可设置icon='assets/icon.ico'
)
