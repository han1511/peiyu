#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚拟筛选 GUI 应用程序 - 优化版本
用于界面化操作虚拟筛选过程，支持选择不同的机器学习模型
优化内容：
1. 详细的模型参数显示
2. 真正的停止功能
3. 正确的 CPU 核数使用
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import pandas as pd
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 辅助函数：计算单个 SMILES 的特征
def calculate_single_smiles_features(smiles, stop_flag=None):
    """
    计算单个 SMILES 字符串的分子特征
    
    参数:
        smiles: SMILES 字符串
        stop_flag: 停止标志，用于中断计算
    
    返回:
        features: 特征向量，如果计算失败返回 None
    """
    try:
        # 检查是否应该停止
        if stop_flag is not None and stop_flag[0]:
            return None
        
        # 创建分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 计算 Morgan 指纹
        morgan_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))
        
        # 计算 MACCS 指纹
        maccs_fp = np.array(MACCSkeys.GenMACCSKeys(mol))
        
        # 计算 RDKit 描述符
        descriptors = {}
        for desc_name, desc_func in Descriptors._descList:
            try:
                value = desc_func(mol)
                # 处理无限值和 NaN 值
                if np.isinf(value) or np.isnan(value):
                    value = 0
                descriptors[desc_name] = value
            except:
                descriptors[desc_name] = 0
        
        # 合并所有特征
        all_features = np.concatenate([
            morgan_fp,
            maccs_fp,
            np.array(list(descriptors.values()))
        ])
        
        return all_features
    except Exception as e:
        print(f"计算特征时出错：{str(e)}")
        return None


class ScreeningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("虚拟筛选工具")
        self.root.geometry("1200x800")
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建滚动区域
        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 绑定鼠标滚轮事件
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # 创建标题
        self.title_label = ttk.Label(self.scrollable_frame, text="虚拟筛选工具", font=("SimHei", 24))
        self.title_label.pack(pady=20)
        
        # 创建标签页
        self.notebook = ttk.Notebook(self.scrollable_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 虚拟筛选标签页
        self.screening_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.screening_tab, text="虚拟筛选")
        
        # 模型构建标签页
        self.model_build_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.model_build_tab, text="模型构建")
        
        # ==================== 虚拟筛选标签页 ====================
        # 创建输入框架
        self.input_frame = ttk.LabelFrame(self.screening_tab, text="输入设置", padding="10")
        self.input_frame.pack(fill=tk.X, pady=10)
        
        # 目标蛋白信息
        self.screening_target_frame = ttk.Frame(self.input_frame)
        self.screening_target_frame.pack(fill=tk.X, pady=5)
        
        self.screening_target_label = ttk.Label(self.screening_target_frame, text="目标蛋白:", width=15)
        self.screening_target_label.pack(side=tk.LEFT, padx=5)
        
        self.screening_target_var = tk.StringVar(value="Dengue Virus NS3 Protease")
        self.screening_target_entry = ttk.Entry(self.screening_target_frame, textvariable=self.screening_target_var, width=50)
        self.screening_target_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # 筛选依据
        self.screening_basis_frame = ttk.Frame(self.input_frame)
        self.screening_basis_frame.pack(fill=tk.X, pady=5)
        
        self.screening_basis_label = ttk.Label(self.screening_basis_frame, text="筛选依据:", width=15)
        self.screening_basis_label.pack(side=tk.LEFT, padx=5)
        
        self.screening_basis_var = tk.StringVar(value="基于训练好的机器学习模型预测")
        self.screening_basis_entry = ttk.Entry(self.screening_basis_frame, textvariable=self.screening_basis_var, width=50)
        self.screening_basis_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # 化合物文件选择
        self.file_frame = ttk.Frame(self.input_frame)
        self.file_frame.pack(fill=tk.X, pady=5)
        
        self.file_label = ttk.Label(self.file_frame, text="化合物文件:", width=15)
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        self.file_var = tk.StringVar()
        self.file_entry = ttk.Entry(self.file_frame, textvariable=self.file_var, width=50)
        self.file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.browse_button = ttk.Button(self.file_frame, text="浏览", command=self.browse_file)
        self.browse_button.pack(side=tk.LEFT, padx=5)
        
        # SMILES 列名设置
        self.smiles_frame = ttk.Frame(self.input_frame)
        self.smiles_frame.pack(fill=tk.X, pady=5)
        
        self.smiles_label = ttk.Label(self.smiles_frame, text="SMILES 列名:", width=15)
        self.smiles_label.pack(side=tk.LEFT, padx=5)
        
        self.smiles_var = tk.StringVar(value="SMILES")
        self.smiles_entry = ttk.Entry(self.smiles_frame, textvariable=self.smiles_var, width=20)
        self.smiles_entry.pack(side=tk.LEFT, padx=5)
        
        # 模型选择框架
        self.model_frame = ttk.LabelFrame(self.screening_tab, text="模型设置", padding="10")
        self.model_frame.pack(fill=tk.X, pady=10)
        
        # 模型选择
        self.model_select_frame = ttk.Frame(self.model_frame)
        self.model_select_frame.pack(fill=tk.X, pady=5)
        
        self.model_label = ttk.Label(self.model_select_frame, text="选择模型:", width=15)
        self.model_label.pack(side=tk.LEFT, padx=5)
        
        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(self.model_select_frame, textvariable=self.model_var, width=20, state="readonly")
        self.model_combobox['values'] = self.get_available_models()
        if self.model_combobox['values']:
            self.model_combobox.current(0)
        self.model_combobox.pack(side=tk.LEFT, padx=5)
        
        # 阈值设置
        self.threshold_frame = ttk.Frame(self.model_frame)
        self.threshold_frame.pack(fill=tk.X, pady=5)
        
        self.threshold_label = ttk.Label(self.threshold_frame, text="预测阈值:", width=15)
        self.threshold_label.pack(side=tk.LEFT, padx=5)
        
        self.threshold_var = tk.DoubleVar(value=0.4)
        self.threshold_scale = ttk.Scale(self.threshold_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                                        variable=self.threshold_var, length=200)
        self.threshold_scale.pack(side=tk.LEFT, padx=5)
        
        self.threshold_entry = ttk.Entry(self.threshold_frame, textvariable=self.threshold_var, width=10)
        self.threshold_entry.pack(side=tk.LEFT, padx=5)
        
        # 模型参数设置
        self.params_frame = ttk.LabelFrame(self.screening_tab, text="模型参数", padding="10")
        self.params_frame.pack(fill=tk.X, pady=10)
        
        # 随机森林参数
        self.rf_frame = ttk.LabelFrame(self.params_frame, text="RandomForest 参数", padding="10")
        self.rf_frame.pack(fill=tk.X, pady=5)
        
        self.n_estimators_frame = ttk.Frame(self.rf_frame)
        self.n_estimators_frame.pack(fill=tk.X, pady=5)
        
        self.n_estimators_label = ttk.Label(self.n_estimators_frame, text="树的数量:", width=15)
        self.n_estimators_label.pack(side=tk.LEFT, padx=5)
        
        self.n_estimators_var = tk.IntVar(value=100)
        self.n_estimators_entry = ttk.Entry(self.n_estimators_frame, textvariable=self.n_estimators_var, width=10)
        self.n_estimators_entry.pack(side=tk.LEFT, padx=5)
        
        #  ==================== 模型构建标签页 ====================
        self.model_build_input_frame = ttk.LabelFrame(self.model_build_tab, text="训练数据设置", padding="10")
        self.model_build_input_frame.pack(fill=tk.X, pady=10)
        
        # 目标蛋白信息
        self.target_protein_frame = ttk.Frame(self.model_build_input_frame)
        self.target_protein_frame.pack(fill=tk.X, pady=5)
        
        self.target_protein_label = ttk.Label(self.target_protein_frame, text="目标蛋白:", width=15)
        self.target_protein_label.pack(side=tk.LEFT, padx=5)
        
        self.target_protein_var = tk.StringVar(value="Dengue Virus NS3 Protease")
        self.target_protein_entry = ttk.Entry(self.target_protein_frame, textvariable=self.target_protein_var, width=50)
        self.target_protein_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # 模型构建依据
        self.model_basis_frame = ttk.Frame(self.model_build_input_frame)
        self.model_basis_frame.pack(fill=tk.X, pady=5)
        
        self.model_basis_label = ttk.Label(self.model_basis_frame, text="构建依据:", width=15)
        self.model_basis_label.pack(side=tk.LEFT, padx=5)
        
        self.model_basis_var = tk.StringVar(value="基于化合物的分子特征和活性标签")
        self.model_basis_entry = ttk.Entry(self.model_basis_frame, textvariable=self.model_basis_var, width=50)
        self.model_basis_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # 训练数据文件选择
        self.train_file_frame = ttk.Frame(self.model_build_input_frame)
        self.train_file_frame.pack(fill=tk.X, pady=5)
        
        self.train_file_label = ttk.Label(self.train_file_frame, text="训练数据文件:", width=15)
        self.train_file_label.pack(side=tk.LEFT, padx=5)
        
        self.train_file_var = tk.StringVar()
        self.train_file_entry = ttk.Entry(self.train_file_frame, textvariable=self.train_file_var, width=50)
        self.train_file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.train_browse_button = ttk.Button(self.train_file_frame, text="浏览", command=self.browse_train_file)
        self.train_browse_button.pack(side=tk.LEFT, padx=5)
        
        # 模型类型选择
        self.model_type_frame = ttk.Frame(self.model_build_input_frame)
        self.model_type_frame.pack(fill=tk.X, pady=5)
        
        self.model_type_label = ttk.Label(self.model_type_frame, text="模型类型:", width=15)
        self.model_type_label.pack(side=tk.LEFT, padx=5)
        
        self.model_type_var = tk.StringVar(value="RandomForest")
        self.model_type_combobox = ttk.Combobox(self.model_type_frame, textvariable=self.model_type_var, width=25, state="readonly")
        self.model_type_combobox['values'] = [
            "RandomForest", 
            "XGBoost", 
            "SVM",
            "LogisticRegression",
            "GradientBoosting",
            "AdaBoost",
            "ExtraTrees",
            "LightGBM",
            "GNN (Graph Neural Network)",
            "Transformer (SMILES-BERT)"
        ]
        self.model_type_combobox.current(0)
        self.model_type_combobox.pack(side=tk.LEFT, padx=5)
        
        # 硬件加速设置
        self.hardware_frame = ttk.LabelFrame(self.model_build_input_frame, text="硬件加速", padding="10")
        self.hardware_frame.pack(fill=tk.X, pady=5)
        
        # GPU 检测
        self.gpu_available = False
        self.gpu_error = ""
        
        # 首先检测 PyTorch GPU
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                self.gpu_error = f"PyTorch GPU 检测成功：{torch.cuda.get_device_name(0)}"
            else:
                # 然后检测 XGBoost GPU
                try:
                    import xgboost as xgb
                    self.gpu_error = f"XGBoost 版本：{xgb.__version__}"
                    if xgb.__version__ >= '1.0.0':
                        import numpy as np
                        # 尝试创建一个小的 GPU DMatrix 来检测 GPU
                        dtest = xgb.DMatrix(np.random.rand(10, 10), label=np.random.randint(0, 2, 10))
                        params = {'tree_method': 'gpu_hist'}
                        # 测试是否可以使用 GPU
                        try:
                            xgb.train(params, dtest, num_boost_round=1)
                            self.gpu_available = True
                            self.gpu_error = "XGBoost GPU 检测成功"
                        except Exception as e:
                            self.gpu_available = False
                            self.gpu_error = f"XGBoost GPU 训练失败：{str(e)}"
                    else:
                        self.gpu_available = False
                        self.gpu_error = "XGBoost 版本过低，需要 1.0.0 或更高"
                except ImportError:
                    self.gpu_available = False
                    self.gpu_error = "未安装 XGBoost 库 (仅影响 XGBoost 和 LightGBM 模型)"
        except ImportError:
            self.gpu_available = False
            self.gpu_error = "未安装 PyTorch 库 (仅影响深度学习模型)"
        except Exception as e:
            self.gpu_available = False
            self.gpu_error = f"检测失败：{str(e)}"
        
        # GPU 选项
        self.gpu_frame = ttk.Frame(self.hardware_frame)
        self.gpu_frame.pack(fill=tk.X, pady=5)
        
        self.gpu_label = ttk.Label(self.gpu_frame, text="GPU 加速:", width=15)
        self.gpu_var = tk.BooleanVar(value=self.gpu_available)
        self.gpu_checkbox = ttk.Checkbutton(self.gpu_frame, text="使用 GPU 加速", variable=self.gpu_var, state=tk.NORMAL if self.gpu_available else tk.DISABLED)
        self.gpu_label.pack(side=tk.LEFT, padx=5)
        self.gpu_checkbox.pack(side=tk.LEFT, padx=5)
        
        if not self.gpu_available:
            # 限制错误信息长度
            error_msg = self.gpu_error
            if len(error_msg) > 50:
                error_msg = error_msg[:50] + "..."
            self.gpu_info_label = ttk.Label(self.gpu_frame, text=f"(未检测到可用 GPU: {error_msg})", foreground="gray", wraplength=400)
            self.gpu_info_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        else:
            self.gpu_info_label = ttk.Label(self.gpu_frame, text="(已检测到可用 GPU)", foreground="green")
            self.gpu_info_label.pack(side=tk.LEFT, padx=5)
        
        # CPU 核数设置
        self.cpu_frame = ttk.Frame(self.hardware_frame)
        self.cpu_frame.pack(fill=tk.X, pady=5)
        
        self.cpu_label = ttk.Label(self.cpu_frame, text="CPU 核数:", width=15)
        self.cpu_label.pack(side=tk.LEFT, padx=5)
        
        self.cpu_var = tk.StringVar(value="全部")
        self.cpu_combobox = ttk.Combobox(self.cpu_frame, textvariable=self.cpu_var, width=20, state="readonly")
        max_cpus = multiprocessing.cpu_count()
        cpu_options = ["全部"] + [str(i) for i in range(1, max_cpus + 1)]
        self.cpu_combobox['values'] = cpu_options
        self.cpu_combobox.current(0)
        self.cpu_combobox.pack(side=tk.LEFT, padx=5)
        
        self.cpu_info_label = ttk.Label(self.cpu_frame, text=f"(当前系统：{max_cpus} 核)", foreground="gray")
        self.cpu_info_label.pack(side=tk.LEFT, padx=5)
        
        # 数据生成框架
        self.data_gen_frame = ttk.LabelFrame(self.model_build_tab, text="数据生成", padding="10")
        self.data_gen_frame.pack(fill=tk.X, pady=10)
        
        # 源文件选择
        self.source_file_frame = ttk.Frame(self.data_gen_frame)
        self.source_file_frame.pack(fill=tk.X, pady=5)
        
        self.source_file_label = ttk.Label(self.source_file_frame, text="源数据文件:", width=15)
        self.source_file_label.pack(side=tk.LEFT, padx=5)
        
        self.source_file_var = tk.StringVar()
        self.source_file_entry = ttk.Entry(self.source_file_frame, textvariable=self.source_file_var, width=50)
        self.source_file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.source_browse_button = ttk.Button(self.source_file_frame, text="浏览", command=self.browse_source_file)
        self.source_browse_button.pack(side=tk.LEFT, padx=5)
        
        # SMILES 列名
        self.gen_smiles_frame = ttk.Frame(self.data_gen_frame)
        self.gen_smiles_frame.pack(fill=tk.X, pady=5)
        
        self.gen_smiles_label = ttk.Label(self.gen_smiles_frame, text="SMILES 列名:", width=15)
        self.gen_smiles_label.pack(side=tk.LEFT, padx=5)
        
        self.gen_smiles_var = tk.StringVar(value="SMILES")
        self.gen_smiles_entry = ttk.Entry(self.gen_smiles_frame, textvariable=self.gen_smiles_var, width=20)
        self.gen_smiles_entry.pack(side=tk.LEFT, padx=5)
        
        # 生成比例设置
        self.ratio_frame = ttk.Frame(self.data_gen_frame)
        self.ratio_frame.pack(fill=tk.X, pady=5)
        
        self.ratio_label = ttk.Label(self.ratio_frame, text="生成比例:", width=15)
        self.ratio_label.pack(side=tk.LEFT, padx=5)
        
        self.ratio_var = tk.DoubleVar(value=1.0)
        self.ratio_scale = ttk.Scale(self.ratio_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, 
                                     variable=self.ratio_var, length=200)
        self.ratio_scale.pack(side=tk.LEFT, padx=5)
        
        self.ratio_entry = ttk.Entry(self.ratio_frame, textvariable=self.ratio_var, width=10)
        self.ratio_entry.pack(side=tk.LEFT, padx=5)
        
        self.ratio_unit_label = ttk.Label(self.ratio_frame, text="(0.1-1.0)")
        self.ratio_unit_label.pack(side=tk.LEFT, padx=5)
        
        # 活性比例设置
        self.active_ratio_frame = ttk.Frame(self.data_gen_frame)
        self.active_ratio_frame.pack(fill=tk.X, pady=5)
        
        self.active_ratio_label = ttk.Label(self.active_ratio_frame, text="活性比例:", width=15)
        self.active_ratio_label.pack(side=tk.LEFT, padx=5)
        
        self.active_ratio_var = tk.DoubleVar(value=0.3)
        self.active_ratio_scale = ttk.Scale(self.active_ratio_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                                           variable=self.active_ratio_var, length=200)
        self.active_ratio_scale.pack(side=tk.LEFT, padx=5)
        
        self.active_ratio_entry = ttk.Entry(self.active_ratio_frame, textvariable=self.active_ratio_var, width=10)
        self.active_ratio_entry.pack(side=tk.LEFT, padx=5)
        
        self.active_ratio_unit_label = ttk.Label(self.active_ratio_frame, text="(0.0-1.0, 默认 0.3)")
        self.active_ratio_unit_label.pack(side=tk.LEFT, padx=5)
        
        # 活性标签方法选择
        self.label_method_frame = ttk.Frame(self.data_gen_frame)
        self.label_method_frame.pack(fill=tk.X, pady=5)
        
        self.label_method_label = ttk.Label(self.label_method_frame, text="标记方法:", width=15)
        self.label_method_label.pack(side=tk.LEFT, padx=5)
        
        self.label_method_var = tk.StringVar(value="random")
        self.label_method_combobox = ttk.Combobox(self.label_method_frame, textvariable=self.label_method_var, width=20, state="readonly")
        self.label_method_combobox['values'] = ["random", "xlogp", "mw", "qed"]
        self.label_method_combobox.current(0)
        self.label_method_combobox.pack(side=tk.LEFT, padx=5)
        
        # 生成数据按钮
        self.generate_button = ttk.Button(self.data_gen_frame, text="生成训练数据", command=self.generate_data)
        self.generate_button.pack(pady=5)
        
        # 模型参数设置框架
        self.model_params_frame = ttk.LabelFrame(self.model_build_tab, text="模型参数详细设置", padding="10")
        self.model_params_frame.pack(fill=tk.X, pady=10)
        
        # 通用参数
        self.common_params_frame = ttk.Frame(self.model_params_frame)
        self.common_params_frame.pack(fill=tk.X, pady=5)
        
        self.n_estimators_label = ttk.Label(self.common_params_frame, text=" estimators(树/迭代次数):", width=20)
        self.n_estimators_label.pack(side=tk.LEFT, padx=5)
        self.n_estimators_build_var = tk.IntVar(value=100)
        self.n_estimators_build_entry = ttk.Entry(self.common_params_frame, textvariable=self.n_estimators_build_var, width=10)
        self.n_estimators_build_entry.pack(side=tk.LEFT, padx=5)
        
        self.max_depth_label = ttk.Label(self.common_params_frame, text="最大深度:", width=10)
        self.max_depth_label.pack(side=tk.LEFT, padx=5)
        self.max_depth_var = tk.IntVar(value=6)
        self.max_depth_entry = ttk.Entry(self.common_params_frame, textvariable=self.max_depth_var, width=10)
        self.max_depth_entry.pack(side=tk.LEFT, padx=5)
        
        self.learning_rate_label = ttk.Label(self.common_params_frame, text="学习率:", width=8)
        self.learning_rate_label.pack(side=tk.LEFT, padx=5)
        self.learning_rate_var = tk.DoubleVar(value=0.1)
        self.learning_rate_entry = ttk.Entry(self.common_params_frame, textvariable=self.learning_rate_var, width=10)
        self.learning_rate_entry.pack(side=tk.LEFT, padx=5)
        
        # 模型构建框架
        self.model_build_control_frame = ttk.LabelFrame(self.model_build_tab, text="模型构建控制", padding="10")
        self.model_build_control_frame.pack(fill=tk.X, pady=10)
        
        # 模型构建按钮和停止按钮
        self.build_button_frame = ttk.Frame(self.model_build_control_frame)
        self.build_button_frame.pack(fill=tk.X, pady=5)
        
        self.model_build_button = ttk.Button(self.build_button_frame, text="构建模型", command=self.build_model, style="Accent.TButton")
        self.model_build_button.pack(side=tk.LEFT, padx=5)
        
        self.model_stop_button = ttk.Button(self.build_button_frame, text="停止构建", command=self.stop_build_model, state=tk.DISABLED)
        self.model_stop_button.pack(side=tk.LEFT, padx=5)
        
        # 模型构建状态栏
        self.model_status_frame = ttk.LabelFrame(self.model_build_tab, text="构建状态", padding="10")
        self.model_status_frame.pack(fill=tk.X, pady=10)
        
        # 状态进度条
        self.model_progress_frame = ttk.Frame(self.model_status_frame)
        self.model_progress_frame.pack(fill=tk.X, pady=5)
        
        self.model_progress_var = tk.DoubleVar(value=0)
        self.model_progress_bar = ttk.Progressbar(self.model_progress_frame, variable=self.model_progress_var, maximum=100, mode='indeterminate')
        self.model_progress_bar.pack(fill=tk.X, padx=5)
        
        # 状态标签
        self.model_status_label = ttk.Label(self.model_status_frame, text="状态：就绪", font=("SimHei", 10))
        self.model_status_label.pack(pady=5)
        
        # 详细信息标签
        self.model_detail_label = ttk.Label(self.model_status_frame, text="", font=("SimHei", 9), foreground="blue")
        self.model_detail_label.pack(pady=2)
        
        # 模型参数显示
        self.model_params_display = ttk.Label(self.model_status_frame, text="", font=("SimHei", 8), foreground="green")
        self.model_params_display.pack(pady=2)
        
        # 运行框架
        self.run_frame = ttk.Frame(self.screening_tab)
        self.run_frame.pack(fill=tk.X, pady=10)
        
        self.run_button = ttk.Button(self.run_frame, text="开始筛选", command=self.start_screening, style="Accent.TButton")
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.cancel_button = ttk.Button(self.run_frame, text="取消", command=self.cancel_screening)
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        
        # 进度条
        self.progress_frame = ttk.Frame(self.screening_tab)
        self.progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5)
        
        self.progress_label = ttk.Label(self.progress_frame, text="准备就绪")
        self.progress_label.pack(pady=5)
        
        # 结果框架
        self.result_frame = ttk.LabelFrame(self.screening_tab, text="筛选结果", padding="10")
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 结果表格
        self.tree = ttk.Treeview(self.result_frame, columns=("compound_id", "smiles", "probability", "status"), show="headings")
        self.tree.heading("compound_id", text="化合物 ID")
        self.tree.heading("smiles", text="SMILES")
        self.tree.heading("probability", text="活性概率")
        self.tree.heading("status", text="状态")
        
        self.tree.column("compound_id", width=100)
        self.tree.column("smiles", width=300)
        self.tree.column("probability", width=100)
        self.tree.column("status", width=100)
        
        self.scrollbar = ttk.Scrollbar(self.result_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 结果统计
        self.stats_frame = ttk.Frame(self.screening_tab)
        self.stats_frame.pack(fill=tk.X, pady=10)
        
        self.stats_label = ttk.Label(self.stats_frame, text="统计信息：总化合物数：0, 活性化合物数：0")
        self.stats_label.pack(side=tk.LEFT, padx=5)
        
        # 保存按钮
        self.save_button = ttk.Button(self.stats_frame, text="保存结果", command=self.save_results, state=tk.DISABLED)
        self.save_button.pack(side=tk.RIGHT, padx=5)
        
        # 线程控制
        self.running = False
        self.thread = None
        self.results = []
        
        # 模型构建线程控制
        self.build_running = False
        self.build_thread = None
        self.build_stop_flag = [False]  # 使用列表以便在嵌套函数中修改
        self.executor = None  # 线程池引用
    
    def _on_mousewheel(self, event):
        """处理鼠标滚轮事件"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def get_available_models(self):
        """获取可用的模型列表"""
        possible_paths = [
            os.path.join(os.getcwd(), "results", "models"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "models"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "models"),
            r"E:\Python\dengue_drug_discovery\results\models"
        ]
        
        models = []
        for models_dir in possible_paths:
            if os.path.exists(models_dir):
                for file in os.listdir(models_dir):
                    if file.endswith("_model.pkl"):
                        model_name = file.replace("_model.pkl", "")
                        models.append(model_name)
                if models:
                    break
        
        if not models:
            models = ["RandomForest", "XGBoost", "SVM", "LogisticRegression", "GradientBoosting"]
        
        return models
    
    def browse_file(self):
        """浏览文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.file_var.set(file_path)
    
    def browse_train_file(self):
        """浏览训练数据文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.train_file_var.set(file_path)
    
    def browse_source_file(self):
        """浏览源数据文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.source_file_var.set(file_path)
    
    def generate_data(self):
        """生成训练数据"""
        source_file = self.source_file_var.get()
        if not source_file:
            messagebox.showerror("错误", "请选择源数据文件")
            return
        
        if not os.path.exists(source_file):
            messagebox.showerror("错误", "源数据文件不存在")
            return
        
        try:
            ratio = float(self.ratio_var.get())
            if ratio < 0.1 or ratio > 1.0:
                messagebox.showerror("错误", "生成比例必须在 0.1 到 1.0 之间")
                return
        except ValueError:
            messagebox.showerror("错误", "生成比例必须是数字")
            return
        
        try:
            active_ratio = float(self.active_ratio_var.get())
            if active_ratio < 0.0 or active_ratio > 1.0:
                messagebox.showerror("错误", "活性比例必须在 0.0 到 1.0 之间")
                return
        except ValueError:
            messagebox.showerror("错误", "活性比例必须是数字")
            return
        
        smiles_column = self.gen_smiles_var.get()
        label_method = self.label_method_var.get()
        
        try:
            df = pd.read_csv(source_file)
            
            # 随机抽样
            sampled_df = df.sample(frac=ratio, random_state=42).reset_index(drop=True)
            
            # 确保包含必要的列
            if smiles_column not in df.columns:
                messagebox.showerror("错误", f"源数据文件必须包含'{smiles_column}'列")
                return
            
            # 添加活性标签
            total_compounds = len(sampled_df)
            active_count = int(total_compounds * active_ratio)
            
            # 初始化 method_msg
            method_msg = "未知方法"
            
            if label_method == "random":
                # 随机标记
                labels = np.zeros(total_compounds, dtype=int)
                labels[:active_count] = 1
                np.random.seed(42)
                np.random.shuffle(labels)
                sampled_df['active'] = labels
                method_msg = "随机标记"
            elif label_method == "xlogp":
                # 基于 XLogP 标记
                if 'XLogP' in sampled_df.columns:
                    sampled_df = sampled_df.sort_values('XLogP', ascending=False)
                    labels = np.zeros(total_compounds, dtype=int)
                    labels[:active_count] = 1
                    sampled_df['active'] = labels
                    sampled_df = sampled_df.sort_index()
                    method_msg = "基于 XLogP（高 XLogP 为活性）"
                else:
                    messagebox.showwarning("警告", "文件中没有 XLogP 列，使用随机标记")
                    labels = np.zeros(total_compounds, dtype=int)
                    labels[:active_count] = 1
                    np.random.seed(42)
                    np.random.shuffle(labels)
                    sampled_df['active'] = labels
                    method_msg = "随机标记（XLogP 不可用）"
            elif label_method == "mw":
                # 基于分子量标记
                if 'MolecularWeight' in sampled_df.columns:
                    sampled_df = sampled_df.sort_values('MolecularWeight', ascending=False)
                    labels = np.zeros(total_compounds, dtype=int)
                    labels[:active_count] = 1
                    sampled_df['active'] = labels
                    sampled_df = sampled_df.sort_index()
                    method_msg = "基于分子量（高分子量为活性）"
                else:
                    messagebox.showwarning("警告", "文件中没有 MolecularWeight 列，使用随机标记")
                    labels = np.zeros(total_compounds, dtype=int)
                    labels[:active_count] = 1
                    np.random.seed(42)
                    np.random.shuffle(labels)
                    sampled_df['active'] = labels
                    method_msg = "随机标记（分子量不可用）"
            elif label_method == "qed":
                # 基于 QED 评分标记（需要计算）
                from rdkit.Chem import QED
                qed_scores = []
                for smiles in sampled_df[smiles_column]:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        qed_scores.append(QED.qed(mol))
                    else:
                        qed_scores.append(0)
                sampled_df['QED'] = qed_scores
                sampled_df = sampled_df.sort_values('QED', ascending=False)
                labels = np.zeros(total_compounds, dtype=int)
                labels[:active_count] = 1
                sampled_df['active'] = labels
                sampled_df = sampled_df.sort_index()
                method_msg = "基于 QED 评分（高 QED 为活性）"
            else:
                # 默认使用随机标记
                messagebox.showwarning("警告", f"未知的标记方法 '{label_method}'，使用随机标记")
                labels = np.zeros(total_compounds, dtype=int)
                labels[:active_count] = 1
                np.random.seed(42)
                np.random.shuffle(labels)
                sampled_df['active'] = labels
                method_msg = "随机标记（默认）"
            
            # 保存到源文件所在目录
            source_dir = os.path.dirname(source_file)
            output_file = os.path.join(source_dir, f"training_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
            
            # 保存数据
            sampled_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # 更新训练文件路径
            self.train_file_var.set(output_file)
            
            # 显示成功信息
            messagebox.showinfo("成功", f"数据生成成功！\n\n" +
                f"源数据总数：{len(df)}\n" +
                f"生成数据数：{len(sampled_df)}\n" +
                f"生成比例：{ratio*100:.1f}%\n" +
                f"活性化合物数：{active_count} ({active_ratio*100:.1f}%)\n" +
                f"标记方法：{method_msg}\n" +
                f"保存位置：{output_file}")
        except Exception as e:
            messagebox.showerror("错误", f"生成数据时出错：{str(e)}")
    
    def stop_build_model(self):
        """停止模型构建"""
        if self.build_running:
            self.build_stop_flag[0] = True
            self.model_status_label.config(text="状态：正在停止...")
            self.model_stop_button.config(state=tk.DISABLED)
            
            # 关闭线程池
            if self.executor is not None:
                self.executor.shutdown(wait=False, cancel_futures=True)
        else:
            messagebox.showinfo("提示", "当前没有在构建模型")
    
    def build_model(self):
        """构建模型"""
        train_file = self.train_file_var.get()
        if not train_file:
            messagebox.showerror("错误", "请选择训练数据文件")
            return
        
        if not os.path.exists(train_file):
            messagebox.showerror("错误", "训练数据文件不存在")
            return
        
        model_type = self.model_type_var.get()
        
        # 获取 CPU 核数 - 修复：确保至少为 1
        cpu_str = self.cpu_var.get()
        if cpu_str == "全部":
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = max(1, int(cpu_str))  # 确保至少为 1
        
        # 获取模型参数
        n_estimators = self.n_estimators_build_var.get()
        max_depth = self.max_depth_var.get()
        learning_rate = self.learning_rate_var.get()
        
        # 禁用按钮
        self.model_build_button.config(state=tk.DISABLED)
        self.model_stop_button.config(state=tk.NORMAL)
        
        # 重置停止标志
        self.build_stop_flag[0] = False
        self.build_running = True
        
        # 更新状态
        self.model_status_label.config(text="状态：正在构建模型...")
        self.model_detail_label.config(text=f"模型类型：{model_type}, CPU 核数：{n_jobs}")
        
        # 显示模型参数
        params_text = f"参数：n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}"
        self.model_params_display.config(text=params_text)
        
        self.model_progress_bar.start()
        
        def build_model_thread():
            try:
                # 导入必要的库
                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.svm import SVC
                from sklearn.preprocessing import StandardScaler
                from sklearn.model_selection import train_test_split
                from imblearn.over_sampling import SMOTE
                
                # 检查是否停止
                if self.build_stop_flag[0]:
                    raise Exception("用户取消构建")
                
                # 更新状态
                self.root.after(0, lambda: self.model_status_label.config(text="状态：正在加载数据..."))
                time.sleep(0.1)
                
                # 加载训练数据
                df = pd.read_csv(train_file)
                
                smiles_column = 'canonical_smiles' if 'canonical_smiles' in df.columns else 'SMILES'
                
                if smiles_column not in df.columns:
                    self.root.after(0, lambda: messagebox.showerror("错误", f"训练数据文件必须包含'{smiles_column}'或'SMILES'列"))
                    self.root.after(0, lambda: self.model_build_button.config(state=tk.NORMAL))
                    return
                
                if 'active' not in df.columns:
                    self.root.after(0, lambda: messagebox.showerror("错误", "训练数据文件必须包含'active'列"))
                    self.root.after(0, lambda: self.model_build_button.config(state=tk.NORMAL))
                    return
                
                # 检查是否停止
                if self.build_stop_flag[0]:
                    raise Exception("用户取消构建")
                
                # 更新状态
                self.root.after(0, lambda: self.model_status_label.config(text="状态：正在计算特征..."))
                self.root.after(0, lambda: self.model_detail_label.config(text=f"总化合物数：{len(df)}, 使用 CPU 核数：{n_jobs}"))
                
                # 计算特征
                features = []
                labels = []
                total = len(df)
                
                # 准备数据
                data = [(row[smiles_column], row['active']) for _, row in df.iterrows()]
                
                # 使用多线程并行计算特征 - 修复：正确使用 CPU 核数
                processed = 0
                
                def process_item(item):
                    nonlocal processed
                    smiles, active = item
                    feat = calculate_single_smiles_features(smiles, self.build_stop_flag)
                    if feat is not None:
                        processed += 1
                        # 每 100 个更新一次进度
                        if processed % 100 == 0 or processed == total:
                            progress = processed / total * 50  # 特征计算占 50%
                            self.root.after(0, lambda p=progress: self.model_progress_var.set(p))
                            self.root.after(0, lambda i=processed: self.model_detail_label.config(text=f"已处理化合物：{i}/{total}, CPU 核数：{n_jobs}"))
                    return (feat, active)
                
                # 创建线程池并执行 - 修复：使用正确的 CPU 核数
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    self.executor = executor
                    # 提交所有任务
                    futures = {executor.submit(process_item, item): item for item in data}
                    
                    # 收集结果
                    for future in as_completed(futures):
                        if self.build_stop_flag[0]:
                            break
                        result = future.result()
                        feat, active = result
                        if feat is not None:
                            features.append(feat)
                            labels.append(active)
                
                self.executor = None
                
                if not features:
                    self.root.after(0, lambda: messagebox.showerror("错误", "无法计算特征，可能是 SMILES 格式错误"))
                    self.root.after(0, lambda: self.model_build_button.config(state=tk.NORMAL))
                    return
                
                # 检查是否停止
                if self.build_stop_flag[0]:
                    raise Exception("用户取消构建")
                
                # 更新状态
                self.root.after(0, lambda: self.model_status_label.config(text="状态：正在处理数据..."))
                time.sleep(0.1)
                
                # 转换为数组
                X = np.array(features)
                y = np.array(labels)
                
                # 处理类别不平衡
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                
                # 分割数据集
                X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
                
                # 特征缩放
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # 检查是否停止
                if self.build_stop_flag[0]:
                    raise Exception("用户取消构建")
                
                # 更新状态
                self.root.after(0, lambda: self.model_status_label.config(text=f"状态：正在训练 {model_type} 模型..."))
                self.root.after(0, lambda: self.model_detail_label.config(
                    text=f"训练集：{len(X_train)}, 测试集：{len(X_test)}, CPU 核数：{n_jobs}"))
                self.root.after(0, lambda: self.model_progress_var.set(60))
                
                # 构建模型
                # 获取 GPU 加速设置
                use_gpu = self.gpu_var.get() and self.gpu_available
                
                # 根据模型类型设置参数
                if model_type == "RandomForest":
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42,
                        n_jobs=n_jobs,  # 使用指定的 CPU 核数
                        verbose=1
                    )
                    params_text = f"RandomForest: n_estimators={n_estimators}, max_depth={max_depth}, n_jobs={n_jobs}"
                elif model_type == "SVM":
                    model = SVC(
                        probability=True,
                        random_state=42,
                        C=1.0,
                        kernel='rbf'
                    )
                    params_text = f"SVM: C=1.0, kernel='rbf'"
                elif model_type == "LogisticRegression":
                    model = LogisticRegression(
                        random_state=42,
                        max_iter=1000,
                        n_jobs=n_jobs
                    )
                    params_text = f"LogisticRegression: max_iter=1000, n_jobs={n_jobs}"
                elif model_type == "GradientBoosting":
                    model = GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        random_state=42
                    )
                    params_text = f"GradientBoosting: n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}"
                elif model_type == "AdaBoost":
                    model = AdaBoostClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        random_state=42
                    )
                    params_text = f"AdaBoost: n_estimators={n_estimators}, lr={learning_rate}"
                elif model_type == "ExtraTrees":
                    model = ExtraTreesClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42,
                        n_jobs=n_jobs
                    )
                    params_text = f"ExtraTrees: n_estimators={n_estimators}, max_depth={max_depth}, n_jobs={n_jobs}"
                elif model_type == "XGBoost":
                    try:
                        from xgboost import XGBClassifier
                        if use_gpu:
                            model = XGBClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                learning_rate=learning_rate,
                                random_state=42,
                                n_jobs=n_jobs,
                                tree_method='gpu_hist',
                                gpu_id=0
                            )
                            params_text = f"XGBoost (GPU): n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}"
                        else:
                            model = XGBClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                learning_rate=learning_rate,
                                random_state=42,
                                n_jobs=n_jobs
                            )
                            params_text = f"XGBoost (CPU): n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}, n_jobs={n_jobs}"
                    except ImportError:
                        self.root.after(0, lambda: messagebox.showerror("错误", "XGBoost 库未安装，请运行：pip install xgboost"))
                        self.root.after(0, lambda: self.model_build_button.config(state=tk.NORMAL))
                        return
                elif model_type == "LightGBM":
                    try:
                        from lightgbm import LGBMClassifier
                        if use_gpu:
                            model = LGBMClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                learning_rate=learning_rate,
                                random_state=42,
                                n_jobs=n_jobs,
                                device='gpu'
                            )
                            params_text = f"LightGBM (GPU): n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}"
                        else:
                            model = LGBMClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                learning_rate=learning_rate,
                                random_state=42,
                                n_jobs=n_jobs
                            )
                            params_text = f"LightGBM (CPU): n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}, n_jobs={n_jobs}"
                    except ImportError:
                        self.root.after(0, lambda: messagebox.showerror("错误", "LightGBM 库未安装，请运行：pip install lightgbm"))
                        self.root.after(0, lambda: self.model_build_button.config(state=tk.NORMAL))
                        return
                elif model_type == "GNN (Graph Neural Network)":
                    try:
                        import torch
                        from torch_geometric.data import DataLoader
                        from torch_geometric.nn import GCNConv
                        from src.modeling.deep_learning_models import GNNModel, prepare_gnn_data
                        
                        # 准备 GNN 数据
                        self.root.after(0, lambda: self.model_status_label.config(text="状态：正在准备 GNN 数据..."))
                        
                        # 准备训练数据
                        smiles_list = df[smiles_column].tolist()
                        labels = df['active'].tolist()
                        
                        # 转换为图数据
                        data_list = []
                        for i, (smiles, label) in enumerate(zip(smiles_list, labels)):
                            mol = Chem.MolFromSmiles(smiles)
                            if mol:
                                # 计算节点特征
                                atom_features = []
                                for atom in mol.GetAtoms():
                                    feature = [
                                        atom.GetAtomicNum(),
                                        atom.GetDegree(),
                                        atom.GetTotalNumHs(),
                                        atom.GetImplicitValence(),
                                        atom.GetFormalCharge(),
                                        atom.GetIsAromatic(),
                                        atom.GetHybridization().real,
                                        atom.GetNumRadicalElectrons(),
                                        atom.IsInRing()
                                    ]
                                    atom_features.append(feature)
                                
                                # 计算边索引
                                edge_index = []
                                for bond in mol.GetBonds():
                                    start = bond.GetBeginAtomIdx()
                                    end = bond.GetEndAtomIdx()
                                    edge_index.append([start, end])
                                    edge_index.append([end, start])  # 无向图
                                
                                if edge_index:
                                    import torch_geometric.data as geo_data
                                    x = torch.tensor(atom_features, dtype=torch.float)
                                    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                                    y = torch.tensor([label], dtype=torch.long)
                                    data = geo_data.Data(x=x, edge_index=edge_index, y=y)
                                    data_list.append(data)
                        
                        if not data_list:
                            self.root.after(0, lambda: messagebox.showerror("错误", "无法准备 GNN 数据，可能是 SMILES 格式错误"))
                            self.root.after(0, lambda: self.model_build_button.config(state=tk.NORMAL))
                            return
                        
                        # 分割数据
                        from sklearn.model_selection import train_test_split
                        train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
                        
                        # 创建数据加载器
                        batch_size = 32
                        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
                        
                        # 创建模型
                        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
                        model = GNNModel().to(device)
                        
                        # 优化器和损失函数
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                        criterion = torch.nn.CrossEntropyLoss()
                        
                        params_text = f"GNN: hidden_dim=64, layers=3, device={device}"
                    except ImportError as e:
                        self.root.after(0, lambda: messagebox.showerror("错误", f"缺少深度学习库：{str(e)}\n请运行：pip install torch torch_geometric torch-scatter torch-sparse"))
                        self.root.after(0, lambda: self.model_build_button.config(state=tk.NORMAL))
                        return
                    except Exception as e:
                        self.root.after(0, lambda: messagebox.showerror("错误", f"GNN 模型初始化失败：{str(e)}"))
                        self.root.after(0, lambda: self.model_build_button.config(state=tk.NORMAL))
                        return
                elif model_type == "Transformer (SMILES-BERT)":
                    try:
                        import torch
                        from torch.utils.data import DataLoader, TensorDataset
                        from src.modeling.deep_learning_models import SMILESTransformer, smiles_to_tensor
                        
                        # 准备 Transformer 数据
                        self.root.after(0, lambda: self.model_status_label.config(text="状态：正在准备 Transformer 数据..."))
                        
                        # 准备训练数据
                        smiles_list = df[smiles_column].tolist()
                        labels = df['active'].tolist()
                        
                        # 转换为张量
                        max_length = 100
                        tensors = []
                        for smiles in smiles_list:
                            tensor = smiles_to_tensor(smiles, max_length)
                            tensors.append(tensor)
                        
                        if not tensors:
                            self.root.after(0, lambda: messagebox.showerror("错误", "无法准备 Transformer 数据，可能是 SMILES 格式错误"))
                            self.root.after(0, lambda: self.model_build_button.config(state=tk.NORMAL))
                            return
                        
                        X = torch.stack(tensors)
                        y = torch.tensor(labels, dtype=torch.long)
                        
                        # 分割数据
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # 创建数据加载器
                        batch_size = 32
                        train_dataset = TensorDataset(X_train, y_train)
                        test_dataset = TensorDataset(X_test, y_test)
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                        
                        # 创建模型
                        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
                        model = SMILESTransformer().to(device)
                        
                        # 优化器和损失函数
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
                        criterion = torch.nn.CrossEntropyLoss()
                        
                        params_text = f"Transformer: embedding_dim=128, heads=4, layers=2, device={device}"
                    except ImportError as e:
                        self.root.after(0, lambda: messagebox.showerror("错误", f"缺少深度学习库：{str(e)}\n请运行：pip install torch"))
                        self.root.after(0, lambda: self.model_build_button.config(state=tk.NORMAL))
                        return
                    except Exception as e:
                        self.root.after(0, lambda: messagebox.showerror("错误", f"Transformer 模型初始化失败：{str(e)}"))
                        self.root.after(0, lambda: self.model_build_button.config(state=tk.NORMAL))
                        return
                else:
                    self.root.after(0, lambda: messagebox.showerror("错误", "不支持的模型类型"))
                    self.root.after(0, lambda: self.model_build_button.config(state=tk.NORMAL))
                    return
                
                # 更新参数显示
                self.root.after(0, lambda p=params_text: self.model_params_display.config(text=p))
                
                # 训练模型 - 支持停止功能
                if model_type in ["XGBoost"]:
                    # XGBoost 支持回调函数
                    from xgboost.callback import EarlyStopping
                    
                    class StopCallback:
                        def __init__(self, stop_flag):
                            self.stop_flag = stop_flag
                        def __call__(self, env):
                            if self.stop_flag[0]:
                                raise Exception("用户取消构建")
                    
                    try:
                        model.fit(X_train_scaled, y_train, 
                                 callbacks=[StopCallback(self.build_stop_flag)],
                                 verbose=1)
                    except Exception as e:
                        if "用户取消" in str(e):
                            raise
                        raise
                elif model_type in ["LightGBM"]:
                    # LightGBM 支持回调函数
                    def stop_callback(env):
                        if self.build_stop_flag[0]:
                            raise Exception("用户取消构建")
                    
                    try:
                        model.fit(X_train_scaled, y_train, 
                                 callbacks=[stop_callback],
                                 verbose=1)
                    except Exception as e:
                        if "用户取消" in str(e):
                            raise
                        raise
                elif model_type in ["GNN (Graph Neural Network)"]:
                    # GNN 模型训练
                    import torch
                    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
                    
                    # 训练循环
                    num_epochs = 50
                    for epoch in range(num_epochs):
                        if self.build_stop_flag[0]:
                            raise Exception("用户取消构建")
                        
                        model.train()
                        total_loss = 0
                        for batch in train_loader:
                            if self.build_stop_flag[0]:
                                raise Exception("用户取消构建")
                            
                            batch = batch.to(device)
                            optimizer.zero_grad()
                            out = model(batch.x, batch.edge_index, batch.batch)
                            loss = criterion(out, batch.y)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()
                        
                        # 每5个epoch更新状态
                        if (epoch + 1) % 5 == 0:
                            self.root.after(0, lambda e=epoch+1, l=total_loss/len(train_loader): 
                                           self.model_status_label.config(text=f"状态：GNN 训练中... Epoch {e}/{num_epochs}, Loss: {l:.4f}"))
                            self.root.after(0, lambda p=60 + (e/num_epochs)*20: self.model_progress_var.set(p))
                elif model_type in ["Transformer (SMILES-BERT)"]:
                    # Transformer 模型训练
                    import torch
                    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
                    
                    # 训练循环
                    num_epochs = 50
                    for epoch in range(num_epochs):
                        if self.build_stop_flag[0]:
                            raise Exception("用户取消构建")
                        
                        model.train()
                        total_loss = 0
                        for batch in train_loader:
                            if self.build_stop_flag[0]:
                                raise Exception("用户取消构建")
                            
                            x, y = batch
                            x, y = x.to(device), y.to(device)
                            optimizer.zero_grad()
                            out = model(x)
                            loss = criterion(out, y)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()
                        
                        # 每5个epoch更新状态
                        if (epoch + 1) % 5 == 0:
                            self.root.after(0, lambda e=epoch+1, l=total_loss/len(train_loader): 
                                           self.model_status_label.config(text=f"状态：Transformer 训练中... Epoch {e}/{num_epochs}, Loss: {l:.4f}"))
                            self.root.after(0, lambda p=60 + (e/num_epochs)*20: self.model_progress_var.set(p))
                else:
                    # 其他模型：定期检查停止标志
                    import threading
                    import time
                    
                    stop_event = threading.Event()
                    
                    def check_stop():
                        while not stop_event.is_set():
                            if self.build_stop_flag[0]:
                                # 对于不支持中断的模型，我们只能等待训练完成
                                # 但至少可以显示停止状态
                                self.root.after(0, lambda: self.model_status_label.config(text="状态：正在停止..."))
                            time.sleep(0.5)
                    
                    stop_thread = threading.Thread(target=check_stop)
                    stop_thread.daemon = True
                    stop_thread.start()
                    
                    # 训练模型
                    model.fit(X_train_scaled, y_train)
                    
                    # 训练完成，停止检查线程
                    stop_event.set()
                    
                    # 检查是否请求停止
                    if self.build_stop_flag[0]:
                        raise Exception("用户取消构建")
                
                # 检查是否停止
                if self.build_stop_flag[0]:
                    raise Exception("用户取消构建")
                
                # 更新状态
                self.root.after(0, lambda: self.model_status_label.config(text="状态：正在评估模型..."))
                self.root.after(0, lambda: self.model_progress_var.set(80))
                time.sleep(0.1)
                
                # 评估模型
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                
                if model_type in ["GNN (Graph Neural Network)"]:
                    # 评估 GNN 模型
                    import torch
                    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
                    model.eval()
                    y_true = []
                    y_pred = []
                    y_pred_proba = []
                    
                    with torch.no_grad():
                        for batch in test_loader:
                            batch = batch.to(device)
                            out = model(batch.x, batch.edge_index, batch.batch)
                            _, predicted = torch.max(out, 1)
                            y_true.extend(batch.y.cpu().numpy())
                            y_pred.extend(predicted.cpu().numpy())
                            y_pred_proba.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
                    
                    y_true = np.array(y_true)
                    y_pred = np.array(y_pred)
                    y_pred_proba = np.array(y_pred_proba)
                elif model_type in ["Transformer (SMILES-BERT)"]:
                    # 评估 Transformer 模型
                    import torch
                    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
                    model.eval()
                    y_true = []
                    y_pred = []
                    y_pred_proba = []
                    
                    with torch.no_grad():
                        for batch in test_loader:
                            x, y = batch
                            x, y = x.to(device), y.to(device)
                            out = model(x)
                            _, predicted = torch.max(out, 1)
                            y_true.extend(y.cpu().numpy())
                            y_pred.extend(predicted.cpu().numpy())
                            y_pred_proba.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
                    
                    y_true = np.array(y_true)
                    y_pred = np.array(y_pred)
                    y_pred_proba = np.array(y_pred_proba)
                else:
                    # 评估传统模型
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    y_true = y_test
                
                # 计算评估指标
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                auc = roc_auc_score(y_true, y_pred_proba)
                
                # 保存模型
                models_dir = r"E:\Python\dengue_drug_discovery\results\models"
                os.makedirs(models_dir, exist_ok=True)
                
                model_path = os.path.join(models_dir, f"{model_type}_model.pth" if model_type in ["GNN (Graph Neural Network)", "Transformer (SMILES-BERT)"] else f"{model_type}_model.pkl")
                scaler_path = os.path.join(models_dir, f"{model_type}_scaler.pkl")
                results_path = os.path.join(models_dir, f"{model_type}_results.json")
                
                self.root.after(0, lambda: self.model_status_label.config(text="状态：正在保存模型..."))
                time.sleep(0.1)
                
                # 保存模型和 scaler
                try:
                    if model_type in ["GNN (Graph Neural Network)", "Transformer (SMILES-BERT)"]:
                        # 保存 PyTorch 模型
                        import torch
                        torch.save(model.state_dict(), model_path)
                        print(f"模型保存成功：{model_path}")
                    else:
                        # 保存传统模型
                        joblib.dump(model, model_path)
                        joblib.dump(scaler, scaler_path)
                        print(f"模型和 scaler 保存成功：{model_path}")
                except Exception as model_save_error:
                    error_msg = f"保存模型文件时出错：{str(model_save_error)}"
                    print(error_msg)
                    self.root.after(0, lambda msg=error_msg: self.model_detail_label.config(text=msg))
                
                # 保存结果到 JSON
                json_save_success = False
                try:
                    import json
                    results = {
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1_score": float(f1),
                        "auc": float(auc),
                        "model_params": {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "learning_rate": learning_rate,
                            "n_jobs": n_jobs,
                            "use_gpu": use_gpu
                        }
                    }
                    
                    # 检查路径是否有效
                    print(f"[DEBUG] 尝试保存结果到：{results_path}")
                    print(f"[DEBUG] 目录是否存在：{os.path.exists(models_dir)}")
                    print(f"[DEBUG] 目录是否可写：{os.access(models_dir, os.W_OK)}")
                    
                    # 使用更安全的文件写入方式
                    print(f"[DEBUG] 正在打开文件...")
                    f = open(results_path, 'w', encoding='utf-8')
                    if f is None:
                        raise IOError(f"无法打开文件：{results_path}")
                    print(f"[DEBUG] 文件句柄：{f}")
                    print(f"[DEBUG] 正在写入 JSON...")
                    json.dump(results, f, indent=4, ensure_ascii=False)
                    print(f"[DEBUG] 正在关闭文件...")
                    f.close()
                    
                    print(f"[DEBUG] 结果保存成功：{results_path}")
                    json_save_success = True
                except Exception as json_save_error:
                    # 如果保存失败，记录错误但继续
                    error_msg = f"保存 JSON 结果时出错：{str(json_save_error)}"
                    print(f"[ERROR] {error_msg}")
                    import traceback
                    traceback.print_exc()
                    self.root.after(0, lambda msg=error_msg: self.model_detail_label.config(text=msg))
                
                # 更新模型列表
                self.root.after(0, lambda: self.model_combobox.config(values=self.get_available_models()))
                
                # 完成
                self.root.after(0, lambda: self.model_progress_bar.stop())
                self.root.after(0, lambda: self.model_progress_var.set(100))
                
                # 根据 JSON 保存状态显示不同的消息
                if json_save_success:
                    self.root.after(0, lambda: self.model_status_label.config(text="状态：构建完成！"))
                    self.root.after(0, lambda: self.model_detail_label.config(
                        text=f"准确率：{accuracy:.4f}, 精确率：{precision:.4f}, 召回率：{recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}"))
                else:
                    self.root.after(0, lambda: self.model_status_label.config(text="状态：构建完成（结果保存失败）"))
                    self.root.after(0, lambda: self.model_detail_label.config(
                        text=f"准确率：{accuracy:.4f}, 精确率：{precision:.4f}, 召回率：{recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f} (结果未保存)"))
                
                self.root.after(0, lambda: messagebox.showinfo("成功", f"模型构建成功！\n\n" +
                    f"模型类型：{model_type}\n" +
                    f"准确率：{accuracy:.4f}\n" +
                    f"精确率：{precision:.4f}\n" +
                    f"召回率：{recall:.4f}\n" +
                    f"F1 分数：{f1:.4f}\n" +
                    f"AUC: {auc:.4f}\n" +
                    f"使用 CPU 核数：{n_jobs}\n" +
                    f"使用 GPU: {use_gpu}"))
            except Exception as e:
                self.root.after(0, lambda: self.model_progress_bar.stop())
                if "用户取消" in str(e):
                    self.root.after(0, lambda: self.model_status_label.config(text="状态：已取消"))
                    self.root.after(0, lambda: self.model_detail_label.config(text="用户取消了模型构建"))
                    self.root.after(0, lambda: messagebox.showinfo("提示", "模型构建已取消"))
                else:
                    self.root.after(0, lambda: self.model_status_label.config(text="状态：构建失败"))
                    self.root.after(0, lambda: self.model_detail_label.config(text=f"错误：{str(e)}"))
                    self.root.after(0, lambda: messagebox.showerror("错误", f"构建模型时出错：{str(e)}"))
            finally:
                self.build_running = False
                self.executor = None
                self.root.after(0, lambda: self.model_build_button.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.model_stop_button.config(state=tk.DISABLED))
        
        # 启动线程
        self.build_thread = threading.Thread(target=build_model_thread)
        self.build_thread.daemon = True
        self.build_thread.start()
    
    def start_screening(self):
        """开始筛选"""
        file_path = self.file_var.get()
        if not file_path:
            messagebox.showerror("错误", "请选择化合物文件")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("错误", "化合物文件不存在")
            return
        
        model_name = self.model_var.get()
        if not model_name:
            messagebox.showerror("错误", "请选择模型")
            return
        
        self.run_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)
        
        self.results = []
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        self.running = True
        self.thread = threading.Thread(target=self.run_screening, args=(file_path, model_name))
        self.thread.daemon = True
        self.thread.start()
    
    def cancel_screening(self):
        """取消筛选"""
        self.running = False
        self.progress_label.config(text="取消中...")
    
    def run_screening(self, file_path, model_name):
        """运行筛选"""
        try:
            possible_paths = [
                os.path.join(os.getcwd(), "results", "models"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "models"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "models"),
                r"E:\Python\dengue_drug_discovery\results\models"
            ]
            
            model_path = None
            scaler_path = None
            for models_dir in possible_paths:
                temp_model_path = os.path.join(models_dir, f"{model_name}_model.pkl")
                temp_scaler_path = os.path.join(models_dir, f"{model_name}_scaler.pkl")
                if os.path.exists(temp_model_path) and os.path.exists(temp_scaler_path):
                    model_path = temp_model_path
                    scaler_path = temp_scaler_path
                    break
            
            if not model_path or not scaler_path:
                self.root.after(0, lambda: messagebox.showerror("错误", f"找不到模型文件：{model_name}"))
                self.root.after(0, self.reset_ui)
                return
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            df = pd.read_csv(file_path)
            smiles_column = self.smiles_var.get()
            
            if smiles_column not in df.columns:
                self.root.after(0, lambda: messagebox.showerror("错误", f"文件中没有'{smiles_column}'列"))
                self.root.after(0, self.reset_ui)
                return
            
            total_compounds = len(df)
            active_count = 0
            
            for i, (_, row) in enumerate(df.iterrows()):
                if not self.running:
                    break
                
                progress = (i + 1) / total_compounds * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                self.root.after(0, lambda i=i, total=total_compounds: 
                               self.progress_label.config(text=f"处理化合物 {i+1}/{total}"))
                
                smiles = row[smiles_column]
                compound_id = f"CMPD_{i+1}"
                
                try:
                    features = calculate_single_smiles_features(smiles)
                    if features is not None:
                        features_scaled = scaler.transform([features])
                        probability = model.predict_proba(features_scaled)[0][1]
                        
                        threshold = self.threshold_var.get()
                        is_active = "活性" if probability >= threshold else "非活性"
                        if is_active == "活性":
                            active_count += 1
                        
                        result = (compound_id, smiles, f"{probability:.4f}", is_active)
                        self.results.append(result)
                        self.root.after(0, lambda r=result: self.tree.insert("", tk.END, values=r))
                    else:
                        result = (compound_id, smiles, "N/A", "计算失败")
                        self.results.append(result)
                        self.root.after(0, lambda r=result: self.tree.insert("", tk.END, values=r))
                except Exception as e:
                    result = (compound_id, smiles, "N/A", f"错误：{str(e)}")
                    self.results.append(result)
                    self.root.after(0, lambda r=result: self.tree.insert("", tk.END, values=r))
            
            # 完成
            self.root.after(0, lambda: self.progress_label.config(text="筛选完成！"))
            self.root.after(0, lambda: self.stats_label.config(
                text=f"统计信息：总化合物数：{total_compounds}, 活性化合物数：{active_count}"))
            self.root.after(0, lambda: self.save_button.config(state=tk.NORMAL))
            
            if not self.running:
                self.root.after(0, lambda: messagebox.showinfo("提示", "筛选已取消"))
            else:
                self.root.after(0, lambda: messagebox.showinfo("完成", f"筛选完成！\n\n" +
                    f"总化合物数：{total_compounds}\n" +
                    f"活性化合物数：{active_count}\n" +
                    f"活性比例：{active_count/total_compounds*100:.2f}%"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"筛选时出错：{str(e)}"))
        finally:
            self.running = False
            self.root.after(0, lambda: self.run_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.cancel_button.config(state=tk.NORMAL))
    
    def reset_ui(self):
        """重置 UI 状态"""
        self.run_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
    
    def save_results(self):
        """保存结果"""
        if not self.results:
            messagebox.showwarning("警告", "没有结果可保存")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                df = pd.DataFrame(self.results, columns=["化合物 ID", "SMILES", "活性概率", "状态"])
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                messagebox.showinfo("成功", f"结果已保存到：{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存结果时出错：{str(e)}")


def main():
    """主函数"""
    root = tk.Tk()
    app = ScreeningGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
