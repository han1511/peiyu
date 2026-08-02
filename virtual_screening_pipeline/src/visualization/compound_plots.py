#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
化合物结构可视化模块

功能：
1. 单个分子结构绘制
2. Top化合物网格展示
3. 分子性质标注图
4. 子结构高亮
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)

COLORS = {
    'palette': ['#2E5C8A', '#E67E22', '#27AE60', '#8E44AD', '#C0392B', 
                '#16A085', '#D35400', '#2980B9', '#27AE60', '#F39C12']
}


class CompoundVisualizer:
    """
    化合物结构可视化类
    
    使用RDKit生成论文级别的分子结构图
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化化合物可视化器
        
        参数:
            output_dir: 图表输出目录
        """
        self.output_dir = output_dir or Path("results/visualization/compounds")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def draw_molecule(self,
                     smiles: str,
                     title: str = None,
                     filename: str = "molecule.png",
                     size: Tuple[int, int] = (400, 400),
                     highlight_atoms: List[int] = None,
                     highlight_bonds: List[int] = None,
                     atom_notes: Dict[int, str] = None) -> str:
        """
        绘制单个分子结构
        
        参数:
            smiles: SMILES字符串
            title: 标题
            filename: 输出文件名
            size: 图片大小
            highlight_atoms: 高亮原子索引列表
            highlight_bonds: 高亮键索引列表
            atom_notes: 原子标注 {原子索引: 标注文本}
            
        返回:
            str: 输出文件路径
        """
        if not HAS_RDKIT:
            logger.error("RDKit not available")
            return ""
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error(f"Invalid SMILES: {smiles}")
            return ""
        
        # 添加氢原子用于更好的展示
        mol = Chem.AddHs(mol)
        
        # 生成2D坐标
        from rdkit.Chem import AllChem
        AllChem.Compute2DCoords(mol)
        
        # 设置绘制选项
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        draw_options = drawer.drawOptions()
        draw_options.addAtomIndices = False
        draw_options.addBondIndices = False
        draw_options.padding = 0.15
        
        # 如果有标注
        if atom_notes:
            for idx, note in atom_notes.items():
                if idx < mol.GetNumAtoms():
                    mol.GetAtomWithIdx(idx).SetProp('atomNote', note)
        
        # 高亮颜色
        highlight_colors = {}
        if highlight_atoms:
            for idx in highlight_atoms:
                if idx < mol.GetNumAtoms():
                    highlight_colors[idx] = (0.9, 0.2, 0.2, 0.6)  # 红色半透明
        
        bond_colors = {}
        if highlight_bonds:
            for idx in highlight_bonds:
                if idx < mol.GetNumBonds():
                    bond_colors[idx] = (0.2, 0.6, 0.9, 0.6)  # 蓝色半透明
        
        # 绘制
        if highlight_atoms or highlight_bonds:
            drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms or [],
                              highlightBonds=highlight_bonds or [],
                              highlightAtomColors=highlight_colors,
                              highlightBondColors=bond_colors)
        else:
            drawer.DrawMolecule(mol)
        
        drawer.FinishDrawing()
        
        output_path = self.output_dir / filename
        drawer.WriteDrawingToFile(str(output_path))
        
        logger.info(f"Molecule image saved to {output_path}")
        return str(output_path)
    
    def draw_compounds_grid(self,
                           smiles_list: List[str],
                           legends: List[str] = None,
                           properties: Dict[str, List] = None,
                           filename: str = "compounds_grid.png",
                           mols_per_row: int = 4,
                           sub_img_size: Tuple[int, int] = (300, 300),
                           title: str = None) -> str:
        """
        绘制化合物网格图
        
        参数:
            smiles_list: SMILES列表
            legends: 图例列表
            properties: 性质字典 {性质名: [值列表]}
            filename: 输出文件名
            mols_per_row: 每行分子数
            sub_img_size: 单个子图大小
            title: 总标题
            
        返回:
            str: 输出文件路径
        """
        if not HAS_RDKIT:
            logger.error("RDKit not available")
            return ""
        
        mols = []
        valid_legends = []
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)
                
                legend = ""
                if legends and i < len(legends):
                    legend = str(legends[i])
                
                # 添加性质到图例
                if properties:
                    prop_parts = []
                    for prop_name, values in properties.items():
                        if i < len(values):
                            prop_parts.append(f"{prop_name}: {values[i]}")
                    if prop_parts:
                        legend += "\n" + "\n".join(prop_parts)
                
                valid_legends.append(legend)
        
        if not mols:
            logger.error("No valid molecules to draw")
            return ""
        
        # 计算网格大小
        n_mols = len(mols)
        n_rows = (n_mols + mols_per_row - 1) // mols_per_row
        
        # 调整整体图片大小
        total_width = mols_per_row * sub_img_size[0]
        total_height = n_rows * sub_img_size[1]
        
        if title:
            total_height += 60
        
        img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row,
                                   subImgSize=sub_img_size,
                                   legends=valid_legends,
                                   useSVG=False)
        
        output_path = self.output_dir / filename
        img.save(str(output_path))
        
        logger.info(f"Compound grid saved to {output_path}")
        return str(output_path)
    
    def draw_top_compounds_with_properties(self,
                                          df: pd.DataFrame,
                                          smiles_col: str = "SMILES",
                                          name_col: str = "compound_name",
                                          property_cols: List[str] = None,
                                          top_n: int = 12,
                                          filename: str = "top_compounds_properties.png",
                                          title: str = "Top Candidates") -> str:
        """
        绘制带性质的Top化合物展示图
        
        参数:
            df: 化合物DataFrame
            smiles_col: SMILES列名
            name_col: 名称列名
            property_cols: 要显示的性质列
            top_n: 前N个化合物
            filename: 输出文件名
            title: 标题
            
        返回:
            str: 输出文件路径
        """
        if not HAS_RDKIT:
            logger.error("RDKit not available")
            return ""
        
        if smiles_col not in df.columns:
            logger.error(f"SMILES column '{smiles_col}' not found")
            return ""
        
        df_top = df.head(top_n)
        
        smiles_list = df_top[smiles_col].tolist()
        names = df_top[name_col].tolist() if name_col in df.columns else [f"Comp_{i}" for i in range(len(df_top))]
        
        # 准备性质
        properties = {}
        if property_cols:
            for col in property_cols:
                if col in df_top.columns:
                    properties[col] = df_top[col].tolist()
        
        return self.draw_compounds_grid(
            smiles_list=smiles_list,
            legends=names,
            properties=properties,
            filename=filename,
            mols_per_row=4,
            sub_img_size=(350, 350),
            title=title
        )
    
    def draw_matched_substructure(self,
                                 smiles: str,
                                 smarts_pattern: str,
                                 filename: str = "substructure_match.png",
                                 size: Tuple[int, int] = (500, 500)) -> str:
        """
        绘制子结构匹配高亮图
        
        参数:
            smiles: 分子SMILES
            smarts_pattern: SMARTS子结构模式
            filename: 输出文件名
            size: 图片大小
            
        返回:
            str: 输出文件路径
        """
        if not HAS_RDKIT:
            logger.error("RDKit not available")
            return ""
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error("Invalid SMILES")
            return ""
        
        pattern = Chem.MolFromSmarts(smarts_pattern)
        if pattern is None:
            logger.error("Invalid SMARTS pattern")
            return ""
        
        matches = mol.GetSubstructMatches(pattern)
        
        if not matches:
            logger.warning("No substructure match found")
            return self.draw_molecule(smiles, filename=filename, size=size)
        
        # 高亮匹配的原子和键
        highlight_atoms = []
        highlight_bonds = []
        
        for match in matches:
            highlight_atoms.extend(match)
            
            # 找到匹配原子之间的键
            for i in range(len(match)):
                for j in range(i+1, len(match)):
                    bond = mol.GetBondBetweenAtoms(match[i], match[j])
                    if bond is not None:
                        highlight_bonds.append(bond.GetIdx())
        
        return self.draw_molecule(smiles, filename=filename, size=size,
                                highlight_atoms=list(set(highlight_atoms)),
                                highlight_bonds=list(set(highlight_bonds)))
    
    def generate_compound_cards(self,
                               df: pd.DataFrame,
                               smiles_col: str = "SMILES",
                               name_col: str = "compound_name",
                               property_cols: List[str] = None,
                               top_n: int = 10,
                               output_dir: Optional[Path] = None) -> List[str]:
        """
        为每个化合物生成单独的卡片图片
        
        参数:
            df: 化合物DataFrame
            smiles_col: SMILES列名
            name_col: 名称列名
            property_cols: 性质列
            top_n: 前N个
            output_dir: 输出目录
            
        返回:
            List[str]: 生成的文件路径列表
        """
        if not HAS_RDKIT or not HAS_MATPLOTLIB:
            logger.error("RDKit or Matplotlib not available")
            return []
        
        output_dir = output_dir or self.output_dir / "cards"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df_top = df.head(top_n)
        generated_files = []
        
        for idx, row in df_top.iterrows():
            smiles = row.get(smiles_col, "")
            name = str(row.get(name_col, f"compound_{idx}"))
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # 创建带信息的组合图
            fig = plt.figure(figsize=(10, 6))
            gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
            
            # 分子结构
            ax_mol = fig.add_subplot(gs[0, :])
            ax_mol.axis('off')
            
            img = Draw.MolToImage(mol, size=(600, 400))
            ax_mol.imshow(img)
            ax_mol.set_title(name, fontweight='bold', fontsize=14, pad=10)
            
            # 性质表格
            ax_table = fig.add_subplot(gs[1, :])
            ax_table.axis('off')
            
            if property_cols:
                table_data = []
                for col in property_cols:
                    if col in row:
                        val = row[col]
                        if isinstance(val, float):
                            val = f"{val:.3f}"
                        table_data.append([col.replace('_', ' ').title(), str(val)])
                
                if table_data:
                    table = ax_table.table(cellText=table_data,
                                          colLabels=['Property', 'Value'],
                                          cellLoc='left',
                                          loc='center',
                                          colWidths=[0.4, 0.4])
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 2)
                    
                    # 设置表头样式
                    for i in range(2):
                        table[(0, i)].set_facecolor('#2E5C8A')
                        table[(0, i)].set_text_props(weight='bold', color='white')
            
            # 保存
            safe_name = "".join(c if c.isalnum() else "_" for c in name)[:50]
            output_path = output_dir / f"{safe_name}_card.png"
            plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            generated_files.append(str(output_path))
        
        logger.info(f"Generated {len(generated_files)} compound cards")
        return generated_files
