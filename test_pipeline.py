#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试完整的虚拟筛选流程

使用少量化合物测试完整的筛选流程是否正常工作
"""

import sys
import os
import numpy as np
import pandas as pd

# 检查RDKit
print("Python version:", sys.version)

try:
    import rdkit
    from rdkit import Chem
    print("RDKit version:", rdkit.__version__)
    print("RDKit loaded successfully")
except ImportError as e:
    print("Error loading RDKit:", e)
    sys.exit(1)

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'virtual_screening_pipeline'))

# 测试完整的筛选流程
def test_full_pipeline():
    print("\n=== Testing Full Pipeline ===")
    
    try:
        from src.compound_library import CompoundLibrary
        from src.molecular_features import FeatureEngineering
        from src.result_analysis import VirtualScreeningReporter
        
        # 创建一个小型化合物库
        test_compounds = [
            {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "name": "Aspirin"},
            {"smiles": "C1=CC=CC=C1", "name": "Benzene"},
            {"smiles": "CCO", "name": "Ethanol"},
            {"smiles": "C1=CC=CC=C1C(=O)O", "name": "Benzoic acid"},
            {"smiles": "CCCCC", "name": "Pentane"}
        ]
        
        # 创建临时SMILES文件
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as f:
            for comp in test_compounds:
                f.write(f"{comp['smiles']} {comp['name']}\n")
            temp_smi_file = f.name
        
        try:
            # 创建化合物库
            library = CompoundLibrary("test_library")
            
            # 从SMILES文件加载
            count = library.load_from_smiles(temp_smi_file)
            print(f"Loaded {count} compounds from file")
        finally:
            # 清理临时文件
            if os.path.exists(temp_smi_file):
                os.unlink(temp_smi_file)
        
        print(f"Loaded {len(library.compounds)} compounds")
        
        # 预处理
        print("\nPreprocessing compounds...")
        removed = library.deduplicate()
        print(f"Removed {removed} duplicates")
        
        removed = library.filter_drug_likeness()
        print(f"Removed {removed} non-drug-like compounds")
        print(f"Remaining compounds: {len(library.compounds)}")
        
        # 生成特征
        print("\nGenerating features...")
        mols = [comp['mol'] for comp in library.compounds if 'mol' in comp]
        print(f"Valid molecules: {len(mols)}")
        
        if mols:
            features = FeatureEngineering()
            feature_matrix, valid_indices, feature_names = features.calculate_all_features(mols)
            print(f"Feature matrix shape: {feature_matrix.shape}")
            print(f"Feature names count: {len(feature_names)}")
        
        # 生成报告
        print("\nGenerating report...")
        import tempfile
        import shutil
        from pathlib import Path
        
        output_dir = tempfile.mkdtemp()
        reporter = VirtualScreeningReporter("NS2A", Path(output_dir))
        
        report_path = reporter.generate_summary_report(
            pipeline_results={
                "target": "NS2A",
                "compounds_processed": len(library.compounds),
                "admet_evaluated": 0
            },
            admet_results=None
        )
        
        print(f"Report generated at: {report_path}")
        
        # 清理
        shutil.rmtree(output_dir)
        
        print("\nFull pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print("Error in full pipeline test:", e)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing full virtual screening pipeline...")
    
    success = test_full_pipeline()
    
    if success:
        print("\n=== All Tests PASSED! ===")
    else:
        print("\n=== Tests FAILED! ===")
