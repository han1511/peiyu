#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载10万个PubChem化合物数据
"""

import os
import sys
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_acquisition.fetch_pubchem_data import fetch_pubchem_compounds
from src.config import DATA_DIR, PUBCHEM_CONFIG

def main():
    print("=" * 80)
    print("下载10万个PubChem化合物数据")
    print("=" * 80)
    
    # 设置下载参数
    total_compounds = 100000
    start_cid = 1
    end_cid = start_cid + total_compounds - 1
    
    # 大规模下载设置
    large_scale = True
    batch_size = 100  # PubChem API建议的最大批量大小
    delay = 0.5  # 增加延迟以避免API限制
    max_retries = 5  # 增加重试次数
    checkpoint_interval = 1000  # 每下载1000个化合物保存一次检查点
    
    print(f"参数设置：")
    print(f"- 总化合物数: {total_compounds}")
    print(f"- CID范围: {start_cid} 到 {end_cid}")
    print(f"- 批次大小: {batch_size}")
    print(f"- 请求延迟: {delay}秒")
    print(f"- 最大重试次数: {max_retries}")
    print(f"- 检查点间隔: {checkpoint_interval}个化合物")
    print(f"- 启用大规模下载模式: {large_scale}")
    
    # 确认下载
    confirm = input("\n开始下载吗？这可能需要几个小时的时间 (y/n): ").strip().lower()
    if confirm != 'y':
        print("下载已取消")
        return
    
    # 计算下载时间估计
    estimated_time_hours = (total_compounds / batch_size) * (delay + 2) / 3600  # 每个批次大约需要delay+2秒
    print(f"\n预计下载时间: {estimated_time_hours:.2f} 小时")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # 生成CID列表
        cid_list = list(range(start_cid, end_cid + 1))
        
        # 下载化合物数据
        df = fetch_pubchem_compounds(
            compound_list=cid_list,
            identifier_type='cid',
            batch_size=batch_size,
            delay=delay,
            max_retries=max_retries,
            output_file='pubchem_100k_compounds.csv',
            large_scale=large_scale,
            resume=True,  # 支持断点续传
            checkpoint_interval=checkpoint_interval
        )
        
        if not df.empty:
            print("\n" + "=" * 80)
            print("下载成功完成！")
            print("=" * 80)
            print(f"共成功获取 {len(df)} 个化合物")
            print(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"数据已保存到: {os.path.join(DATA_DIR['raw'], 'pubchem_100k_compounds.csv')}")
        else:
            print("\n下载失败，未获取到任何化合物数据")
            
    except KeyboardInterrupt:
        print("\n\n下载已中断")
        print("您可以使用相同的命令继续下载")
    except Exception as e:
        print(f"\n\n下载过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("脚本执行完毕")
    print("=" * 80)


if __name__ == "__main__":
    main()
