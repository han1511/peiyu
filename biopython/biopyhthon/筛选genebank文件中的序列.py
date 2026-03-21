from Bio import SeqIO
import os

def filter_nucleocapsid_sequences(input_gb, output_gb):
    """
    从GenBank文件中筛选出编码核衣壳蛋白(nucleocapsid protein)的序列
    并保存为新的GenBank文件
    
    参数:
        input_gb (str): 输入的GenBank文件路径
        output_gb (str): 输出的GenBank文件路径
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_gb):
            raise FileNotFoundError(f"输入文件不存在: {input_gb}")
        
        # 读取GenBank文件
        records = SeqIO.parse(input_gb, "genbank")
        
        # 存储符合条件的记录
        filtered_records = []
        
        # 遍历每条记录
        for record in records:
            # 标记是否为核衣壳蛋白编码序列
            is_nucleocapsid = False
            
            # 检查记录中的特征
            for feature in record.features:
                # 查找CDS特征
                if feature.type == "CDS":
                    # 检查是否有product注释
                    if "product" in feature.qualifiers:
                        products = feature.qualifiers["product"]
                        # 检查product中是否包含核衣壳蛋白相关描述
                        for product in products:
                            if "nucleocapsid" in product.lower() and "protein" in product.lower():
                                is_nucleocapsid = True
                                break
                        if is_nucleocapsid:
                            break
            
            # 如果是核衣壳蛋白编码序列，则添加到结果列表
            if is_nucleocapsid:
                filtered_records.append(record)
                print(f"已筛选出序列: {record.id}")
        
        # 保存筛选结果到新的GenBank文件
        if filtered_records:
            SeqIO.write(filtered_records, output_gb, "genbank")
            print(f"筛选完成，共找到 {len(filtered_records)} 条核衣壳蛋白编码序列")
            print(f"结果已保存到: {output_gb}")
        else:
            print("未找到编码核衣壳蛋白的序列")
            
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")

if __name__ == "__main__":
    # 设置文件路径（请替换为实际文件名）
    input_file = r"H:\研究生\硕士研究生\陈雨虹\汉坦病毒科时间进化\refseq147.gb"
    output_file = r"H:\研究生\硕士研究生\陈雨虹\汉坦病毒科时间进化\refseq147_filtered.gb"

    # 执行筛选
    filter_nucleocapsid_sequences(input_file, output_file)
    