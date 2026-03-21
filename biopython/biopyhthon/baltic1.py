import baltic as bt

# 你的zika.nwk文件路径（注意Windows路径用r前缀避免转义问题）
nwk_path = r'H:\研究生\硕士研究生\陈雨虹\baltic\tests\data\zika.nwk'

# 加载Newick格式树
zika_tree = bt.loadNewick(nwk_path)

# 简单查看树的基本信息（比如节点数、分支数）
print(f"树的节点数：{len(zika_tree.nodes)}") 
print(f"树的分支数：{len(zika_tree.edges)}")
