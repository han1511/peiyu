#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度学习模型实现
支持 GNN 和 Transformer 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GNNModel(nn.Module):
    """图神经网络模型"""
    def __init__(self, input_dim=9, hidden_dim=64, output_dim=2):
        super(GNNModel, self).__init__()
        # 动态导入 torch_geometric
        from torch_geometric.nn import GCNConv, global_max_pool
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.conv3 = GCNConv(hidden_dim * 2, hidden_dim * 4)
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x, edge_index, batch):
        # 动态导入 torch_geometric
        from torch_geometric.nn import global_max_pool
        # 图卷积层
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # 全局池化
        x = global_max_pool(x, batch)
        
        # 全连接层
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class SMILESTransformer(nn.Module):
    """SMILES Transformer 模型"""
    def __init__(self, vocab_size=100, embedding_dim=128, num_heads=4, hidden_dim=256, num_layers=2, output_dim=2):
        super(SMILESTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = self._get_positional_encoding(100, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embedding_dim, output_dim)
    
    def _get_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        seq_len = x.size(1)
        
        # 嵌入层
        x = self.embedding(x)
        
        # 添加位置编码
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer 层（需要转置为 [seq_len, batch_size, embedding_dim]）
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        
        # 池化
        x = x.mean(dim=1)
        
        # 输出层
        x = self.fc(x)
        return x

# SMILES 字符映射
def get_smiles_vocab():
    """获取 SMILES 字符词汇表"""
    vocab = {
        'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 'Cl': 6, 'Br': 7, 'I': 8,
        '(': 9, ')': 10, '[': 11, ']': 12, '.': 13, '=': 14, '#': 15, '@': 16, 
        'H': 17, '+': 18, '-': 19, '1': 20, '2': 21, '3': 22, '4': 23, '5': 24,
        '6': 25, '7': 26, '8': 27, '9': 28, '0': 29, '/': 30, '\\': 31, '%': 32,
        'B': 33, 'c': 34, 'n': 35, 'o': 36, 's': 37, 'p': 38, 'A': 39, 'a': 40,
        'X': 41, 'Y': 42, 'Z': 43, 'U': 44, 'V': 45, 'W': 46, 'K': 47, 'L': 48,
        'M': 49, 'Q': 50, 'R': 51, 'T': 52, 'u': 53, 'v': 54, 'w': 55, 'k': 56,
        'l': 57, 'm': 58, 'q': 59, 'r': 60, 't': 61, 'G': 62, 'g': 63, 'E': 64,
        'e': 65, 'D': 66, 'd': 67, ' ': 68, '<pad>': 69, '<unk>': 70
    }
    return vocab

def smiles_to_tensor(smiles, max_length=100):
    """将 SMILES 转换为张量"""
    vocab = get_smiles_vocab()
    tensor = []
    
    for char in smiles:
        if char in vocab:
            tensor.append(vocab[char])
        else:
            tensor.append(vocab['<unk>'])
    
    # 填充到最大长度
    while len(tensor) < max_length:
        tensor.append(vocab['<pad>'])
    
    # 截断到最大长度
    if len(tensor) > max_length:
        tensor = tensor[:max_length]
    
    return torch.tensor(tensor, dtype=torch.long)

def mol_to_graph(mol):
    """将分子转换为图数据"""
    if mol is None:
        return None
    
    # 动态导入 torch_geometric
    from torch_geometric.data import Data
    
    # 节点特征（使用原子类型作为特征）
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
    
    # 边索引
    edge_index = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])  # 无向图
    
    if not edge_index:
        return None
    
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)

def prepare_gnn_data(smiles_list):
    """准备 GNN 数据"""
    data_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            data = mol_to_graph(mol)
            if data:
                data_list.append(data)
    return data_list

def prepare_transformer_data(smiles_list, max_length=100):
    """准备 Transformer 数据"""
    tensors = []
    for smiles in smiles_list:
        tensor = smiles_to_tensor(smiles, max_length)
        tensors.append(tensor)
    return torch.stack(tensors)

# 模型训练函数
def train_gnn(model, dataloader, optimizer, criterion, device):
    """训练 GNN 模型"""
    model.train()
    total_loss = 0
    
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train_transformer(model, dataloader, optimizer, criterion, device):
    """训练 Transformer 模型"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 模型评估函数
def evaluate_gnn(model, dataloader, device):
    """评估 GNN 模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            _, predicted = torch.max(out, 1)
            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()
    
    return correct / total

def evaluate_transformer(model, dataloader, device):
    """评估 Transformer 模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, predicted = torch.max(out, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    return correct / total

# 模型保存和加载
def save_model(model, path):
    """保存模型"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """加载模型"""
    model.load_state_dict(torch.load(path, map_location=device))
    return model
