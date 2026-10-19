import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# 模型定义：LSTM + Attention + MLP
# ============================================
class LSTMAttentionRegressor(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, num_layers=4, output_dim=2, dropout=0.3):
        super(LSTMAttentionRegressor, self).__init__()

        # LSTM 层：提取时序特征
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True  # 双向增强特征
        )

        # 注意力层：计算每个时间步的权重
        self.attn = nn.Linear(hidden_dim * 2, 1)  # 双向 → hidden_dim*2

        # MLP 头：深度非线性映射
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # 计算注意力权重
        attn_weights = F.softmax(self.attn(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # 加权求和 → (batch, hidden*2)

        # MLP 输出预测
        y_pred = self.mlp(context)
        return y_pred
