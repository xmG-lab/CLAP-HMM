import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricGatedFusion(nn.Module):

    def __init__(self, seq_dim=128, aux_dim=2):
        super(AsymmetricGatedFusion, self).__init__()
        self.seq_dim = seq_dim
        self.aux_dim = aux_dim

        # 用 aux 信息生成 gate
        self.gate_fc = nn.Sequential(
            nn.Linear(aux_dim, seq_dim),
            nn.Sigmoid()
        )

    def forward(self, x_seq, x_aux):
        gate = self.gate_fc(x_aux)  # [B, 128]，从2维生成门控权重
        gated_seq = x_seq * gate  # [B, 128]，应用门控
        fused = torch.cat([gated_seq, x_aux], dim=-1)  # 拼接输出 [B, 130]
        return fused


if __name__ == "__main__":
    batch_size = 4
    seq_dim = 128
    aux_dim = 2

    # 模拟主特征和辅助特征
    x_seq = torch.randn(batch_size, seq_dim)  # 来自CNN-LSTM的特征
    x_aux = torch.randn(batch_size, aux_dim)  # 来自ProtHint的特征

    fusion_model = AsymmetricGatedFusion(seq_dim=seq_dim, aux_dim=aux_dim)
    fused_output = fusion_model(x_seq, x_aux)

    print("融合输出形状:", fused_output.shape)  # 应输出 [4, 130]
    print("样例输出:", fused_output[0].data)
