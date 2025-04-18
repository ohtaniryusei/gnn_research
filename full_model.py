import torch
import torch.nn as nn
from static_gnn import StaticGraphEncoder
from dynamic_gnn import DynamicGraphEncoder
from dynamic_gnn_graphmixer import GraphMixerEncoder  # ← 追加
from fusion import FusionLayer
from classifier import CreditRiskClassifier

class FullModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.in_dim = config['in_dim']
        self.time_config = config['time_config']  # (d_re, d_ab, d_se, d_time, hidden)
        self.dynamic_type = config.get('dynamic_type', 'tgat')  # デフォルトは tgat

        # 静的グラフ用エンコーダ（GraphSAGE）
        self.static_encoder = StaticGraphEncoder(
            in_channels=self.in_dim,
            hidden_channels=self.in_dim,
            out_channels=self.in_dim
        )

        # 動的グラフ用エンコーダ（選択式）
        if self.dynamic_type == 'graphmixer':
            self.dynamic_encoder = GraphMixerEncoder(
                node_dim=self.in_dim,
                time_config=self.time_config,
                out_dim=self.in_dim
            )
        else:
            self.dynamic_encoder = DynamicGraphEncoder(
                node_dim=self.in_dim,
                time_config=self.time_config,
                out_dim=self.in_dim
            )

        self.fusion = FusionLayer(self.in_dim)
        self.classifier = CreditRiskClassifier(self.in_dim)

    def forward(
        self,
        h_static,            # [B, D]
        h_neighbors,         # [B, N, D]
        t,                   # [B]
        t_prime,             # [B, N]
        semantic_feat,       # [B, N, d_se]
    ):
        """
        Returns:
            y_pred: [B, 1] - デフォルト確率
        """
        h_d = self.dynamic_encoder(h_neighbors, t, t_prime, semantic_feat)
        h_fused = self.fusion(h_static, h_d)
        y_pred = self.classifier(h_fused)
        return y_pred
