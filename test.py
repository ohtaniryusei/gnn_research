import argparse
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from full_model import FullModel
from utils.data_loader import load_transaction_batch, load_cora_batch

def get_input_dim(dataset_name):
    if dataset_name == 'cora':
        dataset = Planetoid(root='data/', name='Cora')
        return dataset[0].x.shape[1]
    else:
        return 128  # デフォルト：synthetic 用

def test(model_name, dataset_name):
    print(f"\n🚀 モデル = {model_name}, データセット = {dataset_name} でAUC評価を開始...\n")

    # モデル設定
    in_dim = get_input_dim(dataset_name)

    config = {
        'in_dim': in_dim,
        'time_config': (16, 16, 8, 32, in_dim),
        'dynamic_type': model_name
    }

    model = FullModel(config)
    model.eval()

    aucs = []
    criterion = nn.BCELoss()

    for i in range(10):  # 10バッチ分の平均で評価
        with torch.no_grad():
            if dataset_name == 'synthetic':
                h_static, h_neighbors, t, t_prime, sem_feat, y_true = load_transaction_batch()
            elif dataset_name == 'cora':
                h_static, h_neighbors, t, t_prime, sem_feat, y_true = load_cora_batch()
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")

            y_pred = model(h_static, h_neighbors, t, t_prime, sem_feat).squeeze()
            y_pred = torch.clamp(y_pred, 0.0, 1.0)  # 念のため

            # 損失 & AUC
            loss = criterion(y_pred, y_true).item()
            try:
                auc = roc_auc_score(y_true.numpy(), y_pred.numpy())
            except:
                auc = 0.0  # 片方のクラスしかないとき

            aucs.append(auc)
            print(f"Batch {i+1:02d} | AUC: {auc:.4f} | Loss: {loss:.4f}")

    print(f"\n✅ 平均AUC ({model_name} on {dataset_name}): {sum(aucs)/len(aucs):.4f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['tgat', 'graphmixer'], default='tgat',
                        help='動的GNNモデルの種類')
    parser.add_argument('--dataset', type=str, choices=['synthetic', 'cora'], default='synthetic',
                        help='評価データセットの種類')
    args = parser.parse_args()

    test(args.model, args.dataset)
