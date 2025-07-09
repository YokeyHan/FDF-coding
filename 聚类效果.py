import torch
import numpy as np
from collections import Counter
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

from data_provider.data_factory import load_data
from args import get_args
from models.FDF import FDF  

def extract_embeddings_labels(model, loader, device):
    all_Z = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out, z_seq, q = model(x, task="train")
            # z_seq: [B, L, N, C]; q: [B*L*N, K]
            labels = q.argmax(1).cpu().numpy()
            Z_flat = z_seq.reshape(-1, z_seq.size(-1)).cpu().numpy()
            all_labels.append(labels)
            all_Z.append(Z_flat)
    return np.concatenate(all_Z, axis=0), np.concatenate(all_labels, axis=0)

def main():
    args = get_args()
    device = args.device

    # 模型实例化
    model = FDF(args).to(device)

    # 加载 checkpoint
    raw_ckpt = torch.load(f"/home/user/plh/Prediction/myFDF-ssl/FDF-main/results/model_checkpoint.pth",
                          map_location=device)
    state = raw_ckpt.get("state_dict", raw_ckpt)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(">>> Loaded checkpoint:")
    print("    missing keys:", missing)
    print("    unexpected keys:", unexpected)

    # 准备 DataLoader
    _, val_loader, _, _ = load_data(args)

    # 提取 embedding 和 label
    Z, labels = extract_embeddings_labels(model, val_loader, device)

    # 保存
    np.save("embeddings2.npy", Z)
    np.save("labels2.npy", labels)

    # 评估
    cnt = Counter(labels)
    print("Cluster counts:", cnt)
    score = silhouette_score(Z, labels)
    print(f"Silhouette score: {score:.4f}")

if __name__ == "__main__":
    main()


