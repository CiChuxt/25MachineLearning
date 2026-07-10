import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------- 1. 加载本地手写数字数据集 ----------------------
print("加载本地手写数字数据集...")
digits = load_digits()
X = digits.data.astype(np.float32)
y = digits.target.astype(np.int32)

sample_num = len(X)
print(f"使用样本数量：{sample_num}, 特征维度：{X.shape[1]}")

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------- 2. PCA 降维到2维 ----------------------
print("\n开始 PCA 降维...")
t0 = time.time()
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
t_pca = time.time() - t0
print(f"PCA 耗时：{t_pca:.2f} s")
print(f"PCA 前两主成分累计方差解释率：{sum(pca.explained_variance_ratio_):.4f}")

# ---------------------- 3. UMAP 降维到2维 ----------------------
print("\n开始 UMAP 降维...")
t0 = time.time()
umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)
t_umap = time.time() - t0
print(f"UMAP 耗时：{t_umap:.2f} s")

# ===================== 独立PCA二维聚类图 =====================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
colors = plt.cm.tab10(np.arange(10))

plt.figure(figsize=(10, 7))
for digit in range(10):
    mask = y == digit
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=20, color=colors[digit], label=str(digit), alpha=0.8)
plt.title(f"PCA 降维手写数字\n耗时:{t_pca:.2f}s 方差贡献率:{sum(pca.explained_variance_ratio_):.3f}", fontsize=13)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="数字类别", markerscale=2)
plt.tight_layout()
plt.savefig("PCA_聚类图.png", dpi=300)  # 自动保存高清图
plt.show()

# ===================== 独立UMAP二维聚类图 =====================
plt.figure(figsize=(10, 7))
for digit in range(10):
    mask = y == digit
    plt.scatter(X_umap[mask, 0], X_umap[mask, 1], s=20, color=colors[digit], label=str(digit), alpha=0.8)
plt.title(f"UMAP 降维手写数字\n耗时:{t_umap:.2f}s", fontsize=13)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend(title="数字类别", markerscale=2)
plt.tight_layout()
plt.savefig("UMAP_聚类图.png", dpi=300)
plt.show()


# ===================== 可视化——PCA方差解释率变化曲线 =====================
pca_full = PCA(n_components=30, random_state=42)
pca_full.fit(X_scaled)
cum_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 31), cum_variance, marker="o", color="#2E86AB")
plt.axhline(y=0.9, color="red", linestyle="--", label="90%信息阈值线")
plt.axvline(x=2, color="orange", linestyle="--", label="本次实验使用前2维")
plt.xlabel("主成分序号")
plt.ylabel("累计方差解释率")
plt.title("PCA前30个主成分累计方差解释率变化曲线")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("PCA方差解释率曲线图.png", dpi=300)
plt.show()

# ---------------------- 5. 分类精度对比 ----------------------
def eval_acc(X_2d, label):
    X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.3, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return accuracy_score(y_test, pred)

acc_pca = eval_acc(X_pca, y)
acc_umap = eval_acc(X_umap, y)
print("\n===== 2维特征逻辑回归分类精度对比 =====")
print(f"PCA 2维特征准确率: {acc_pca:.4f}")
print(f"UMAP 2维特征准确率: {acc_umap:.4f}")