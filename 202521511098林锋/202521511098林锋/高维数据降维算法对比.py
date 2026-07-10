# -*- coding: utf-8 -*-
# 字体规范：绘图英文Times New Roman，中文宋体
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as MDA
from sklearn.manifold import Isomap, TSNE
import umap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# --------------------------全局绘图字体设置（满足格式要求）--------------------------
plt.rcParams["font.family"] = ["SimSun"]    # 中文宋体
plt.rcParams["font.size"] = 10.5            # 正文10.5号
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif'] #英文罗马体

# --------------------------加载高维MNIST数据(784维)--------------------------
print("加载MNIST高维数据集...")
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = X / 255.0  #归一化
# 采样10000样本加速实验
sample_idx = np.random.permutation(len(X))[:10000]
X, y = X[sample_idx], y[sample_idx].astype(int)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"数据集维度：样本{X.shape[0]}，特征维数{X.shape[1]}")

# --------------------------定义评估函数--------------------------
def evaluate_embedding(X_emb, label, name, train_ratio=0.7):
    n_train = int(len(X_emb)*train_ratio)
    Xtr, Xte = X_emb[:n_train], X_emb[n_train:]
    ytr, yte = label[:n_train], label[n_train:]
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(Xtr, ytr)
    acc = knn.score(Xte, yte)
    print(f"{name} 5-NN分类准确率：{acc:.4f}")
    return acc

def plot_emb(emb, label, title):
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(emb[:,0], emb[:,1], c=label, s=3, cmap="tab10")
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()

# --------------------------依次运行全部降维算法--------------------------
dim_out = 2  #降维至2维可视化
alg_result = {}


# 1.PCA
t0 = time.time()
pca = PCA(n_components=dim_out)
X_pca = pca.fit_transform(X_scaled)
t1 = time.time()
acc_pca = evaluate_embedding(X_pca, y, "PCA")
alg_result["PCA"] = {"time":t1-t0, "acc":acc_pca}
plot_emb(X_pca, y, "PCA主成分分析降维结果")

# 2.MDA
t0 = time.time()
mda = MDA(n_components=dim_out)
X_mda = mda.fit_transform(X_scaled, y)
t1 = time.time()
acc_mda = evaluate_embedding(X_mda, y, "MDA多判别分析")
alg_result["MDA"] = {"time":t1-t0, "acc":acc_mda}
plot_emb(X_mda, y, "MDA有监督降维结果")

# 3.KPCA高斯核
t0 = time.time()
kpca = KernelPCA(n_components=dim_out, kernel="rbf", gamma=0.01)
X_kpca = kpca.fit_transform(X_scaled)
t1 = time.time()
acc_kpca = evaluate_embedding(X_kpca, y, "KPCA核主成分")
alg_result["KPCA"] = {"time":t1-t0, "acc":acc_kpca}
plot_emb(X_kpca, y, "KPCA非线性降维结果")

# 4.ISOMAP
t0 = time.time()
iso = Isomap(n_components=dim_out, n_neighbors=10)
X_iso = iso.fit_transform(X_scaled)
t1 = time.time()
acc_iso = evaluate_embedding(X_iso, y, "ISOMAP")
alg_result["ISOMAP"] = {"time":t1-t0, "acc":acc_iso}
plot_emb(X_iso, y, "ISOMAP流形降维")

# 5.T-SNE
t0 = time.time()
tsne = TSNE(n_components=dim_out, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
t1 = time.time()
acc_tsne = evaluate_embedding(X_tsne, y, "T-SNE")
alg_result["TSNE"] = {"time":t1-t0, "acc":acc_tsne}
plot_emb(X_tsne, y, "T-SNE可视化结果")

# 6.UMAP
t0 = time.time()
umap_model = umap.UMAP(n_components=dim_out, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)
t1 = time.time()
acc_umap = evaluate_embedding(X_umap, y, "UMAP")
alg_result["UMAP"] = {"time":t1-t0, "acc":acc_umap}
plot_emb(X_umap, y, "UMAP降维可视化")

# --------------------------输出汇总对比表格--------------------------
print("\n========算法耗时与分类精度汇总========")
print(f"{'算法':<8}{'运行时间(s)':<12}{'5NN准确率':<10}")
for name, res in alg_result.items():
    print(f"{name:<8}{res['time']:<12.3f}{res['acc']:<10.4f}")