import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA # MDA
from sklearn.manifold import Isomap, TSNE
from sklearn.metrics import silhouette_score
import umap
# 内置人脸数据集，无需Kaggle、无需手动下载
from sklearn.datasets import fetch_olivetti_faces

# 设置中文绘图
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 加载高维人脸数据
data = fetch_olivetti_faces()
X = data.data  # (400, 4096) 400样本，4096维
y = data.target # 0~39共40类人脸标签
print("原始数据集维度：", X.shape)
print("人脸类别总数：", len(np.unique(y)))

# 【提速关键】截取前200个样本，大幅减少运算量
X = X[:200]
y = y[:200]
print("截取后数据集维度：", X.shape)

# 标准化预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 存储各算法量化指标
metrics = []

# 绘图函数：仅保存图片，关闭弹窗，防止程序阻塞
def plot_embedding(emb, label, title, save_name):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=label, palette="tab20", s=30, legend=False)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_name}.png", dpi=300)
    # 注释掉弹窗，不会卡住程序
    # plt.show()

# 1. PCA线性降维
t0 = time.time()
pca = PCA(n_components=2)
emb_pca = pca.fit_transform(X_scaled)
t1 = time.time()
sil_pca = silhouette_score(emb_pca, y)
res_err = 1 - np.sum(pca.explained_variance_ratio_)
metrics.append(["PCA", round(t1-t0,2), round(sil_pca,4), round(res_err,4)])
plot_embedding(emb_pca, y, "PCA降维可视化结果", "PCA")
print("PCA运行完成")

# 2. MDA(LDA)线性判别降维
t0 = time.time()
lda = LDA(n_components=2)
emb_lda = lda.fit_transform(X_scaled, y)
t1 = time.time()
sil_lda = silhouette_score(emb_lda, y)
metrics.append(["MDA(LDA)", round(t1-t0,2), round(sil_lda,4), np.nan])
plot_embedding(emb_lda, y, "MDA线性判别降维可视化结果", "MDA")
print("MDA运行完成")

# 3. KPCA核主成分分析（RBF核捕捉非线性）
t0 = time.time()
kpca = KernelPCA(n_components=2, kernel="rbf")
emb_kpca = kpca.fit_transform(X_scaled)
t1 = time.time()
sil_kpca = silhouette_score(emb_kpca, y)
metrics.append(["KPCA", round(t1-t0,2), round(sil_kpca,4), np.nan])
plot_embedding(emb_kpca, y, "KPCA核主成分降维可视化结果", "KPCA")
print("KPCA运行完成")

# 4. ISOMAP流形学习：n_neighbors调低，提速
t0 = time.time()
iso = Isomap(n_components=2, n_neighbors=5)
emb_iso = iso.fit_transform(X_scaled)
t1 = time.time()
sil_iso = silhouette_score(emb_iso, y)
metrics.append(["ISOMAP", round(t1-t0,2), round(sil_iso,4), np.nan])
plot_embedding(emb_iso, y, "ISOMAP流形降维可视化结果", "ISOMAP")
print("ISOMAP运行完成")

# 5. t-SNE：perplexity调低，减少迭代计算
t0 = time.time()
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
emb_tsne = tsne.fit_transform(X_scaled)
t1 = time.time()
sil_tsne = silhouette_score(emb_tsne, y)
metrics.append(["t-SNE", round(t1-t0,2), round(sil_tsne,4), np.nan])
plot_embedding(emb_tsne, y, "t-SNE降维可视化结果", "TSNE")
print("t-SNE运行完成")

# 6. UMAP拓扑保持降维
t0 = time.time()
umap_model = umap.UMAP(n_components=2, n_neighbors=8, random_state=42)
emb_umap = umap_model.fit_transform(X_scaled)
t1 = time.time()
sil_umap = silhouette_score(emb_umap, y)
metrics.append(["UMAP", round(t1-t0,2), round(sil_umap,4), np.nan])
plot_embedding(emb_umap, y, "UMAP拓扑降维可视化结果", "UMAP")
print("UMAP运行完成")

# 输出指标表格并保存
df_metric = pd.DataFrame(metrics, columns=["算法名称","运行时间(s)","轮廓系数","未解释方差占比"])
df_metric.to_csv("降维算法性能指标对比.csv", index=False, encoding="utf-8-sig")
print("\n全部实验运行完成！指标对比表格：")
print(df_metric)

# 导出完整原始数据集，交给老师
df_export = pd.DataFrame(X)
df_export["label"] = y
df_export.to_csv("Olivetti_人脸完整高维数据集.csv", index=False, encoding="utf-8-sig")
print("\n原始数据集已导出：Olivetti_人脸完整高维数据集.csv")