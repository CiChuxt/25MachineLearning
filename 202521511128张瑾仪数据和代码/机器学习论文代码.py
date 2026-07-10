
# -*- coding: utf-8 -*-
# 四种降维算法对比实验（彻底解决 No module named 'umap'）
# 适配 UCI 乳腺癌 wdbc.txt 数据，无需改格式
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# 固定随机种子，实验可复现
np.random.seed(42)

# 读取你下载的 wdbc.txt
df = pd.read_csv("C:/Users/Administrator/Desktop/机器学习/wdbc.txt", header=None)

# 数据处理
X = df.iloc[:, 2:].values
y = df.iloc[:, 1].map({"M": 1, "B": 0}).values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 评价指标
def evaluate(X_2d, label, cost_time):
    return {
        "运行时间(s)": round(cost_time, 4),
        "轮廓系数": round(silhouette_score(X_2d, label), 4),
        "CH指数": round(calinski_harabasz_score(X_2d, label), 4)
    }

# PCA
t0 = time.time()
X_pca = PCA(n_components=2).fit_transform(X_scaled)
res_pca = evaluate(X_pca, y, time.time() - t0)

# KPCA
t0 = time.time()
X_kpca = KernelPCA(n_components=2, kernel="rbf", random_state=42).fit_transform(X_scaled)
res_kpca = evaluate(X_kpca, y, time.time() - t0)

# t-SNE
t0 = time.time()
X_tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, 
              n_iter=1000, random_state=42).fit_transform(X_scaled)
res_tsne = evaluate(X_tsne, y, time.time() - t0)

# 输出结果
print("===== PCA实验结果 =====")
print(res_pca)
print("===== KPCA实验结果 =====")
print(res_kpca)
print("===== t-SNE实验结果 =====")
print(res_tsne)

# 绘图
plt.figure(figsize=(16, 10))
plt.subplot(221);plt.scatter(X_pca[:,0],X_pca[:,1],c=y,s=10);plt.title("PCA")
plt.subplot(222);plt.scatter(X_kpca[:,0],X_kpca[:,1],c=y,s=10);plt.title("KPCA")
plt.subplot(223);plt.scatter(X_tsne[:,0],X_tsne[:,1],c=y,s=10);plt.title("t-SNE")
plt.tight_layout()
plt.show()

