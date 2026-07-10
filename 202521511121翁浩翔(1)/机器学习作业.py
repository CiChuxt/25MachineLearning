import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import Isomap, TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import umap

#数据下载地址https://ndownloader.figshare.com/files/5976027

faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = faces.data, faces.target   # X: (400, 4096), y: 0~39
print(f"数据形状: X = {X.shape}, y = {y.shape}")
print(f"类别数: {len(np.unique(y))}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- 图三：PCA 方差解释比例 ----------
pca_full = PCA().fit(X_scaled)
evr = pca_full.explained_variance_ratio_
cumsum = np.cumsum(evr)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(range(1, 51), evr[:50], alpha=0.7, label='单个主成分方差占比')
ax1.set_xlabel('主成分序号')
ax1.set_ylabel('方差占比', color='blue')
ax2 = ax1.twinx()
ax2.plot(range(1, 51), cumsum[:50], 'r-o', label='累计方差占比')
ax2.set_ylabel('累计方差占比', color='red')
plt.title('PCA 前 50 个主成分的方差解释')
fig.legend(loc='center right')
plt.show()

def apply_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    t0 = time.time()
    X_emb = pca.fit_transform(X)
    return X_emb, time.time() - t0

def apply_mda(X, y, n_components=2):
    lda = LDA(n_components=n_components)
    t0 = time.time()
    X_emb = lda.fit_transform(X, y)
    return X_emb, time.time() - t0

def apply_kpca(X, n_components=2, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=gamma)
    t0 = time.time()
    X_emb = kpca.fit_transform(X)
    return X_emb, time.time() - t0

def apply_isomap(X, n_components=2, n_neighbors=8):   # 增大近邻数
    iso = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    t0 = time.time()
    X_emb = iso.fit_transform(X)
    return X_emb, time.time() - t0

def apply_tsne(X, n_components=2, perplexity=30, n_iter=500, method='auto'):
    if method == 'auto':
        # 如果降维维度 > 3，自动切换为 exact 方法
        method = 'barnes_hut' if n_components <= 3 else 'exact'
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                max_iter=n_iter, method=method, random_state=42)
    t0 = time.time()
    X_emb = tsne.fit_transform(X)
    t = time.time() - t0
    return X_emb, t

def apply_umap_func(X, n_components=2, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=42)
    t0 = time.time()
    X_emb = reducer.fit_transform(X)
    return X_emb, time.time() - t0

# ---------- 计算二维嵌入 ----------
methods = {
    'PCA': apply_pca,
    'MDA': lambda X: apply_mda(X, y),
    'KPCA': lambda X: apply_kpca(X, gamma=0.0001),
    'ISOMAP': lambda X: apply_isomap(X, n_neighbors=8),
    't-SNE': lambda X: apply_tsne(X, perplexity=30, n_iter=500),  # 二维仍然用 barnes_hut
    'UMAP': lambda X: apply_umap_func(X, n_neighbors=15, min_dist=0.1)
}

embeddings_2d = {}
times_2d = {}
for name, func in methods.items():
    print(f"计算 {name} 2D 嵌入...")
    X_2d, t = func(X_scaled)
    embeddings_2d[name] = X_2d
    times_2d[name] = t
    print(f"  耗时: {t:.2f} 秒")

import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘制二维嵌入图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()
for ax, (name, X_2d) in zip(axes, embeddings_2d.items()):
    sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab20', s=15, alpha=0.8)
    ax.set_title(name)
    ax.axis('off')

# 统一添加颜色条，并手动调整子图间距
cbar = fig.colorbar(sc, ax=axes, orientation='horizontal', fraction=0.02, pad=0.06)
fig.suptitle('Olivetti Faces 二维嵌入可视化', fontsize=16)
fig.subplots_adjust(top=0.92, bottom=0.08, left=0.03, right=0.97, hspace=0.2, wspace=0.1)
plt.show()

# 分类性能评估（降维至 10 维）
dim = 10
knn = KNeighborsClassifier(n_neighbors=1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
for name, func in methods.items():
    n_comp = min(dim, 9) if name == 'MDA' else dim

    if name == 'PCA':
        X_emb, _ = apply_pca(X_scaled, n_comp)
    elif name == 'MDA':
        X_emb, _ = apply_mda(X_scaled, y, n_comp)
    elif name == 'KPCA':
        X_emb, _ = apply_kpca(X_scaled, n_comp, gamma=0.0001)
    elif name == 'ISOMAP':
        X_emb, _ = apply_isomap(X_scaled, n_comp, n_neighbors=8)   # 增大近邻
    elif name == 't-SNE':
        X_emb, _ = apply_tsne(X_scaled, n_comp, perplexity=30, n_iter=500, method='exact')
    elif name == 'UMAP':
        X_emb, _ = apply_umap_func(X_scaled, n_comp, n_neighbors=15, min_dist=0.1)
    else:
        continue

    acc = cross_val_score(knn, X_emb, y, cv=cv, scoring='accuracy').mean()
    f1 = cross_val_score(knn, X_emb, y, cv=cv, scoring='f1_macro').mean()
    results.append((name, n_comp, acc, f1, times_2d[name]))

print("\n降维后 1‑NN 分类结果（5 折交叉验证，降维至 10 维）:")
print("方法\t维度\t准确率\t宏观F1\t降维时间(秒)")
for r in results:
    print(f"{r[0]}\t{r[1]}\t{r[2]:.3f}\t{r[3]:.3f}\t{r[4]:.2f}")
