# =========================
# 1. 基础库
# =========================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import umap

# =========================
# 2. 数据加载
# =========================
data = fetch_openml(name='leukemia', version=1, as_frame=False)

X = data.data
y = data.target

# 标签转换（ALL=0, AML=1）
y = np.array([0 if i == 'ALL' else 1 for i in y])

# 标准化
X = StandardScaler().fit_transform(X)

print("数据维度:", X.shape)

# =========================
# 3. PCA
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# =========================
# 4. KPCA
# =========================
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=1e-6)
X_kpca = kpca.fit_transform(X)

# =========================
# 5. ISOMAP
# =========================
iso = Isomap(n_neighbors=5, n_components=2)
X_iso = iso.fit_transform(X)

# =========================
# 6. t-SNE
# =========================
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# =========================
# 7. UMAP
# =========================
umap_model = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)
X_umap = umap_model.fit_transform(X)

# =========================
# 8. MDA（LDA）
# =========================
lda = LDA(n_components=1)
X_lda = lda.fit_transform(X, y)

# =========================
# 9. 单独绘图函数
# =========================
def plot(X_emb, title):
    plt.figure()
    plt.scatter(X_emb[:, 0], X_emb[:, 1] if X_emb.shape[1] > 1 else np.zeros_like(X_emb[:, 0]),
                c=y, s=15)
    plt.title(title)
    plt.show()

# =========================
# 10. 单独图（论文用）
# =========================
plot(X_pca, "PCA")
plot(X_kpca, "KPCA")
plot(X_iso, "ISOMAP")
plot(X_tsne, "t-SNE")
plot(X_umap, "UMAP")

# LDA特殊处理（1维）
plt.figure()
plt.scatter(X_lda[:, 0], np.zeros_like(X_lda[:, 0]), c=y, s=15)
plt.title("MDA / LDA")
plt.show()

# =========================
# 11. 总对比图（论文必用）
# =========================
methods = {
    "PCA": X_pca,
    "KPCA": X_kpca,
    "ISOMAP": X_iso,
    "t-SNE": X_tsne,
    "UMAP": X_umap
}

plt.figure(figsize=(15, 10))

for i, (name, X_emb) in enumerate(methods.items()):
    plt.subplot(2, 3, i + 1)
    plt.scatter(X_emb[:, 0], X_emb[:, 1], c=y, s=10)
    plt.title(name)

plt.tight_layout()
plt.show()

plt.figure()

plt.scatter(
    X_lda[:, 0],
    np.random.normal(0, 0.01, len(X_lda)),  # 加一点扰动
    c=y,
    s=15
)

plt.title("MDA / LDA (1D projection)")
plt.show()