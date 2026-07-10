# 1. 设置matplotlib内联显示
%matplotlib inline

# 2. 导入所有需要的库
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import Isomap, TSNE
from sklearn.neighbors import NearestNeighbors
import time
import warnings
warnings.filterwarnings('ignore')

# 3. 尝试导入UMAP
try:
    import umap
    UMAP_AVAILABLE = True
    print("✓ UMAP已安装")
except ImportError:
    UMAP_AVAILABLE = False
    print("✗ UMAP未安装，请执行: !pip install umap-learn")

print("✓ 所有库导入完成，matplotlib内联显示已启用")
# 加载200个样本
print("正在加载数据...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, 
                    as_frame=False, parser='auto')
X = X[:200]
y = y[:200].astype(int)
print(f"数据形状: {X.shape}")

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"降维后形状: {X_pca.shape}")
print(f"方差解释率: {pca.explained_variance_ratio_}")

# 绘制散点图
fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.tab10(np.linspace(0, 1, 10))

for digit in range(10):
    mask = y == digit
    if np.sum(mask) > 0:
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[colors[digit]], label=str(digit), alpha=0.7, s=30)

ax.set_title('PCA降维结果 - 快速测试 (200个样本)', fontsize=14, fontweight='bold')
ax.legend(title='数字', markerscale=2)
ax.grid(True, alpha=0.3)
ax.set_xlabel('主成分1')
ax.set_ylabel('主成分2')

plt.tight_layout()
plt.show()  

print("✓ 如果上方显示了散点图，说明环境正常！")
# ==========================================
# Cell 2: Define all functions
# ==========================================

def load_mnist_subset(n_samples=5000, random_state=42):
    """Load MNIST dataset subset"""
    print(f"Loading MNIST dataset (sampling {n_samples} images)...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, 
                        as_frame=False, parser='auto')
    np.random.seed(random_state)
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_subset = X[indices]
    y_subset = y[indices].astype(int)
    print(f"Data loaded, shape: {X_subset.shape}")
    return X_subset, y_subset

def preprocess_data(X, standardize=True):
    """Standardize data"""
    if standardize:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
        return X_processed, scaler
    return X, None

def plot_2d_embedding(X_2d, y, title, ax=None):
    """Plot 2D embedding scatter plot"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for digit in range(10):
        mask = y == digit
        if np.sum(mask) > 0:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       c=[colors[digit]], label=str(digit), 
                       alpha=0.6, s=8, edgecolors='none')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend(markerscale=3, loc='best', title='Digit', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax

def run_pca(X, y=None, n_components=2):
    """Principal Component Analysis"""
    print("\n" + "="*50)
    print("[PCA] Principal Component Analysis")
    start_time = time.time()
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    elapsed = time.time() - start_time
    print(f"Done, time: {elapsed:.2f}s")
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
    return X_pca

def run_lda(X, y, n_components=2):
    """Linear Discriminant Analysis"""
    print("\n" + "="*50)
    print("[LDA] Linear Discriminant Analysis")
    start_time = time.time()
    n_classes = len(np.unique(y))
    n_components = min(n_components, n_classes - 1)
    lda = LDA(n_components=n_components)
    X_lda = lda.fit_transform(X, y)
    elapsed = time.time() - start_time
    print(f"Done, time: {elapsed:.2f}s")
    return X_lda

def run_kpca(X, y=None, n_components=2, kernel='rbf', gamma=0.001):
    """Kernel Principal Component Analysis"""
    print("\n" + "="*50)
    print(f"[KPCA] Kernel PCA (kernel={kernel})")
    start_time = time.time()
    kpca = KernelPCA(n_components=n_components, kernel=kernel, 
                     gamma=gamma, random_state=42)
    X_kpca = kpca.fit_transform(X)
    elapsed = time.time() - start_time
    print(f"Done, time: {elapsed:.2f}s")
    return X_kpca

def run_isomap(X, y=None, n_components=2, n_neighbors=30):
    """Isometric Mapping"""
    print("\n" + "="*50)
    print(f"[ISOMAP] Isometric Mapping (k={n_neighbors})")
    start_time = time.time()
    isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors, n_jobs=-1)
    X_isomap = isomap.fit_transform(X)
    elapsed = time.time() - start_time
    print(f"Done, time: {elapsed:.2f}s")
    return X_isomap

def run_tsne(X, y=None, n_components=2, perplexity=30, n_iter=1000):
    """t-Distributed Stochastic Neighbor Embedding"""
    print("\n" + "="*50)
    print(f"[t-SNE] perplexity={perplexity}")
    start_time = time.time()
    
    # Compatible with both old and new scikit-learn versions
    try:
        tsne = TSNE(n_components=n_components, 
                    perplexity=perplexity, 
                    max_iter=n_iter,
                    random_state=42, 
                    n_jobs=-1, 
                    verbose=1)
    except TypeError:
        tsne = TSNE(n_components=n_components, 
                    perplexity=perplexity, 
                    n_iter=n_iter,
                    random_state=42, 
                    n_jobs=-1, 
                    verbose=1)
    
    X_tsne = tsne.fit_transform(X)
    elapsed = time.time() - start_time
    print(f"Done, time: {elapsed:.2f}s")
    return X_tsne

def run_umap(X, y=None, n_components=2, n_neighbors=15, min_dist=0.1):
    """Uniform Manifold Approximation and Projection"""
    if not UMAP_AVAILABLE:
        print("\nUMAP not installed")
        return None
    print("\n" + "="*50)
    print(f"[UMAP] k={n_neighbors}, min_dist={min_dist}")
    start_time = time.time()
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=42, verbose=True)
    X_umap = reducer.fit_transform(X)
    elapsed = time.time() - start_time
    print(f"Done, time: {elapsed:.2f}s")
    return X_umap

print("All functions defined successfully")
# ==========================================
save_dir = 'E:/python/result/mnist_results./results'  

import os
os.makedirs(save_dir, exist_ok=True)

# Load data
X, y = load_mnist_subset(n_samples=1000)
X_scaled, _ = preprocess_data(X, standardize=True)

results = {}
methods = [
    ('PCA', run_pca, 'PCA - Principal Component Analysis', 'mnist_pca.png', {'n_components': 2}),
    ('LDA', run_lda, 'LDA - Linear Discriminant Analysis', 'mnist_lda.png', {'n_components': 2}),
    ('KPCA', run_kpca, 'KPCA - Kernel PCA (RBF)', 'mnist_kpca.png', {'n_components': 2}),
    ('ISOMAP', run_isomap, 'ISOMAP - Isometric Mapping', 'mnist_isomap.png', {'n_components': 2}),
    ('t-SNE', run_tsne, 't-SNE', 'mnist_tsne.png', {'n_components': 2, 'perplexity': 30, 'n_iter': 500}),
]
if UMAP_AVAILABLE:
    methods.append(('UMAP', run_umap, 'UMAP', 'mnist_umap.png', {'n_components': 2}))

for name, func, title, filename, kwargs in methods:
    X_emb = func(X_scaled, y, **kwargs) if name == 'LDA' else func(X_scaled, y=None, **kwargs)
    results[name] = X_emb
    
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_2d_embedding(X_emb, y, title, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.show()

print(f"\nDone! Images saved to: {os.path.abspath(save_dir)}")
# ==========================================
# 定量表1：降维后的KNN分类准确率
# 评估降维数据对分类任务的保留程度
# ==========================================
print("\n" + "="*70)
print("定量表1：降维后KNN分类准确率对比")
print("="*70)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from tabulate import tabulate

# 准备原始高维数据的baseline
knn = KNeighborsClassifier(n_neighbors=5)
scores_orig = cross_val_score(knn, X_scaled, y, cv=5)
print(f"原始784维数据 - 5折交叉验证准确率: {scores_orig.mean():.4f} (+/- {scores_orig.std():.4f})")

# 对每种降维方法评估分类准确率
classification_results = []

for name, X_emb in results.items():
    if X_emb is None:
        continue
    try:
        scores = cross_val_score(knn, X_emb, y, cv=5)
        classification_results.append([
            name,
            f"{scores.mean():.4f}",
            f"{scores.std():.4f}",
            f"{scores.mean():.2%}"
        ])
        print(f"{name:<10} -> Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    except Exception as e:
        print(f"{name:<10} -> Error: {e}")

# 打印表格
print("\n" + "-"*70)
headers = ["Method", "Mean Accuracy", "Std", "Accuracy %"]
print(tabulate(classification_results, headers=headers, tablefmt="grid"))
print("-"*70)
print("说明：使用5-NN分类器，5折交叉验证。LDA因利用标签信息通常最优。")
# ==========================================
# 定量表2：降维后聚类质量指标
# 评估无监督降维的类内紧凑度和类间分离度
# ==========================================
print("\n" + "="*70)
print("定量表2：降维后聚类质量指标对比")
print("="*70)

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

clustering_results = []

for name, X_emb in results.items():
    if X_emb is None:
        continue
    try:
        # 计算三个聚类指标（使用真实标签作为聚类标签）
        sil = silhouette_score(X_emb, y)
        db = davies_bouldin_score(X_emb, y)
        ch = calinski_harabasz_score(X_emb, y)
        
        clustering_results.append([
            name,
            f"{sil:.4f}",
            f"{db:.4f}",
            f"{ch:.2f}"
        ])
        print(f"{name:<10} -> Silhouette: {sil:.4f} | Davies-Bouldin: {db:.4f} | Calinski-Harabasz: {ch:.2f}")
    except Exception as e:
        print(f"{name:<10} -> Error: {e}")

# 打印表格
print("\n" + "-"*70)
headers = ["Method", "Silhouette ↑", "Davies-Bouldin ↓", "Calinski-Harabasz ↑"]
print(tabulate(clustering_results, headers=headers, tablefmt="grid"))
print("-"*70)
print("说明：↑ 表示越高越好，↓ 表示越低越好。使用真实数字标签作为聚类标签。")
print("Silhouette范围[-1,1]，越接近1表示聚类质量越好。")
# ==========================================
# 定量表3：计算时间与k近邻保持率
# 评估算法效率和局部结构保持能力
# ==========================================
print("\n" + "="*70)
print("定量表3：计算时间与结构保持度对比")
print("="*70)

from sklearn.neighbors import NearestNeighbors

def compute_knn_preservation(X_high, X_low, n_neighbors=10):
    """
    计算降维前后的k近邻保持率
    返回：信任度(Trustworthiness)和连续度(Continuity)的近似值
    """
    n_samples = len(X_high)
    
    # 高维空间k近邻
    nn_high = NearestNeighbors(n_neighbors=n_neighbors+1)
    nn_high.fit(X_high)
    _, idx_high = nn_high.kneighbors(X_high)
    idx_high = idx_high[:, 1:]  # 去掉自身
    
    # 低维空间k近邻
    nn_low = NearestNeighbors(n_neighbors=n_neighbors+1)
    nn_low.fit(X_low)
    _, idx_low = nn_low.kneighbors(X_low)
    idx_low = idx_low[:, 1:]  # 去掉自身
    
    # 计算重叠率（简化的信任度/连续度）
    overlap = 0
    for i in range(n_samples):
        overlap += len(set(idx_high[i]) & set(idx_low[i]))
    
    preservation_rate = overlap / (n_samples * n_neighbors)
    return preservation_rate

# 重新计时并计算结构保持度
print("\n重新运行各方法并记录精确时间...")
print("(使用相同数据规模以确保公平比较)\n")

timing_results = []

for name, func, kwargs in [
    ('PCA', run_pca, {'n_components': 2}),
    ('LDA', run_lda, {'n_components': 2}),
    ('KPCA', run_kpca, {'n_components': 2}),
    ('ISOMAP', run_isomap, {'n_components': 2}),
    ('t-SNE', run_tsne, {'n_components': 2, 'perplexity': 30, 'n_iter': 500}),
]:
    # 计时
    start_time = time.time()
    if name == 'LDA':
        X_emb = func(X_scaled, y, **kwargs)
    else:
        X_emb = func(X_scaled, y=None, **kwargs)
    elapsed = time.time() - start_time
    
    # 计算k近邻保持率
    preservation = compute_knn_preservation(X_scaled, X_emb, n_neighbors=10)
    
    timing_results.append([
        name,
        f"{elapsed:.3f}s",
        f"{preservation:.4f}",
        "O(n)" if name in ['PCA', 'LDA'] else "O(n²)" if name == 'KPCA' else "O(kn²logn)" if name == 'ISOMAP' else "O(n²)"
    ])
    print(f"{name:<10} -> Time: {elapsed:.3f}s | KNN Preservation (k=10): {preservation:.4f}")

if UMAP_AVAILABLE and 'UMAP' in results:
    start_time = time.time()
    X_umap = run_umap(X_scaled, y=None, n_components=2)
    elapsed = time.time() - start_time
    preservation = compute_knn_preservation(X_scaled, X_umap, n_neighbors=10)
    timing_results.append([
        'UMAP',
        f"{elapsed:.3f}s",
        f"{preservation:.4f}",
        "O(kn)"
    ])
    print(f"{'UMAP':<10} -> Time: {elapsed:.3f}s | KNN Preservation (k=10): {preservation:.4f}")

# 打印表格
print("\n" + "-"*70)
headers = ["Method", "Computation Time", "KNN Preservation ↑", "Complexity"]
print(tabulate(timing_results, headers=headers, tablefmt="grid"))
print("-"*70)
print("说明：KNN保持率越高，说明降维后局部邻域结构保留越好。")
print("PCA/LDA有解析解，速度最快；t-SNE/ISOMAP需迭代优化，速度较慢。")