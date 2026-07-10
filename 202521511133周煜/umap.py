"""
UMAP 降维方法对比实验
复现论文：基于UMAP的高维数据降维研究
包含：PCA / t-SNE / UMAP 三种方法对比、量化指标、参数敏感性分析
"""

import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# ========== 1. 数据加载与预处理 ==========
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_mnist_data(n_per_class=500, random_state=42):
    """加载MNIST数据集，每类抽取n_per_class个样本"""
    print("正在加载MNIST数据集...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.astype(np.float32) / 255.0  # 归一化到[0,1]
    y = mnist.target.astype(int)

    # 分层抽样，每类抽取n_per_class个样本
    np.random.seed(random_state)
    indices = []
    for label in range(10):
        label_idx = np.where(y == label)[0]
        selected = np.random.choice(label_idx, size=n_per_class, replace=False)
        indices.extend(selected)

    X_sample = X[indices]
    y_sample = y[indices]

    print(f"采样后数据维度: {X_sample.shape}, 类别数: {len(np.unique(y_sample))}")
    return X_sample, y_sample


# ========== 2. 降维方法 ==========
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def run_pca(X, n_components=2):
    """PCA降维"""
    start = time.time()
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    elapsed = time.time() - start
    print(f"PCA 完成，耗时: {elapsed:.2f}秒")
    return X_pca, elapsed


def run_tsne(X, n_components=2, perplexity=30, max_iter=1000, learning_rate=200):
    """t-SNE降维"""
    start = time.time()
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        max_iter=max_iter,
        learning_rate=learning_rate,
        init='pca',
        random_state=42
    )
    X_tsne = tsne.fit_transform(X)
    elapsed = time.time() - start
    print(f"t-SNE 完成，耗时: {elapsed:.2f}秒")
    return X_tsne, elapsed


def run_umap(X, n_components=2, n_neighbors=15, min_dist=0.1):
    """UMAP降维"""
    start = time.time()
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    X_umap = reducer.fit_transform(X)
    elapsed = time.time() - start
    print(f"UMAP 完成，耗时: {elapsed:.2f}秒")
    return X_umap, elapsed


# ========== 3. 评价指标 ==========
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors


def knn_accuracy(X_low, y, k=5, cv=5):
    """KNN分类准确率（5折交叉验证）"""
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_low, y, cv=cv, scoring='accuracy')
    return scores.mean() * 100


def local_preservation_accuracy(X_high, X_low, k=15):
    """局部保持准确率 LPA
    统计高维k近邻在低维中仍为近邻的比例
    """
    n_samples = X_high.shape[0]

    # 高维k近邻
    nn_high = NearestNeighbors(n_neighbors=k + 1).fit(X_high)
    distances_high, indices_high = nn_high.kneighbors(X_high)
    indices_high = indices_high[:, 1:]  # 去掉自身

    # 低维k近邻
    nn_low = NearestNeighbors(n_neighbors=k + 1).fit(X_low)
    distances_low, indices_low = nn_low.kneighbors(X_low)
    indices_low = indices_low[:, 1:]

    # 计算重叠比例
    overlap_count = 0
    for i in range(n_samples):
        high_neighbors = set(indices_high[i])
        low_neighbors = set(indices_low[i])
        overlap_count += len(high_neighbors & low_neighbors)

    lpa = overlap_count / (n_samples * k) * 100
    return lpa


# ========== 4. 可视化 ==========
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_comparison(results, y, save_path='comparison.png'):
    """三种方法可视化对比"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ax, (name, X_low, _) in zip(axes, results):
        for i, label in enumerate(range(10)):
            mask = y == label
            ax.scatter(X_low[mask, 0], X_low[mask, 1],
                       c=[colors[i]], label=str(label), s=10, alpha=0.7)
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Digit')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"可视化结果已保存至: {save_path}")
    plt.close()


def plot_parameter_sensitivity(param_name, param_values, metrics, save_path):
    """参数敏感性分析图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    x = range(len(param_values))
    xticks = [str(v) for v in param_values]

    # KNN准确率
    ax1.plot(x, metrics['knn_acc'], 'o-', linewidth=2, markersize=8, color='#2196F3')
    ax1.set_title(f'{param_name} 对 KNN 分类准确率的影响', fontsize=12)
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('KNN Accuracy (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(xticks)
    ax1.grid(True, alpha=0.3)

    # 局部保持准确率
    ax2.plot(x, metrics['lpa'], 'o-', linewidth=2, markersize=8, color='#FF5722')
    ax2.set_title(f'{param_name} 对局部保持准确率的影响', fontsize=12)
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('LPA (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(xticks)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"参数敏感性分析图已保存至: {save_path}")
    plt.close()


# ========== 5. 主实验流程 ==========
def main():
    print("=" * 60)
    print("UMAP 降维方法对比实验")
    print("=" * 60)

    # 加载数据
    X, y = load_mnist_data(n_per_class=500, random_state=42)

    # ---- 实验1：三种方法对比 ----
    print("\n" + "=" * 60)
    print("实验1：PCA / t-SNE / UMAP 对比")
    print("=" * 60)

    X_pca, time_pca = run_pca(X)
    X_tsne, time_tsne = run_tsne(X)
    X_umap, time_umap = run_umap(X)

    results = [
        ('PCA', X_pca, time_pca),
        ('t-SNE', X_tsne, time_tsne),
        ('UMAP', X_umap, time_umap)
    ]

    # 计算量化指标
    print("\n计算量化指标...")
    metrics_table = []
    for name, X_low, elapsed in results:
        knn_acc = knn_accuracy(X_low, y)
        lpa = local_preservation_accuracy(X, X_low, k=15)
        metrics_table.append({
            '方法': name,
            'KNN准确率(%)': round(knn_acc, 2),
            '局部保持准确率(%)': round(lpa, 2),
            '运行时间(秒)': round(elapsed, 2)
        })

    # 打印量化对比表
    print("\n" + "-" * 60)
    print(f"{'方法':<10} {'KNN准确率(%)':<15} {'LPA(%)':<15} {'运行时间(秒)':<15}")
    print("-" * 60)
    for row in metrics_table:
        print(f"{row['方法']:<10} {row['KNN准确率(%)']:<15} {row['局部保持准确率(%)']:<15} {row['运行时间(秒)']:<15}")
    print("-" * 60)

    # 可视化对比
    plot_comparison(results, y, 'comparison_3methods.png')

    # ---- 实验2：n_neighbors 参数敏感性 ----
    print("\n" + "=" * 60)
    print("实验2：n_neighbors 参数敏感性分析")
    print("=" * 60)

    n_neighbors_list = [5, 15, 30, 50]
    nn_metrics = {'knn_acc': [], 'lpa': []}

    for nn in n_neighbors_list:
        print(f"\nn_neighbors = {nn}")
        X_umap_nn, _ = run_umap(X, n_neighbors=nn, min_dist=0.1)
        nn_metrics['knn_acc'].append(knn_accuracy(X_umap_nn, y))
        nn_metrics['lpa'].append(local_preservation_accuracy(X, X_umap_nn))

    plot_parameter_sensitivity('n_neighbors', n_neighbors_list, nn_metrics,
                               'sensitivity_n_neighbors.png')

    # ---- 实验3：min_dist 参数敏感性 ----
    print("\n" + "=" * 60)
    print("实验3：min_dist 参数敏感性分析")
    print("=" * 60)

    min_dist_list = [0.0, 0.1, 0.3, 0.5]
    md_metrics = {'knn_acc': [], 'lpa': []}

    for md in min_dist_list:
        print(f"\nmin_dist = {md}")
        X_umap_md, _ = run_umap(X, n_neighbors=15, min_dist=md)
        md_metrics['knn_acc'].append(knn_accuracy(X_umap_md, y))
        md_metrics['lpa'].append(local_preservation_accuracy(X, X_umap_md))

    plot_parameter_sensitivity('min_dist', min_dist_list, md_metrics,
                               'sensitivity_min_dist.png')

    # ---- 实验4：UMAP 新样本变换能力演示 ----
    print("\n" + "=" * 60)
    print("实验4：UMAP 新样本变换能力（归纳式学习）")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # UMAP 支持 transform
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    reducer.fit(X_train)
    X_train_umap = reducer.transform(X_train)
    X_test_umap = reducer.transform(X_test)
    print(f"UMAP 训练集降维: {X_train_umap.shape}")
    print(f"UMAP 测试集降维: {X_test_umap.shape}")
    print("UMAP 支持对新样本直接变换，具备归纳式学习能力")

    print("\n" + "=" * 60)
    print("所有实验完成！生成的文件：")
    print("  - comparison_3methods.png  (三种方法可视化对比)")
    print("  - sensitivity_n_neighbors.png (n_neighbors参数敏感性)")
    print("  - sensitivity_min_dist.png (min_dist参数敏感性)")
    print("=" * 60)


if __name__ == '__main__':
    main()
