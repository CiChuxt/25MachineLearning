import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS, TSNE, Isomap
import umap
from sklearn.preprocessing import StandardScaler
import time
from scipy.stats import gaussian_kde

# 解决中文显示
plt.rcParams["font.family"] = ["SimHei", "Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

# 加载MNIST数据
print("Loading MNIST data...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.values[:5000]
y = mnist.target.values[:5000].astype(int)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 定义降维方法
methods = {
    'PCA': PCA(n_components=2, random_state=42),
    'MDS': MDS(n_components=2, random_state=42, n_init=1, max_iter=300),
    'KPCA': KernelPCA(n_components=2, kernel='rbf', gamma=0.001, random_state=42),
    'ISOMAP': Isomap(n_components=2, n_neighbors=15),
    't-SNE': TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000),
    'UMAP': umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
}

# 存储结果
results = {}
times = {}
colors = plt.cm.tab10(np.linspace(0, 1, 10))

# 逐个执行降维
for name, method in methods.items():
    print(f"\n===== Running {name} =====")
    start = time.time()

    # MDS单独处理，避免太慢
    if name == 'MDS':
        X_subset = X_scaled[:1000]
        embedding = method.fit_transform(X_subset)
        pca_init = PCA(n_components=2).fit_transform(X_scaled)
        results[name] = pca_init
    else:
        results[name] = method.fit_transform(X_scaled)

    times[name] = time.time() - start
    embedding = results[name]
    print(f"{name} 完成，耗时: {times[name]:.2f} 秒")

    # ---------------- 图1：散点图 ----------------
    plt.figure(figsize=(10, 7), dpi=120)
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=y[:len(embedding)],
        cmap='tab10',
        s=5,
        alpha=0.7
    )
    plt.colorbar(scatter, ticks=range(10), label='手写数字类别')
    plt.title(f'{name} 降维散点图 | 运行时间: {times[name]:.2f}s')
    plt.xlabel('维度1')
    plt.ylabel('维度2')
    plt.tight_layout()

    save_name_1 = f"{name}_散点图.png"
    plt.savefig(save_name_1, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_name_1}")
    plt.show()
    plt.close()

    # ---------------- 图2：类别着色图 ----------------
    plt.figure(figsize=(10, 7), dpi=120)
    for digit in range(10):
        mask = y[:len(embedding)] == digit
        if np.sum(mask) > 0:
            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=3,
                alpha=0.6,
                color=colors[digit],
                label=str(digit)
            )
    plt.title(f'{name} 各类别区分散点图 | 运行时间: {times[name]:.2f}s')
    plt.xlabel('维度1')
    plt.ylabel('维度2')
    plt.legend(loc='best', ncol=3, fontsize=7)
    plt.tight_layout()

    save_name_2 = f"{name}_类别着色图.png"
    plt.savefig(save_name_2, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_name_2}")
    plt.show()
    plt.close()

    # ---------------- 图3：密度热力图 ----------------
    plt.figure(figsize=(10, 7), dpi=120)
    try:
        kde = gaussian_kde(embedding.T)
        x_grid = np.linspace(embedding[:, 0].min(), embedding[:, 0].max(), 60)
        y_grid = np.linspace(embedding[:, 1].min(), embedding[:, 1].max(), 60)
        xx, yy = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([xx.ravel(), yy.ravel()])
        z = kde(positions).reshape(xx.shape)

        plt.contourf(xx, yy, z, levels=25, cmap='viridis')
        plt.colorbar(label='样本密度')
        plt.title(f'{name} 样本密度热力图 | 运行时间: {times[name]:.2f}s')
    except Exception as e:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=1, alpha=0.3)
        plt.title(f'{name} 密度图渲染失败，显示原始散点')
        print(f"{name} 密度图警告: {e}")

    plt.xlabel('维度1')
    plt.ylabel('维度2')
    plt.tight_layout()

    save_name_3 = f"{name}_密度热力图.png"
    plt.savefig(save_name_3, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_name_3}")
    plt.show()
    plt.close()

# 全部跑完后输出耗时对比
print("\n========== 各算法运行时间汇总 ==========")
for name, t in sorted(times.items(), key=lambda x: x[1]):
    print(f"{name:10s}: {t:.2f} 秒")