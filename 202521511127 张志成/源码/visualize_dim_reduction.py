import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_and_sample(dataset_name, n_per_class=500):
    from sklearn.datasets import fetch_openml
    if dataset_name == 'MNIST':
        print("加载 MNIST...")
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True,
                            as_frame=False, parser='liac-arff')
    else:
        print("加载 Fashion-MNIST...")
        X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True,
                            as_frame=False, parser='liac-arff')
    X = X.astype(np.float32) / 255.0
    y = y.astype(int)


    rng = np.random.RandomState(42)
    classes = np.unique(y)
    indices = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        c_idx = rng.choice(c_idx, min(n_per_class, len(c_idx)), replace=False)
        indices.extend(c_idx)
    indices = np.array(indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


def plot_embedding(X_emb, y, method_name, dataset_name, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    classes = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    for i, c in enumerate(classes):
        mask = y == c
        ax.scatter(X_emb[mask, 0], X_emb[mask, 1],
                   c=[colors[i]], label=str(c), s=1.5, alpha=0.7)

    ax.set_title(f'{method_name} - {dataset_name}', fontsize=14)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend(markerscale=5, fontsize=7, loc='upper right',
              bbox_to_anchor=(1.15, 1.0), ncol=1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def run_visualization(dataset_name, load_func):
    """在一个数据集上运行所有方法并生成散点图"""
    X, y = load_func(dataset_name)
    print(f"数据: {dataset_name}, 样本: {X.shape[0]}, 维度: {X.shape[1]}")

    import os
    os.makedirs('figures', exist_ok=True)

    from sklearn.decomposition import PCA, KernelPCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.manifold import Isomap, TSNE

    methods = [
        ('PCA', PCA(n_components=2, random_state=42)),
        ('LDA', LinearDiscriminantAnalysis(n_components=2)),
        ('KPCA_RBF', KernelPCA(n_components=2, kernel='rbf', n_jobs=-1, random_state=42)),
        ('ISOMAP', Isomap(n_components=2, n_neighbors=10, n_jobs=-1)),
        ('TSNE', TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)),
    ]

    # UMAP
    try:
        import umap
        methods.append(('UMAP', umap.UMAP(n_components=2, n_neighbors=15,
                                           min_dist=0.1, random_state=42)))
    except ImportError:
        print("  UMAP不可用，跳过")

    for name, model in methods:
        print(f"  运行 {name}...")
        try:
            start = time.time()
            if name == 'LDA':
                X_emb = model.fit_transform(X, y)
            else:
                X_emb = model.fit_transform(X)
            elapsed = time.time() - start
            print(f"    耗时: {elapsed:.1f}s, 嵌入形状: {X_emb.shape}")
            plot_embedding(X_emb, y, name, dataset_name,
                           f'figures/{name}_{dataset_name}.png')
        except Exception as e:
            print(f"    {name} 失败: {e}")


if __name__ == '__main__':
    print("降维可视化实验")
    print("=" * 50)

    run_visualization('MNIST', load_and_sample)
    run_visualization('Fashion-MNIST', load_and_sample)

    print("\n完成！图片保存在 figures/ 目录下")
