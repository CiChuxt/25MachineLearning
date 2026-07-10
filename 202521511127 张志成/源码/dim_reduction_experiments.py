"""
高维数据降维方法对比实验
======================
方法: PCA, MDA(LDA), KPCA, ISOMAP, T-SNE, UMAP
数据集: MNIST, Fashion-MNIST, COIL-20
评估指标: kNN准确率, 信任度(Trustworthiness), 连续性(Continuity), 计算时间
"""

import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 数据加载
# ============================================================

def load_mnist_subset(n_samples_per_class=300, random_state=42):
    """加载MNIST子集"""
    from sklearn.datasets import fetch_openml
    print("正在加载MNIST数据集...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
    X = X.astype(np.float32) / 255.0
    y = y.astype(int)
    return stratified_sample(X, y, n_samples_per_class, random_state)

def load_fashion_mnist_subset(n_samples_per_class=300, random_state=42):
    """加载Fashion-MNIST子集"""
    from sklearn.datasets import fetch_openml
    print("正在加载Fashion-MNIST数据集...")
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
    X = X.astype(np.float32) / 255.0
    y = y.astype(int)
    return stratified_sample(X, y, n_samples_per_class, random_state)

def load_coil20_subset(random_state=42):
    """尝试加载COIL-20，失败则使用Swiss Roll合成流形数据"""
    print("正在尝试加载COIL-20数据集...")
    try:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml('coil20', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
        X = X.astype(np.float32) / 255.0
        y = y.astype(int)
        rng = np.random.RandomState(random_state)
        if X.shape[0] > 2000:
            indices = rng.choice(X.shape[0], 1440, replace=False)
            X, y = X[indices], y[indices]
        return X, y
    except Exception:
        print("COIL-20无法直接加载，使用Swiss Roll合成流形数据作为替代...")
        from sklearn.datasets import make_swiss_roll
        X, t = make_swiss_roll(n_samples=1500, noise=0.0, random_state=random_state)
        # 使用角度作为类别标签（每10度一个类别）
        y = np.digitize(t, bins=np.linspace(t.min(), t.max(), 20)) - 1
        y = np.clip(y, 0, 19)
        return X.astype(np.float32), y

def stratified_sample(X, y, n_per_class, random_state=42):
    """分层抽样"""
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)
    indices = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        if len(c_idx) > n_per_class:
            c_idx = rng.choice(c_idx, n_per_class, replace=False)
        indices.extend(c_idx)
    indices = np.array(indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


# ============================================================
# 降维方法
# ============================================================

class DimReductionMethods:
    """统一封装各降维方法"""

    @staticmethod
    def pca(X, n_components=2):
        from sklearn.decomposition import PCA
        start = time.time()
        model = PCA(n_components=n_components, random_state=42)
        X_emb = model.fit_transform(X)
        elapsed = time.time() - start
        return X_emb, elapsed, model

    @staticmethod
    def lda(X, y, n_components=2):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        start = time.time()
        # LDA 最多降至 min(n_features, n_classes - 1)
        max_dim = min(X.shape[1], len(np.unique(y)) - 1)
        n_comp = min(n_components, max_dim)
        if n_comp < 2:
            # 若只能降至1维，先降至1维再补零
            model = LinearDiscriminantAnalysis(n_components=1)
            X_emb_1d = model.fit_transform(X, y)
            X_emb = np.hstack([X_emb_1d, np.zeros_like(X_emb_1d)])
            elapsed = time.time() - start
            return X_emb, elapsed, model
        model = LinearDiscriminantAnalysis(n_components=n_comp)
        X_emb = model.fit_transform(X, y)
        elapsed = time.time() - start
        return X_emb, elapsed, model

    @staticmethod
    def kpca(X, n_components=2, kernel='rbf', gamma=None):
        from sklearn.decomposition import KernelPCA
        start = time.time()
        model = KernelPCA(n_components=n_components, kernel=kernel,
                          gamma=gamma, n_jobs=-1, random_state=42)
        X_emb = model.fit_transform(X)
        elapsed = time.time() - start
        return X_emb, elapsed, model

    @staticmethod
    def isomap(X, n_components=2, n_neighbors=10):
        from sklearn.manifold import Isomap
        start = time.time()
        model = Isomap(n_components=n_components, n_neighbors=n_neighbors, n_jobs=-1)
        X_emb = model.fit_transform(X)
        elapsed = time.time() - start
        return X_emb, elapsed, model

    @staticmethod
    def tsne(X, n_components=2, perplexity=30, random_state=42):
        from sklearn.manifold import TSNE
        start = time.time()
        # 先用PCA降至50维加速
        from sklearn.decomposition import PCA
        if X.shape[1] > 50:
            X_reduced = PCA(n_components=50, random_state=random_state).fit_transform(X)
        else:
            X_reduced = X
        model = TSNE(n_components=n_components, perplexity=perplexity,
                     random_state=random_state, n_jobs=-1, verbose=0)
        X_emb = model.fit_transform(X_reduced)
        elapsed = time.time() - start
        return X_emb, elapsed, model

    @staticmethod
    def umap(X, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
        try:
            import umap
        except ImportError:
            print("  [警告] umap-learn 未安装，将跳过UMAP")
            return None, 0, None
        start = time.time()
        model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                          min_dist=min_dist, random_state=random_state, verbose=False)
        X_emb = model.fit_transform(X)
        elapsed = time.time() - start
        return X_emb, elapsed, model


# ============================================================
# 评估指标
# ============================================================

def evaluate_knn_accuracy(X_emb, y, n_neighbors=5):
    """在降维空间中评估kNN分类准确率"""
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    scores = cross_val_score(knn, X_emb, y, cv=5, scoring='accuracy')
    return scores.mean()

def evaluate_trustworthiness(X_high, X_low, n_neighbors=5):
    """计算信任度 — 使用sklearn内置高效实现"""
    from sklearn.manifold import trustworthiness
    return trustworthiness(X_high, X_low, n_neighbors=n_neighbors)

def evaluate_continuity(X_high, X_low, n_neighbors=5):
    """计算连续性 — 交换参数调用trustworthiness"""
    from sklearn.manifold import trustworthiness
    return trustworthiness(X_low, X_high, n_neighbors=n_neighbors)


# ============================================================
# 实验主流程
# ============================================================

def run_experiment(dataset_name, load_func, methods_to_run=None, sample_size=3000):
    """在单个数据集上运行所有降维方法的实验"""
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*60}")

    # 加载数据
    X, y = load_func()
    if X is None:
        print(f"  [跳过] {dataset_name} 数据加载失败")
        return None

    print(f"  样本数: {X.shape[0]}, 维度: {X.shape[1]}, 类别数: {len(np.unique(y))}")

    # 若样本过多，进一步采样以加速T-SNE
    if X.shape[0] > sample_size:
        rng = np.random.RandomState(42)
        classes = np.unique(y)
        indices = []
        n_per = sample_size // len(classes)
        for c in classes:
            c_idx = np.where(y == c)[0]
            c_idx = rng.choice(c_idx, n_per, replace=False)
            indices.extend(c_idx)
        indices = np.array(indices)
        rng.shuffle(indices)
        X, y = X[indices], y[indices]
        print(f"  采样后样本数: {X.shape[0]}")

    # 定义要运行的方法
    all_methods = {
        'PCA':        lambda: DimReductionMethods.pca(X),
        'MDA (LDA)':  lambda: DimReductionMethods.lda(X, y),
        'KPCA (RBF)': lambda: DimReductionMethods.kpca(X, kernel='rbf', gamma=None),
        'ISOMAP':     lambda: DimReductionMethods.isomap(X, n_neighbors=10),
        'T-SNE':      lambda: DimReductionMethods.tsne(X),
        'UMAP':       lambda: DimReductionMethods.umap(X),
    }

    results = {}
    for method_name, method_func in all_methods.items():
        print(f"\n  运行 {method_name}...")
        try:
            X_emb, elapsed, model = method_func()
            if X_emb is None:
                print(f"    [跳过] {method_name} 不可用")
                continue

            print(f"    嵌入耗时: {elapsed:.1f} 秒")

            # 评估指标
            knn_acc = evaluate_knn_accuracy(X_emb, y)
            print(f"    kNN准确率 (5折CV): {knn_acc:.4f}")

            trust = evaluate_trustworthiness(X, X_emb)
            print(f"    信任度: {trust:.4f}")

            cont = evaluate_continuity(X, X_emb)
            print(f"    连续性: {cont:.4f}")

            results[method_name] = {
                'knn_accuracy': knn_acc,
                'trustworthiness': trust,
                'continuity': cont,
                'time': elapsed,
                'embedding': X_emb,
                'model': model
            }
        except Exception as e:
            print(f"    [错误] {method_name}: {e}")

    return results


def print_results_table(dataset_name, results):
    """打印结果表格"""
    if results is None:
        return
    print(f"\n{'='*80}")
    print(f"  {dataset_name} 实验结果汇总")
    print(f"{'='*80}")
    header = f"  {'方法':<15} {'kNN准确率':>10} {'信任度':>8} {'连续性':>8} {'时间(秒)':>10}"
    print(header)
    print("  " + "-" * len(header))
    for method, metrics in results.items():
        print(f"  {method:<15} {metrics['knn_accuracy']:>10.4f} "
              f"{metrics['trustworthiness']:>8.4f} {metrics['continuity']:>8.4f} "
              f"{metrics['time']:>10.1f}")


def generate_latex_table(dataset_name, results):
    """生成LaTeX格式的表格"""
    if results is None:
        return ""
    rows = []
    for method, m in results.items():
        rows.append(
            f"  {method} & {m['knn_accuracy']:.1f}\\% & "
            f"{m['trustworthiness']:.3f} & {m['continuity']:.3f} & "
            f"{m['time']:.0f} \\\\"
        )
    return "\n".join(rows)


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("  高维数据降维方法对比实验")
    print("  方法: PCA, MDA(LDA), KPCA, ISOMAP, T-SNE, UMAP")
    print("=" * 60)

    # 检查依赖
    print("\n检查依赖库...")
    deps_ok = True
    for lib in ['sklearn', 'numpy']:
        try:
            __import__(lib.replace('-', '_'))
            print(f"  ✓ {lib}")
        except ImportError:
            print(f"  ✗ {lib} 未安装")
            deps_ok = False
    try:
        import umap
        print(f"  ✓ umap-learn")
    except ImportError:
        print(f"  ⚠ umap-learn 未安装，将跳过UMAP方法")

    if not deps_ok:
        print("\n请先安装缺失的依赖: pip install scikit-learn numpy")
        sys.exit(1)

    all_results = {}

    # ---- MNIST ----
    try:
        results_mnist = run_experiment("MNIST", load_mnist_subset)
        print_results_table("MNIST", results_mnist)
        all_results['MNIST'] = results_mnist
    except Exception as e:
        print(f"\nMNIST实验失败: {e}")
        all_results['MNIST'] = None

    # ---- Fashion-MNIST ----
    try:
        results_fmnist = run_experiment("Fashion-MNIST", load_fashion_mnist_subset)
        print_results_table("Fashion-MNIST", results_fmnist)
        all_results['Fashion-MNIST'] = results_fmnist
    except Exception as e:
        print(f"\nFashion-MNIST实验失败: {e}")
        all_results['Fashion-MNIST'] = None

    # ---- COIL-20 ----
    try:
        results_coil = run_experiment("COIL-20", load_coil20_subset, sample_size=1500)
        print_results_table("COIL-20", results_coil)
        all_results['COIL-20'] = results_coil
    except Exception as e:
        print(f"\nCOIL-20实验失败: {e}")
        all_results['COIL-20'] = None

    # ============================================================
    # 最终汇总
    # ============================================================
    print(f"\n\n{'='*60}")
    print("  最终汇总")
    print(f"{'='*60}")

    for ds_name, results in all_results.items():
        if results is None:
            print(f"\n{ds_name}: 数据不可用")
            continue
        print(f"\n--- {ds_name} ---")
        print_results_table(ds_name, results)

    # 保存数值结果到文件
    print(f"\n\n正在保存结果到 result_tables.txt ...")
    with open('result_tables.txt', 'w', encoding='utf-8') as f:
        f.write("降维方法对比实验结果\n")
        f.write("=" * 60 + "\n\n")
        for ds_name, results in all_results.items():
            if results is None:
                f.write(f"\n{ds_name}: 数据不可用\n")
                continue
            f.write(f"\n{ds_name}:\n")
            f.write(f"{'方法':<15} {'kNN准确率':>10} {'信任度':>8} {'连续性':>8} {'时间(秒)':>10}\n")
            f.write("-" * 55 + "\n")
            for method, m in results.items():
                f.write(f"{method:<15} {m['knn_accuracy']:>10.4f} "
                        f"{m['trustworthiness']:>8.4f} {m['continuity']:>8.4f} "
                        f"{m['time']:>10.1f}\n")

    print("完成! 结果已保存到 result_tables.txt")
