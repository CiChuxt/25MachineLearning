import os, json, time, warnings, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap, trustworthiness
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import umap

warnings.filterwarnings("ignore")


fm.fontManager.addfont('/usr/share/fonts/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.sans-serif'] = ['Noto Serif SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['savefig.bbox'] = 'tight'


DATA_DIR   = "/home/z/my-project/data"
FIG_DIR    = "/home/z/my-project/figures"
RESULT_DIR = "/home/z/my-project/results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# Subsample size for fair comparison (ISOMAP/t-SNE 限制)
N_VIS = 3000      # 用于2D可视化
N_TIME = 3000     # 用于运行时间测试
N_CLS_TRAIN = 10000  # 用于分类训练
N_CLS_TEST = 2000
N_PARAM = 2000    # 参数敏感性分析

# MNIST 10类数字颜色
DIGIT_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))



# 1. Data Loading

def load_mnist():
    """加载MNIST数据，缓存到本地。"""
    cache_path = os.path.join(DATA_DIR, "mnist.pkl")
    if os.path.exists(cache_path):
        print("[LOAD] 从缓存读取MNIST...")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["X"], data["y"]

    print("[LOAD] 从OpenML下载MNIST ...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff")
    X = X.astype(np.float32) / 255.0   # 归一化到[0,1]
    y = y.astype(np.int64)
    with open(cache_path, "wb") as f:
        pickle.dump({"X": X, "y": y}, f)
    print(f"[LOAD] 完成: X.shape={X.shape}, y.shape={y.shape}")
    return X, y



# 2. Subsample helper

def subsample(X, y, n, seed=SEED):
    """分层随机子采样，保持类别比例。"""
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        n_c = max(1, int(round(n * len(c_idx) / len(y))))
        n_c = min(n_c, len(c_idx))
        idx.extend(rng.choice(c_idx, n_c, replace=False))
    rng.shuffle(idx)
    return X[idx], y[idx]



# 3. Six Dimensionality Reduction Methods

def run_pca(X, n=2):
    return PCA(n_components=n, random_state=SEED).fit_transform(X)

def run_lda(X, y, n=2):
    # LDA 最多 min(n_classes-1, n_features) = 9 维
    n_eff = min(n, len(np.unique(y)) - 1)
    lda = LinearDiscriminantAnalysis(n_components=n_eff)
    return lda.fit_transform(X, y)

def run_kpca(X, n=2):
    return KernelPCA(n_components=n, kernel="rbf", gamma=1e-3,
                     random_state=SEED, eigen_solver="arpack").fit_transform(X)

def run_isomap(X, n=2, n_neighbors=15):
    return Isomap(n_components=n, n_neighbors=n_neighbors,
                  n_jobs=-1).fit_transform(X)

def run_tsne(X, n=2, perplexity=30.0, learning_rate=200, max_iter=1000):
    return TSNE(n_components=n, perplexity=perplexity, learning_rate=learning_rate,
                max_iter=max_iter, random_state=SEED, init="pca",
                metric="euclidean", n_jobs=-1).fit_transform(X)

def run_umap(X, n=2, n_neighbors=15, min_dist=0.1):
    return umap.UMAP(n_components=n, n_neighbors=n_neighbors, min_dist=min_dist,
                     random_state=SEED, n_jobs=-1).fit_transform(X)



# 4. Experiment A: 2D Visualization Comparison

def experiment_visualization(X_sub, y_sub):
    print("\n" + "="*60)
    print("实验A: 2D 可视化对比")
    print("="*60)
    results = {}

    methods = [
        ("PCA",     lambda: run_pca(X_sub)),
        ("LDA",     lambda: run_lda(X_sub, y_sub)),
        ("KPCA",    lambda: run_kpca(X_sub)),
        ("ISOMAP",  lambda: run_isomap(X_sub)),
        ("t-SNE",   lambda: run_tsne(X_sub)),
        ("UMAP",    lambda: run_umap(X_sub)),
    ]

    embeddings = {}
    for name, fn in methods:
        t0 = time.time()
        try:
            Z = fn()
            dt = time.time() - t0
            embeddings[name] = Z
            print(f"  [{name:8s}] shape={Z.shape}  time={dt:.2f}s")
            results[name] = {"shape": list(Z.shape), "time": dt, "status": "ok"}
        except Exception as e:
            print(f"  [{name:8s}] FAILED: {e}")
            results[name] = {"status": "fail", "error": str(e)}

    # 2x3 grid 散点图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    order = ["PCA", "LDA", "KPCA", "ISOMAP", "t-SNE", "UMAP"]
    for ax, name in zip(axes, order):
        if name not in embeddings:
            ax.text(0.5, 0.5, f"{name}\nFAILED", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(name, fontsize=14)
            continue
        Z = embeddings[name]
        for c in range(10):
            m = y_sub == c
            ax.scatter(Z[m, 0], Z[m, 1], c=[DIGIT_COLORS[c]], s=4, alpha=0.6,
                       label=str(c), rasterized=True)
        ax.set_title(name, fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("Component 1", fontsize=9)
        ax.set_ylabel("Component 2", fontsize=9)
    # 通用图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=10,
               fontsize=11, markerscale=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("六种降维方法在 MNIST 上的 2D 可视化对比 (N=3000)",
                 fontsize=15, y=1.00)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "expA_2d_visualization.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] {fig_path}")

    # 保存嵌入供后续使用
    with open(os.path.join(RESULT_DIR, "embeddings.pkl"), "wb") as f:
        pickle.dump({"embeddings": embeddings, "y": y_sub}, f)

    return results, embeddings



# 5. Experiment B: Quantitative Metrics
#    Trustworthiness, Continuity, KNN preservation (k=5,15)

def experiment_metrics(X_sub, y_sub, embeddings):
    print("\n" + "="*60)
    print("实验B: 定量指标 (Trustworthiness / Continuity / KNN保持率)")
    print("="*60)

    from sklearn.metrics import pairwise_distances
    D_high = pairwise_distances(X_sub)
    n = X_sub.shape[0]

    # KNN保持率: 高维 k近邻 在低维仍是 k近邻 的比例
    def knn_preservation(D_high, D_low, k=5):
        # 高维每个样本的k近邻索引
        nn_high = np.argsort(D_high, axis=1)[:, 1:k+1]
        nn_low  = np.argsort(D_low,  axis=1)[:, 1:k+1]
        preserve = []
        for i in range(D_high.shape[0]):
            inter = len(set(nn_high[i]) & set(nn_low[i]))
            preserve.append(inter / k)
        return float(np.mean(preserve))

    # Continuity: 与Trustworthiness对称 (用sklearn的trustworthiness对低维评估,
    # 然后交换 D 的角色来近似 continuity)
    metrics = {}
    for name, Z in embeddings.items():
        D_low = pairwise_distances(Z)
        tw = trustworthiness(X_sub, Z, n_neighbors=5)
        tw15 = trustworthiness(X_sub, Z, n_neighbors=15)
        # Continuity: 交换高维/低维 (因为trustworthiness衡量低维近邻在高维也是近邻;
        # continuity 衡量高维近邻在低维也是近邻,等价于trustworthiness(X=Z, X_embedded=X))
        ct = trustworthiness(Z, X_sub, n_neighbors=5)
        ct15 = trustworthiness(Z, X_sub, n_neighbors=15)
        kp5  = knn_preservation(D_high, D_low, k=5)
        kp15 = knn_preservation(D_high, D_low, k=15)
        metrics[name] = {
            "trustworthiness_k5":  float(tw),
            "trustworthiness_k15": float(tw15),
            "continuity_k5":       float(ct),
            "continuity_k15":      float(ct15),
            "knn_preservation_k5":  float(kp5),
            "knn_preservation_k15": float(kp15),
        }
        print(f"  [{name:8s}] T@5={tw:.3f} T@15={tw15:.3f}  C@5={ct:.3f} C@15={ct15:.3f}  KNN@5={kp5:.3f} KNN@15={kp15:.3f}")

    # 绘制柱状图
    names = list(metrics.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(names))
    w = 0.18

    # 子图1: Trustworthiness & Continuity
    ax = axes[0]
    ax.bar(x - 1.5*w, [metrics[n]["trustworthiness_k5"]  for n in names], w, label="Trustworthiness k=5",  color="#4C72B0")
    ax.bar(x - 0.5*w, [metrics[n]["trustworthiness_k15"] for n in names], w, label="Trustworthiness k=15", color="#DD8452")
    ax.bar(x + 0.5*w, [metrics[n]["continuity_k5"]       for n in names], w, label="Continuity k=5",       color="#55A868")
    ax.bar(x + 1.5*w, [metrics[n]["continuity_k15"]      for n in names], w, label="Continuity k=15",      color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Trustworthiness & Continuity")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # 子图2: KNN preservation
    ax = axes[1]
    ax.bar(x - w/2, [metrics[n]["knn_preservation_k5"]  for n in names], w, label="KNN preservation k=5",  color="#4C72B0")
    ax.bar(x + w/2, [metrics[n]["knn_preservation_k15"] for n in names], w, label="KNN preservation k=15", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Preservation ratio")
    ax.set_title("KNN Preservation Rate")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("六种降维方法的定量指标对比 (N=3000)", fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "expB_quantitative_metrics.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] {fig_path}")

    return metrics



# 6. Experiment C: Runtime Comparison

def experiment_runtime(X_sub, y_sub):
    print("\n" + "="*60)
    print("实验C: 运行时间对比 (N=3000)")
    print("="*60)

    methods = [
        ("PCA",     lambda: PCA(n_components=2, random_state=SEED).fit_transform(X_sub)),
        ("LDA",     lambda: LinearDiscriminantAnalysis(n_components=2).fit_transform(X_sub, y_sub)),
        ("KPCA",    lambda: KernelPCA(n_components=2, kernel="rbf", gamma=1e-3,
                                       random_state=SEED, eigen_solver="arpack").fit_transform(X_sub)),
        ("ISOMAP",  lambda: Isomap(n_components=2, n_neighbors=15, n_jobs=-1).fit_transform(X_sub)),
        ("t-SNE",   lambda: TSNE(n_components=2, perplexity=30, max_iter=1000,
                                  random_state=SEED, init="pca", n_jobs=-1).fit_transform(X_sub)),
        ("UMAP",    lambda: umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                                       random_state=SEED, n_jobs=-1).fit_transform(X_sub)),
    ]

    timings = {}
    for name, fn in methods:
        # warmup for short methods
        if name == "PCA":
            _ = fn()
        t0 = time.time()
        try:
            _ = fn()
            dt = time.time() - t0
            timings[name] = dt
            print(f"  [{name:8s}] {dt:.3f} s")
        except Exception as e:
            timings[name] = float('nan')
            print(f"  [{name:8s}] FAILED: {e}")

    # 柱状图
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(timings.keys())
    times = [timings[n] for n in names]
    colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3","#937860"]
    bars = ax.bar(names, times, color=colors[:len(names)])
    ax.set_ylabel("Runtime (seconds, log scale)")
    ax.set_yscale("log")
    ax.set_title("六种降维方法的训练时间对比 (N=3000, single core)")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.05,
                f"{t:.2f}s", ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", alpha=0.3, which="both")
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "expC_runtime.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] {fig_path}")
    return timings



#    t-SNE perplexity: 5, 30, 100
#    UMAP n_neighbors: 5, 15, 50

def experiment_param_sensitivity(X_sub, y_sub):
    print("\n" + "="*60)
    print("实验D: 参数敏感性分析 (N=2000)")
    print("="*60)

    # t-SNE perplexity
    perplexities = [5, 30, 100]
    tsne_results = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, p in zip(axes, perplexities):
        # perplexity 必须小于 n_samples-1
        p_eff = min(p, X_sub.shape[0] - 1)
        t0 = time.time()
        Z = TSNE(n_components=2, perplexity=p_eff, max_iter=1000,
                 random_state=SEED, init="pca", n_jobs=-1).fit_transform(X_sub)
        dt = time.time() - t0
        tsne_results[p] = {"Z": Z, "time": dt}
        for c in range(10):
            m = y_sub == c
            ax.scatter(Z[m, 0], Z[m, 1], c=[DIGIT_COLORS[c]], s=4, alpha=0.6, rasterized=True)
        ax.set_title(f"t-SNE (perplexity={p_eff})\nt={dt:.1f}s", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("t-SNE 的 perplexity 参数敏感性", fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "expD_tsne_perplexity.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] {fig_path}")

    # UMAP n_neighbors
    n_neighbors_list = [5, 15, 50]
    umap_results = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, nn in zip(axes, n_neighbors_list):
        nn_eff = min(nn, X_sub.shape[0] - 1)
        t0 = time.time()
        Z = umap.UMAP(n_components=2, n_neighbors=nn_eff, min_dist=0.1,
                      random_state=SEED, n_jobs=-1).fit_transform(X_sub)
        dt = time.time() - t0
        umap_results[nn] = {"Z": Z, "time": dt}
        for c in range(10):
            m = y_sub == c
            ax.scatter(Z[m, 0], Z[m, 1], c=[DIGIT_COLORS[c]], s=4, alpha=0.6, rasterized=True)
        ax.set_title(f"UMAP (n_neighbors={nn_eff})\nt={dt:.1f}s", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("UMAP 的 n_neighbors 参数敏感性", fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "expD_umap_nneighbors.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] {fig_path}")

    return {
        "tsne_perplexity_times": {str(k): v["time"] for k, v in tsne_results.items()},
        "umap_nneighbors_times": {str(k): v["time"] for k, v in umap_results.items()},
    }



# 8. Experiment E: Downstream Classification (KNN after DR)

def experiment_classification(X, y):
    print("\n" + "="*60)
    print("实验E: 降维+KNN 下游分类性能对比")
    print("="*60)

    # 训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=N_CLS_TRAIN, test_size=N_CLS_TEST,
        random_state=SEED, stratify=y
    )

    # 测试维度
    dims_to_test = [2, 10, 50]
    methods_factory = {
        "PCA":     lambda d: PCA(n_components=d, random_state=SEED),
        "LDA":     lambda d: LinearDiscriminantAnalysis(n_components=min(d, 9)),
        "KPCA":    lambda d: KernelPCA(n_components=d, kernel="rbf", gamma=1e-3,
                                       random_state=SEED, eigen_solver="arpack"),
        # ISOMAP/t-SNE/UMAP 在 10000 样本 + 多维度上太慢,跳过或用更小子集
    }

    results = {}
    for name, factory in methods_factory.items():
        results[name] = {}
        for d in dims_to_test:
            try:
                t0 = time.time()
                reducer = factory(d)
                # LDA 需要 y
                if name == "LDA":
                    Z_tr = reducer.fit_transform(X_train, y_train)
                    Z_te = reducer.transform(X_test)
                else:
                    Z_tr = reducer.fit_transform(X_train)
                    Z_te = reducer.transform(X_test)
                # KNN 分类
                clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
                clf.fit(Z_tr, y_train)
                acc = accuracy_score(y_test, clf.predict(Z_te))
                dt = time.time() - t0
                results[name][d] = {"accuracy": float(acc), "time": float(dt)}
                print(f"  [{name:8s}] d={d:3d}  acc={acc:.4f}  time={dt:.1f}s")
            except Exception as e:
                results[name][d] = {"accuracy": float('nan'), "time": float('nan'), "error": str(e)}
                print(f"  [{name:8s}] d={d:3d}  FAILED: {e}")

    # 对 ISOMAP/t-SNE/UMAP 用较小子集 (训练5000,测试1000)
    print("  --- 非线性方法用 N_train=5000, N_test=1000 ---")
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X, y, train_size=5000, test_size=1000,
        random_state=SEED, stratify=y
    )
    nonlinear_factory = {
        "ISOMAP":  lambda d: Isomap(n_components=d, n_neighbors=15, n_jobs=-1),
        # t-SNE 没有transform,不能用train/test split,这里跳过或用替代方案
        "UMAP":    lambda d: umap.UMAP(n_components=d, n_neighbors=15, min_dist=0.1,
                                       random_state=SEED, n_jobs=-1),
    }
    for name, factory in nonlinear_factory.items():
        results[name] = {}
        for d in dims_to_test:
            try:
                t0 = time.time()
                reducer = factory(d)
                Z_tr = reducer.fit_transform(X_train2)
                Z_te = reducer.transform(X_test2)
                clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
                clf.fit(Z_tr, y_train2)
                acc = accuracy_score(y_test2, clf.predict(Z_te))
                dt = time.time() - t0
                results[name][d] = {"accuracy": float(acc), "time": float(dt)}
                print(f"  [{name:8s}] d={d:3d}  acc={acc:.4f}  time={dt:.1f}s")
            except Exception as e:
                results[name][d] = {"accuracy": float('nan'), "time": float('nan'), "error": str(e)}
                print(f"  [{name:8s}] d={d:3d}  FAILED: {e}")

    # t-SNE 没有 transform 方法,跳过分类实验,在结果中标注
    results["t-SNE"] = {d: {"accuracy": float('nan'), "time": float('nan'),
                            "note": "t-SNE 无 transform 方法,无法用于监督下游分类"} for d in dims_to_test}

    # 绘图
    fig, ax = plt.subplots(figsize=(11, 6))
    all_methods = ["PCA", "LDA", "KPCA", "ISOMAP", "UMAP", "t-SNE"]
    colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3","#937860"]
    markers = ["o","s","^","D","v","*"]
    for i, name in enumerate(all_methods):
        accs = [results.get(name, {}).get(d, {}).get("accuracy", float('nan')) for d in dims_to_test]
        valid = [(d, a) for d, a in zip(dims_to_test, accs) if not np.isnan(a)]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, marker=markers[i], color=colors[i], label=name,
                    linewidth=2, markersize=10)
    ax.set_xlabel("降维维度 d")
    ax.set_ylabel("KNN (k=5) 分类准确率")
    ax.set_title("降维后 KNN 下游分类性能 (PCA/LDA/KPCA: N_train=10000; ISOMAP/UMAP: N_train=5000)")
    ax.set_xticks(dims_to_test)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "expE_classification.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] {fig_path}")
    return results



# 9. Data Description Figures

def make_data_description_figures(X, y):
    print("\n" + "="*60)
    print("数据描述图表")
    print("="*60)

    # 1) 类别分布柱状图
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = np.bincount(y)
    bars = ax.bar(range(10), counts, color="#4C72B0")
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit class")
    ax.set_ylabel("Sample count")
    ax.set_title("MNIST 类别分布 (共 70,000 样本)")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                str(c), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "data_class_distribution.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [SAVE] data_class_distribution.png")

    # 2) 每类一个示例
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for c in range(10):
        idx = np.where(y == c)[0][0]
        axes[c].imshow(X[idx].reshape(28, 28), cmap="gray")
        axes[c].set_title(str(c), fontsize=11)
        axes[c].axis("off")
    plt.suptitle("MNIST 各类别示例图像 (28×28=784 维)", y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "data_samples.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [SAVE] data_samples.png")

    # 3) 像素方差热图
    pixel_var = X.var(axis=0).reshape(28, 28)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(pixel_var, cmap="viridis")
    ax.set_title("MNIST 像素方差热图 (显示数字笔迹的分布区域)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "data_pixel_variance.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [SAVE] data_pixel_variance.png")

    # 4) 像素稀疏性
    sparsity_per_sample = (X == 0).mean(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sparsity_per_sample, bins=50, color="#55A868", edgecolor="white")
    ax.set_xlabel("Zero-pixel ratio per sample")
    ax.set_ylabel("Frequency")
    ax.set_title(f"MNIST 像素稀疏性 (平均 {(X==0).mean()*100:.1f}% 像素为 0)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "data_sparsity.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [SAVE] data_sparsity.png")



# 10. Main

def main():
    X, y = load_mnist()

    # 数据描述图表 (用全集)
    make_data_description_figures(X, y)

    # 子采样
    X_vis, y_vis = subsample(X, y, N_VIS)
    print(f"\n[SUBSAMPLE] 可视化子集: {X_vis.shape}, 类别分布: {np.bincount(y_vis)}")

    # Experiment A: 2D 可视化
    vis_results, embeddings = experiment_visualization(X_vis, y_vis)

    # Experiment B: 定量指标
    metrics = experiment_metrics(X_vis, y_vis, embeddings)

    # Experiment C: 运行时间
    X_time, y_time = subsample(X, y, N_TIME)
    timings = experiment_runtime(X_time, y_time)

    # Experiment D: 参数敏感性
    X_param, y_param = subsample(X, y, N_PARAM)
    param_results = experiment_param_sensitivity(X_param, y_param)

    # Experiment E: 下游分类
    cls_results = experiment_classification(X, y)

    # 汇总保存
    summary = {
        "data": {
            "name": "MNIST",
            "n_samples_total": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": 10,
            "source": "http://yann.lecun.com/exdb/mnist/  (OpenML id=554)",
            "class_distribution": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        },
        "subsample_sizes": {
            "visualization": N_VIS,
            "runtime": N_TIME,
            "param_sensitivity": N_PARAM,
            "classification_train": N_CLS_TRAIN,
            "classification_test": N_CLS_TEST,
        },
        "expA_visualization": vis_results,
        "expB_metrics": metrics,
        "expC_runtime": {k: float(v) for k, v in timings.items()},
        "expD_param": param_results,
        "expE_classification": cls_results,
        "seed": SEED,
    }

    with open(os.path.join(RESULT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[SUMMARY] 保存到 {RESULT_DIR}/summary.json")
    print("\n所有实验完成!")


if __name__ == "__main__":
    main()