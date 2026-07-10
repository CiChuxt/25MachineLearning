

import os
import time
import json
import warnings
warnings.filterwarnings("ignore")

os.makedirs("figs", exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
import umap


DATA_DIR = r"d:\桌面\机器学习报告\罗思凡202521511107\罗思凡20252151107 代码+数据+图结果\数据"


def load_digits_csv():
    """从本地 CSV 读取 Digits 数据集，返回 (X, y)"""
    path = os.path.join(DATA_DIR, "digits_手写数字数据集.csv")
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ("target", "target_name")]
    X = df[feature_cols].values.astype(np.float64)
    y = df["target"].values.astype(int)
    return X, y


def load_wine_csv():
    """从本地 CSV 读取 UCI Wine 数据集，返回 (X, y)"""
    path = os.path.join(DATA_DIR, "wine_UCI葡萄酒数据集.csv")
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ("target", "target_name")]
    X = df[feature_cols].values.astype(np.float64)
    y = df["target"].values.astype(int)
    return X, y


def load_iris_csv():
    """从本地 CSV 读取 UCI Iris 数据集，返回 (X, y)"""
    path = os.path.join(DATA_DIR, "iris_UCI鸢尾花数据集.csv")
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ("target", "target_name")]
    X = df[feature_cols].values.astype(np.float64)
    y = df["target"].values.astype(int)
    return X, y

# ---------- 中文字体设置（若系统无中文字体，则回退英文标题） ----------
plt.rcParams["axes.unicode_minus"] = False
CN_FONT = None
for name in ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei"]:
    try:
        font_manager.findfont(name, fallback_to_default=False)
        CN_FONT = name
        break
    except Exception:
        continue
if CN_FONT:
    plt.rcParams["font.sans-serif"] = [CN_FONT]

METHOD_NAMES = ["PCA", "KPCA(RBF)", "ISOMAP", "t-SNE", "UMAP"]

# 是否在运行时弹出图像窗口（PyCharm/Jupyter 中会显示在 SciView 或独立窗口）
SHOW_FIGS = True



def run_five_methods(X, n_neighbors_isomap=10, perplexity=30, umap_neighbors=15):
    """对输入特征矩阵 X 依次运行 PCA / KPCA / ISOMAP / t-SNE / UMAP，
    返回 {方法名: (二维嵌入, 计算耗时秒)}"""
    results = {}

    t0 = time.time()
    emb = PCA(n_components=2, random_state=42).fit_transform(X)
    results["PCA"] = (emb, time.time() - t0)

    t0 = time.time()
    emb = KernelPCA(n_components=2, kernel="rbf", random_state=42).fit_transform(X)
    results["KPCA(RBF)"] = (emb, time.time() - t0)

    t0 = time.time()
    emb = Isomap(n_neighbors=n_neighbors_isomap, n_components=2).fit_transform(X)
    results["ISOMAP"] = (emb, time.time() - t0)

    t0 = time.time()
    emb = TSNE(n_components=2, perplexity=perplexity, learning_rate=200,
               init="pca", random_state=42, max_iter=1000).fit_transform(X)
    results["t-SNE"] = (emb, time.time() - t0)

    t0 = time.time()
    emb = umap.UMAP(n_neighbors=umap_neighbors, min_dist=0.1,
                     n_components=2, random_state=42).fit_transform(X)
    results["UMAP"] = (emb, time.time() - t0)

    return results


def quantitative_eval(results, y):
    """计算每种方法的 KNN 交叉验证准确率与轮廓系数（真实计算，非引用文献数值）"""
    metrics = {}
    for name, (emb, t) in results.items():
        knn = KNeighborsClassifier(n_neighbors=10)
        acc = cross_val_score(knn, emb, y, cv=5).mean()
        sil = silhouette_score(emb, y)
        metrics[name] = {
            "knn_acc": round(float(acc) * 100, 2),
            "silhouette": round(float(sil), 3),
            "time_sec": round(float(t), 3),
        }
    return metrics


def plot_five_panel(results, y, title, save_path, class_names=None):
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    cmap = plt.get_cmap("tab10")
    for ax, name in zip(axes, METHOD_NAMES):
        emb, t = results[name]
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap=cmap, s=8, alpha=0.75)
        ax.set_title(f"{name}\n({t:.2f}s)", fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=15)
    handles, labels = sc.legend_elements()
    if class_names is not None:
        labels = [class_names[i] for i in range(len(class_names))]
    fig.legend(handles, labels, loc="lower center", ncol=min(10, len(labels)),
               bbox_to_anchor=(0.5, -0.05), fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print("saved", save_path)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


def fig3_perplexity(X, y, save_path):
    perps = [5, 15, 30, 50, 100]
    fig, axes = plt.subplots(1, len(perps), figsize=(24, 5))
    cmap = plt.get_cmap("tab10")
    sil_scores = {}
    for ax, p in zip(axes, perps):
        t0 = time.time()
        emb = TSNE(n_components=2, perplexity=p, learning_rate=200,
                   init="pca", random_state=42, max_iter=1000).fit_transform(X)
        dt = time.time() - t0
        sil = silhouette_score(emb, y)
        sil_scores[p] = round(float(sil), 3)
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap=cmap, s=8, alpha=0.75)
        ax.set_title(f"Perplexity={p}\nSilhouette={sil:.3f}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("不同困惑度对 t-SNE 结果的影响（Digits 数据集）", fontsize=15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print("saved", save_path)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)
    return sil_scores


def fig4_barnes_hut(X_full, y_full, save_path):
    """标准 t-SNE（method='exact'）与 Barnes-Hut t-SNE（method='barnes_hut'）
    在不同样本量下的真实运行耗时对比"""
    sizes = [300, 600, 900, 1200]
    exact_times, bh_times = [], []
    rng = np.random.RandomState(42)
    for n in sizes:
        idx = rng.choice(len(X_full), size=min(n, len(X_full)), replace=False)
        Xs = X_full[idx]

        t0 = time.time()
        TSNE(n_components=2, method="exact", perplexity=30,
             init="pca", random_state=42, max_iter=500).fit_transform(Xs)
        exact_times.append(time.time() - t0)

        t0 = time.time()
        TSNE(n_components=2, method="barnes_hut", perplexity=30,
             init="pca", random_state=42, max_iter=500).fit_transform(Xs)
        bh_times.append(time.time() - t0)
        print(f"n={n} exact={exact_times[-1]:.2f}s bh={bh_times[-1]:.2f}s")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, exact_times, marker="o", label="标准 t-SNE (exact)")
    ax.plot(sizes, bh_times, marker="s", label="Barnes-Hut t-SNE")
    ax.set_xlabel("样本量 n")
    ax.set_ylabel("运行时间 (秒)")
    ax.set_title("标准 t-SNE 与 Barnes-Hut t-SNE 计算效率对比\n(Digits 数据集真实实测)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print("saved", save_path)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)
    return sizes, exact_times, bh_times


def main():
    all_metrics = {}

    # ---------------- 数据集一：Digits（真实数据，从本地 CSV 读取） ----------------
    Xd, yd = load_digits_csv()
    Xd = StandardScaler().fit_transform(Xd)
    Xd_pca50 = PCA(n_components=min(50, Xd.shape[1]), random_state=42).fit_transform(Xd)

    res_digits = run_five_methods(Xd_pca50, n_neighbors_isomap=10, perplexity=30)
    all_metrics["digits"] = quantitative_eval(res_digits, yd)
    plot_five_panel(res_digits, yd,
                     "图1 五种降维方法在手写数字数据集（Digits，真实数据）上的二维可视化对比",
                     "figs/fig1_digits_five_methods.png",
                     class_names=[str(i) for i in range(10)])

    # ---------------- 数据集二：UCI Wine（真实数据，从本地 CSV 读取） ----------------
    Xw, yw = load_wine_csv()
    Xw = StandardScaler().fit_transform(Xw)
    res_wine = run_five_methods(Xw, n_neighbors_isomap=10, perplexity=30, umap_neighbors=15)
    all_metrics["wine"] = quantitative_eval(res_wine, yw)
    plot_five_panel(res_wine, yw,
                     "图2 五种降维方法在 UCI Wine 数据集（真实数据）上的二维可视化对比",
                     "figs/fig2_wine_five_methods.png",
                     class_names=["Class1", "Class2", "Class3"])

    # ---------------- 困惑度实验（图3，基于 Digits 真实数据） ----------------
    sil_scores = fig3_perplexity(Xd_pca50, yd, "figs/fig3_perplexity_effect.png")
    all_metrics["perplexity_silhouette"] = sil_scores

    # ---------------- Barnes-Hut 加速实验（图4，真实计时） ----------------
    sizes, exact_t, bh_t = fig4_barnes_hut(Xd_pca50, yd, "figs/fig4_barnes_hut_speed.png")
    all_metrics["barnes_hut"] = {
        "sizes": sizes, "exact_times": exact_t, "bh_times": bh_t
    }

    with open("figs/real_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
