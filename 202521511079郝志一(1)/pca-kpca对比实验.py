"""
PCA 与 KPCA 降维方法对比研究 —— 完整实验
=========================================
一键运行：实验 → 全部 16 张图表

包含:
  Part A: PCA / KPCA 从零实现 (numpy)
  Part B: sklearn 交叉验证
  Part C: MNIST / Fashion-MNIST 基础实验 (10 张图)
  Part D: 补充分析图 (6 张图: 耗时/特征谱/γ敏感性/仪表盘)
"""

# ============================================================
# 导入与全局配置
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

import warnings
warnings.filterwarnings('ignore', message=".*does not have a glyph.*")
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.decomposition import KernelPCA as SklearnKPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import os

FIGURES_DIR = 'figures'

# ============================================================
# Part A: PCA 从零实现
# ============================================================
class PCA_from_scratch:
    """主成分分析 —— 从零实现（最大方差视角）"""

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None

    def fit(self, X):
        n, p = X.shape
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        Sigma = (1 / n) * (X_centered.T @ X_centered)
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = eigenvalues / np.sum(eigenvalues)
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ============================================================
# Part B: KPCA 从零实现
# ============================================================
class KPCA_from_scratch:
    """核主成分分析 —— 从零实现"""

    def __init__(self, n_components=2, kernel='rbf', gamma=None, degree=3, coef0=1):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alphas_ = None
        self.lambdas_ = None
        self.X_train_ = None

    def _compute_kernel(self, X, Y=None):
        if Y is None:
            Y = X
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
        if self.kernel == 'linear':
            return X @ Y.T
        elif self.kernel == 'poly':
            return (self.gamma * X @ Y.T + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
            Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
            sq_dist = X_norm + Y_norm - 2 * (X @ Y.T)
            return np.exp(-self.gamma * sq_dist)
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * X @ Y.T + self.coef0)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def fit(self, X):
        n = X.shape[0]
        self.X_train_ = X
        K = self._compute_kernel(X)
        one_n = np.ones((n, n)) / n
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        pos_idx = eigenvalues > 1e-10
        eigenvalues = eigenvalues[pos_idx]
        eigenvectors = eigenvectors[:, pos_idx]
        d = min(self.n_components, len(eigenvalues))
        self.lambdas_ = eigenvalues[:d]
        self.alphas_ = np.zeros((n, d))
        for k in range(d):
            self.alphas_[:, k] = eigenvectors[:, k] / np.sqrt(eigenvalues[k])
        total_var = np.sum(np.abs(eigenvalues))
        self.explained_variance_ratio_ = eigenvalues / total_var
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
        return self

    def transform(self, X):
        K_new = self._compute_kernel(self.X_train_, X)
        return K_new.T @ self.alphas_

    def fit_transform(self, X):
        self.fit(X)
        n = X.shape[0]
        K = self._compute_kernel(X)
        one_n = np.ones((n, n)) / n
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        return K_centered @ self.alphas_


# ============================================================
# Part C: 数据加载
# ============================================================
def load_mnist(n_samples=3000):
    print(f"[加载] MNIST 数据集（{n_samples} 样本）...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True,
                         as_frame=False, parser='auto')
    np.random.seed(42)
    idx = np.random.choice(len(X), size=n_samples, replace=False)
    X, y = X[idx], y[idx].astype(int)
    print(f"  形状: {X.shape}, 类别: {len(np.unique(y))}")
    return X, y


def load_fashion_mnist(n_samples=3000):
    print(f"[加载] Fashion-MNIST 数据集（{n_samples} 样本）...")
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True,
                         as_frame=False, parser='auto')
    np.random.seed(42)
    idx = np.random.choice(len(X), size=n_samples, replace=False)
    X, y = X[idx], y[idx].astype(int)
    print(f"  形状: {X.shape}, 类别: {len(np.unique(y))}")
    return X, y


# ============================================================
# Part D: 可视化函数（基础实验 + 补充分析）
# ============================================================

# ---- D1: 基础实验图 ----

def plot_2d_scatter(X_2d, y, title, save_path=None, n_classes=10):
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.tab10
    for c in range(n_classes):
        mask = y == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[cmap(c)], label=str(c), s=8, alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Component 1'); ax.set_ylabel('Component 2')
    ax.legend(loc='upper right', ncol=2, markerscale=2, fontsize=8)
    ax.set_aspect('equal')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  已保存: {save_path}")
    plt.close()


def plot_variance_ratio(pca_model, kpca_models, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cum_ratio = pca_model.cumulative_variance_ratio_[:50]
    ax1.plot(range(1, len(cum_ratio) + 1), cum_ratio * 100, 'b-', linewidth=2)
    ax1.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='80%')
    ax1.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90%')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative Explained Variance (%)')
    ax1.set_title('PCA: Cumulative Explained Variance')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    colors = {'rbf': 'red', 'poly': 'green', 'sigmoid': 'purple'}
    for name, kpca in kpca_models.items():
        cr = kpca.cumulative_variance_ratio_[:min(50, len(kpca.cumulative_variance_ratio_))]
        ax2.plot(range(1, len(cr) + 1), cr * 100,
                 color=colors.get(name, 'gray'), linewidth=2, label=f'KPCA ({name})')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Ratio of Eigenvalues (%)')
    ax2.set_title('KPCA: Eigenvalue Ratio (by kernel)')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  已保存: {save_path}")
    plt.close()


def plot_full_comparison(X_pca, X_kpca_rbf, X_kpca_poly, X_kpca_sigmoid,
                         y, dataset_name, save_dir, n_classes=10):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    cmap = plt.cm.tab10
    data = [
        (X_pca, 'PCA (Linear)'),
        (X_kpca_rbf, 'KPCA (RBF Kernel)'),
        (X_kpca_poly, 'KPCA (Polynomial Kernel, d=3)'),
        (X_kpca_sigmoid, 'KPCA (Sigmoid Kernel)'),
    ]
    for ax, (X_2d, title) in zip(axes.flat, data):
        for c in range(n_classes):
            mask = y == c
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[cmap(c)], label=str(c), s=6, alpha=0.7)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Component 1'); ax.set_ylabel('Component 2')
        ax.legend(loc='upper right', ncol=2, markerscale=2, fontsize=7)
        ax.set_aspect('equal')
    fig.suptitle(f'{dataset_name}: PCA vs KPCA 2D Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    sp = f'{save_dir}/{dataset_name}_comparison.png'
    plt.savefig(sp, dpi=200, bbox_inches='tight')
    print(f"  已保存: {sp}")
    plt.close()


def plot_classification_comparison(results_df, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = results_df['Method'].values
    accuracies = results_df['Accuracy'].values
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    bars = ax.bar(methods, accuracies, color=colors[:len(methods)],
                  edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
    ax.set_title('KNN Classification Accuracy After Dimensionality Reduction',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(accuracies) * 1.2)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  已保存: {save_path}")
    plt.close()


# ---- D2: 补充分析图 ----

def fig_timing_comparison(X_scaled, save_path):
    print("[补充图] 计算耗时对比...")
    sample_sizes = [500, 1000, 1500, 2000, 2500, 3000]
    methods = {
        'PCA': ('#3498db', lambda X: PCA_from_scratch(n_components=50).fit_transform(X)),
        'KPCA-RBF': ('#e74c3c', lambda X: KPCA_from_scratch(n_components=50, kernel='rbf').fit_transform(X)),
        'KPCA-Poly': ('#2ecc71', lambda X: KPCA_from_scratch(n_components=50, kernel='poly', degree=3).fit_transform(X)),
        'KPCA-Sigmoid': ('#9b59b6', lambda X: KPCA_from_scratch(n_components=50, kernel='sigmoid').fit_transform(X)),
    }
    times = {name: [] for name in methods}
    for n in sample_sizes:
        X_sub = X_scaled[:n]
        for name, (_, func) in methods.items():
            t0 = time.time()
            func(X_sub)
            times[name].append(time.time() - t0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    markers = ['o', 's', '^', 'D']
    for (name, (color, _)), marker in zip(methods.items(), markers):
        ax1.plot(sample_sizes, times[name], color=color, marker=marker,
                 linewidth=2.5, markersize=8, label=name, alpha=0.9)
    ax1.set_xlabel('Number of Samples', fontsize=12)
    ax1.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax1.set_title('Computation Time vs Sample Size\n(Dimensionality Reduction to 50D)',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, framealpha=0.9); ax1.grid(True, alpha=0.3)

    final_times = [times[name][-1] for name in methods]
    bar_colors = [methods[name][0] for name in methods]
    bars = ax2.bar(range(len(methods)), final_times, color=bar_colors,
                   edgecolor='black', linewidth=0.8, width=0.55)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(list(methods.keys()), fontsize=11)
    ax2.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax2.set_title('Time at n=3000 Samples (50 Components)', fontsize=13, fontweight='bold')
    for bar, t in zip(bars, final_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_times)*0.02,
                 f'{t:.1f}s', ha='center', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  已保存: {save_path}")
    plt.close()


def fig_eigenvalue_spectrum(X_scaled, save_path):
    print("[补充图] 特征值谱衰减...")
    n = X_scaled.shape[0]
    pca = PCA_from_scratch(n_components=min(n, X_scaled.shape[1])).fit(X_scaled)
    kernels = {
        'Linear': ('#3498db', 'linear'),
        'RBF': ('#e74c3c', 'rbf'),
        'Poly (d=3)': ('#2ecc71', 'poly'),
        'Sigmoid': ('#9b59b6', 'sigmoid'),
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    top_k = 50
    x = np.arange(1, top_k + 1)
    pca_ev = pca.explained_variance_[:top_k]
    ax1.semilogy(x, pca_ev, 'b-', linewidth=2, marker='o', markersize=4,
                 label='PCA (covariance eigenvalues)', alpha=0.8)
    for label, (color, kernel) in kernels.items():
        kpca = KPCA_from_scratch(n_components=top_k, kernel=kernel).fit(X_scaled)
        ev = kpca.lambdas_[:top_k]
        ev = ev / ev[0] * pca_ev[0]
        ax1.semilogy(x, ev, color=color, linewidth=2, marker='s', markersize=3,
                     label=f'KPCA {label}', alpha=0.7)
    ax1.set_xlabel('Component Index', fontsize=12)
    ax1.set_ylabel('Eigenvalue (log scale)', fontsize=12)
    ax1.set_title('Eigenvalue Spectrum (Normalized, Top 50)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, framealpha=0.9); ax1.grid(True, alpha=0.3, which='both')

    ax2.plot(x, np.cumsum(pca.explained_variance_ratio_[:top_k]) * 100,
             'b-', linewidth=2.5, label='PCA', alpha=0.9)
    for label, (color, kernel) in kernels.items():
        kpca = KPCA_from_scratch(n_components=top_k, kernel=kernel).fit(X_scaled)
        cum_ratio = np.cumsum(kpca.explained_variance_ratio_[:top_k]) * 100
        ax2.plot(x, cum_ratio, color=color, linewidth=2.5, linestyle='--',
                 label=f'KPCA {label}', alpha=0.8)
    ax2.axhline(y=80, color='gray', linestyle=':', alpha=0.6, linewidth=1)
    ax2.text(top_k + 0.5, 80, '80%', fontsize=9, va='center', color='gray')
    ax2.axhline(y=90, color='gray', linestyle=':', alpha=0.6, linewidth=1)
    ax2.text(top_k + 0.5, 90, '90%', fontsize=9, va='center', color='gray')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Ratio (%)', fontsize=12)
    ax2.set_title('Cumulative Explained Ratio (Top 50)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, framealpha=0.9); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  已保存: {save_path}")
    plt.close()


def fig_gamma_sensitivity(X_scaled, y, ds_name, save_path):
    print(f"[补充图] γ 参数敏感性 ({ds_name})...")
    default_gamma = 1.0 / X_scaled.shape[1]
    gammas = np.logspace(-5, -1, 20)
    accuracies, variances = [], []
    for gamma in gammas:
        kpca = KPCA_from_scratch(n_components=2, kernel='rbf', gamma=gamma)
        X_2d = kpca.fit_transform(X_scaled)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_2d, y, test_size=0.3, random_state=42, stratify=y)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_tr, y_tr)
        accuracies.append(accuracy_score(y_te, knn.predict(X_te)) * 100)
        vr = np.sum(kpca.lambdas_[:2]) / np.sum(kpca.lambdas_) * 100 if len(kpca.lambdas_) > 2 else 100
        variances.append(vr)

    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    ax1.semilogx(gammas, accuracies, color='#3498db', marker='o', markersize=7,
                 linewidth=2.5, label='KNN Classification Accuracy (5-NN)')
    ax1.set_xlabel('Gamma', fontsize=13)
    ax1.set_ylabel('Classification Accuracy (%)', fontsize=13, color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1.set_ylim(0, max(accuracies) * 1.15)

    best_idx = np.argmax(accuracies)
    best_gamma, best_acc = gammas[best_idx], accuracies[best_idx]
    ax1.axvline(x=best_gamma, color='green', linestyle='--', alpha=0.4)
    ax1.annotate(f'Best gamma={best_gamma:.1e}\nAcc={best_acc:.1f}%',
                 xy=(best_gamma, best_acc),
                 xytext=(best_gamma * 30, best_acc - 8),
                 arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                 fontsize=11, fontweight='bold', color='green',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax1.axvline(x=default_gamma, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
    ax1.text(default_gamma * 1.3, max(accuracies) * 0.15,
             f'Default gamma = 1/p = {default_gamma:.1e}',
             fontsize=9, color='gray', bbox=dict(facecolor='white', alpha=0.7))

    ax2 = ax1.twinx()
    ax2.semilogx(gammas, variances, color='#e74c3c', marker='s', markersize=5,
                 linewidth=2, linestyle='--', alpha=0.7, label='Top-2 Eigenvalue Ratio (%)')
    ax2.set_ylabel('Top-2 Eigenvalue Ratio (%)', fontsize=13, color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10, framealpha=0.9)
    ax1.axvspan(min(gammas), gammas[1], alpha=0.08, color='red')
    ax1.text(gammas[2], max(accuracies)*0.08, 'gamma too small\n(degenerates to PCA)',
             fontsize=8, color='red', alpha=0.7)
    ax1.axvspan(gammas[-2], max(gammas), alpha=0.08, color='red')
    ax1.text(gammas[-5], max(accuracies)*0.08, 'gamma too large\n(Gram -> Identity)',
             fontsize=8, color='red', alpha=0.7)
    ax1.set_title(f'KPCA-RBF: Gamma Parameter Sensitivity ({ds_name}, 3000 samples)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  已保存: {save_path}")
    plt.close()


def fig_comprehensive_dashboard(X_scaled, y, dataset_name, save_path):
    print(f"[补充图] 综合仪表盘 ({dataset_name})...")
    n_classes = len(np.unique(y))
    pca_2d = PCA_from_scratch(n_components=2).fit_transform(X_scaled)
    kpca_rbf = KPCA_from_scratch(n_components=2, kernel='rbf').fit_transform(X_scaled)
    kpca_poly = KPCA_from_scratch(n_components=2, kernel='poly', degree=3).fit_transform(X_scaled)
    kpca_sigmoid = KPCA_from_scratch(n_components=2, kernel='sigmoid').fit_transform(X_scaled)

    methods_2d = {'PCA': pca_2d, 'KPCA RBF': kpca_rbf,
                  'KPCA Poly': kpca_poly, 'KPCA Sigmoid': kpca_sigmoid}
    accuracies = {}
    for name, X_2d in methods_2d.items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_2d, y, test_size=0.3, random_state=42, stratify=y)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_tr, y_tr)
        accuracies[name] = accuracy_score(y_te, knn.predict(X_te)) * 100

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
    cmap = plt.cm.tab10
    for idx, (name, X_2d) in enumerate(methods_2d.items()):
        ax = fig.add_subplot(gs[0, idx])
        for c in range(n_classes):
            mask = y == c
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[cmap(c)], label=str(c), s=5, alpha=0.7)
        ax.set_title(f'{name}\nAcc: {accuracies[name]:.1f}%', fontsize=11, fontweight='bold')
        ax.set_xlabel('C1'); ax.set_ylabel('C2')
        ax.set_aspect('equal')
        ax.legend(loc='upper right', ncol=2, markerscale=2.5, fontsize=6, framealpha=0.8)

    ax_bar = fig.add_subplot(gs[1, :2])
    colors_bar = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    bars = ax_bar.bar(list(accuracies.keys()), list(accuracies.values()),
                      color=colors_bar, edgecolor='black', linewidth=0.8, width=0.5)
    for bar, acc in zip(bars, accuracies.values()):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f'{acc:.1f}%', ha='center', fontsize=12, fontweight='bold')
    ax_bar.set_ylabel('KNN Accuracy (%)', fontsize=12)
    ax_bar.set_title('2D KNN Classification Accuracy', fontsize=13, fontweight='bold')
    ax_bar.set_ylim(0, max(accuracies.values()) * 1.2)
    ax_bar.grid(axis='y', alpha=0.3)

    ax_var = fig.add_subplot(gs[1, 2:])
    pca_full = PCA_from_scratch(n_components=50).fit(X_scaled)
    kpca_rbf_full = KPCA_from_scratch(n_components=50, kernel='rbf').fit(X_scaled)
    comps = [1, 2, 5, 10, 20, 50]
    x_pos = np.arange(len(comps))
    width = 0.35
    pca_cum = [pca_full.cumulative_variance_ratio_[c-1]*100 for c in comps]
    kpca_cum = [kpca_rbf_full.cumulative_variance_ratio_[c-1]*100 for c in comps]
    bars1 = ax_var.bar(x_pos - width/2, pca_cum, width, color='#3498db',
                       label='PCA', edgecolor='black', linewidth=0.5)
    bars2 = ax_var.bar(x_pos + width/2, kpca_cum, width, color='#e74c3c',
                       label='KPCA (RBF)', edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars1, pca_cum):
        ax_var.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold')
    for bar, val in zip(bars2, kpca_cum):
        ax_var.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold')
    ax_var.set_xticks(x_pos)
    ax_var.set_xticklabels([f'{c}' for c in comps])
    ax_var.set_xlabel('Number of Components', fontsize=12)
    ax_var.set_ylabel('Cumulative Explained Ratio (%)', fontsize=12)
    ax_var.set_title('Cumulative Explained Ratio by Components', fontsize=13, fontweight='bold')
    ax_var.legend(fontsize=11); ax_var.grid(axis='y', alpha=0.3)

    ax_time = fig.add_subplot(gs[2, :2])
    methods_time = {
        'PCA': ('#3498db', lambda X: PCA_from_scratch(n_components=50).fit_transform(X)),
        'KPCA-RBF': ('#e74c3c', lambda X: KPCA_from_scratch(n_components=50, kernel='rbf').fit_transform(X)),
        'KPCA-Poly': ('#2ecc71', lambda X: KPCA_from_scratch(n_components=50, kernel='poly', degree=3).fit_transform(X)),
        'KPCA-Sigmoid': ('#9b59b6', lambda X: KPCA_from_scratch(n_components=50, kernel='sigmoid').fit_transform(X)),
    }
    times = {}
    for name, (color, func) in methods_time.items():
        t0 = time.time(); func(X_scaled); times[name] = time.time() - t0
    bars_t = ax_time.bar(times.keys(), times.values(),
                         color=[m[0] for m in methods_time.values()],
                         edgecolor='black', linewidth=0.8, width=0.5)
    for bar, t in zip(bars_t, times.values()):
        ax_time.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times.values())*0.02,
                     f'{t:.2f}s', ha='center', fontsize=11, fontweight='bold')
    ax_time.set_ylabel('Time (seconds)', fontsize=12)
    ax_time.set_title('Computation Time (50 Components)', fontsize=13, fontweight='bold')
    ax_time.grid(axis='y', alpha=0.3)

    ax_summary = fig.add_subplot(gs[2, 2:]); ax_summary.axis('off')
    best_method = max(accuracies, key=accuracies.get)
    summary_text = (
        f"Dataset: {dataset_name}\n"
        f"Samples: {X_scaled.shape[0]}, Original Dim: {X_scaled.shape[1]}\n"
        + '-' * 40 + '\n'
        f"Best method: {best_method}\n"
        f"  Accuracy: {accuracies[best_method]:.1f}%\n"
        f"PCA Accuracy: {accuracies['PCA']:.1f}%\n"
        f"Improvement: {accuracies[best_method] - accuracies['PCA']:.1f} pp\n"
        + '-' * 40 + '\n'
        f"PCA 50D time: {times['PCA']:.2f}s\n"
        f"KPCA-RBF 50D time: {times['KPCA-RBF']:.2f}s\n"
        + '-' * 40 + '\n'
        f"PCA top-50 var ratio: {pca_full.cumulative_variance_ratio_[49]*100:.1f}%\n"
        f"KPCA-RBF top-50 ratio: {kpca_rbf_full.cumulative_variance_ratio_[49]*100:.1f}%"
    )
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#ccc', alpha=0.9))
    fig.suptitle(f'{dataset_name} - PCA vs KPCA Comprehensive Dashboard',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  已保存: {save_path}")
    plt.close()


# ============================================================
# Part E: 分类评估 + 验证
# ============================================================
def evaluate_classification(X_2d, y, method_name):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_2d, y, test_size=0.3, random_state=42, stratify=y)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, knn.predict(X_te)) * 100
    print(f"  {method_name}: KNN 分类精度 = {acc:.2f}%")
    return acc


def verify_implementation():
    print("\n" + "=" * 60)
    print("验证自实现代码与 sklearn 的一致性")
    print("=" * 60)
    np.random.seed(0)
    X_test = np.random.randn(200, 50)

    pca_mine = PCA_from_scratch(n_components=5).fit(X_test)
    X_mine = pca_mine.transform(X_test)
    pca_sk = SklearnPCA(n_components=5).fit(X_test)
    X_sk = pca_sk.transform(X_test)
    diff_pca = np.mean(np.abs(np.abs(np.corrcoef(X_mine.T, X_sk.T)[:5, 5:]) - 1))
    print(f"\n[PCA] 自实现 vs sklearn 相关系数偏差: {diff_pca:.6f} (越接近0越好)")

    for kernel in ['rbf', 'poly', 'sigmoid']:
        kpca_mine = KPCA_from_scratch(n_components=5, kernel=kernel)
        X_mine = kpca_mine.fit_transform(X_test)
        kpca_sk = SklearnKPCA(n_components=5, kernel=kernel, fit_inverse_transform=False)
        X_sk = kpca_sk.fit_transform(X_test)
        corr = np.abs(np.corrcoef(X_mine.T, X_sk.T)[:5, 5:])
        diag_corr = np.abs(np.diag(np.abs(corr)))
        print(f"[KPCA-{kernel}] 对应成分相关性: {np.round(diag_corr, 4)}, "
              f"均值: {np.mean(diag_corr):.4f} (越接近1越好)")
    print("\n验证完成！\n")


# ============================================================
# Part F: 主流程
# ============================================================
def main():
    print("=" * 60)
    print("PCA 与 KPCA 降维方法对比 —— 完整实验 + 论文生成")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ---------- Step 1: 验证 ----------
    verify_implementation()

    # ---------- Step 2: 基础实验 ----------
    print("\n" + "=" * 60)
    print("Step 1/3: 基础实验（10 张基础图）")
    print("=" * 60)

    datasets = {
        'MNIST': load_mnist(n_samples=3000),
        'Fashion-MNIST': load_fashion_mnist(n_samples=3000),
    }
    all_results = []
    scalers = {}

    for ds_name, (X, y) in datasets.items():
        print(f"\n{'='*60}\n数据集: {ds_name}\n{'='*60}")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scalers[ds_name] = (X_scaled, y)
        n_classes = len(np.unique(y))

        # PCA
        print("\n[PCA]"); t0 = time.time()
        pca = PCA_from_scratch(n_components=50)
        X_pca = pca.fit_transform(X_scaled)
        print(f"  耗时: {time.time()-t0:.2f}s, 前10维累计方差: {pca.cumulative_variance_ratio_[9]*100:.1f}%")

        # KPCA variants
        kpca_models = {}
        for kernel, kwargs in [('rbf', {}), ('poly', {'degree': 3}), ('sigmoid', {})]:
            print(f"\n[KPCA-{kernel}]"); t0 = time.time()
            kpca = KPCA_from_scratch(n_components=50, kernel=kernel, **kwargs)
            X_kpca = kpca.fit_transform(X_scaled)
            kpca_models[kernel] = (kpca, X_kpca)
            print(f"  耗时: {time.time()-t0:.2f}s")

        # 2D projections
        pca_2d = PCA_from_scratch(n_components=2).fit_transform(X_scaled)
        kpca_rbf_2d = KPCA_from_scratch(n_components=2, kernel='rbf').fit_transform(X_scaled)
        kpca_poly_2d = KPCA_from_scratch(n_components=2, kernel='poly', degree=3).fit_transform(X_scaled)
        kpca_sigmoid_2d = KPCA_from_scratch(n_components=2, kernel='sigmoid').fit_transform(X_scaled)

        # Plots
        plot_2d_scatter(pca_2d, y, f'{ds_name} - PCA',
                        f'{FIGURES_DIR}/{ds_name}_pca.png', n_classes)
        plot_2d_scatter(kpca_rbf_2d, y, f'{ds_name} - KPCA (RBF)',
                        f'{FIGURES_DIR}/{ds_name}_kpca_rbf.png', n_classes)
        plot_full_comparison(pca_2d, kpca_rbf_2d, kpca_poly_2d, kpca_sigmoid_2d,
                             y, ds_name, FIGURES_DIR, n_classes)
        kpca_var_models = {k: m[0] for k, m in kpca_models.items()}
        plot_variance_ratio(pca, kpca_var_models,
                            f'{FIGURES_DIR}/{ds_name}_variance.png')

        # Classification
        print("\n[分类评估]")
        for name, X_2d in [('PCA', pca_2d), ('KPCA-RBF', kpca_rbf_2d),
                            ('KPCA-Poly', kpca_poly_2d), ('KPCA-Sigmoid', kpca_sigmoid_2d)]:
            acc = evaluate_classification(X_2d, y, name)
            all_results.append({'Dataset': ds_name, 'Method': name, 'Accuracy': acc})

    # Classification summary
    for ds_name in ['MNIST', 'Fashion-MNIST']:
        ds_results = [r for r in all_results if r['Dataset'] == ds_name]
        if ds_results:
            plot_classification_comparison(
                pd.DataFrame(ds_results),
                f'{FIGURES_DIR}/{ds_name}_classification.png')

    # ---------- Step 3: 补充分析图 ----------
    print("\n" + "=" * 60)
    print("Step 2/3: 补充分析图（6 张）")
    print("=" * 60)

    X_mnist_s, y_mnist = scalers['MNIST']
    X_fashion_s, y_fashion = scalers['Fashion-MNIST']

    fig_timing_comparison(X_mnist_s, f'{FIGURES_DIR}/supp_timing_comparison.png')
    fig_eigenvalue_spectrum(X_mnist_s, f'{FIGURES_DIR}/supp_eigenvalue_spectrum.png')
    fig_gamma_sensitivity(X_mnist_s, y_mnist, 'MNIST',
                          f'{FIGURES_DIR}/supp_gamma_sensitivity_MNIST.png')
    fig_gamma_sensitivity(X_fashion_s, y_fashion, 'Fashion-MNIST',
                          f'{FIGURES_DIR}/supp_gamma_sensitivity_FashionMNIST.png')
    fig_comprehensive_dashboard(X_mnist_s, y_mnist, 'MNIST',
                                f'{FIGURES_DIR}/supp_dashboard_MNIST.png')
    fig_comprehensive_dashboard(X_fashion_s, y_fashion, 'Fashion-MNIST',
                                f'{FIGURES_DIR}/supp_dashboard_FashionMNIST.png')

    print("\n" + "=" * 60)
    print("全部完成！")
    print(f"  图表: {FIGURES_DIR}/ 目录 ({len(os.listdir(FIGURES_DIR))} 张)")
    print("=" * 60)


if __name__ == '__main__':
    main()
