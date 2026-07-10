"""
基于PCA与t-SNE降维的手写数字识别性能对比研究
主实验程序
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import time
import os

# 设置结果保存路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_and_preprocess_data():
    """加载并预处理手写数字数据集"""
    print("=" * 60)
    print("1. 数据加载与预处理")
    print("=" * 60)
    
    # 加载sklearn自带的手写数字数据集
    digits = load_digits()
    X = digits.data  # 特征矩阵: (1797, 64)
    y = digits.target  # 标签: (1797,)
    
    print(f"数据集名称: sklearn手写数字数据集 (Digits Dataset)")
    print(f"样本数量: {X.shape[0]}")
    print(f"特征维度: {X.shape[1]} (8×8像素图像展开为64维向量)")
    print(f"类别数量: {len(np.unique(y))} (数字0-9)")
    print(f"各类别样本分布:")
    for i in range(10):
        print(f"  数字 {i}: {np.sum(y == i)} 个样本")
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n训练集样本数: {X_train.shape[0]}")
    print(f"测试集样本数: {X_test.shape[0]}")
    
    # 保存数据信息
    data_info = {
        'total_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)),
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0]
    }
    
    return X_train, X_test, y_train, y_test, data_info


def pca_reduction(X_train, X_test, n_components=2):
    """PCA降维"""
    print(f"\n执行PCA降维，目标维度: {n_components}")
    start_time = time.time()
    
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    elapsed = time.time() - start_time
    print(f"PCA降维完成，耗时: {elapsed:.4f}秒")
    print(f"前{n_components}个主成分的累计方差解释率: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return X_train_pca, X_test_pca, pca, elapsed


def tsne_reduction(X_train, X_test, y_train, n_components=2, perplexity=30):
    """t-SNE降维"""
    print(f"\n执行t-SNE降维，目标维度: {n_components}, 困惑度: {perplexity}")
    start_time = time.time()
    
    # t-SNE是无监督的，通常对整个数据集进行降维用于可视化
    # 为了分类任务，我们分别对训练集和测试集进行处理
    # 注意：t-SNE没有transform方法，需要使用完整数据拟合
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                random_state=42, init='pca', learning_rate='auto')
    
    # 合并训练集和测试集进行t-SNE（因为t-SNE不支持单独transform）
    X_combined = np.vstack([X_train, X_test])
    X_combined_tsne = tsne.fit_transform(X_combined)
    
    # 重新分割
    X_train_tsne = X_combined_tsne[:len(X_train)]
    X_test_tsne = X_combined_tsne[len(X_train):]
    
    elapsed = time.time() - start_time
    print(f"t-SNE降维完成，耗时: {elapsed:.4f}秒")
    
    return X_train_tsne, X_test_tsne, tsne, elapsed


def train_and_evaluate_classifier(X_train, y_train, X_test, y_test, method_name):
    """训练并评估分类器"""
    print(f"\n--- {method_name} 降维后的分类实验 ---")
    
    classifiers = {
        'SVM (RBF核)': SVC(kernel='rbf', random_state=42),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        '随机森林': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for clf_name, clf in classifiers.items():
        start_time = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 5折交叉验证
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
        
        results[clf_name] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_time': train_time
        }
        
        print(f"{clf_name}:")
        print(f"  测试集准确率: {accuracy:.4f}")
        print(f"  5折交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  训练时间: {train_time:.4f}秒")
    
    return results


def visualize_2d_projection(X, y, title, save_name):
    """二维降维结果可视化"""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', 
                         alpha=0.7, s=20, edgecolors='k', linewidth=0.5)
    plt.colorbar(scatter, ticks=range(10), label='数字类别')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('维度1', fontsize=12)
    plt.ylabel('维度2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {save_path}")


def plot_pca_variance_explained(pca, save_name):
    """绘制PCA方差解释率曲线"""
    plt.figure(figsize=(10, 6))
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
             'b-o', markersize=4, linewidth=2)
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% 方差')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95% 方差')
    
    plt.xlabel('主成分数量', fontsize=12)
    plt.ylabel('累计方差解释率', fontsize=12)
    plt.title('PCA累计方差解释率曲线', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(FIG_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {save_path}")


def plot_accuracy_comparison(baseline_results, pca_results, tsne_results, save_name):
    """绘制不同降维方法的分类准确率对比"""
    classifiers = list(baseline_results.keys())
    baseline_acc = [baseline_results[clf]['accuracy'] for clf in classifiers]
    pca_acc = [pca_results[clf]['accuracy'] for clf in classifiers]
    tsne_acc = [tsne_results[clf]['accuracy'] for clf in classifiers]
    
    x = np.arange(len(classifiers))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, baseline_acc, width, label='原始数据 (64维)', color='#2E86AB')
    plt.bar(x, pca_acc, width, label='PCA降维 (2维)', color='#A23B72')
    plt.bar(x + width, tsne_acc, width, label='t-SNE降维 (2维)', color='#F18F01')
    
    plt.xlabel('分类器', fontsize=12)
    plt.ylabel('测试集准确率', fontsize=12)
    plt.title('不同降维方法下的分类准确率对比', fontsize=14, fontweight='bold')
    plt.xticks(x, classifiers, fontsize=10)
    plt.ylim([0, 1.1])
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上标注数值
    for i, v in enumerate(baseline_acc):
        plt.text(i - width, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    for i, v in enumerate(pca_acc):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    for i, v in enumerate(tsne_acc):
        plt.text(i + width, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {save_path}")


def plot_time_comparison(baseline_results, pca_results, tsne_results, 
                         pca_time, tsne_time, save_name):
    """绘制计算时间对比"""
    methods = ['原始数据', 'PCA降维', 't-SNE降维']
    # 总时间 = 降维时间 + 分类训练时间（取SVM的时间）
    times = [
        baseline_results['SVM (RBF核)']['train_time'],
        pca_time + pca_results['SVM (RBF核)']['train_time'],
        tsne_time + tsne_results['SVM (RBF核)']['train_time']
    ]
    
    plt.figure(figsize=(8, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = plt.bar(methods, times, color=colors, width=0.5)
    
    plt.xlabel('方法', fontsize=12)
    plt.ylabel('总计算时间 (秒)', fontsize=12)
    plt.title('不同方法的计算时间对比 (SVM分类器)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{t:.3f}s', ha='center', fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {save_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("基于PCA与t-SNE降维的手写数字识别性能对比研究")
    print("=" * 60)
    
    # 1. 数据加载与预处理
    X_train, X_test, y_train, y_test, data_info = load_and_preprocess_data()
    
    # 2. 原始数据分类（基准）
    print("\n" + "=" * 60)
    print("2. 原始数据分类实验（基准）")
    print("=" * 60)
    baseline_results = train_and_evaluate_classifier(
        X_train, y_train, X_test, y_test, "原始数据(64维)"
    )
    
    # 3. PCA降维
    print("\n" + "=" * 60)
    print("3. PCA降维实验")
    print("=" * 60)
    
    # 3.1 二维PCA（用于可视化）
    X_train_pca_2d, X_test_pca_2d, pca_2d, pca_time_2d = pca_reduction(
        X_train, X_test, n_components=2
    )
    
    # 3.2 完整PCA用于方差分析
    pca_full = PCA(n_components=None, random_state=42)
    pca_full.fit(X_train)
    plot_pca_variance_explained(pca_full, 'pca_variance_explained.png')
    
    # 3.3 PCA降维后的分类
    pca_results = train_and_evaluate_classifier(
        X_train_pca_2d, y_train, X_test_pca_2d, y_test, "PCA(2维)"
    )
    
    # 3.4 PCA可视化
    visualize_2d_projection(X_train_pca_2d, y_train, 
                           'PCA二维降维结果（训练集）', 
                           'pca_2d_visualization.png')
    
    # 4. t-SNE降维
    print("\n" + "=" * 60)
    print("4. t-SNE降维实验")
    print("=" * 60)
    
    X_train_tsne, X_test_tsne, tsne_model, tsne_time = tsne_reduction(
        X_train, X_test, y_train, n_components=2, perplexity=30
    )
    
    tsne_results = train_and_evaluate_classifier(
        X_train_tsne, y_train, X_test_tsne, y_test, "t-SNE(2维)"
    )
    
    # t-SNE可视化
    visualize_2d_projection(X_train_tsne, y_train, 
                           't-SNE二维降维结果（训练集）', 
                           'tsne_2d_visualization.png')
    
    # 5. 结果对比与可视化
    print("\n" + "=" * 60)
    print("5. 结果对比与可视化")
    print("=" * 60)
    
    plot_accuracy_comparison(baseline_results, pca_results, tsne_results,
                            'accuracy_comparison.png')
    
    plot_time_comparison(baseline_results, pca_results, tsne_results,
                        pca_time_2d, tsne_time, 'time_comparison.png')
    
    # 6. 保存结果表格
    print("\n" + "=" * 60)
    print("6. 保存实验结果")
    print("=" * 60)
    
    # 整理结果表格
    all_results = []
    for clf_name in baseline_results.keys():
        all_results.append({
            '分类器': clf_name,
            '降维方法': '无(原始64维)',
            '测试准确率': baseline_results[clf_name]['accuracy'],
            '交叉验证均值': baseline_results[clf_name]['cv_mean'],
            '交叉验证标准差': baseline_results[clf_name]['cv_std'],
            '训练时间(秒)': baseline_results[clf_name]['train_time']
        })
        all_results.append({
            '分类器': clf_name,
            '降维方法': 'PCA(2维)',
            '测试准确率': pca_results[clf_name]['accuracy'],
            '交叉验证均值': pca_results[clf_name]['cv_mean'],
            '交叉验证标准差': pca_results[clf_name]['cv_std'],
            '训练时间(秒)': pca_results[clf_name]['train_time']
        })
        all_results.append({
            '分类器': clf_name,
            '降维方法': 't-SNE(2维)',
            '测试准确率': tsne_results[clf_name]['accuracy'],
            '交叉验证均值': tsne_results[clf_name]['cv_mean'],
            '交叉验证标准差': tsne_results[clf_name]['cv_std'],
            '训练时间(秒)': tsne_results[clf_name]['train_time']
        })
    
    df_results = pd.DataFrame(all_results)
    results_path = os.path.join(DATA_DIR, 'experiment_results.csv')
    df_results.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"实验结果已保存: {results_path}")
    
    # 保存降维时间统计
    time_stats = pd.DataFrame({
        '降维方法': ['PCA(2维)', 't-SNE(2维)'],
        '降维耗时(秒)': [pca_time_2d, tsne_time]
    })
    time_path = os.path.join(DATA_DIR, 'reduction_time.csv')
    time_stats.to_csv(time_path, index=False, encoding='utf-8-sig')
    print(f"降维时间统计已保存: {time_path}")
    
    # 保存数据信息
    data_info_df = pd.DataFrame([data_info])
    info_path = os.path.join(DATA_DIR, 'dataset_info.csv')
    data_info_df.to_csv(info_path, index=False, encoding='utf-8-sig')
    print(f"数据集信息已保存: {info_path}")
    
    print("\n" + "=" * 60)
    print("所有实验完成！")
    print("=" * 60)
    print(f"数据文件保存在: {DATA_DIR}")
    print(f"图表文件保存在: {FIG_DIR}")


if __name__ == '__main__':
    main()
