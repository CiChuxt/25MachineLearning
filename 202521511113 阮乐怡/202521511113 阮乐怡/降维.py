#!/usr/bin/env python
# coding: utf-8

# In[7]:


#isomap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances

# ===================== 全局配置：支持中文显示 =====================
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号'-'显示为方块的问题
l

# ---------------------- 2. 待遍历的候选低维维度：1~10全部遍历 ----------------------
dim_candidates = list(range(2, 11))  # 1,2,3,4,5,6,7,8,9,10
result_dict = {}  # 存储每个维度的全部指标

# ---------------------- 工具函数 ----------------------
def compute_stress(high_dist, low_dist):
    """PPT Isomap/MDS应力函数Stress（Frobenius范数形式）"""
    diff = high_dist - low_dist
    numerator = np.sum(diff ** 2)
    denominator = np.sum(high_dist ** 2)
    stress = np.sqrt(numerator / denominator)
    return stress

# TOPSIS综合评价函数
def topsis_evaluation(matrix, benefit_idx, cost_idx, weights=None):
    """
    matrix: m行n列，m个方案(维度)，n个评价指标
    benefit_idx: 效益指标下标列表（越大越好）
    cost_idx: 成本指标下标列表（越小越好）
    weights: 各指标权重，None为等权重
    return: 各方案贴近度得分
    """
    m, n = matrix.shape
    # 1. 正向化处理
    norm_matrix = matrix.copy()
    # 成本指标正向化：max - x
    for idx in cost_idx:
        col_max = np.max(norm_matrix[:, idx])
        norm_matrix[:, idx] = col_max - norm_matrix[:, idx]
    # 2. 向量标准化
    sqrt_sum = np.sqrt(np.sum(norm_matrix ** 2, axis=0))
    norm_matrix = norm_matrix / sqrt_sum
    # 3. 加权
    if weights is None:
        weights = np.ones(n) / n
    weight_matrix = norm_matrix * weights
    # 4. 正负理想解
    pos_ideal = np.max(weight_matrix, axis=0)
    neg_ideal = np.min(weight_matrix, axis=0)
    # 5. 距离计算
    d_pos = np.sqrt(np.sum((weight_matrix - pos_ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weight_matrix - neg_ideal) ** 2, axis=1))
    # 6. 贴近度（综合得分，越高越好）
    score = d_neg / (d_pos + d_neg)
    return score

# 原始高维距离矩阵（全局一次计算，复用）
dist_high = euclidean_distances(X_raw)

# ---------------------- 3. 循环遍历每个d'，计算全部指标 ----------------------
for d in dim_candidates:
    print(f"===== 正在计算低维维度 d' = {d} =====")
    # Isomap降维
    iso = Isomap(n_components=d, n_neighbors=10)
    X_low = iso.fit_transform(X_raw)
    
    # 1. 聚类指标：轮廓系数、ARI
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    y_pred = kmeans.fit_predict(X_low)
    sil = silhouette_score(X_low, y_pred)
    ari = adjusted_rand_score(y_real, y_pred)
    
    # 2. Stress应力函数
    dist_low = euclidean_distances(X_low)
    stress_val = compute_stress(dist_high, dist_low)
    
    # Isomap无重构误差，舍弃
    result_dict[d] = {
        "silhouette": sil,
        "ARI": ari,
        "Stress": stress_val
    }
    print(f"轮廓系数(Silhouette): {sil:.4f}")
    print(f"调整兰德指数(ARI): {ari:.4f}")
    print(f"应力函数Stress: {stress_val:.4f} (越小越好)\n")

# ---------------------- 4. TOPSIS综合评价计算 ----------------------
# 构建评价矩阵：行=维度1~10，列=[轮廓系数, ARI, Stress]
d_list = list(result_dict.keys())
sil_list = [result_dict[d]["silhouette"] for d in d_list]
ari_list = [result_dict[d]["ARI"] for d in d_list]
stress_list = [result_dict[d]["Stress"] for d in d_list]

eval_matrix = np.column_stack([sil_list, ari_list, stress_list])
# 指标下标：0轮廓系数(效益)，1ARI(效益)，2Stress(成本)
benefit_cols = [0, 1]
cost_cols = [2]
# 计算TOPSIS综合贴近度
topsis_scores = topsis_evaluation(eval_matrix, benefit_cols, cost_cols)

# 绑定维度与综合得分
dim_score_dict = dict(zip(d_list, topsis_scores))
# 取综合得分最高为最优维度
best_d = max(dim_score_dict.items(), key=lambda x: x[1])

# ---------------------- 5. 打印完整汇总表 ----------------------
print("==================== 维度遍历汇总（含TOPSIS综合得分） ====================")
for idx, d in enumerate(d_list):
    metrics = result_dict[d]
    score = dim_score_dict[d]
    print(f"d'={d:2d} | 轮廓系数={metrics['silhouette']:.4f} | ARI={metrics['ARI']:.4f} | Stress={metrics['Stress']:.4f} | TOPSIS综合得分={score:.4f}")
print(f"\n【TOPSIS综合评价最优低维维度 d' = {best_d[0]}】综合贴近度得分={best_d[1]:.4f}")

# ====================== 多指标可视化 + TOPSIS得分曲线 ======================
plt.figure(figsize=(14, 10))
# 1. 轮廓系数
plt.subplot(2,2,1)
plt.plot(d_list, sil_list, marker='o', color='#2E86AB', linewidth=2)
plt.xlabel("Isomap 低维维度 $d'$")
plt.ylabel("轮廓系数")
plt.title("不同维度下轮廓系数变化趋势")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 2. ARI
plt.subplot(2,2,2)
plt.plot(d_list, ari_list, marker='s', color='#A23B72', linewidth=2)
plt.xlabel("Isomap 低维维度 $d'$")
plt.ylabel("调整兰德指数 ARI")
plt.title("不同维度下ARI变化趋势")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 3. Stress应力函数
plt.subplot(2,2,3)
plt.plot(d_list, stress_list, marker='^', color='#F18F01', linewidth=2)
plt.xlabel("Isomap 低维维度 $d'$")
plt.ylabel("应力函数 Stress")
plt.title("不同维度下应力函数变化（越小距离保留越好）")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 4. TOPSIS综合得分（核心新增）
plt.subplot(2,2,4)
plt.plot(d_list, topsis_scores, marker='D', color='#28a745', linewidth=2, markersize=7)
# 标注最优维度
plt.scatter(best_d[0], best_d[1], c='red', s=120, zorder=5, label=f'最优维度d\'={best_d[0]}')
plt.xlabel("Isomap 低维维度 $d'$")
plt.ylabel("TOPSIS综合贴近度得分")
plt.title("TOPSIS多指标综合评价得分（越高综合效果越好）")
plt.xticks(d_list)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------- 6. 最优维度降维可视化 ----------------------
# 先用TOPSIS选出的最优d'做Isomap
iso_best = Isomap(n_components=best_d[0], n_neighbors=10)
X_best = iso_best.fit_transform(X_raw)
# 二次降维到2维绘图
iso_2d = Isomap(n_components=2, n_neighbors=10)
X_vis = iso_2d.fit_transform(X_best)

plt.figure(figsize=(8,6))
sc = plt.scatter(X_vis[:,0], X_vis[:,1], c=y_real, cmap="tab10", s=15)
plt.legend(*sc.legend_elements(), title="数字类别标签")
plt.title(f"TOPSIS最优维度$d'$={best_d[0]} 二维样本分布可视化")
plt.xlabel("Isomap 第一维特征")
plt.ylabel("Isomap 第二维特征")
plt.show()


# In[8]:


#pca
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances

# ===================== 全局配置：支持中文显示 =====================
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号'-'显示为方块的问题

# ---------------------- 1. 加载内置高维数据集(8*8=64维手写数字) ----------------------
data = load_digits()
X_raw = data.data  # 原始高维数据 (1797, 64)
y_real = data.target  # 真实标签，用于计算ARI
n_cluster = len(np.unique(y_real))  # 聚类簇数=10

# ---------------------- 2. 待遍历的候选低维维度：2~10 ----------------------
dim_candidates = list(range(2, 11))
result_dict = {}  # 存储每个维度的全部指标

# ---------------------- 工具函数 ----------------------
def compute_stress(high_dist, low_dist):
    """应力函数Stress（Frobenius范数形式，衡量高低维距离保真度）"""
    diff = high_dist - low_dist
    numerator = np.sum(diff ** 2)
    denominator = np.sum(high_dist ** 2)
    stress = np.sqrt(numerator / denominator)
    return stress

# TOPSIS综合评价函数
def topsis_evaluation(matrix, benefit_idx, cost_idx, weights=None):
    """
    matrix: m行n列，m个方案(维度)，n个评价指标
    benefit_idx: 效益指标下标列表（越大越好）
    cost_idx: 成本指标下标列表（越小越好）
    weights: 各指标权重，None为等权重
    return: 各方案贴近度得分
    """
    m, n = matrix.shape
    # 1. 正向化处理
    norm_matrix = matrix.copy()
    # 成本指标正向化：max - x
    for idx in cost_idx:
        col_max = np.max(norm_matrix[:, idx])
        norm_matrix[:, idx] = col_max - norm_matrix[:, idx]
    # 2. 向量标准化
    sqrt_sum = np.sqrt(np.sum(norm_matrix ** 2, axis=0))
    norm_matrix = norm_matrix / sqrt_sum
    # 3. 加权
    if weights is None:
        weights = np.ones(n) / n
    weight_matrix = norm_matrix * weights
    # 4. 正负理想解
    pos_ideal = np.max(weight_matrix, axis=0)
    neg_ideal = np.min(weight_matrix, axis=0)
    # 5. 距离计算
    d_pos = np.sqrt(np.sum((weight_matrix - pos_ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weight_matrix - neg_ideal) ** 2))
    # 6. 贴近度（综合得分，越高越好）
    score = d_neg / (d_pos + d_neg)
    return score

# 原始高维距离矩阵（全局一次计算，复用）
dist_high = euclidean_distances(X_raw)

# ---------------------- 3. 循环遍历每个d'，计算全部指标 ----------------------
for d in dim_candidates:
    print(f"===== 正在计算低维维度 d' = {d} =====")
    # 替换为PCA降维
    pca = PCA(n_components=d, random_state=42)
    X_low = pca.fit_transform(X_raw)
    
    # 1. 聚类指标：轮廓系数、ARI
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    y_pred = kmeans.fit_predict(X_low)
    sil = silhouette_score(X_low, y_pred)
    ari = adjusted_rand_score(y_real, y_pred)
    
    # 2. Stress应力函数
    dist_low = euclidean_distances(X_low)
    stress_val = compute_stress(dist_high, dist_low)

    # 新增：累计方差解释率
    explained_var = sum(pca.explained_variance_ratio_)
    
    result_dict[d] = {
        "silhouette": sil,
        "ARI": ari,
        "Stress": stress_val,
        "VarRatio": explained_var
    }
    print(f"轮廓系数(Silhouette): {sil:.4f}")
    print(f"调整兰德指数(ARI): {ari:.4f}")
    print(f"应力函数Stress: {stress_val:.4f} (越小越好)")
    print(f"累计方差解释率: {explained_var:.4f}\n")

# ---------------------- 4. TOPSIS综合评价计算 ----------------------
# 构建评价矩阵：[轮廓系数, ARI, 累计方差解释率, Stress]
# 0:sil(效益)  1:ARI(效益)  2:VarRatio(效益)  3:Stress(成本)
d_list = list(result_dict.keys())
sil_list = [result_dict[d]["silhouette"] for d in d_list]
ari_list = [result_dict[d]["ARI"] for d in d_list]
var_list = [result_dict[d]["VarRatio"] for d in d_list]
stress_list = [result_dict[d]["Stress"] for d in d_list]

eval_matrix = np.column_stack([sil_list, ari_list, var_list, stress_list])
# 效益指标：0,1,2；成本指标：3
benefit_cols = [0, 1, 2]
cost_cols = [3]
# 计算TOPSIS综合贴近度
topsis_scores = topsis_evaluation(eval_matrix, benefit_cols, cost_cols)

# 绑定维度与综合得分
dim_score_dict = dict(zip(d_list, topsis_scores))
# 取综合得分最高为最优维度
best_d = max(dim_score_dict.items(), key=lambda x: x[1])

# ---------------------- 5. 打印完整汇总表 ----------------------
print("==================== 维度遍历汇总（含TOPSIS综合得分） ====================")
for idx, d in enumerate(d_list):
    metrics = result_dict[d]
    score = dim_score_dict[d]
    print(f"d'={d:2d} | 轮廓系数={metrics['silhouette']:.4f} | ARI={metrics['ARI']:.4f} | 方差解释率={metrics['VarRatio']:.4f} | Stress={metrics['Stress']:.4f} | TOPSIS综合得分={score:.4f}")
print(f"\n【TOPSIS综合评价最优低维维度 d' = {best_d[0]}】综合贴近度得分={best_d[1]:.4f}")

# ====================== 多指标可视化 + TOPSIS得分曲线 ======================
plt.figure(figsize=(16, 10))
# 1. 轮廓系数
plt.subplot(2,3,1)
plt.plot(d_list, sil_list, marker='o', color='#2E86AB', linewidth=2)
plt.xlabel("PCA 低维维度 $d'$")
plt.ylabel("轮廓系数")
plt.title("不同维度下轮廓系数变化趋势")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 2. ARI
plt.subplot(2,3,2)
plt.plot(d_list, ari_list, marker='s', color='#A23B72', linewidth=2)
plt.xlabel("PCA 低维维度 $d'$")
plt.ylabel("调整兰德指数 ARI")
plt.title("不同维度下ARI变化趋势")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 新增3：累计方差解释率
plt.subplot(2,3,3)
plt.plot(d_list, var_list, marker='*', color='#009944', linewidth=2)
plt.xlabel("PCA 低维维度 $d'$")
plt.ylabel("累计方差解释率")
plt.title("不同维度下方差解释率变化（越大保留信息越多）")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 4. Stress应力函数
plt.subplot(2,3,4)
plt.plot(d_list, stress_list, marker='^', color='#F18F01', linewidth=2)
plt.xlabel("PCA 低维维度 $d'$")
plt.ylabel("应力函数 Stress")
plt.title("不同维度下应力函数变化（越小距离保留越好）")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 5. TOPSIS综合得分
plt.subplot(2,3,5)
plt.plot(d_list, topsis_scores, marker='D', color='#28a745', linewidth=2, markersize=7)
# 标注最优维度
plt.scatter(best_d[0], best_d[1], c='red', s=120, zorder=5, label=f'最优维度d\'={best_d[0]}')
plt.xlabel("PCA 低维维度 $d'$")
plt.ylabel("TOPSIS综合贴近度得分")
plt.title("TOPSIS多指标综合评价得分（越高综合效果越好）")
plt.xticks(d_list)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------- 6. 最优维度降维可视化 ----------------------
# 先用TOPSIS选出的最优d'做PCA
pca_best = PCA(n_components=best_d[0], random_state=42)
X_best = pca_best.fit_transform(X_raw)
# 二次PCA降到2维绘图
pca_2d = PCA(n_components=2, random_state=42)
X_vis = pca_2d.fit_transform(X_best)

plt.figure(figsize=(8,6))
sc = plt.scatter(X_vis[:,0], X_vis[:,1], c=y_real, cmap="tab10", s=15)
plt.legend(*sc.legend_elements(), title="数字类别标签")
plt.title(f"TOPSIS最优维度$d'$={best_d[0]} PCA二维样本分布可视化")
plt.xlabel("PCA 第一主成分")
plt.ylabel("PCA 第二主成分")
plt.show()


# In[11]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances

# ===================== 全局配置：支持中文显示 =====================
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号'-'显示为方块的问题

# ---------------------- 1. 加载内置高维数据集(8*8=64维手写数字) ----------------------
data = load_digits()
X_raw = data.data  # 原始高维数据 (1797, 64)
y_real = data.target  # 真实标签，用于计算ARI
n_cluster = len(np.unique(y_real))  # 聚类簇数=10

# ---------------------- 2. 待遍历的候选低维维度：2~10 ----------------------
dim_candidates = list(range(2, 11))
result_dict = {}  # 存储每个维度的全部指标

# ---------------------- 工具函数 ----------------------
def compute_stress(high_dist, low_dist):
    """应力函数Stress（Frobenius范数形式，衡量高低维距离保真度）"""
    diff = high_dist - low_dist
    numerator = np.sum(diff ** 2)
    denominator = np.sum(high_dist ** 2)
    stress = np.sqrt(numerator / denominator)
    return stress

# TOPSIS综合评价函数
def topsis_evaluation(matrix, benefit_idx, cost_idx, weights=None):
    """
    matrix: m行n列，m个方案(维度)，n个评价指标
    benefit_idx: 效益指标下标列表（越大越好）
    cost_idx: 成本指标下标列表（越小越好）
    weights: 各指标权重，None为等权重
    return: 各方案贴近度得分
    """
    m, n = matrix.shape
    # 1. 正向化处理
    norm_matrix = matrix.copy()
    # 成本指标正向化：max - x
    for idx in cost_idx:
        col_max = np.max(norm_matrix[:, idx])
        norm_matrix[:, idx] = col_max - norm_matrix[:, idx]
    # 2. 向量标准化
    sqrt_sum = np.sqrt(np.sum(norm_matrix ** 2, axis=0))
    norm_matrix = norm_matrix / sqrt_sum
    # 3. 加权
    if weights is None:
        weights = np.ones(n) / n
    weight_matrix = norm_matrix * weights
    # 4. 正负理想解
    pos_ideal = np.max(weight_matrix, axis=0)
    neg_ideal = np.min(weight_matrix, axis=0)
    # 5. 距离计算
    d_pos = np.sqrt(np.sum((weight_matrix - pos_ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weight_matrix - neg_ideal) ** 2))
    # 6. 贴近度（综合得分，越高越好）
    score = d_neg / (d_pos + d_neg)
    return score

# 原始高维距离矩阵（全局一次计算，复用）
dist_high = euclidean_distances(X_raw)

# ---------------------- 3. 循环遍历每个d'，计算全部指标 ----------------------
for d in dim_candidates:
    print(f"===== 正在计算低维维度 d' = {d} =====")
    # KPCA：开启fit_inverse_transform才能读取eigenvalues_
    kpca = KernelPCA(n_components=d, kernel="rbf", random_state=42, fit_inverse_transform=True)
    X_low = kpca.fit_transform(X_raw)
    
    # 1. 聚类指标：轮廓系数、ARI
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    y_pred = kmeans.fit_predict(X_low)
    sil = silhouette_score(X_low, y_pred)
    ari = adjusted_rand_score(y_real, y_pred)
    
    # 2. Stress应力函数
    dist_low = euclidean_distances(X_low)
    stress_val = compute_stress(dist_high, dist_low)

    # 修复属性：lambdas_ → eigenvalues_
    eig_vals = kpca.eigenvalues_
    total_eig = np.sum(eig_vals)
    explained_var = np.sum(eig_vals[:d]) / total_eig
    
    result_dict[d] = {
        "silhouette": sil,
        "ARI": ari,
        "Stress": stress_val,
        "VarRatio": explained_var
    }
    print(f"轮廓系数(Silhouette): {sil:.4f}")
    print(f"调整兰德指数(ARI): {ari:.4f}")
    print(f"应力函数Stress: {stress_val:.4f} (越小越好)")
    print(f"核特征值累计占比(等效方差解释率): {explained_var:.4f}\n")

# ---------------------- 4. TOPSIS综合评价计算 ----------------------
# 评价矩阵：[轮廓系数, ARI, 核特征值累计占比, Stress]
# 0:sil(效益)  1:ARI(效益)  2:VarRatio(效益)  3:Stress(成本)
d_list = list(result_dict.keys())
sil_list = [result_dict[d]["silhouette"] for d in d_list]
ari_list = [result_dict[d]["ARI"] for d in d_list]
var_list = [result_dict[d]["VarRatio"] for d in d_list]
stress_list = [result_dict[d]["Stress"] for d in d_list]

eval_matrix = np.column_stack([sil_list, ari_list, var_list, stress_list])
benefit_cols = [0, 1, 2]
cost_cols = [3]
topsis_scores = topsis_evaluation(eval_matrix, benefit_cols, cost_cols)

# 绑定维度与综合得分
dim_score_dict = dict(zip(d_list, topsis_scores))
# 取综合得分最高为最优维度
best_d = max(dim_score_dict.items(), key=lambda x: x[1])

# ---------------------- 5. 打印完整汇总表 ----------------------
print("==================== 维度遍历汇总（含TOPSIS综合得分） ====================")
for idx, d in enumerate(d_list):
    metrics = result_dict[d]
    score = dim_score_dict[d]
    print(f"d'={d:2d} | 轮廓系数={metrics['silhouette']:.4f} | ARI={metrics['ARI']:.4f} | 核特征值占比={metrics['VarRatio']:.4f} | Stress={metrics['Stress']:.4f} | TOPSIS综合得分={score:.4f}")
print(f"\n【TOPSIS综合评价最优低维维度 d' = {best_d[0]}】综合贴近度得分={best_d[1]:.4f}")

# ====================== 多指标可视化 + TOPSIS得分曲线 ======================
plt.figure(figsize=(16, 10))
# 1. 轮廓系数
plt.subplot(2,3,1)
plt.plot(d_list, sil_list, marker='o', color='#2E86AB', linewidth=2)
plt.xlabel("KPCA 低维维度 $d'$")
plt.ylabel("轮廓系数")
plt.title("不同维度下轮廓系数变化趋势")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 2. ARI
plt.subplot(2,3,2)
plt.plot(d_list, ari_list, marker='s', color='#A23B72', linewidth=2)
plt.xlabel("KPCA 低维维度 $d'$")
plt.ylabel("调整兰德指数 ARI")
plt.title("不同维度下ARI变化趋势")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 3. 核特征值累计占比（等效方差解释率）
plt.subplot(2,3,3)
plt.plot(d_list, var_list, marker='*', color='#009944', linewidth=2)
plt.xlabel("KPCA 低维维度 $d'$")
plt.ylabel("核特征值累计占比")
plt.title("不同维度下核特征值占比变化（越大保留信息越多）")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 4. Stress应力函数
plt.subplot(2,3,4)
plt.plot(d_list, stress_list, marker='^', color='#F18F01', linewidth=2)
plt.xlabel("KPCA 低维维度 $d'$")
plt.ylabel("应力函数 Stress")
plt.title("不同维度下应力函数变化（越小距离保留越好）")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 5. TOPSIS综合得分
plt.subplot(2,3,5)
plt.plot(d_list, topsis_scores, marker='D', color='#28a745', linewidth=2, markersize=7)
# 标注最优维度
plt.scatter(best_d[0], best_d[1], c='red', s=120, zorder=5, label=f'最优维度d\'={best_d[0]}')
plt.xlabel("KPCA 低维维度 $d'$")
plt.ylabel("TOPSIS综合贴近度得分")
plt.title("TOPSIS多指标综合评价得分（越高综合效果越好）")
plt.xticks(d_list)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------- 6. 最优维度降维可视化 ----------------------
# 先用TOPSIS选出的最优d'做KPCA
kpca_best = KernelPCA(n_components=best_d[0], kernel="rbf", random_state=42, fit_inverse_transform=True)
X_best = kpca_best.fit_transform(X_raw)
# 二次KPCA降到2维绘图
kpca_2d = KernelPCA(n_components=2, kernel="rbf", random_state=42, fit_inverse_transform=True)
X_vis = kpca_2d.fit_transform(X_best)

plt.figure(figsize=(8,6))
sc = plt.scatter(X_vis[:,0], X_vis[:,1], c=y_real, cmap="tab10", s=15)
plt.legend(*sc.legend_elements(), title="数字类别标签")
plt.title(f"TOPSIS最优维度$d'$={best_d[0]} KPCA二维样本分布可视化")
plt.xlabel("KPCA 第一核主成分")
plt.ylabel("KPCA 第二核主成分")
plt.show()


# In[13]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances

# ===================== 全局配置：支持中文显示 =====================
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号'-'显示为方块的问题

# ---------------------- 1. 加载内置高维数据集(8*8=64维手写数字) ----------------------
data = load_digits()
X_raw = data.data  # 原始高维数据 (1797, 64)
y_real = data.target  # 真实标签，用于计算ARI
n_cluster = len(np.unique(y_real))  # 聚类簇数=10

# ---------------------- 2. 待遍历的候选低维维度：2~10 ----------------------
dim_candidates = list(range(2, 4))
result_dict = {}  # 存储每个维度的全部指标

# ---------------------- 工具函数 ----------------------
def compute_stress(high_dist, low_dist):
    """应力函数Stress（Frobenius范数形式，衡量高低维距离保真度）"""
    diff = high_dist - low_dist
    numerator = np.sum(diff ** 2)
    denominator = np.sum(high_dist ** 2)
    stress = np.sqrt(numerator / denominator)
    return stress

# TOPSIS综合评价函数
def topsis_evaluation(matrix, benefit_idx, cost_idx, weights=None):
    """
    matrix: m行n列，m个方案(维度)，n个评价指标
    benefit_idx: 效益指标下标列表（越大越好）
    cost_idx: 成本指标下标列表（越小越好）
    weights: 各指标权重，None为等权重
    return: 各方案贴近度得分
    """
    m, n = matrix.shape
    # 1. 正向化处理
    norm_matrix = matrix.copy()
    # 成本指标正向化：max - x
    for idx in cost_idx:
        col_max = np.max(norm_matrix[:, idx])
        norm_matrix[:, idx] = col_max - norm_matrix[:, idx]
    # 2. 向量标准化
    sqrt_sum = np.sqrt(np.sum(norm_matrix ** 2, axis=0))
    norm_matrix = norm_matrix / sqrt_sum
    # 3. 加权
    if weights is None:
        weights = np.ones(n) / n
    weight_matrix = norm_matrix * weights
    # 4. 正负理想解
    pos_ideal = np.max(weight_matrix, axis=0)
    neg_ideal = np.min(weight_matrix, axis=0)
    # 5. 距离计算
    d_pos = np.sqrt(np.sum((weight_matrix - pos_ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weight_matrix - neg_ideal) ** 2))
    # 6. 贴近度（综合得分，越高越好）
    score = d_neg / (d_pos + d_neg)
    return score

# 原始高维距离矩阵（全局一次计算，复用）
dist_high = euclidean_distances(X_raw)

# ---------------------- 3. 循环遍历每个d'，计算全部指标 ----------------------
for d in dim_candidates:
    print(f"===== 正在计算低维维度 d' = {d} =====")
    # t-SNE降维模型
    tsne = TSNE(n_components=d, random_state=42, perplexity=30)
    X_low = tsne.fit_transform(X_raw)
    
    # 1. 聚类指标：轮廓系数、ARI
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    y_pred = kmeans.fit_predict(X_low)
    sil = silhouette_score(X_low, y_pred)
    ari = adjusted_rand_score(y_real, y_pred)
    
    # 2. Stress应力函数
    dist_low = euclidean_distances(X_low)
    stress_val = compute_stress(dist_high, dist_low)

    # TSNE无方差、特征值相关指标，仅保存三项指标
    result_dict[d] = {
        "silhouette": sil,
        "ARI": ari,
        "Stress": stress_val
    }
    print(f"轮廓系数(Silhouette): {sil:.4f}")
    print(f"调整兰德指数(ARI): {ari:.4f}")
    print(f"应力函数Stress: {stress_val:.4f} (越小越好)\n")

# ---------------------- 4. TOPSIS综合评价计算 ----------------------
# 评价矩阵：[轮廓系数, ARI, Stress]
# 0:sil(效益)  1:ARI(效益)  2:Stress(成本)
d_list = list(result_dict.keys())
sil_list = [result_dict[d]["silhouette"] for d in d_list]
ari_list = [result_dict[d]["ARI"] for d in d_list]
stress_list = [result_dict[d]["Stress"] for d in d_list]

eval_matrix = np.column_stack([sil_list, ari_list, stress_list])
benefit_cols = [0, 1]
cost_cols = [2]
topsis_scores = topsis_evaluation(eval_matrix, benefit_cols, cost_cols)

# 绑定维度与综合得分
dim_score_dict = dict(zip(d_list, topsis_scores))
# 取综合得分最高为最优维度
best_d = max(dim_score_dict.items(), key=lambda x: x[1])

# ---------------------- 5. 打印完整汇总表 ----------------------
print("==================== 维度遍历汇总（含TOPSIS综合得分） ====================")
for idx, d in enumerate(d_list):
    metrics = result_dict[d]
    score = dim_score_dict[d]
    print(f"d'={d:2d} | 轮廓系数={metrics['silhouette']:.4f} | ARI={metrics['ARI']:.4f} | Stress={metrics['Stress']:.4f} | TOPSIS综合得分={score:.4f}")
print(f"\n【TOPSIS综合评价最优低维维度 d' = {best_d[0]}】综合贴近度得分={best_d[1]:.4f}")

# ====================== 多指标可视化 2×2布局 ======================
plt.figure(figsize=(14, 10))
# 1. 轮廓系数
plt.subplot(2,2,1)
plt.plot(d_list, sil_list, marker='o', color='#2E86AB', linewidth=2)
plt.xlabel("t-SNE 低维维度 $d'$")
plt.ylabel("轮廓系数")
plt.title("不同维度下轮廓系数变化趋势")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 2. ARI
plt.subplot(2,2,2)
plt.plot(d_list, ari_list, marker='s', color='#A23B72', linewidth=2)
plt.xlabel("t-SNE 低维维度 $d'$")
plt.ylabel("调整兰德指数 ARI")
plt.title("不同维度下ARI变化趋势")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 3. Stress应力函数
plt.subplot(2,2,3)
plt.plot(d_list, stress_list, marker='^', color='#F18F01', linewidth=2)
plt.xlabel("t-SNE 低维维度 $d'$")
plt.ylabel("应力函数 Stress")
plt.title("不同维度下应力函数变化（越小距离保留越好）")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 4. TOPSIS综合得分
plt.subplot(2,2,4)
plt.plot(d_list, topsis_scores, marker='D', color='#28a745', linewidth=2, markersize=7)
# 标注最优维度
plt.scatter(best_d[0], best_d[1], c='red', s=120, zorder=5, label=f'最优维度d\'={best_d[0]}')
plt.xlabel("t-SNE 低维维度 $d'$")
plt.ylabel("TOPSIS综合贴近度得分")
plt.title("TOPSIS多指标综合评价得分（越高综合效果越好）")
plt.xticks(d_list)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------- 最优维度降维可视化 ----------------------
# 先用TOPSIS选出的最优d'做t-SNE
tsne_best = TSNE(n_components=best_d[0], random_state=42, perplexity=30)
X_best = tsne_best.fit_transform(X_raw)
# 二次t-SNE降到2维绘图
tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
X_vis = tsne_2d.fit_transform(X_best)

plt.figure(figsize=(8,6))
sc = plt.scatter(X_vis[:,0], X_vis[:,1], c=y_real, cmap="tab10", s=15)
plt.legend(*sc.legend_elements(), title="数字类别标签")
plt.title(f"TOPSIS最优维度$d'$={best_d[0]} t-SNE二维样本分布可视化")
plt.xlabel("t-SNE 第一嵌入维度")
plt.ylabel("t-SNE 第二嵌入维度")
plt.show()


# In[2]:


# LLE (Locally Linear Embedding)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

# ===================== 全局配置：支持中文显示 =====================
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号'-'显示为方块的问题

# ---------------------- 1. 加载内置高维数据集(8*8=64维手写数字) ----------------------
data = load_digits()
X_raw = data.data  # 原始高维数据 (1797, 64)
y_real = data.target  # 真实标签，用于计算ARI
n_cluster = len(np.unique(y_real))  # 聚类簇数=10

# 数据标准化（LLE对尺度敏感，建议标准化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ---------------------- 2. 待遍历的候选低维维度：2~10 ----------------------
dim_candidates = list(range(2, 11))  # 2,3,4,5,6,7,8,9,10
result_dict = {}  # 存储每个维度的全部指标

# ---------------------- 工具函数 ----------------------
def compute_stress(high_dist, low_dist):
    """应力函数Stress（Frobenius范数形式）"""
    diff = high_dist - low_dist
    numerator = np.sum(diff ** 2)
    denominator = np.sum(high_dist ** 2)
    stress = np.sqrt(numerator / denominator)
    return stress

# TOPSIS综合评价函数
def topsis_evaluation(matrix, benefit_idx, cost_idx, weights=None):
    """
    matrix: m行n列，m个方案(维度)，n个评价指标
    benefit_idx: 效益指标下标列表（越大越好）
    cost_idx: 成本指标下标列表（越小越好）
    weights: 各指标权重，None为等权重
    return: 各方案贴近度得分
    """
    m, n = matrix.shape
    # 1. 正向化处理
    norm_matrix = matrix.copy()
    # 成本指标正向化：max - x
    for idx in cost_idx:
        col_max = np.max(norm_matrix[:, idx])
        norm_matrix[:, idx] = col_max - norm_matrix[:, idx]
    # 2. 向量标准化
    sqrt_sum = np.sqrt(np.sum(norm_matrix ** 2, axis=0))
    # 避免除以0
    sqrt_sum[sqrt_sum == 0] = 1
    norm_matrix = norm_matrix / sqrt_sum
    # 3. 加权
    if weights is None:
        weights = np.ones(n) / n
    weight_matrix = norm_matrix * weights
    # 4. 正负理想解
    pos_ideal = np.max(weight_matrix, axis=0)
    neg_ideal = np.min(weight_matrix, axis=0)
    # 5. 距离计算
    d_pos = np.sqrt(np.sum((weight_matrix - pos_ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weight_matrix - neg_ideal) ** 2, axis=1))
    # 6. 贴近度（综合得分，越高越好）
    score = d_neg / (d_pos + d_neg + 1e-10)
    return score

# 原始高维距离矩阵（全局一次计算，复用）
dist_high = euclidean_distances(X_scaled)

# ---------------------- 3. 循环遍历每个d'，计算全部指标 ----------------------
print("正在使用LLE进行降维...")
print("LLE参数: n_neighbors=30, method='standard'")
print("-" * 60)

for d in dim_candidates:
    print(f"===== 正在计算低维维度 d' = {d} =====")
    
    # LLE降维
    # 注意：n_neighbors需要大于d，否则会出错
    n_neighbors = min(30, max(d + 1, 10))  # 确保邻居数大于维度
    lle = LocallyLinearEmbedding(
        n_components=d,
        n_neighbors=n_neighbors,
        method='standard',  # 'standard', 'hessian', 'modified', 'ltsa'
        random_state=42
    )
    X_low = lle.fit_transform(X_scaled)
    
    # 1. 聚类指标：轮廓系数、ARI
    kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X_low)
    sil = silhouette_score(X_low, y_pred)
    ari = adjusted_rand_score(y_real, y_pred)
    
    # 2. Stress应力函数
    dist_low = euclidean_distances(X_low)
    stress_val = compute_stress(dist_high, dist_low)
    
    # 3. 重构误差（LLE特有）
    reconstruction_error = lle.reconstruction_error_
    
    result_dict[d] = {
        "silhouette": sil,
        "ARI": ari,
        "Stress": stress_val,
        "reconstruction_error": reconstruction_error,
        "n_neighbors": n_neighbors
    }
    print(f"轮廓系数(Silhouette): {sil:.4f}")
    print(f"调整兰德指数(ARI): {ari:.4f}")
    print(f"应力函数Stress: {stress_val:.4f} (越小越好)")
    print(f"重构误差: {reconstruction_error:.6f} (越小越好)")
    print(f"使用的邻居数: {n_neighbors}\n")

# ---------------------- 4. TOPSIS综合评价计算 ----------------------
# 构建评价矩阵：行=维度2~10，列=[轮廓系数, ARI, Stress]
# 注意：这里只使用聚类指标和应力指标
d_list = list(result_dict.keys())
sil_list = [result_dict[d]["silhouette"] for d in d_list]
ari_list = [result_dict[d]["ARI"] for d in d_list]
stress_list = [result_dict[d]["Stress"] for d in d_list]
error_list = [result_dict[d]["reconstruction_error"] for d in d_list]

eval_matrix = np.column_stack([sil_list, ari_list, stress_list])
# 指标下标：0轮廓系数(效益)，1ARI(效益)，2Stress(成本)
benefit_cols = [0, 1]
cost_cols = [2]
# 计算TOPSIS综合贴近度
topsis_scores = topsis_evaluation(eval_matrix, benefit_cols, cost_cols)

# 绑定维度与综合得分
dim_score_dict = dict(zip(d_list, topsis_scores))
# 取综合得分最高为最优维度
best_d = max(dim_score_dict.items(), key=lambda x: x[1])

# ---------------------- 5. 打印完整汇总表 ----------------------
print("=" * 80)
print("维度遍历汇总（含TOPSIS综合得分）")
print("=" * 80)
print(f"{'维度':^6} | {'轮廓系数':^10} | {'ARI':^10} | {'Stress':^10} | {'重构误差':^12} | {'TOPSIS得分':^12}")
print("-" * 80)
for idx, d in enumerate(d_list):
    metrics = result_dict[d]
    score = dim_score_dict[d]
    print(f"d'={d:2d}   | {metrics['silhouette']:^10.4f} | {metrics['ARI']:^10.4f} | {metrics['Stress']:^10.4f} | {metrics['reconstruction_error']:^12.6f} | {score:^12.4f}")
print("=" * 80)
print(f"\n【TOPSIS综合评价最优低维维度 d' = {best_d[0]}】综合贴近度得分 = {best_d[1]:.4f}")

# ====================== 多指标可视化 + TOPSIS得分曲线 ======================
plt.figure(figsize=(16, 10))

# 1. 轮廓系数
plt.subplot(2,3,1)
plt.plot(d_list, sil_list, marker='o', color='#2E86AB', linewidth=2, markersize=8)
plt.xlabel("LLE 低维维度 $d'$", fontsize=12)
plt.ylabel("轮廓系数", fontsize=12)
plt.title("不同维度下轮廓系数变化趋势", fontsize=12)
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 2. ARI
plt.subplot(2,3,2)
plt.plot(d_list, ari_list, marker='s', color='#A23B72', linewidth=2, markersize=8)
plt.xlabel("LLE 低维维度 $d'$", fontsize=12)
plt.ylabel("调整兰德指数 ARI", fontsize=12)
plt.title("不同维度下ARI变化趋势", fontsize=12)
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 3. Stress应力函数
plt.subplot(2,3,3)
plt.plot(d_list, stress_list, marker='^', color='#F18F01', linewidth=2, markersize=8)
plt.xlabel("LLE 低维维度 $d'$", fontsize=12)
plt.ylabel("应力函数 Stress", fontsize=12)
plt.title("不同维度下应力函数变化（越小越好）", fontsize=12)
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 4. 重构误差
plt.subplot(2,3,4)
plt.plot(d_list, error_list, marker='*', color='#009944', linewidth=2, markersize=10)
plt.xlabel("LLE 低维维度 $d'$", fontsize=12)
plt.ylabel("重构误差", fontsize=12)
plt.title("不同维度下重构误差变化（越小越好）", fontsize=12)
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 5. TOPSIS综合得分
plt.subplot(2,3,5)
plt.plot(d_list, topsis_scores, marker='D', color='#28a745', linewidth=2, markersize=8)
# 标注最优维度
plt.scatter(best_d[0], best_d[1], c='red', s=150, zorder=5, 
            label=f'最优维度 d\'={best_d[0]}', edgecolors='darkred', linewidth=2)
plt.xlabel("LLE 低维维度 $d'$", fontsize=12)
plt.ylabel("TOPSIS综合贴近度得分", fontsize=12)
plt.title("TOPSIS多指标综合评价得分（越高越好）", fontsize=12)
plt.xticks(d_list)
plt.legend(loc='best', fontsize=10)
plt.grid(alpha=0.3)

# 6. 汇总信息
plt.subplot(2,3,6)
plt.axis('off')
info_text = f"LLE降维结果汇总\n\n"
info_text += f"数据集: 手写数字 (1797样本, 64维)\n"
info_text += f"最优维度: d' = {best_d[0]}\n"
info_text += f"TOPSIS得分: {best_d[1]:.4f}\n"
info_text += f"轮廓系数: {result_dict[best_d[0]]['silhouette']:.4f}\n"
info_text += f"ARI: {result_dict[best_d[0]]['ARI']:.4f}\n"
info_text += f"Stress: {result_dict[best_d[0]]['Stress']:.4f}\n"
info_text += f"重构误差: {result_dict[best_d[0]]['reconstruction_error']:.6f}\n"
info_text += f"邻居数: {result_dict[best_d[0]]['n_neighbors']}"
plt.text(0.1, 0.5, info_text, transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# ---------------------- 6. 最优维度降维可视化 ----------------------
print("\n正在生成最优维度的可视化...")

# 先用TOPSIS选出的最优d'做LLE
n_neighbors_best = min(30, max(best_d[0] + 1, 10))
lle_best = LocallyLinearEmbedding(
    n_components=best_d[0],
    n_neighbors=n_neighbors_best,
    method='standard',
    random_state=42
)
X_best = lle_best.fit_transform(X_scaled)

from sklearn.decomposition import PCA
# 最优10维LLE特征直接PCA映射2维
pca_2d = PCA(n_components=2)
X_vis = pca_2d.fit_transform(X_best)

plt.figure(figsize=(10, 8))
sc = plt.scatter(X_vis[:,0], X_vis[:,1], c=y_real, cmap="tab10", s=20, alpha=0.7)
plt.legend(*sc.legend_elements(), title="数字类别", loc='best')
plt.title(f"LLE最优维度 d\'={best_d[0]}，PCA二维可视化", fontsize=14)
plt.xlabel("PCA第一主成分", fontsize=12)
plt.ylabel("PCA第二主成分", fontsize=12)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# ---------------------- 7. 额外：LLE参数敏感性分析 ----------------------
print("\n===== LLE参数敏感性分析 =====")
print("LLE对邻居数n_neighbors敏感，不同邻居数可能影响降维效果")
print(f"当前使用的邻居数范围: 10-30")

# 测试不同邻居数对最优维度的影响
test_d = 5  # 固定维度5进行测试
neighbor_options = [10, 15, 20, 25, 30, 35, 40]
stress_values = []
ari_values = []

print(f"\n固定维度 d'={test_d}，不同邻居数的影响:")
print(f"{'邻居数':^8} | {'Stress':^10} | {'ARI':^10}")
print("-" * 35)

for n_neigh in neighbor_options:
    try:
        lle_test = LocallyLinearEmbedding(
            n_components=test_d,
            n_neighbors=n_neigh,
            method='standard',
            random_state=42
        )
        X_test = lle_test.fit_transform(X_scaled)
        dist_test = euclidean_distances(X_test)
        stress_test = compute_stress(dist_high, dist_test)
        
        # 聚类
        kmeans_test = KMeans(n_clusters=n_cluster, random_state=42, n_init=10)
        y_pred_test = kmeans_test.fit_predict(X_test)
        ari_test = adjusted_rand_score(y_real, y_pred_test)
        
        stress_values.append(stress_test)
        ari_values.append(ari_test)
        print(f" {n_neigh:^8} | {stress_test:^10.4f} | {ari_test:^10.4f}")
    except Exception as e:
        print(f" {n_neigh:^8} | {'错误':^10} | {'错误':^10}")

# 可视化参数敏感性
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(neighbor_options[:len(stress_values)], stress_values, 'o-', color='#F18F01', linewidth=2)
plt.xlabel('邻居数 n_neighbors')
plt.ylabel('Stress')
plt.title(f'LLE Stress vs 邻居数 (d\'={test_d})')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(neighbor_options[:len(ari_values)], ari_values, 's-', color='#A23B72', linewidth=2)
plt.xlabel('邻居数 n_neighbors')
plt.ylabel('ARI')
plt.title(f'LLE ARI vs 邻居数 (d\'={test_d})')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n===== 分析完成 =====")
print(f"LLE参数: method='standard', n_neighbors动态调整")
print(f"最优维度: {best_d[0]}")
print(f"在该维度下的性能:")
print(f"  - 轮廓系数: {result_dict[best_d[0]]['silhouette']:.4f}")
print(f"  - ARI: {result_dict[best_d[0]]['ARI']:.4f}")
print(f"  - Stress: {result_dict[best_d[0]]['Stress']:.4f}")
print(f"  - 重构误差: {result_dict[best_d[0]]['reconstruction_error']:.6f}")
print(f"  - 使用的邻居数: {result_dict[best_d[0]]['n_neighbors']}")


# In[16]:


#mds
# MDS (Multidimensional Scaling)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
import time

# ===================== 全局配置：支持中文显示 =====================
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号'-'显示为方块的问题

# ---------------------- 1. 加载内置高维数据集(8*8=64维手写数字) ----------------------
data = load_digits()
X_raw = data.data  # 原始高维数据 (1797, 64)
y_real = data.target  # 真实标签，用于计算ARI
n_cluster = len(np.unique(y_real))  # 聚类簇数=10

# 数据标准化（MDS对尺度敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ---------------------- 2. 待遍历的候选低维维度：2~10 ----------------------
dim_candidates = list(range(2, 11))  # 2,3,4,5,6,7,8,9,10
result_dict = {}  # 存储每个维度的全部指标

# ---------------------- 工具函数 ----------------------
def compute_stress(high_dist, low_dist):
    """应力函数Stress（Frobenius范数形式）"""
    diff = high_dist - low_dist
    numerator = np.sum(diff ** 2)
    denominator = np.sum(high_dist ** 2)
    stress = np.sqrt(numerator / denominator)
    return stress

# TOPSIS综合评价函数
def topsis_evaluation(matrix, benefit_idx, cost_idx, weights=None):
    """
    matrix: m行n列，m个方案(维度)，n个评价指标
    benefit_idx: 效益指标下标列表（越大越好）
    cost_idx: 成本指标下标列表（越小越好）
    weights: 各指标权重，None为等权重
    return: 各方案贴近度得分
    """
    m, n = matrix.shape
    # 1. 正向化处理
    norm_matrix = matrix.copy()
    # 成本指标正向化：max - x
    for idx in cost_idx:
        col_max = np.max(norm_matrix[:, idx])
        norm_matrix[:, idx] = col_max - norm_matrix[:, idx]
    # 2. 向量标准化
    sqrt_sum = np.sqrt(np.sum(norm_matrix ** 2, axis=0))
    # 避免除以0
    sqrt_sum[sqrt_sum == 0] = 1
    norm_matrix = norm_matrix / sqrt_sum
    # 3. 加权
    if weights is None:
        weights = np.ones(n) / n
    weight_matrix = norm_matrix * weights
    # 4. 正负理想解
    pos_ideal = np.max(weight_matrix, axis=0)
    neg_ideal = np.min(weight_matrix, axis=0)
    # 5. 距离计算
    d_pos = np.sqrt(np.sum((weight_matrix - pos_ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weight_matrix - neg_ideal) ** 2, axis=1))
    # 6. 贴近度（综合得分，越高越好）
    score = d_neg / (d_pos + d_neg + 1e-10)
    return score

# 原始高维距离矩阵（全局一次计算，复用）
dist_high = euclidean_distances(X_scaled)

# ---------------------- 3. 循环遍历每个d'，计算全部指标 ----------------------
print("正在使用MDS进行降维...")
print("MDS参数: metric=True (经典MDS), n_init=4, max_iter=300")
print("注意：MDS计算复杂度较高，可能需要较长时间")
print("-" * 60)

for d in dim_candidates:
    print(f"===== 正在计算低维维度 d' = {d} =====")
    
    start_time = time.time()
    
    # MDS降维
    mds = MDS(
        n_components=d,
        metric=True,  # True: 经典MDS（保持距离），False: 非度量MDS（保持序关系）
        n_init=4,     # 初始化次数
        max_iter=300,
        random_state=42,
        dissimilarity='euclidean'  # 使用欧氏距离
    )
    X_low = mds.fit_transform(X_scaled)
    
    # 获取MDS的stress值（sklearn的MDS返回的是最小化的stress值）
    mds_stress = mds.stress_
    
    # 1. 聚类指标：轮廓系数、ARI
    kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X_low)
    sil = silhouette_score(X_low, y_pred)
    ari = adjusted_rand_score(y_real, y_pred)
    
    # 2. Stress应力函数（使用我们自己定义的）
    dist_low = euclidean_distances(X_low)
    stress_val = compute_stress(dist_high, dist_low)
    
    elapsed_time = time.time() - start_time
    
    result_dict[d] = {
        "silhouette": sil,
        "ARI": ari,
        "Stress": stress_val,
        "MDS_Stress": mds_stress,
        "time": elapsed_time
    }
    print(f"轮廓系数(Silhouette): {sil:.4f}")
    print(f"调整兰德指数(ARI): {ari:.4f}")
    print(f"应力函数Stress: {stress_val:.4f} (越小越好)")
    print(f"MDS内置Stress: {mds_stress:.4f}")
    print(f"计算时间: {elapsed_time:.2f} 秒\n")

# ---------------------- 4. TOPSIS综合评价计算 ----------------------
d_list = list(result_dict.keys())
sil_list = [result_dict[d]["silhouette"] for d in d_list]
ari_list = [result_dict[d]["ARI"] for d in d_list]
stress_list = [result_dict[d]["Stress"] for d in d_list]
mds_stress_list = [result_dict[d]["MDS_Stress"] for d in d_list]
time_list = [result_dict[d]["time"] for d in d_list]

eval_matrix = np.column_stack([sil_list, ari_list, stress_list])
# 指标下标：0轮廓系数(效益)，1ARI(效益)，2Stress(成本)
benefit_cols = [0, 1]
cost_cols = [2]
# 计算TOPSIS综合贴近度
topsis_scores = topsis_evaluation(eval_matrix, benefit_cols, cost_cols)

# 绑定维度与综合得分
dim_score_dict = dict(zip(d_list, topsis_scores))
# 取综合得分最高为最优维度
best_d = max(dim_score_dict.items(), key=lambda x: x[1])

# ---------------------- 5. 打印完整汇总表 ----------------------
print("=" * 95)
print("维度遍历汇总（含TOPSIS综合得分）")
print("=" * 95)
print(f"{'维度':^6} | {'轮廓系数':^10} | {'ARI':^10} | {'Stress':^10} | {'MDS_Stress':^12} | {'时间(秒)':^10} | {'TOPSIS':^10}")
print("-" * 95)
for idx, d in enumerate(d_list):
    metrics = result_dict[d]
    score = dim_score_dict[d]
    print(f"d'={d:2d}   | {metrics['silhouette']:^10.4f} | {metrics['ARI']:^10.4f} | {metrics['Stress']:^10.4f} | {metrics['MDS_Stress']:^12.4f} | {metrics['time']:^10.2f} | {score:^10.4f}")
print("=" * 95)
print(f"\n【TOPSIS综合评价最优低维维度 d' = {best_d[0]}】综合贴近度得分 = {best_d[1]:.4f}")

# ====================== 多指标可视化 ======================
plt.figure(figsize=(18, 10))

# 1. 轮廓系数
plt.subplot(2,3,1)
plt.plot(d_list, sil_list, marker='o', color='#2E86AB', linewidth=2, markersize=8)
plt.xlabel("MDS 低维维度 $d'$", fontsize=12)
plt.ylabel("轮廓系数", fontsize=12)
plt.title("不同维度下轮廓系数变化趋势", fontsize=12)
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 2. ARI
plt.subplot(2,3,2)
plt.plot(d_list, ari_list, marker='s', color='#A23B72', linewidth=2, markersize=8)
plt.xlabel("MDS 低维维度 $d'$", fontsize=12)
plt.ylabel("调整兰德指数 ARI", fontsize=12)
plt.title("不同维度下ARI变化趋势", fontsize=12)
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 3. Stress应力函数
plt.subplot(2,3,3)
plt.plot(d_list, stress_list, marker='^', color='#F18F01', linewidth=2, markersize=8)
plt.xlabel("MDS 低维维度 $d'$", fontsize=12)
plt.ylabel("应力函数 Stress", fontsize=12)
plt.title("不同维度下应力函数变化（越小越好）", fontsize=12)
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 4. MDS内置Stress
plt.subplot(2,3,4)
plt.plot(d_list, mds_stress_list, marker='*', color='#009944', linewidth=2, markersize=10)
plt.xlabel("MDS 低维维度 $d'$", fontsize=12)
plt.ylabel("MDS 内置 Stress", fontsize=12)
plt.title("MDS内置Stress变化（越小越好）", fontsize=12)
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 5. TOPSIS综合得分
plt.subplot(2,3,5)
plt.plot(d_list, topsis_scores, marker='D', color='#28a745', linewidth=2, markersize=8)
# 标注最优维度
plt.scatter(best_d[0], best_d[1], c='red', s=150, zorder=5, 
            label=f'最优维度 d\'={best_d[0]}', edgecolors='darkred', linewidth=2)
plt.xlabel("MDS 低维维度 $d'$", fontsize=12)
plt.ylabel("TOPSIS综合贴近度得分", fontsize=12)
plt.title("TOPSIS多指标综合评价得分（越高越好）", fontsize=12)
plt.xticks(d_list)
plt.legend(loc='best', fontsize=10)
plt.grid(alpha=0.3)

# 6. 汇总信息
plt.subplot(2,3,6)
plt.axis('off')
info_text = f"MDS降维结果汇总\n\n"
info_text += f"数据集: 手写数字 (1797样本, 64维)\n"
info_text += f"最优维度: d' = {best_d[0]}\n"
info_text += f"TOPSIS得分: {best_d[1]:.4f}\n"
info_text += f"轮廓系数: {result_dict[best_d[0]]['silhouette']:.4f}\n"
info_text += f"ARI: {result_dict[best_d[0]]['ARI']:.4f}\n"
info_text += f"Stress: {result_dict[best_d[0]]['Stress']:.4f}\n"
info_text += f"MDS Stress: {result_dict[best_d[0]]['MDS_Stress']:.4f}\n"
info_text += f"计算时间: {result_dict[best_d[0]]['time']:.2f}秒"
plt.text(0.1, 0.5, info_text, transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# ---------------------- 6. 最优维度降维可视化 ----------------------
print("\n正在生成最优维度的可视化...")

# 先用TOPSIS选出的最优d'做MDS
mds_best = MDS(
    n_components=best_d[0],
    metric=True,
    n_init=4,
    max_iter=300,
    random_state=42,
    dissimilarity='euclidean'
)
X_best = mds_best.fit_transform(X_scaled)

# 二次降维到2维绘图（使用MDS）
mds_2d = MDS(
    n_components=2,
    metric=True,
    n_init=4,
    max_iter=300,
    random_state=42,
    dissimilarity='euclidean'
)
X_vis = mds_2d.fit_transform(X_best)

plt.figure(figsize=(10, 8))
sc = plt.scatter(X_vis[:,0], X_vis[:,1], c=y_real, cmap="tab10", s=20, alpha=0.7)
plt.legend(*sc.legend_elements(), title="数字类别", loc='best')
plt.title(f"MDS最优维度 d\'={best_d[0]} 的二维可视化", fontsize=14)
plt.xlabel("MDS 第一维特征", fontsize=12)
plt.ylabel("MDS 第二维特征", fontsize=12)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# ---------------------- 7. 额外：MDS vs 其他方法对比 ----------------------
print("\n===== MDS特性分析 =====")
print("1. MDS是一种经典的距离保持降维方法")
print("2. 与Isomap不同，MDS使用欧氏距离而非测地线距离")
print("3. MDS的计算复杂度为O(N³)，适合中小型数据集")
print("4. MDS对噪声敏感度中等")

# 比较不同维度的计算时间
plt.figure(figsize=(10, 6))
plt.bar(d_list, time_list, color='#6C3082', alpha=0.7)
plt.xlabel("MDS 低维维度 $d'$", fontsize=12)
plt.ylabel("计算时间 (秒)", fontsize=12)
plt.title("不同维度下MDS计算时间对比", fontsize=14)
for i, v in enumerate(time_list):
    plt.text(d_list[i], v + 0.1, f'{v:.1f}s', ha='center', va='bottom')
plt.xticks(d_list)
plt.grid(alpha=0.2, axis='y')
plt.tight_layout()
plt.show()

# ---------------------- 8. 对比不同Stress指标 ----------------------
plt.figure(figsize=(10, 6))
plt.plot(d_list, stress_list, 'o-', color='#F18F01', linewidth=2, markersize=8, label='自定义Stress')
plt.plot(d_list, mds_stress_list, 's-', color='#009944', linewidth=2, markersize=8, label='MDS内置Stress')
plt.xlabel("MDS 低维维度 $d'$", fontsize=12)
plt.ylabel("Stress值", fontsize=12)
plt.title("自定义Stress vs MDS内置Stress对比", fontsize=14)
plt.xticks(d_list)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\n===== 分析完成 =====")
print(f"MDS参数: metric=True, n_init=4, max_iter=300")
print(f"最优维度: {best_d[0]}")
print(f"在该维度下的性能:")
print(f"  - 轮廓系数: {result_dict[best_d[0]]['silhouette']:.4f}")
print(f"  - ARI: {result_dict[best_d[0]]['ARI']:.4f}")
print(f"  - Stress: {result_dict[best_d[0]]['Stress']:.4f}")
print(f"  - MDS Stress: {result_dict[best_d[0]]['MDS_Stress']:.4f}")
print(f"  - 计算时间: {result_dict[best_d[0]]['time']:.2f}秒")


# In[17]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances

# ===================== 全局配置：支持中文显示 =====================
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号'-'显示为方块的问题

# ---------------------- 1. 加载内置高维数据集(8*8=64维手写数字) ----------------------
data = load_digits()
X_raw = data.data  # 原始高维数据 (1797, 64)
y_real = data.target  # 真实标签，用于计算ARI
n_cluster = len(np.unique(y_real))  # 聚类簇数=10

# ---------------------- 2. 待遍历的候选低维维度：2~10 ----------------------
dim_candidates = list(range(2, 11))
result_dict = {}  # 存储每个维度的全部指标

# ---------------------- 工具函数 ----------------------
def compute_stress(high_dist, low_dist):
    """应力函数Stress（Frobenius范数形式，衡量高低维距离保真度）"""
    diff = high_dist - low_dist
    numerator = np.sum(diff ** 2)
    denominator = np.sum(high_dist ** 2)
    stress = np.sqrt(numerator / denominator)
    return stress

# TOPSIS综合评价函数
def topsis_evaluation(matrix, benefit_idx, cost_idx, weights=None):
    """
    matrix: m行n列，m个方案(维度)，n个评价指标
    benefit_idx: 效益指标下标列表（越大越好）
    cost_idx: 成本指标下标列表（越小越好）
    weights: 各指标权重，None为等权重
    return: 各方案贴近度得分
    """
    m, n = matrix.shape
    # 1. 正向化处理
    norm_matrix = matrix.copy()
    # 成本指标正向化：max - x
    for idx in cost_idx:
        col_max = np.max(norm_matrix[:, idx])
        norm_matrix[:, idx] = col_max - norm_matrix[:, idx]
    # 2. 向量标准化
    sqrt_sum = np.sqrt(np.sum(norm_matrix ** 2, axis=0))
    norm_matrix = norm_matrix / sqrt_sum
    # 3. 加权
    if weights is None:
        weights = np.ones(n) / n
    weight_matrix = norm_matrix * weights
    # 4. 正负理想解
    pos_ideal = np.max(weight_matrix, axis=0)
    neg_ideal = np.min(weight_matrix, axis=0)
    # 5. 距离计算
    d_pos = np.sqrt(np.sum((weight_matrix - pos_ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weight_matrix - neg_ideal) ** 2))
    # 6. 贴近度（综合得分，越高越好）
    score = d_neg / (d_pos + d_neg)
    return score

# 原始高维距离矩阵（全局一次计算，复用）
dist_high = euclidean_distances(X_raw)

# ---------------------- 3. 循环遍历每个d'，计算全部指标 ----------------------
for d in dim_candidates:
    print(f"===== 正在计算低维维度 d' = {d} =====")
    # SVD降维 TruncatedSVD
    svd = TruncatedSVD(n_components=d, random_state=42)
    X_low = svd.fit_transform(X_raw)
    
    # 1. 聚类指标：轮廓系数、ARI
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    y_pred = kmeans.fit_predict(X_low)
    sil = silhouette_score(X_low, y_pred)
    ari = adjusted_rand_score(y_real, y_pred)
    
    # 2. Stress应力函数
    dist_low = euclidean_distances(X_low)
    stress_val = compute_stress(dist_high, dist_low)
    
    # SVD无方差解释率内置属性，仅保存三项指标
    result_dict[d] = {
        "silhouette": sil,
        "ARI": ari,
        "Stress": stress_val
    }
    print(f"轮廓系数(Silhouette): {sil:.4f}")
    print(f"调整兰德指数(ARI): {ari:.4f}")
    print(f"应力函数Stress: {stress_val:.4f} (越小越好)\n")

# ---------------------- 4. TOPSIS综合评价计算 ----------------------
# 评价矩阵：[轮廓系数, ARI, Stress]
# 0:sil(效益)  1:ARI(效益)  2:Stress(成本)
d_list = list(result_dict.keys())
sil_list = [result_dict[d]["silhouette"] for d in d_list]
ari_list = [result_dict[d]["ARI"] for d in d_list]
stress_list = [result_dict[d]["Stress"] for d in d_list]

eval_matrix = np.column_stack([sil_list, ari_list, stress_list])
benefit_cols = [0, 1]
cost_cols = [2]
topsis_scores = topsis_evaluation(eval_matrix, benefit_cols, cost_cols)

# 绑定维度与综合得分
dim_score_dict = dict(zip(d_list, topsis_scores))
# 取综合得分最高为最优维度
best_d = max(dim_score_dict.items(), key=lambda x: x[1])

# ---------------------- 5. 打印完整汇总表 ----------------------
print("==================== 维度遍历汇总（含TOPSIS综合得分） ====================")
for idx, d in enumerate(d_list):
    metrics = result_dict[d]
    score = dim_score_dict[d]
    print(f"d'={d:2d} | 轮廓系数={metrics['silhouette']:.4f} | ARI={metrics['ARI']:.4f} | Stress={metrics['Stress']:.4f} | TOPSIS综合得分={score:.4f}")
print(f"\n【TOPSIS综合评价最优低维维度 d' = {best_d[0]}】综合贴近度得分={best_d[1]:.4f}")

# ====================== 多指标可视化 2×2布局 ======================
plt.figure(figsize=(14, 10))
# 1. 轮廓系数
plt.subplot(2,2,1)
plt.plot(d_list, sil_list, marker='o', color='#2E86AB', linewidth=2)
plt.xlabel("SVD(TruncatedSVD) 低维维度 $d'$")
plt.ylabel("轮廓系数")
plt.title("不同维度下轮廓系数变化趋势")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 2. ARI
plt.subplot(2,2,2)
plt.plot(d_list, ari_list, marker='s', color='#A23B72', linewidth=2)
plt.xlabel("SVD(TruncatedSVD) 低维维度 $d'$")
plt.ylabel("调整兰德指数 ARI")
plt.title("不同维度下ARI变化趋势")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 3. Stress应力函数
plt.subplot(2,2,3)
plt.plot(d_list, stress_list, marker='^', color='#F18F01', linewidth=2)
plt.xlabel("SVD(TruncatedSVD) 低维维度 $d'$")
plt.ylabel("应力函数 Stress")
plt.title("不同维度下应力函数变化（越小距离保留越好）")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 4. TOPSIS综合得分
plt.subplot(2,2,4)
plt.plot(d_list, topsis_scores, marker='D', color='#28a745', linewidth=2, markersize=7)
# 标注最优维度
plt.scatter(best_d[0], best_d[1], c='red', s=120, zorder=5, label=f'最优维度d\'={best_d[0]}')
plt.xlabel("SVD(TruncatedSVD) 低维维度 $d'$")
plt.ylabel("TOPSIS综合贴近度得分")
plt.title("TOPSIS多指标综合评价得分（越高综合效果越好）")
plt.xticks(d_list)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------- 最优维度降维可视化 ----------------------
# 先用TOPSIS选出的最优d'做SVD
svd_best = TruncatedSVD(n_components=best_d[0], random_state=42)
X_best = svd_best.fit_transform(X_raw)
# 二次SVD降到2维绘图
svd_2d = TruncatedSVD(n_components=2, random_state=42)
X_vis = svd_2d.fit_transform(X_best)

plt.figure(figsize=(8,6))
sc = plt.scatter(X_vis[:,0], X_vis[:,1], c=y_real, cmap="tab10", s=15)
plt.legend(*sc.legend_elements(), title="数字类别标签")
plt.title(f"TOPSIS最优维度$d'$={best_d[0]} SVD二维样本分布可视化")
plt.xlabel("SVD 第一奇异分量")
plt.ylabel("SVD 第二奇异分量")
plt.show()


# In[ ]:


#有监督学习


# In[18]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances

# 中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 加载数据
data = load_digits()
X_raw = data.data  # (1797, 64)
y_real = data.target
n_class = len(np.unique(y_real))  # 10类
max_lda_dim = n_class - 1        # LDA最大维度 9

# 1. 遍历LDA可用维度 2~9
dim_candidates = list(range(2, max_lda_dim + 1))
result_dict = {}
dist_high = euclidean_distances(X_raw)

def compute_stress(high_dist, low_dist):
    diff = high_dist - low_dist
    numerator = np.sum(diff ** 2)
    denominator = np.sum(high_dist ** 2)
    stress = np.sqrt(numerator / denominator)
    return stress

# TOPSIS函数复用你原有逻辑
def topsis_evaluation(matrix, benefit_idx, cost_idx, weights=None):
    m, n = matrix.shape
    norm_matrix = matrix.copy()
    for idx in cost_idx:
        col_max = np.max(norm_matrix[:, idx])
        norm_matrix[:, idx] = col_max - norm_matrix[:, idx]
    sqrt_sum = np.sqrt(np.sum(norm_matrix ** 2, axis=0))
    norm_matrix = norm_matrix / sqrt_sum
    if weights is None:
        weights = np.ones(n) / n
    weight_matrix = norm_matrix * weights
    pos_ideal = np.max(weight_matrix, axis=0)
    neg_ideal = np.min(weight_matrix, axis=0)
    d_pos = np.sqrt(np.sum((weight_matrix - pos_ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weight_matrix - neg_ideal) ** 2))
    score = d_neg / (d_pos + d_neg)
    return score

# 遍历各低维维度计算指标
for d in dim_candidates:
    print(f"===== LDA 低维维度 d' = {d} =====")
    # LDA训练：必须传入标签y_real
    lda = LinearDiscriminantAnalysis(n_components=d)
    X_low = lda.fit_transform(X_raw, y_real)

    # Kmeans聚类评估
    kmeans = KMeans(n_clusters=n_class, random_state=42, n_init="auto")
    y_pred = kmeans.fit_predict(X_low)
    sil = silhouette_score(X_low, y_pred)
    ari = adjusted_rand_score(y_real, y_pred)

    # Stress
    dist_low = euclidean_distances(X_low)
    stress_val = compute_stress(dist_high, dist_low)

    result_dict[d] = {
        "silhouette": sil,
        "ARI": ari,
        "Stress": stress_val
    }
    print(f"轮廓系数: {sil:.4f} | ARI: {ari:.4f} | Stress: {stress_val:.4f}\n")

# TOPSIS综合评价
d_list = list(result_dict.keys())
sil_list = [result_dict[d]["silhouette"] for d in d_list]
ari_list = [result_dict[d]["ARI"] for d in d_list]
stress_list = [result_dict[d]["Stress"] for d in d_list]
eval_matrix = np.column_stack([sil_list, ari_list, stress_list])
benefit_cols = [0, 1]
cost_cols = [2]
topsis_scores = topsis_evaluation(eval_matrix, benefit_cols, cost_cols)
dim_score_dict = dict(zip(d_list, topsis_scores))
best_d = max(dim_score_dict.items(), key=lambda x: x[1])

# 输出汇总
print("==================== LDA维度遍历汇总 ====================")
for idx, d in enumerate(d_list):
    m = result_dict[d]
    s = dim_score_dict[d]
    print(f"d'={d:2d} | 轮廓={m['silhouette']:.4f} | ARI={m['ARI']:.4f} | Stress={m['Stress']:.4f} | TOPSIS={s:.4f}")
print(f"\nTOPSIS最优LDA维度 d' = {best_d[0]}，综合得分={best_d[1]:.4f}")

# 绘图
plt.figure(figsize=(14, 10))
plt.subplot(2,2,1)
plt.plot(d_list, sil_list, marker='o', c="#2E86AB", lw=2)
plt.xlabel("LDA低维维度 $d'$")
plt.ylabel("轮廓系数")
plt.title("各维度轮廓系数变化")
plt.xticks(d_list)
plt.grid(alpha=0.3)

plt.subplot(2,2,2)
plt.plot(d_list, ari_list, marker='s', c="#A23B72", lw=2)
plt.xlabel("LDA低维维度 $d'$")
plt.ylabel("ARI")
plt.title("各维度ARI变化")
plt.xticks(d_list)
plt.grid(alpha=0.3)

plt.subplot(2,2,3)
plt.plot(d_list, stress_list, marker='^', c="#F18F01", lw=2)
plt.xlabel("LDA低维维度 $d'$")
plt.ylabel("Stress")
plt.title("距离保留应力值")
plt.xticks(d_list)
plt.grid(alpha=0.3)

plt.subplot(2,2,4)
plt.plot(d_list, topsis_scores, marker='D', c="#28a745", lw=2, ms=7)
plt.scatter(best_d[0], best_d[1], c="red", s=120, zorder=5, label=f"最优d'={best_d[0]}")
plt.xlabel("LDA低维维度 $d'$")
plt.ylabel("TOPSIS综合得分")
plt.title("LDA多指标综合评价")
plt.legend()
plt.xticks(d_list)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 最优LDA维度降维后，再投影到2维可视化
lda_best = LinearDiscriminantAnalysis(n_components=best_d[0])
X_best = lda_best.fit_transform(X_raw, y_real)

# 取前两维直接画图（LDA本身可以直接降到2）
plt.figure(figsize=(8,6))
sc = plt.scatter(X_best[:,0], X_best[:,1], c=y_real, cmap="tab10", s=15)
plt.legend(*sc.legend_elements(), title="数字类别")
plt.title(f"LDA最优维度$d'$={best_d[0]} 前两维样本分布")
plt.xlabel("LDA第一判别轴")
plt.ylabel("LDA第二判别轴")
plt.show()


# In[6]:


#lasso
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap  # 新增Isomap用于统一可视化

# ===================== 全局中文配置 =====================
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------- 1. 加载并预处理数据 ----------------------
data = load_digits()
X_raw = data.data  # (1797, 64)
y_real = data.target
n_cluster = len(np.unique(y_real))

# 标准化（L1正则对尺度敏感，但统一流程）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
dist_high = euclidean_distances(X_scaled)  # 原始高维距离矩阵

# ---------------------- 工具函数复用 ----------------------
def compute_stress(high_dist, low_dist):
    diff = high_dist - low_dist
    numerator = np.sum(diff ** 2)
    denominator = np.sum(high_dist ** 2)
    stress = np.sqrt(numerator / denominator)
    return stress

def topsis_evaluation(matrix, benefit_idx, cost_idx, weights=None):
    m, n = matrix.shape
    norm_matrix = matrix.copy()
    # 成本指标正向化
    for idx in cost_idx:
        col_max = np.max(norm_matrix[:, idx])
        norm_matrix[:, idx] = col_max - norm_matrix[:, idx]
    # 向量归一化
    sqrt_sum = np.sqrt(np.sum(norm_matrix ** 2, axis=0))
    norm_matrix = norm_matrix / sqrt_sum
    # 权重
    if weights is None:
        weights = np.ones(n) / n
    weight_matrix = norm_matrix * weights
    pos_ideal = np.max(weight_matrix, axis=0)
    neg_ideal = np.min(weight_matrix, axis=0)
    d_pos = np.sqrt(np.sum((weight_matrix - pos_ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weight_matrix - neg_ideal) ** 2))
    score = d_neg / (d_pos + d_neg)
    return score

# ---------------------- 2. 遍历保留特征数量 2~10 ----------------------
dim_candidates = list(range(2, 11))
result_dict = {}

for keep_dim in dim_candidates:
    print(f"===== Lasso筛选，保留特征数 = {keep_dim} =====")
    # L1正则多分类逻辑回归，控制稀疏度筛选特征
    lasso_clf = LogisticRegression(penalty="l1", solver="saga", random_state=42, max_iter=5000)
    lasso_clf.fit(X_scaled, y_real)
    # 提取所有类别特征系数绝对值均值，代表特征重要性
    feat_importance = np.mean(np.abs(lasso_clf.coef_), axis=0)
    # 取前keep_dim个最重要特征下标
    top_idx = np.argsort(feat_importance)[-keep_dim:]
    X_low = X_scaled[:, top_idx]

    # KMeans聚类评估
    kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init="auto")
    y_pred = kmeans.fit_predict(X_low)
    sil = silhouette_score(X_low, y_pred)
    ari = adjusted_rand_score(y_real, y_pred)

    # 计算Stress
    dist_low = euclidean_distances(X_low)
    stress_val = compute_stress(dist_high, dist_low)

    result_dict[keep_dim] = {
        "silhouette": sil,
        "ARI": ari,
        "Stress": stress_val
    }
    print(f"轮廓系数: {sil:.4f} | ARI: {ari:.4f} | Stress: {stress_val:.4f}\n")

# ---------------------- 3. TOPSIS综合评价 ----------------------
d_list = list(result_dict.keys())
sil_list = [result_dict[d]["silhouette"] for d in d_list]
ari_list = [result_dict[d]["ARI"] for d in d_list]
stress_list = [result_dict[d]["Stress"] for d in d_list]

eval_matrix = np.column_stack([sil_list, ari_list, stress_list])
benefit_cols = [0, 1]  # 轮廓、ARI越大越好
cost_cols = [2]        # Stress越小越好
topsis_scores = topsis_evaluation(eval_matrix, benefit_cols, cost_cols)
dim_score_dict = dict(zip(d_list, topsis_scores))
best_d = max(dim_score_dict.items(), key=lambda x: x[1])

# ---------------------- 4. 输出汇总表格 ----------------------
print("==================== Lasso特征筛选维度汇总 ====================")
for idx, d in enumerate(d_list):
    metrics = result_dict[d]
    score = dim_score_dict[d]
    print(f"保留特征数={d:2d} | 轮廓={metrics['silhouette']:.4f} | ARI={metrics['ARI']:.4f} | Stress={metrics['Stress']:.4f} | TOPSIS={score:.4f}")
print(f"\n【TOPSIS最优保留特征数量 d' = {best_d[0]}】综合得分={best_d[1]:.4f}")

# ====================== 多指标可视化 ======================
plt.figure(figsize=(14, 10))
# 1. 轮廓系数
plt.subplot(2,2,1)
plt.plot(d_list, sil_list, marker='o', color='#2E86AB', linewidth=2)
plt.xlabel("Lasso保留原始特征数量 $d'$")
plt.ylabel("轮廓系数")
plt.title("不同特征数量下轮廓系数变化趋势")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 2. ARI
plt.subplot(2,2,2)
plt.plot(d_list, ari_list, marker='s', color='#A23B72', linewidth=2)
plt.xlabel("Lasso保留原始特征数量 $d'$")
plt.ylabel("调整兰德指数 ARI")
plt.title("不同特征数量下ARI变化趋势")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 3. Stress应力函数
plt.subplot(2,2,3)
plt.plot(d_list, stress_list, marker='^', color='#F18F01', linewidth=2)
plt.xlabel("Lasso保留原始特征数量 $d'$")
plt.ylabel("应力函数 Stress")
plt.title("距离保留应力值（越小越好）")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 4. TOPSIS综合得分
plt.subplot(2,2,4)
plt.plot(d_list, topsis_scores, marker='D', color='#28a745', linewidth=2, markersize=7)
plt.scatter(best_d[0], best_d[1], c='red', s=120, zorder=5, label=f'最优特征数={best_d[0]}')
plt.xlabel("Lasso保留原始特征数量 $d'$")
plt.ylabel("TOPSIS综合贴近度得分")
plt.title("TOPSIS多指标综合评价得分")
plt.legend()
plt.xticks(d_list)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------- 最优特征子集二维可视化（修复Isomap报错，删除random_state） ----------------------
best_k = best_d[0]
# 提取互信息最高的best_k个原始特征
top_idx_best = np.argsort(feat_importance)[-best_k:]
X_best = X_scaled[:, top_idx_best]

# Isomap 删除不支持的 random_state 参数
iso_2d = Isomap(n_components=2, n_neighbors=10)
X_vis = iso_2d.fit_transform(X_best)

plt.figure(figsize=(10,8))
sc = plt.scatter(X_vis[:,0], X_vis[:,1], c=y_real, cmap="tab10", s=20, alpha=0.8)
plt.legend(*sc.legend_elements(), title="数字类别标签", loc="best")
plt.title(f"Lasso最优保留{best_k}个特征 Isomap二维投影可视化")
plt.xlabel("Isomap投影维度1")
plt.ylabel("Isomap投影维度2")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap  # 新增Isomap用于统一可视化标准

# ===================== 全局中文配置 =====================
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------- 1. 加载并预处理数据 ----------------------
data = load_digits()
X_raw = data.data  # (1797, 64)
y_real = data.target
n_cluster = len(np.unique(y_real))

# 标准化（互信息对尺度不敏感，但统一流程）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
dist_high = euclidean_distances(X_scaled)  # 原始高维距离矩阵

# ---------------------- 工具函数复用 ----------------------
def compute_stress(high_dist, low_dist):
    diff = high_dist - low_dist
    numerator = np.sum(diff ** 2)
    denominator = np.sum(high_dist ** 2)
    stress = np.sqrt(numerator / denominator)
    return stress

def topsis_evaluation(matrix, benefit_idx, cost_idx, weights=None):
    m, n = matrix.shape
    norm_matrix = matrix.copy()
    # 成本指标正向化
    for idx in cost_idx:
        col_max = np.max(norm_matrix[:, idx])
        norm_matrix[:, idx] = col_max - norm_matrix[:, idx]
    # 向量归一化
    sqrt_sum = np.sqrt(np.sum(norm_matrix ** 2, axis=0))
    norm_matrix = norm_matrix / sqrt_sum
    # 权重
    if weights is None:
        weights = np.ones(n) / n
    weight_matrix = norm_matrix * weights
    pos_ideal = np.max(weight_matrix, axis=0)
    neg_ideal = np.min(weight_matrix, axis=0)
    d_pos = np.sqrt(np.sum((weight_matrix - pos_ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weight_matrix - neg_ideal) ** 2))
    score = d_neg / (d_pos + d_neg)
    return score

# ---------------------- 2. 遍历保留特征数量 2~10 ----------------------
dim_candidates = list(range(2, 11))
result_dict = {}

# 一次性计算所有特征与标签的互信息（过滤式核心）
mi_scores = mutual_info_classif(X_scaled, y_real, random_state=42)

for keep_dim in dim_candidates:
    print(f"===== 互信息过滤，保留特征数 = {keep_dim} =====")
    # 取互信息最大的前keep_dim个特征下标
    top_idx = np.argsort(mi_scores)[-keep_dim:]
    X_low = X_scaled[:, top_idx]

    # KMeans聚类评估
    kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init="auto")
    y_pred = kmeans.fit_predict(X_low)
    sil = silhouette_score(X_low, y_pred)
    ari = adjusted_rand_score(y_real, y_pred)

    # 计算Stress
    dist_low = euclidean_distances(X_low)
    stress_val = compute_stress(dist_high, dist_low)

    result_dict[keep_dim] = {
        "silhouette": sil,
        "ARI": ari,
        "Stress": stress_val
    }
    print(f"轮廓系数: {sil:.4f} | ARI: {ari:.4f} | Stress: {stress_val:.4f}\n")

# ---------------------- 3. TOPSIS综合评价 ----------------------
d_list = list(result_dict.keys())
sil_list = [result_dict[d]["silhouette"] for d in d_list]
ari_list = [result_dict[d]["ARI"] for d in d_list]
stress_list = [result_dict[d]["Stress"] for d in d_list]

eval_matrix = np.column_stack([sil_list, ari_list, stress_list])
benefit_cols = [0, 1]  # 轮廓、ARI越大越好
cost_cols = [2]        # Stress越小越好
topsis_scores = topsis_evaluation(eval_matrix, benefit_cols, cost_cols)
dim_score_dict = dict(zip(d_list, topsis_scores))
best_d = max(dim_score_dict.items(), key=lambda x: x[1])

# ---------------------- 4. 输出汇总表格 ----------------------
print("==================== 互信息过滤特征选择汇总 ====================")
for idx, d in enumerate(d_list):
    metrics = result_dict[d]
    score = dim_score_dict[d]
    print(f"保留特征数={d:2d} | 轮廓={metrics['silhouette']:.4f} | ARI={metrics['ARI']:.4f} | Stress={metrics['Stress']:.4f} | TOPSIS={score:.4f}")
print(f"\n【TOPSIS最优保留特征数量 d' = {best_d[0]}】综合得分={best_d[1]:.4f}")

# ====================== 多指标可视化 ======================
plt.figure(figsize=(14, 10))
# 1. 轮廓系数
plt.subplot(2,2,1)
plt.plot(d_list, sil_list, marker='o', color='#2E86AB', linewidth=2)
plt.xlabel("互信息筛选保留特征数量 $d'$")
plt.ylabel("轮廓系数")
plt.title("不同特征数量下轮廓系数变化趋势")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 2. ARI
plt.subplot(2,2,2)
plt.plot(d_list, ari_list, marker='s', color='#A23B72', linewidth=2)
plt.xlabel("互信息筛选保留特征数量 $d'$")
plt.ylabel("调整兰德指数 ARI")
plt.title("不同特征数量下ARI变化趋势")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 3. Stress应力函数
plt.subplot(2,2,3)
plt.plot(d_list, stress_list, marker='^', color='#F18F01', linewidth=2)
plt.xlabel("互信息筛选保留特征数量 $d'$")
plt.ylabel("应力函数 Stress")
plt.title("距离保留应力值（越小越好）")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# 4. TOPSIS综合得分
plt.subplot(2,2,4)
plt.plot(d_list, topsis_scores, marker='D', color='#28a745', linewidth=2, markersize=7)
plt.scatter(best_d[0], best_d[1], c='red', s=120, zorder=5, label=f'最优特征数={best_d[0]}')
plt.xlabel("互信息筛选保留特征数量 $d'$")
plt.ylabel("TOPSIS综合贴近度得分")
plt.title("TOPSIS多指标综合评价得分")
plt.legend()
plt.xticks(d_list)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------- 最优特征子集二维可视化（修复Isomap报错，移除random_state） ----------------------
best_k = best_d[0]
# 提取互信息最高的best_k个原始特征
top_idx_best = np.argsort(mi_scores)[-best_k:]
X_best = X_scaled[:, top_idx_best]

# Isomap 移除不支持的 random_state 参数
iso_2d = Isomap(n_components=2, n_neighbors=10)
X_vis = iso_2d.fit_transform(X_best)

plt.figure(figsize=(10,8))
sc = plt.scatter(X_vis[:,0], X_vis[:,1], c=y_real, cmap="tab10", s=20, alpha=0.8)
plt.legend(*sc.legend_elements(), title="数字类别标签", loc="best")
plt.title(f"互信息过滤最优{best_k}个特征 Isomap二维投影可视化")
plt.xlabel("Isomap投影维度1")
plt.ylabel("Isomap投影维度2")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# In[32]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

# 中文显示配置
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 加载数据
data = load_digits()
X_raw = data.data
y_real = data.target
n_cluster = len(np.unique(y_real))

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
dist_high = euclidean_distances(X_scaled)

# Stress计算函数
def compute_stress(high_dist, low_dist):
    diff = high_dist - low_dist
    numerator = np.sum(diff ** 2)
    denominator = np.sum(high_dist ** 2)
    stress = np.sqrt(numerator / denominator)
    return stress

# TOPSIS综合评价
def topsis_evaluation(matrix, benefit_idx, cost_idx, weights=None):
    m, n = matrix.shape
    norm_matrix = matrix.copy()
    for idx in cost_idx:
        col_max = np.max(norm_matrix[:, idx])
        norm_matrix[:, idx] = col_max - norm_matrix[:, idx]
    sqrt_sum = np.sqrt(np.sum(norm_matrix ** 2, axis=0))
    norm_matrix = norm_matrix / sqrt_sum
    if weights is None:
        weights = np.ones(n) / n
    weight_matrix = norm_matrix * weights
    pos_ideal = np.max(weight_matrix, axis=0)
    neg_ideal = np.min(weight_matrix, axis=0)
    d_pos = np.sqrt(np.sum((weight_matrix - pos_ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weight_matrix - neg_ideal) ** 2))
    score = d_neg / (d_pos + d_neg)
    return score

# 遍历保留特征数量 2~10
dim_candidates = list(range(2, 11))
result_dict = {}

for keep_dim in dim_candidates:
    print(f"===== RFE递归特征消除，保留特征数 = {keep_dim} =====")
    # 基模型：逻辑回归
    base_model = LogisticRegression(max_iter=5000, solver="saga", random_state=42)
    # RFE构建
    rfe = RFE(estimator=base_model, n_features_to_select=keep_dim, step=1)
    X_low = rfe.fit_transform(X_scaled, y_real)

    # Kmeans聚类评估
    kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init="auto")
    y_pred = kmeans.fit_predict(X_low)
    sil = silhouette_score(X_low, y_pred)
    ari = adjusted_rand_score(y_real, y_pred)

    # Stress
    dist_low = euclidean_distances(X_low)
    stress_val = compute_stress(dist_high, dist_low)

    result_dict[keep_dim] = {
        "silhouette": sil,
        "ARI": ari,
        "Stress": stress_val
    }
    print(f"轮廓系数: {sil:.4f} | ARI: {ari:.4f} | Stress: {stress_val:.4f}\n")

# TOPSIS综合打分
d_list = list(result_dict.keys())
sil_list = [result_dict[d]["silhouette"] for d in d_list]
ari_list = [result_dict[d]["ARI"] for d in d_list]
stress_list = [result_dict[d]["Stress"] for d in d_list]
eval_matrix = np.column_stack([sil_list, ari_list, stress_list])
benefit_cols = [0, 1]
cost_cols = [2]
topsis_scores = topsis_evaluation(eval_matrix, benefit_cols, cost_cols)
dim_score_dict = dict(zip(d_list, topsis_scores))
best_d = max(dim_score_dict.items(), key=lambda x: x[1])

# 输出汇总表
print("==================== RFE递归特征消除维度汇总 ====================")
for idx, d in enumerate(d_list):
    m = result_dict[d]
    s = dim_score_dict[d]
    print(f"保留特征数={d:2d} | 轮廓={m['silhouette']:.4f} | ARI={m['ARI']:.4f} | Stress={m['Stress']:.4f} | TOPSIS={s:.4f}")
print(f"\nTOPSIS最优保留特征数量 d' = {best_d[0]}，综合得分={best_d[1]:.4f}")

# 绘制四张子图
plt.figure(figsize=(14, 10))
plt.subplot(2,2,1)
plt.plot(d_list, sil_list, marker='o', c="#2E86AB", lw=2)
plt.xlabel("RFE保留原始特征数量 $d'$")
plt.ylabel("轮廓系数")
plt.title("不同特征数量下轮廓系数变化")
plt.xticks(d_list)
plt.grid(alpha=0.3)

plt.subplot(2,2,2)
plt.plot(d_list, ari_list, marker='s', c="#A23B72", lw=2)
plt.xlabel("RFE保留原始特征数量 $d'$")
plt.ylabel("ARI")
plt.title("不同特征数量下ARI变化")
plt.xticks(d_list)
plt.grid(alpha=0.3)

plt.subplot(2,2,3)
plt.plot(d_list, stress_list, marker='^', c="#F18F01", lw=2)
plt.xlabel("RFE保留原始特征数量 $d'$")
plt.ylabel("Stress")
plt.title("距离保留应力值")
plt.xticks(d_list)
plt.grid(alpha=0.3)

plt.subplot(2,2,4)
plt.plot(d_list, topsis_scores, marker='D', c="#28a745", lw=2, ms=7)
plt.scatter(best_d[0], best_d[1], c="red", s=120, zorder=5, label=f"最优特征数={best_d[0]}")
plt.xlabel("RFE保留原始特征数量 $d'$")
plt.ylabel("TOPSIS综合得分")
plt.title("RFE多指标综合评价")
plt.legend()
plt.xticks(d_list)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---------------------- 6. 最优特征子集二维可视化（修复投影版） ----------------------
from sklearn.manifold import Isomap
best_k = best_d[0]
top_idx_best = np.argsort(fusion_importance)[-best_k:]
X_best = X_scaled[:, top_idx_best]

# 对筛选后的多维特征进行Isomap二维投影
iso_vis = Isomap(n_components=2, n_neighbors=10)
X_2d_proj = iso_vis.fit_transform(X_best)

plt.figure(figsize=(8,6))
sc = plt.scatter(X_2d_proj[:,0], X_2d_proj[:,1], c=y_real, cmap="tab10", s=15)
plt.legend(*sc.legend_elements(), title="数字类别标签")
plt.title(f"多树模型加权融合最优{best_k}个特征 Isomap二维投影可视化")
plt.xlabel("Isomap投影维度1")
plt.ylabel("Isomap投影维度2")
plt.grid(alpha=0.3)
plt.show()


# In[31]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

# ===================== 中文显示配置 =====================
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------- 1. 加载预处理数据 ----------------------
data = load_digits()
X_raw = data.data
y_real = data.target
n_cluster = len(np.unique(y_real))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
dist_high = euclidean_distances(X_scaled)
n_feat = X_scaled.shape[1]

# ---------------------- 工具函数复用 ----------------------
def compute_stress(high_dist, low_dist):
    diff = high_dist - low_dist
    numerator = np.sum(diff ** 2)
    denominator = np.sum(high_dist ** 2)
    stress = np.sqrt(numerator / denominator)
    return stress

def topsis_evaluation(matrix, benefit_idx, cost_idx, weights=None):
    m, n = matrix.shape
    norm_matrix = matrix.copy()
    for idx in cost_idx:
        col_max = np.max(norm_matrix[:, idx])
        norm_matrix[:, idx] = col_max - norm_matrix[:, idx]
    sqrt_sum = np.sqrt(np.sum(norm_matrix ** 2, axis=0))
    norm_matrix = norm_matrix / sqrt_sum
    if weights is None:
        weights = np.ones(n) / n
    weight_matrix = norm_matrix * weights
    pos_ideal = np.max(weight_matrix, axis=0)
    neg_ideal = np.min(weight_matrix, axis=0)
    d_pos = np.sqrt(np.sum((weight_matrix - pos_ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weight_matrix - neg_ideal) ** 2))
    score = d_neg / (d_pos + d_neg)
    return score

# ---------------------- 2. 定义5种树模型（已删除CatBoost） ----------------------
models = [
    ("决策树DT", DecisionTreeClassifier(random_state=42)),
    ("Bagging", BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, random_state=42)),
    ("随机森林RF", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("AdaBoost", AdaBoostClassifier(n_estimators=100, random_state=42)),
    ("GBDT", GradientBoostingClassifier(n_estimators=100, random_state=42))
]

# 存储每个模型原始特征重要性
importance_list = []
model_names = []

for name, model in models:
    print(f"正在训练 {name} ...")
    model.fit(X_scaled, y_real)
    # Bagging特殊处理：取基树重要性均值
    if name == "Bagging":
        imps = []
        for est in model.estimators_:
            imps.append(est.feature_importances_)
        feat_imp = np.mean(np.array(imps), axis=0)
    else:
        feat_imp = model.feature_importances_
    importance_list.append(feat_imp)
    model_names.append(name)

# ---------------------- 3. 统一归一化所有模型重要性（关键：全部缩放到总和=1） ----------------------
norm_importance_list = []
for imp in importance_list:
    sum_imp = np.sum(imp)
    if sum_imp > 1e-8:
        norm_importance_list.append(imp / sum_imp)
    else:
        norm_importance_list.append(imp)
imp_stack = np.vstack(norm_importance_list)
# 等权重融合
fusion_importance = np.mean(imp_stack, axis=0)

# 遍历保留特征数量 2~10
dim_candidates = list(range(2, 11))
result_dict = {}

for keep_dim in dim_candidates:
    print(f"\n===== 多树模型加权融合筛选，保留特征数 = {keep_dim} =====")
    top_idx = np.argsort(fusion_importance)[-keep_dim:]
    X_low = X_scaled[:, top_idx]

    # KMeans无监督聚类评估
    kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init="auto")
    y_pred = kmeans.fit_predict(X_low)
    sil = silhouette_score(X_low, y_pred)
    ari = adjusted_rand_score(y_real, y_pred)

    dist_low = euclidean_distances(X_low)
    stress_val = compute_stress(dist_high, dist_low)

    result_dict[keep_dim] = {
        "silhouette": sil,
        "ARI": ari,
        "Stress": stress_val
    }
    print(f"轮廓系数: {sil:.4f} | ARI: {ari:.4f} | Stress: {stress_val:.4f}")

# ---------------------- 4. TOPSIS综合评价选最优维度 ----------------------
d_list = list(result_dict.keys())
sil_list = [result_dict[d]["silhouette"] for d in d_list]
ari_list = [result_dict[d]["ARI"] for d in d_list]
stress_list = [result_dict[d]["Stress"] for d in d_list]

eval_matrix = np.column_stack([sil_list, ari_list, stress_list])
benefit_cols = [0, 1]
cost_cols = [2]
topsis_scores = topsis_evaluation(eval_matrix, benefit_cols, cost_cols)
dim_score_dict = dict(zip(d_list, topsis_scores))
best_d = max(dim_score_dict.items(), key=lambda x: x[1])

# 输出汇总表
print("\n==================== 多树模型加权融合特征筛选汇总 ====================")
for idx, d in enumerate(d_list):
    m = result_dict[d]
    s = dim_score_dict[d]
    print(f"保留特征数={d:2d} | 轮廓={m['silhouette']:.4f} | ARI={m['ARI']:.4f} | Stress={m['Stress']:.4f} | TOPSIS={s:.4f}")
print(f"\n【TOPSIS最优保留特征数量 d' = {best_d[0]}】综合得分={best_d[1]:.4f}")

# ---------------------- 5. 四指标可视化 ----------------------
plt.figure(figsize=(14, 10))
# 轮廓系数
plt.subplot(2,2,1)
plt.plot(d_list, sil_list, marker='o', c="#2E86AB", lw=2)
plt.xlabel("多模型融合筛选保留特征数量 $d'$")
plt.ylabel("轮廓系数")
plt.title("不同特征数量下轮廓系数变化")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# ARI
plt.subplot(2,2,2)
plt.plot(d_list, ari_list, marker='s', c="#A23B72", lw=2)
plt.xlabel("多模型融合筛选保留特征数量 $d'$")
plt.ylabel("调整兰德指数 ARI")
plt.title("不同特征数量下ARI变化趋势")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# Stress
plt.subplot(2,2,3)
plt.plot(d_list, stress_list, marker='^', c="#F18F01", lw=2)
plt.xlabel("多模型融合筛选保留特征数量 $d'$")
plt.ylabel("应力函数 Stress")
plt.title("距离保留应力值（越小失真越小）")
plt.xticks(d_list)
plt.grid(alpha=0.3)

# TOPSIS综合得分
plt.subplot(2,2,4)
plt.plot(d_list, topsis_scores, marker='D', c="#28a745", linewidth=2, markersize=7)
plt.scatter(best_d[0], best_d[1], c='red', s=120, zorder=5, label=f'最优特征数={best_d[0]}')
plt.xlabel("多模型融合筛选保留特征数量 $d'$")
plt.ylabel("TOPSIS综合贴近度得分")
plt.title("TOPSIS多指标综合评价得分")
plt.legend()
plt.xticks(d_list)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------- 6. 最优特征子集二维可视化（修复投影版） ----------------------
from sklearn.manifold import Isomap
best_k = best_d[0]
top_idx_best = np.argsort(fusion_importance)[-best_k:]
X_best = X_scaled[:, top_idx_best]

# 对筛选后的多维特征进行Isomap二维投影
iso_vis = Isomap(n_components=2, n_neighbors=10)
X_2d_proj = iso_vis.fit_transform(X_best)

plt.figure(figsize=(8,6))
sc = plt.scatter(X_2d_proj[:,0], X_2d_proj[:,1], c=y_real, cmap="tab10", s=15)
plt.legend(*sc.legend_elements(), title="数字类别标签")
plt.title(f"多树模型加权融合最优{best_k}个特征 Isomap二维投影可视化")
plt.xlabel("Isomap投影维度1")
plt.ylabel("Isomap投影维度2")
plt.grid(alpha=0.3)
plt.show()

# 输出全局融合Top10特征下标
top10_idx = np.argsort(fusion_importance)[-10:]
top10_idx_sorted = np.sort(top10_idx)
print("\n==================== 多模型加权融合全局Top10重要特征下标 ====================")
print(top10_idx_sorted)
print("各特征融合重要性得分：")
for idx in top10_idx_sorted:
    print(f"特征{idx:2d} 融合重要性 = {fusion_importance[idx]:.6f}")

# ---------------------- 绘制特征重要性双柱状图 ----------------------
# 提取融合后Top10特征，从高到低排序
top10_idx = np.argsort(fusion_importance)[-10:][::-1]
top10_fusion_imp = fusion_importance[top10_idx]
feat_labels = [f"特征{i}" for i in top10_idx]

plt.figure(figsize=(16, 6))
# 左图：融合综合重要性
ax1 = plt.subplot(1, 2, 1)
bars = ax1.bar(feat_labels, top10_fusion_imp, color="#28a745", width=0.6)
ax1.set_title("多树模型加权融合 Top10 特征重要性", fontsize=13)
ax1.set_xlabel("特征编号")
ax1.set_ylabel("综合融合重要性得分（所有特征总和=1）")
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis="y", alpha=0.3)
for bar in bars:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h, f"{h:.4f}", ha="center", va="bottom", fontsize=9)

# 右图：5模型分组对比（已删除CatBoost，颜色同步缩减）
ax2 = plt.subplot(1, 2, 2)
top10_model_imp = imp_stack[:, top10_idx]
x = np.arange(len(feat_labels))
width = 0.15
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

for i, (model_name, c) in enumerate(zip(model_names, colors)):
    offset = width * (i - 2)
    ax2.bar(x + offset, top10_model_imp[i], width, label=model_name, color=c)

ax2.set_title("5种树模型 Top10特征 单模型归一化重要性对比", fontsize=13)
ax2.set_xlabel("特征编号")
ax2.set_ylabel("单模型归一化特征重要性（单模型总和=1）")
ax2.set_xticks(x)
ax2.set_xticklabels(feat_labels, rotation=45)
ax2.legend(fontsize=8)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# In[ ]:


#绘图


# In[7]:


import matplotlib.pyplot as plt
import numpy as np

# 中文显示配置
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False
# ---------------------- 1. 完整数据集 ----------------------
# 线性模型
linear_data = {
    "name": ["LDA", "PCA", "SVD", "MDS", "Lasso", "RFE"],
    "ari": [0.9162, 0.6505, 0.5993, 0.5498, 0.4586, 0.5505],
    "topsis": [0.9901, 0.9089, 0.9265, 0.6764, 0.8086, 0.8121],
    "dim": [9, 9, 9, 9, 8, 8]
}
# 非线性模型
nonlinear_data = {
    "name": ["t-SNE", "LLE", "KPCA", "ISOMAP", "融合树", "互信息"],
    "ari": [0.8901, 0.4878, 0.0048, 0.4833, 0.4369, 0.5165],
    "topsis": [0.9931, 0.9021, 0.8646, 0.8347, 0.8120, 0.8044],
    "dim": [2, 10, 6, 2, 8, 10]
}

# ---------------------- 2. 绘图 ----------------------
plt.figure(figsize=(12, 7), dpi=120)

# 绘制线性点（蓝色）
sc1 = plt.scatter(
    linear_data["ari"], linear_data["topsis"],
    s=np.array(linear_data["dim"]) * 18,
    c="#1f77b4", alpha=0.7, label="线性算法"
)
# 绘制非线性点（红色）
sc2 = plt.scatter(
    nonlinear_data["ari"], nonlinear_data["topsis"],
    s=np.array(nonlinear_data["dim"]) * 18,
    c="#d62728", alpha=0.7, label="非线性算法"
)

# 标注每个点模型名称
for i, txt in enumerate(linear_data["name"]):
    plt.text(linear_data["ari"][i]+0.006, linear_data["topsis"][i], txt, fontsize=8)
for i, txt in enumerate(nonlinear_data["name"]):
    plt.text(nonlinear_data["ari"][i]+0.006, nonlinear_data["topsis"][i], txt, fontsize=8)

# 图表配置
plt.xlabel("ARI（聚类匹配精度）", fontsize=11)
plt.ylabel("TOPSIS综合得分", fontsize=11)
plt.title("线性/非线性算法 ARI-TOPSIS 双指标散点图（圆点大小=最优维度）", fontsize=13)
plt.grid(alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# In[8]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# 中文设置
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# ========== 数据集 ==========
# 无监督模型
unsup_name = ["PCA", "SVD", "KPCA", "MDS", "ISOMAP", "LLE", "t-SNE"]
unsup_ari = [0.6505, 0.5993, 0.0048, 0.5498, 0.4833, 0.4878, 0.8901]
unsup_topsis = [0.9089, 0.9265, 0.8646, 0.6764, 0.8347, 0.9021, 0.9931]
unsup_dim = [9, 9, 6, 9, 2, 10, 2]

# 有监督模型
sup_name = ["LDA", "RFE", "Lasso", "互信息", "融合树"]
sup_ari = [0.9162, 0.5505, 0.4586, 0.5165, 0.4369]
sup_topsis = [0.9901, 0.8121, 0.8086, 0.8044, 0.8120]
sup_dim = [9, 8, 8, 10, 8]

# ========== 绘图 ==========
plt.figure(figsize=(14, 8), dpi=120)
# 无监督：蓝色
sc1 = plt.scatter(unsup_ari, unsup_topsis, s=np.array(unsup_dim)*18, c="#1f77b4", alpha=0.7)
# 有监督：橙色
sc2 = plt.scatter(sup_ari, sup_topsis, s=np.array(sup_dim)*18, c="#ff7f0e", alpha=0.7)

# 标注模型名称
for i, txt in enumerate(unsup_name):
    plt.text(unsup_ari[i]+0.007, unsup_topsis[i], txt, fontsize=9)
for i, txt in enumerate(sup_name):
    plt.text(sup_ari[i]+0.007, sup_topsis[i], txt, fontsize=9)

# 坐标轴与标题
plt.xlabel("ARI（聚类匹配精度）", fontsize=12)
plt.ylabel("TOPSIS综合贴近度得分", fontsize=12)
plt.title("有监督/无监督降维算法 ARI-TOPSIS 双指标散点图（圆点大小=最优维度）", fontsize=14)
plt.grid(alpha=0.3)

# 图例
legend = [
    Patch(color="#1f77b4", label="无监督降维算法"),
    Patch(color="#ff7f0e", label="有监督降维算法")
]
plt.legend(handles=legend, loc="lower right", fontsize=11)
plt.tight_layout()
plt.show()


# In[ ]:




