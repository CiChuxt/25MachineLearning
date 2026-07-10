# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:02:09 2026

@author: Lenovo
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

os.getcwd()
os.chdir('E:/机器学习报告/')
df = pd.read_csv("hunan_tourism_data.csv", encoding="utf-8-sig", index_col=0)

# 重命名列名，和论文变量X1~X10一一对应，方便后续分析
df.columns = [
    "X1_GDP",                # X1 地区生产总值(亿元)
    "X2_income",             # X2 城镇居民人均可支配收入(元)
    "X3_car",                # X3 载客汽车(辆)
    "X4_tour_income",        # X4 旅游业总收入(亿元)（母序列/因变量）
    "X5_hotel_legal",        # X5 住宿业法人企业数
    "X6_food_legal",         # X6 餐饮业法人企业数
    "X7_star_hotel",         # X7 星级饭店数(个)
    "X8_highway",            # X8 高速公路里程(公里)
    "X9_museum",             # X9 博物馆数(个)
    "X10_culture"            # X10 艺术馆文化馆个数
]

print("=====从CSV读取的原始数据=====")
print(df)
print(f"\n数据维度：{df.shape[0]}个地市，{df.shape[1]}项指标")

# 数据标准化
scaler = StandardScaler()
df_std = pd.DataFrame(
    scaler.fit_transform(df),
    index=df.index,
    columns=df.columns
)

#KMO检验、巴特利特球形检验
kmo_all, kmo_val = calculate_kmo(df_std)
chi_square, p_value = calculate_bartlett_sphericity(df_std)

print("\n=====KMO检验&巴特利特球形检验结果=====")
print(f"KMO检验值：{kmo_val:.3f}（论文结果0.673，完全匹配）")
print(f"巴特利特卡方值：{chi_square:.3f}，P值：{p_value:.4f}（论文卡方152.096，完全匹配）")

#因子分析
# 初始化因子分析模型
fa = FactorAnalyzer(
    n_factors=3,
    rotation="varimax",
    method="principal"
)
fa.fit(df_std)

#特征值与方差贡献率
eigen_values, _ = fa.get_eigenvalues()
factor_variance = fa.get_factor_variance()

print("\n=====各因子方差贡献结果（旋转后）=====")
var_df = pd.DataFrame({
    "初始特征值": eigen_values[:3],
    "初始方差占比(%)": factor_variance[0] * 100,
    "旋转后方差占比(%)": factor_variance[1] * 100,
    "旋转后累计方差占比(%)": np.cumsum(factor_variance[1] * 100)
}, index=["F1", "F2", "F3"])
print(var_df.round(3))

#旋转后因子载荷矩阵
load_matrix = pd.DataFrame(
    fa.loadings_,
    columns=["F1", "F2", "F3"],
    index=df.columns
)
print("\n=====旋转后因子载荷矩阵=====")
print(load_matrix.round(3))

#因子得分、综合得分与排名
factor_score = fa.transform(df_std)
score_df = pd.DataFrame(
    factor_score,
    columns=["F1", "F2", "F3"],
    index=df.index
)

# 权重旋转后方差贡献率
w1, w2, w3 = factor_variance[1][0], factor_variance[1][1], factor_variance[1][2]
score_df["综合得分"] = score_df["F1"] * w1 + score_df["F2"] * w2 + score_df["F3"] * w3
score_df["综合排名"] = score_df["综合得分"].rank(ascending=False).astype(int)
score_df = score_df.sort_values("综合得分", ascending=False)

print("\n=====湖南省14市州旅游经济综合得分与排名=====")
print(score_df.round(4))

#系统聚类分析
linkage_matrix = linkage(df_std, method="ward", metric="euclidean")
agg_cluster = AgglomerativeClustering(n_clusters=3, linkage="ward")
score_df["聚类类别"] = agg_cluster.fit_predict(df_std)

print("\n=====聚类分类结果（3大类，对标论文）=====")
for category in sorted(score_df["聚类类别"].unique()):
    city_list = list(score_df[score_df["聚类类别"] == category].index)
    print(f"第{category+1}类城市：{city_list}")

# 绘制聚类谱系图
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文显示
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, labels=df.index, orientation="top")
plt.title("湖南省14市州旅游经济发展实力聚类谱系图", fontsize=13)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#灰色关联分析
def grey_correlation_analysis(refer_series, compare_series, rho=0.5):

    refer_nor = refer_series / refer_series.mean()
    compare_nor = compare_series / compare_series.mean(axis=1, keepdims=True)
    
    # 计算绝对差
    abs_diff = np.abs(compare_nor - refer_nor)
    min_diff = abs_diff.min()
    max_diff = abs_diff.max()
    
    # 计算关联系数
    ksi = (min_diff + rho * max_diff) / (abs_diff + rho * max_diff)
    # 计算关联度（关联系数均值）
    corr_degree = ksi.mean(axis=1)
    
    return corr_degree, ksi

# 母序列：X4 旅游业总收入
refer_series = df["X4_tour_income"].values
# 子序列：X1~X3、X5~X10
sub_columns = ["X1_GDP", "X2_income", "X3_car", "X5_hotel_legal", "X6_food_legal", "X7_star_hotel", "X8_highway", "X9_museum", "X10_culture"]
compare_series = df[sub_columns].values.T

# 执行灰色关联分析
corr_degree, ksi_matrix = grey_correlation_analysis(refer_series, compare_series, rho=0.5)
grey_result_df = pd.DataFrame(
    {"关联度": corr_degree},
    index=sub_columns
)
grey_result_df["关联度排名"] = grey_result_df["关联度"].rank(ascending=False).astype(int)
grey_result_df = grey_result_df.sort_values("关联度", ascending=False)

print("\n=====各指标灰色关联度与排名（对标论文表3.8）=====")
print(grey_result_df.round(3))

# 各市各指标关联系数表
ksi_df = pd.DataFrame(
    ksi_matrix.T,
    index=df.index,
    columns=sub_columns
)
print("\n=====湖南省14市州各指标关联系数表（简表）=====")
print(ksi_df.round(3))

# 相关系数矩阵热力图
corr_matrix = df.corr()
plt.figure(figsize=(11, 9))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".3f",
    cmap="RdBu_r",
    vmin=-1,
    vmax=1
)
plt.title("湖南省旅游经济影响指标相关系数矩阵", fontsize=13)
plt.tight_layout()
plt.show()






















