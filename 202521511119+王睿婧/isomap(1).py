# 导入全部依赖库
import os
print("当前工作目录:", os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd   # 用于导出 CSV

# Windows专用中文配置，仅使用系统自带黑体，不会报字体缺失
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

# ===================== 模块1：Swiss Roll仿真流形数据集实验 =====================
print("========== 一、Swiss Roll 仿真数据集加载与ISOMAP降维 ==========")
# 1. 生成数据集
n_samples = 1500
noise = 0.05
X_swiss, color_label = datasets.make_swiss_roll(n_samples=n_samples, noise=noise, random_state=42)
print(f"Swiss Roll数据集形状：样本数{X_swiss.shape[0]}, 原始维度{X_swiss.shape[1]}")
print(f"连续色度标签shape：{color_label.shape}")

# 2. 标准化预处理
scaler = StandardScaler()
X_swiss_scaled = scaler.fit_transform(X_swiss)

# 3. ISOMAP模型训练
isomap_swiss = manifold.Isomap(n_neighbors=12, n_components=2)
Y_swiss_isomap = isomap_swiss.fit_transform(X_swiss_scaled)
recon_err_swiss = isomap_swiss.reconstruction_error()
print(f"Swiss Roll ISOMAP重构误差 J_ISOMAP = {recon_err_swiss:.4f}")

# 4. PCA对照
pca_swiss = PCA(n_components=2)
Y_swiss_pca = pca_swiss.fit_transform(X_swiss_scaled)

# 5. 绘图
fig1 = plt.figure(figsize=(18, 5))
ax1 = fig1.add_subplot(1, 3, 1, projection="3d")
ax1.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2], c=color_label, cmap=plt.cm.Spectral, s=7)
ax1.set_title("原始3维Swiss Roll卷曲流形", fontsize=12)

ax2 = fig1.add_subplot(1, 3, 2)
ax2.scatter(Y_swiss_isomap[:, 0], Y_swiss_isomap[:, 1], c=color_label, cmap=plt.cm.Spectral, s=7)
ax2.set_title(f"ISOMAP降维(k=12) 重构误差={recon_err_swiss:.4f}", fontsize=12)

ax3 = fig1.add_subplot(1, 3, 3)
ax3.scatter(Y_swiss_pca[:, 0], Y_swiss_pca[:, 1], c=color_label, cmap=plt.cm.Spectral, s=7)
ax3.set_title("PCA线性降维（流形结构扭曲）", fontsize=12)
plt.tight_layout()
plt.show()

# ===================== 模块2：UCI手写数字真实高维数据集 =====================
print("\n========== 二、UCI手写数字真实高维数据集加载与ISOMAP降维 ==========")
digits = datasets.load_digits()
X_digit = digits.data
y_digit = digits.target
print(f"UCI手写数字数据集：样本数{X_digit.shape[0]}, 原始高维特征维度{X_digit.shape[1]}")
print(f"分类标签类别数：{len(np.unique(y_digit))}")

X_digit_scaled = scaler.fit_transform(X_digit)

isomap_digit = manifold.Isomap(n_neighbors=10, n_components=2)
Y_digit_isomap = isomap_digit.fit_transform(X_digit_scaled)
recon_err_digit = isomap_digit.reconstruction_error()
sil_score = silhouette_score(Y_digit_isomap, y_digit)
print(f"UCI数据集ISOMAP重构误差 J_ISOMAP = {recon_err_digit:.4f}")
print(f"ISOMAP降维后轮廓系数(聚类效果) = {sil_score:.4f}")

pca_digit = PCA(n_components=2)
Y_digit_pca = pca_digit.fit_transform(X_digit_scaled)
pca_sil = silhouette_score(Y_digit_pca, y_digit)
print(f"PCA降维后轮廓系数 = {pca_sil:.4f}")

# 绘图
fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))
sc1 = ax_a.scatter(Y_digit_isomap[:, 0], Y_digit_isomap[:, 1], c=y_digit, cmap="tab10", s=12)
ax_a.set_title(f"UCI手写数字 ISOMAP降维2维\n重构误差={recon_err_digit:.4f},轮廓系数={sil_score:.4f}", fontsize=12)
plt.colorbar(sc1, ax=ax_a, label="数字标签 0~9")

sc2 = ax_b.scatter(Y_digit_pca[:, 0], Y_digit_pca[:, 1], c=y_digit, cmap="tab10", s=12)
ax_b.set_title(f"UCI手写数字 PCA降维2维\n轮廓系数={pca_sil:.4f}", fontsize=12)
plt.colorbar(sc2, ax=ax_b, label="数字标签 0~9")
plt.tight_layout()
plt.show()

# ===================== 输出模型核心参数 =====================
print("\n========== ISOMAP模型核心参数输出 ==========")
print("【Swiss Roll实验模型参数】")
print(f"近邻超参 k = {isomap_swiss.n_neighbors}")
print(f"降维目标维度 d = {isomap_swiss.n_components}")
print(f"测地线距离矩阵尺寸：{isomap_swiss.dist_matrix_.shape}")

print("\n【UCI手写数字实验模型参数】")
print(f"近邻超参 k = {isomap_digit.n_neighbors}")
print(f"降维目标维度 d = {isomap_digit.n_components}")
print(f"测地线距离矩阵尺寸：{isomap_digit.dist_matrix_.shape}")

# ===================== 导出降维结果为 CSV =====================
print("\n========== 导出 CSV 文件 ==========")

# 1. Swiss Roll 数据（ISOMAP 降维结果 + 颜色标签）
df_swiss = pd.DataFrame({
    'component_1': Y_swiss_isomap[:, 0],
    'component_2': Y_swiss_isomap[:, 1],
    'color_label': color_label
})
df_swiss.to_csv('swiss_roll_isomap.csv', index=False)
print("已保存：swiss_roll_isomap.csv")

# 2. UCI 手写数字数据（ISOMAP 降维结果 + 数字标签）
df_digit = pd.DataFrame({
    'component_1': Y_digit_isomap[:, 0],
    'component_2': Y_digit_isomap[:, 1],
    'digit_label': y_digit
})
df_digit.to_csv('digits_isomap.csv', index=False)
print("已保存：digits_isomap.csv")

print("\n所有 CSV 文件已导出至当前工作目录。")