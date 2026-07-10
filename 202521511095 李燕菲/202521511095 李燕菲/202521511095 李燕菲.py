import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 设置中文字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#  1. 加载数据集与预处理
print("=" * 60)
print("加载 Olivetti 人脸数据集")
print("=" * 60)

faces = fetch_olivetti_faces()
X = faces.data  # 400样本 × 4096维特征
y = faces.target  # 40个类别（0-39）
img_shape = (64, 64)  # 图像原始尺寸

print(f"样本总数: {X.shape[0]}")
print(f"特征维度: {X.shape[1]}")
print(f"类别数量: {len(np.unique(y))}")
print(f"每类样本数: 10个\n")

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("数据标准化完成\n")

#  2. 全量PCA计算与方差解释率分析
print("=" * 60)
print("方差解释率分析（全量主成分）")
print("=" * 60)

# 不指定维度，计算全部主成分
pca_full = PCA()
pca_full.fit(X_scaled)

# 计算累计方差解释率
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

# 输出关键节点表格
key_points = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
print("\n表1：累计方差解释率对应主成分数量")
print("-" * 50)
print(f"{'累计方差保留率':<18} | {'所需主成分数量':<14} | {'压缩比例':<10}")
print("-" * 50)
for threshold in key_points:
    n_comp = np.argmax(cum_var >= threshold) + 1
    compress_ratio = n_comp / X.shape[1] * 100
    print(f"{threshold * 100:>10.0f}%        | {n_comp:>10}个    | {compress_ratio:>6.2f}%")
print("-" * 50)

# 打印前10个主成分的详细方差
print("\n表2：前10个主成分方差解释率详情")
print("-" * 55)
print(f"{'主成分序号':<10} | {'单一方差解释率':<16} | {'累计方差解释率':<16}")
print("-" * 55)
for i in range(10):
    single = pca_full.explained_variance_ratio_[i] * 100
    cum = cum_var[i] * 100
    print(f"第{i + 1:>2}主成分    |     {single:>6.2f}%        |     {cum:>6.2f}%")
print("-" * 55)
print()

# 绘制方差解释率曲线图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 左图：碎石图 + 单一方差解释率
ax1.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
         pca_full.explained_variance_ratio_, 'b-', linewidth=1.5)
ax1.set_xlabel('主成分序号', fontsize=11)
ax1.set_ylabel('方差解释率', fontsize=11)
ax1.set_title('碎石图：各主成分方差解释率', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 200)

# 右图：累计方差解释率曲线
ax2.plot(range(1, len(cum_var) + 1), cum_var, 'r-', linewidth=2)
for threshold in [0.7, 0.9, 0.95]:
    ax2.axhline(y=threshold, color='gray', linestyle='--', alpha=0.7)
    ax2.text(5, threshold + 0.01, f'{int(threshold * 100)}%', fontsize=10)
ax2.set_xlabel('主成分数量', fontsize=11)
ax2.set_ylabel('累计方差解释率', fontsize=11)
ax2.set_title('累计方差解释率曲线', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)
ax2.set_xlim(0, 300)

plt.tight_layout()
plt.savefig('PCA方差解释率分析.png', dpi=300, bbox_inches='tight')
print("方差解释率曲线图已保存为：PCA方差解释率分析.png")
plt.show()

#  3. 特征脸可视化
print("\n" + "=" * 60)
print("特征脸可视化")
print("=" * 60)

# 提取前12个特征脸
n_eigenfaces = 12
eigenfaces = pca_full.components_[:n_eigenfaces]

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i in range(n_eigenfaces):
    ax = axes[i]
    eigenface_img = eigenfaces[i].reshape(img_shape)
    im = ax.imshow(eigenface_img, cmap='gray')
    ax.set_title(f'第{i + 1}特征脸', fontsize=12)
    ax.axis('off')

plt.suptitle('PCA提取的特征脸（Eigenfaces）', fontsize=15, y=0.98)
plt.tight_layout()
plt.savefig('特征脸可视化.png', dpi=300, bbox_inches='tight')
print("特征脸可视化图已保存为：特征脸可视化.png")
plt.show()

#  4. 二维降维散点图
print("\n" + "=" * 60)
print("步骤4：二维降维可视化")
print("=" * 60)

# 降维到2维
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)

print(f"降维前维度: {X_scaled.shape[1]}")
print(f"降维后维度: {X_pca_2d.shape[1]}")
print(f"前2个主成分累计方差解释率: {pca_2d.explained_variance_ratio_.sum() * 100:.2f}%\n")

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='tab20',
                      alpha=0.8, s=40, edgecolors='k', linewidth=0.3)
plt.xlabel('第一主成分 (PC1)', fontsize=12)
plt.ylabel('第二主成分 (PC2)', fontsize=12)
plt.title('PCA二维降维散点图（Olivetti人脸数据集）', fontsize=14)
plt.grid(True, alpha=0.3)
plt.colorbar(scatter, label='人脸类别编号')
plt.tight_layout()
plt.savefig('PCA二维降维散点图.png', dpi=300, bbox_inches='tight')
print("二维降维散点图已保存为：PCA二维降维散点图.png")
plt.show()

#  5. 人脸重构效果对比
print("\n" + "=" * 60)
print("人脸重构效果对比")
print("=" * 60)

# 选取第0号样本作为示例
sample_idx = 0
original_face = X[sample_idx].reshape(img_shape)

# 不同维度下的重构
n_components_list = [2, 10, 30, 50, 100, 200]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

# 原始图
axes[0].imshow(original_face, cmap='gray')
axes[0].set_title('原始人脸 (4096维)', fontsize=12)
axes[0].axis('off')

# 各维度重构
for i, n_comp in enumerate(n_components_list):
    pca_temp = PCA(n_components=n_comp, random_state=42)
    X_reduced = pca_temp.fit_transform(X_scaled)
    X_reconstructed = pca_temp.inverse_transform(X_reduced)

    # 反标准化
    recon_scaled = X_reconstructed[sample_idx].reshape(1, -1)
    recon_face = scaler.inverse_transform(recon_scaled)[0].reshape(img_shape)

    ax = axes[i + 1]
    ax.imshow(recon_face, cmap='gray')
    var_ratio = sum(pca_temp.explained_variance_ratio_) * 100
    ax.set_title(f'{n_comp}个主成分\n方差保留: {var_ratio:.1f}%', fontsize=11)
    ax.axis('off')

# 最后一个位置留白
axes[-1].axis('off')

plt.suptitle('不同主成分数量下的人脸重构效果对比', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig('人脸重构效果对比.png', dpi=300, bbox_inches='tight')
print("人脸重构对比图已保存为：人脸重构效果对比.png")
plt.show()

print("\n" + "=" * 60)
print("PCA降维分析全部完成！")
print("=" * 60)
# 6. MDA（多类判别分析/LDA）降维实验
print("\n" + "=" * 60)
print("MDA多类判别分析降维实验")
print("=" * 60)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 数据集基础参数
C = len(np.unique(y))
max_mda_dim = C - 1
print(f"人脸类别总数 C = {C}，MDA理论最大可降维维度：{max_mda_dim}")

# 6.1 求解全部判别方向，计算判别特征方差贡献
lda_full = LinearDiscriminantAnalysis()
X_mda_all = lda_full.fit_transform(X_scaled, y)  # MDA必须输入标签y
mda_explained_ratio = lda_full.explained_variance_ratio_
cum_mda_var = np.cumsum(mda_explained_ratio)

# 输出MDA前10个判别分量方差详情
print("\n表3：MDA前10个判别分量方差解释率详情")
print("-" * 58)
print(f"{'判别分量序号':<12} | {'单一方差解释率':<16} | {'累计方差解释率':<16}")
print("-" * 58)
for i in range(10):
    single_mda = mda_explained_ratio[i] * 100
    cum_mda = cum_mda_var[i] * 100
    print(f"第{i + 1:>2}判别分量    |     {single_mda:>6.2f}%        |     {cum_mda:>6.2f}%")
print("-" * 58)

# 6.2 MDA二维降维散点可视化（对比PCA分类效果）
lda_2d = LinearDiscriminantAnalysis(n_components=2)
X_mda_2d = lda_2d.fit_transform(X_scaled, y)
print(f"\nMDA二维降维前两分量累计方差解释率：{sum(lda_2d.explained_variance_ratio_)*100:.2f}%")

plt.figure(figsize=(10, 8))
scatter_mda = plt.scatter(X_mda_2d[:, 0], X_mda_2d[:, 1], c=y, cmap='tab20',
                         alpha=0.8, s=40, edgecolors='k', linewidth=0.3)
plt.xlabel('第一判别分量 (LD1)', fontsize=12)
plt.ylabel('第二判别分量 (LD2)', fontsize=12)
plt.title('MDA二维降维散点图（Olivetti人脸数据集）', fontsize=14)
plt.grid(True, alpha=0.3)
plt.colorbar(scatter_mda, label='人脸类别编号')
plt.tight_layout()
plt.savefig('MDA二维降维散点图.png', dpi=300, bbox_inches='tight')
print("MDA二维降维散点图已保存为：MDA二维降维散点图.png")
plt.show()

# 6.3 MDA与PCA二维分布对比可视化
fig, (ax_pca, ax_mda) = plt.subplots(1, 2, figsize=(18, 7))
# PCA子图
ax_pca.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='tab20', alpha=0.7, s=30)
ax_pca.set_title("PCA二维投影", fontsize=13)
ax_pca.set_xlabel("PC1")
ax_pca.set_ylabel("PC2")
ax_pca.grid(alpha=0.3)
# MDA子图
ax_mda.scatter(X_mda_2d[:, 0], X_mda_2d[:, 1], c=y, cmap='tab20', alpha=0.7, s=30)
ax_mda.set_title("MDA二维投影", fontsize=13)
ax_mda.set_xlabel("LD1")
ax_mda.set_ylabel("LD2")
ax_mda.grid(alpha=0.3)

plt.suptitle("PCA与MDA二维降维样本分布对比", fontsize=15)
plt.tight_layout()
plt.savefig('PCA_MDA二维对比图.png', dpi=300, bbox_inches='tight')
print("PCA与MDA对比图已保存为：PCA_MDA二维对比图.png")
plt.show()

# 6.4 输出维度上限说明
print(f"\nMDA特性说明：仅存在{max_mda_dim}个有效判别维度，无超过该数值的分量；")
print("MDA为有监督降维，依靠类别标签优化类间/类内散度比值，侧重提升分类性能。")

print("\n" + "=" * 60)
print("PCA + MDA线性降维对比实验全部完成！")
print("=" * 60)