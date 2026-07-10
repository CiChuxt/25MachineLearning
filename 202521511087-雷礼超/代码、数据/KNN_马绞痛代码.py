import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "SimHei"

# 1. 定义数据集28列完整字段名（匹配官方Readme）
col_names = [
    "surgery","age","hospital_num","rectal_temp","pulse","respiratory_rate",
    "extremities_temp","peripheral_pulse","mucous_membrane","cap_refill_time",
    "pain_level","peristalsis","abdistension","nasogastric_gas",
    "nasogastric_reflex","nasogastric_ph","rectal_feces","abdomen_exam",
    "packed_cell_volume","total_protein","abdominocentesis_app",
    "abdominocentesis_pro","outcome","surgical_lesion","lesion_site",
    "lesion_type","lesion_subtype","cp_data"
]
# 读取原始CSV数据集
df_raw = pd.read_csv("data/horse-colic.csv", names=col_names)
df_raw = df_raw.replace("?", np.nan)
print(f"原始总样本：{len(df_raw)}")

# 绘制缺失值热力图
plt.figure(figsize=(12,7))
sns.heatmap(df_raw.isnull(), cbar=False, cmap="gray")
plt.title("数据集缺失值分布热力图")
plt.xlabel("特征字段")
plt.ylabel("样本编号")
plt.savefig("./result/missing_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# 筛选建模特征与标签
feature_cols = [
    "surgery","age","rectal_temp","pulse","respiratory_rate",
    "extremities_temp","peripheral_pulse","mucous_membrane","cap_refill_time",
    "pain_level","peristalsis","abdistension","nasogastric_gas",
    "nasogastric_reflex","nasogastric_ph","rectal_feces","abdomen_exam",
    "packed_cell_volume","total_protein","abdominocentesis_app",
    "abdominocentesis_pro","surgical_lesion"
]
label_col = "outcome"
df_model = df_raw[feature_cols + [label_col]].copy()
# 删除含缺失样本
df_clean = df_model.dropna()
# 统一转为数值
for col in df_clean.columns:
    df_clean[col] = pd.to_numeric(df_clean[col])
print(f"清洗后有效样本：{len(df_clean)}")
print("标签分布：\n", df_clean[label_col].value_counts())

# 绘制标签分布柱状图
plt.figure(figsize=(8,5))
label_count = df_clean[label_col].value_counts()
sns.barplot(x=label_count.index.astype(str), y=label_count.values)
plt.title("马匹生存结局样本分布")
plt.xlabel("标签(1存活/2死亡)")
plt.ylabel("样本数量")
plt.savefig("./result/label_dist.png", dpi=300)
plt.show()

# 划分特征、标签，分层7:3分割
X = df_clean[feature_cols]
y = df_clean[label_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"训练集{len(X_train)}条，测试集{len(X_test)}条")

# Z-score标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. 5折交叉验证筛选最优K
k_list = [3,5,7,9,11,13,15]
cv_mean = []
cv_std = []
best_k = 3
best_acc = 0.0
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, cv=5, scoring="accuracy")
    mean_s = scores.mean()
    std_s = scores.std()
    cv_mean.append(mean_s)
    cv_std.append(std_s)
    print(f"K={k} 平均准确率:{mean_s:.4f} 标准差:{std_s:.4f}")
    if mean_s > best_acc:
        best_acc = mean_s
        best_k = k
print(f"最优近邻参数 K = {best_k}")

# K值折线图
plt.figure(figsize=(10,6))
plt.plot(k_list, cv_mean, marker="o", linewidth=2, color="#2E86AB")
plt.fill_between(k_list, np.array(cv_mean)-np.array(cv_std),
                 np.array(cv_mean)+np.array(cv_std), alpha=0.2)
plt.xlabel("近邻数量 K")
plt.ylabel("5折交叉验证平均准确率")
plt.title("K候选值与模型泛化性能变化曲线")
plt.grid(True, alpha=0.3)
plt.savefig("./result/k_curve.png", dpi=300)
plt.show()

# 3. 模型评估封装函数
def model_eval(Xtr, Xte, ytr, yte, exp_name):
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(Xtr, ytr)
    y_pred = knn.predict(Xte)
    acc = accuracy_score(yte, y_pred)
    cm = confusion_matrix(yte, y_pred)
    print(f"\n========{exp_name} 模型评估========")
    print(f"测试集总体准确率：{acc:.2f}")
    print(classification_report(yte, y_pred, target_names=["存活(1)","死亡(2)"]))
    # 混淆矩阵绘图
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["预测存活","预测死亡"],
                yticklabels=["真实存活","真实死亡"])
    plt.title(f"{exp_name} 混淆矩阵热力图")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.savefig(f"./result/cm_{exp_name}.png", dpi=300)
    plt.show()
    return acc, cm

# 三组对照实验
acc_raw, cm_raw = model_eval(X_train_scaled, X_test_scaled, y_train, y_test, "原始特征KNN")
# PCA降维至8维
pca = PCA(n_components=8, random_state=42)
Xtr_pca = pca.fit_transform(X_train_scaled)
Xte_pca = pca.transform(X_test_scaled)
acc_pca, cm_pca = model_eval(Xtr_pca, Xte_pca, y_train, y_test, "PCA降维+KNN")
# KPCA高斯核降维8维
kpca = KernelPCA(n_components=8, kernel="rbf", gamma=0.1, random_state=42)
Xtr_kpca = kpca.fit_transform(X_train_scaled)
Xte_kpca = kpca.transform(X_test_scaled)
acc_kpca, cm_kpca = model_eval(Xtr_kpca, Xte_kpca, y_train, y_test, "KPCA降维+KNN")

# PCA/KPCA二维可视化对比
pca_2d = PCA(n_components=2)
X2_pca = pca_2d.fit_transform(X_train_scaled)
kpca_2d = KernelPCA(n_components=2, kernel="rbf", gamma=0.1)
X2_kpca = kpca_2d.fit_transform(X_train_scaled)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6))
sc1 = ax1.scatter(X2_pca[:,0], X2_pca[:,1], c=y_train, cmap="coolwarm", s=30)
ax1.set_title("PCA二维线性投影样本分布")
ax1.set_xlabel("主成分PC1")
ax1.set_ylabel("主成分PC2")
ax1.grid(True, alpha=0.3)
sc2 = ax2.scatter(X2_kpca[:,0], X2_kpca[:,1], c=y_train, cmap="coolwarm", s=30)
ax2.set_title("KPCA高斯核二维投影样本分布")
ax2.set_xlabel("核主成分KPC1")
ax2.set_ylabel("核主成分KPC2")
ax2.grid(True, alpha=0.3)
plt.colorbar(sc1, ax=ax1, label="1=存活 / 2=死亡")
plt.colorbar(sc2, ax=ax2, label="1=存活 / 2=死亡")
plt.tight_layout()
plt.savefig("./result/dim_compare_scatter.png", dpi=300)
plt.show()

# 输出三组准确率汇总
print("\n=====三组模型总体准确率汇总=====")
print(f"原始22维KNN：{acc_raw:.2f}")
print(f"PCA(8维)+KNN：{acc_pca:.2f}")
print(f"KPCA(RBF核8维)+KNN：{acc_kpca:.2f}")
