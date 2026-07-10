import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import Isomap, TSNE
import umap
import trimap
from sklearn.metrics import homogeneity_score
import time

wine = load_wine()
X = wine.data
y = wine.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def dr_pca(X_data):
    start = time.time()
    pca_model = PCA(n_components=2, random_state=42)
    embedding = pca_model.fit_transform(X_data)
    run_time = time.time() - start
    homo = homogeneity_score(y, embedding[:, 0].round(2))
    return embedding, run_time, homo

def dr_kpca(X_data):
    start = time.time()
    kpca_model = KernelPCA(n_components=2, kernel="rbf", gamma=0.1, random_state=42)
    embedding = kpca_model.fit_transform(X_data)
    run_time = time.time() - start
    homo = homogeneity_score(y, embedding[:, 0].round(2))
    return embedding, run_time, homo

def dr_lda(X_data, label):
    start = time.time()
    lda_model = LDA(n_components=2)
    embedding = lda_model.fit_transform(X_data, label)
    run_time = time.time() - start
    homo = homogeneity_score(y, embedding[:, 0].round(2))
    return embedding, run_time, homo

def dr_isomap(X_data):
    start = time.time()
    isomap_model = Isomap(n_components=2, n_neighbors=8)
    embedding = isomap_model.fit_transform(X_data)
    run_time = time.time() - start
    homo = homogeneity_score(y, embedding[:, 0].round(2))
    return embedding, run_time, homo

def dr_tsne(X_data):
    start = time.time()
    tsne_model = TSNE(n_components=2, perplexity=20, random_state=42)
    embedding = tsne_model.fit_transform(X_data)
    run_time = time.time() - start
    homo = homogeneity_score(y, embedding[:, 0].round(2))
    return embedding, run_time, homo

def dr_umap(X_data):
    start = time.time()
    umap_model = umap.UMAP(n_components=2, n_neighbors=10, random_state=42)
    embedding = umap_model.fit_transform(X_data)
    run_time = time.time() - start
    homo = homogeneity_score(y, embedding[:, 0].round(2))
    return embedding, run_time, homo

def dr_trimap(X_data):
    start = time.time()
    tm_model = trimap.TRIMAP(n_dims=2)
    embedding = tm_model.fit_transform(X_data)
    run_time = time.time() - start
    homo = homogeneity_score(y, embedding[:, 0].round(2))
    return embedding, run_time, homo

#运行所有降维算法，存储结果
dr_all = {
    "PCA": dr_pca,
    "KPCA(RBF核)": dr_kpca,
    "LDA(监督线性判别)": dr_lda,
    "Isomap全局流形": dr_isomap,
    "t-SNE局部可视化": dr_tsne,
    "UMAP均衡拓扑": dr_umap,
    "TriMAP高速流形": dr_trimap
}
res_dict = {}
for name, func in dr_all.items():
    if name == "LDA(监督线性判别)":
        emb, t_cost, homo = func(X_scaled, y)
    else:
        emb, t_cost, homo = func(X_scaled)
    res_dict[name] = {"emb": emb, "time": t_cost, "homo": homo}


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
color = ["#e74c3c", "#27ae60", "#3498db"]
class_label = ["类别1", "类别2", "类别3"]


fig1, axes1 = plt.subplots(1, 2, figsize=(13, 7), dpi=150)
fig1.subplots_adjust(bottom=0.22)  
alg1 = ["PCA", "KPCA(RBF核)"]
for idx, name in enumerate(alg1):
    ax = axes1[idx]
    emb = res_dict[name]["emb"]
    for c in range(3):
        mask = y == c
        ax.scatter(emb[mask, 0], emb[mask, 1], c=color[c], s=20, alpha=0.7, label=class_label[c])
    ax.set_title(f"{name}", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


plt.savefig("线性降维_PCA_KPCA.png", bbox_inches="tight", dpi=150)
plt.show()


fig2, axes2 = plt.subplots(2, 3, figsize=(16, 8), dpi=150)
fig2.subplots_adjust(bottom=0.23)
axes2 = axes2.flatten()
alg2 = ["LDA(监督线性判别)", "Isomap全局流形", "t-SNE局部可视化", "UMAP均衡拓扑", "TriMAP高速流形"]
for idx, name in enumerate(alg2):
    ax = axes2[idx]
    emb = res_dict[name]["emb"]
    for c in range(3):
        mask = y == c
        ax.scatter(emb[mask, 0], emb[mask, 1], c=color[c], s=20, alpha=0.7)
    ax.set_title(f"{name}", fontsize=10)
    ax.grid(alpha=0.3)
axes2[5].axis("off")

plt.savefig("非线性流形降维对比.png", bbox_inches="tight", dpi=150)
plt.show()

print("="*60)
print(f"{'降维算法':<18}{'运行时间(秒)':<12}{'聚类同质性':<12}")
print("-"*60)
for name, info in res_dict.items():
    print(f"{name:<18}{info['time']:<12.3f}{info['homo']:.3f}")
print("="*60)