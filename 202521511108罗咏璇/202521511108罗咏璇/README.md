# 降维方法比较实验

本文件夹包含课程论文所需的实验代码和结果，默认使用 `sklearn digits` 内置手写数字数据集，比较 PCA、MDS、KPCA、Isomap、LLE、t-SNE 和 UMAP 七种降维方法。

## 运行环境

先安装依赖：

```bash
pip install -r requirements.txt
```

如果不安装 `umap-learn`，UMAP 会运行失败，但其他方法仍可正常运行。

## 一键运行

```bash
python run_all.py --dataset digits --sample-size 1200 --random-state 42
```

结果会输出到：

```text
experiment_results/
├── figures/
│   ├── pca_result.png
│   ├── mds_result.png
│   ├── kpca_result.png
│   ├── isomap_result.png
│   ├── lle_result.png
│   ├── t_sne_result.png
│   └── umap_result.png
└── results/
    ├── metrics.csv
    └── summary.md
```

## 单独运行

```bash
python pca.py
python mds.py
python kpca.py
python isomap.py
python lle.py
python tsne.py
python umap_method.py
```

## 当前结果说明

当前 `experiment_results` 中已经生成了 PCA、MDS、KPCA、Isomap、LLE、t-SNE 和 UMAP 七种方法的完整结果。图像文件保存在 `experiment_results/figures/`，量化指标保存在 `experiment_results/results/metrics.csv`。

从当前结果看，UMAP 的 Silhouette Score 最高，t-SNE 的 Trustworthiness 最高，PCA 运行速度最快，MDS 运行时间最长。这些结论可以直接写入论文的结果分析部分。
