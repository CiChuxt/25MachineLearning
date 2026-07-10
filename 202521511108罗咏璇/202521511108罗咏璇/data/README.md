# 数据集说明

本实验默认使用 `sklearn.datasets.load_digits` 内置手写数字数据集，因此不需要额外下载原始数据文件。

数据集基本信息：

| 项目 | 内容 |
|---|---|
| 数据集名称 | sklearn digits |
| 数据来源 | scikit-learn 内置数据集 |
| 样本数量 | 1797 |
| 实验抽样数量 | 1200 |
| 图像大小 | 8 x 8 |
| 特征维度 | 64 |
| 类别数量 | 10 |
| 类别含义 | 数字 0 到 9 |

代码中通过以下方式加载数据：

```python
from sklearn.datasets import load_digits

data = load_digits()
X = data.data
y = data.target
```

本实验选择该数据集的原因是：它具有图像数据的高维特征形式，同时规模适中，适合在普通电脑上快速比较 PCA、MDS、KPCA、Isomap、LLE、t-SNE 和 UMAP 等降维方法。

