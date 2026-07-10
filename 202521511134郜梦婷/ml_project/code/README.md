# 代码运行说明

报告实验使用 `sklearn.datasets.load_digits` 提供的内置手写数字数据。代码在该数据基础上固定随机种子进行分层抽样，共得到 200 个样本；随后将每个原始 8×8 图像通过三次插值重采样为 28×28 图像，并展平为 784 维特征向量。UCI 原始数据来源链接记录在 `../data/SOURCE.md` 中。

## 环境依赖

```bash
pip install numpy pandas scipy scikit-learn matplotlib umap-learn
```

## 运行顺序

```bash
python prepare_data.py       # 生成 200×784 的处理后输入数据
```

```bash
python run_base_methods.py   # 运行 PCA、MDA、KPCA、Isomap
python run_tsne.py           # 运行 t-SNE（先使用 PCA 预降至 50 维）
python run_umap.py           # 运行 UMAP
python make_figures.py       # 生成图表与汇总指标
```

各脚本默认读取 `../data/processed/digits_28x28_stratified_200.npz`。如需从头复现实验，请先运行 `prepare_data.py`；也可以直接使用压缩包中已经提供的处理后数据文件。所有脚本固定随机种子为 42。生成的二维嵌入结果、运行配置和评价指标 CSV 文件会保存到 `../data/processed/` 目录。
