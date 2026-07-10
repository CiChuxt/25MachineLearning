# 处理后数据说明

- 数据来源：UCI Optical Recognition of Handwritten Digits 数据；实验中通过 `sklearn.datasets.load_digits` 读取其内置版本。
- 抽样方式：固定 `random_state=42` 进行分层抽样，共 200 个样本。由于使用近似分层抽样，各类别数量并非完全相同；当前标签计数为 `[20, 20, 20, 21, 20, 20, 20, 20, 19, 20]`。
- 特征构造：原始 8×8 灰度块图像先进行归一化，再通过三次插值重采样为 28×28 图像，最后展平为 784 维特征。
- 文件 `digits_28x28_stratified_200.npz`：包含 `X`（200×784 的归一化图像向量）、`y`（数字标签）和 `original_indices`（原始样本索引）。
- 文件 `embeddings_all_2d.npz`：包含各降维方法得到的二维嵌入坐标。
- 文件 `metrics_all.csv`：包含各降维方法的评价指标结果。
- 文件 `run_config.json` 与 `environment.json`：分别记录运行参数和环境信息。
