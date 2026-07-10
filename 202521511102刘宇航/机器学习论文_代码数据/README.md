# 基于PCA与t-SNE降维的手写数字识别性能对比研究
## 代码与数据说明

### 文件夹结构
```
机器学习论文_代码数据/
├── code/                    # 代码目录
│   └── main_experiment.py   # 主实验程序
├── data/                    # 数据与结果目录
│   ├── dataset_info.csv     # 数据集基本信息
│   ├── experiment_results.csv  # 分类实验结果
│   └── reduction_time.csv   # 降维算法耗时统计
└── figures/                 # 图表目录
    ├── pca_variance_explained.png    # PCA方差解释率曲线
    ├── pca_2d_visualization.png      # PCA二维可视化结果
    ├── tsne_2d_visualization.png     # t-SNE二维可视化结果
    ├── accuracy_comparison.png       # 分类准确率对比柱状图
    └── time_comparison.png           # 计算时间对比图
```

### 运行环境要求
- Python 3.7+
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

### 安装依赖
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### 运行方式
```bash
cd code
python main_experiment.py
```

### 数据集说明
本实验使用 scikit-learn 内置的手写数字数据集（Digits Dataset）：
- 样本数量：1797个
- 特征维度：64维（8×8像素灰度图像展开）
- 类别数量：10类（数字0-9）
- 数据来源：scikit-learn.datasets.load_digits()
- 原始数据集来源：UCI Machine Learning Repository

### 实验内容
1. **数据预处理**：标准化处理，按7:3划分训练集和测试集
2. **基准实验**：原始64维数据上使用SVM、KNN、随机森林进行分类
3. **PCA降维实验**：将数据降至2维，进行可视化和分类
4. **t-SNE降维实验**：将数据降至2维，进行可视化和分类
5. **对比分析**：准确率、计算时间、可视化效果对比

### 降维方法
- **PCA（主成分分析）**：线性降维方法，基于方差最大化原则
- **t-SNE（t分布随机邻域嵌入）**：非线性降维方法，基于流形学习理论
