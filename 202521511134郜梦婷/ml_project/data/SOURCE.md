# 数据来源与预处理说明

- UCI Machine Learning Repository：**Optical Recognition of Handwritten Digits**  
  https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits

- 本实验代码实际使用的 scikit-learn 内置数据加载器：  
  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

UCI 原始设计说明中，手写数字图像被划分为 8×8 个整数像素块，每个像素块的取值范围为 0–16。本文提交的实验中，代码通过 `sklearn.datasets.load_digits` 读取其内置手写数字数据副本，并在此基础上进行固定随机种子抽样。

需要说明的是，本文实验输入并不是直接使用 UCI 官网下载文件中的全部原始样本，而是使用 scikit-learn 内置的 8×8 手写数字数据副本。随后，实验将每张 8×8 图像通过三次插值重采样为 28×28 图像，并展平为 784 维派生图像特征向量。

因此，784 维特征是本文实验流程中明确构造的派生表征，用于统一高维降维方法比较；它不表示 UCI 原始文件天然具有 784 个独立属性，也不意味着插值操作产生了新的原始观测信息。
