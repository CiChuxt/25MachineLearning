#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 导入依赖库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# 解决绘图负号、中文乱码
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 自动下载并加载手写数字数据集（无需手动下载文件，sklearn内置在线获取）
data = load_digits()
X_raw = data.data       # 原始高维特征 (1797, 64)，8×8像素展平
y_real = data.target    # 真实数字标签 0~9
n_cluster = len(np.unique(y_real))  # 聚类簇数=10


# In[2]:


data


# In[ ]:




