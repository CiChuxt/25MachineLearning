# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 22:10:43 2026

@author: HP
"""

"""
SAS代码：

回归分析：
proc reg data=Work.Hbdata;
model y=x1-x11;
run;

主成分分析+回归分析
proc princomp data=Work.Hbdata out=prin;
var x1-x11;
proc sort;
by prin1;
proc print;
id date;
var prin1 prin2 prin3;
proc sort;
by prin2;
proc print;
id date;
var prin1 prin2 prin3;
proc sort;
by prin3;
proc print;
id date;
var prin1 prin2 prin3;
run;
proc reg data=prin;
model y=prin1 prin2 prin3;
run;
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
def load_and_preprocess_data():
    file_path = "D:/python/pca/data.xlsx"
    # 读取Excel数据
    df = pd.read_excel(file_path)
    # 跳过首行单位说明行
    df = df.iloc[1:].reset_index(drop=True)
    # 批量转换数据类型
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # 设置年份为索引
    df.set_index('年份', inplace=True)
    df.index = df.index.astype(int)
    y_col = "地区生产总值(Y)"
    x_cols = [
        "人口总数(X1)", "第三产业(X2)", "平均每人可支配收入(X3)",
        "固定资产投资额(X4)", "进出口总额(X5)", "地方一般公共预算收入(X6)",
        "地方一般公共预算支出(X7)", "工业企业单位数(X8)", "社会消费品零售总额(X9)",
        "卫生人员数(X10)", "普通高等学校毕业生人数(X11)"
    ]
    X = df[x_cols]
    Y = df[[y_col]]
    
    print("="*60)
    print(f"时间范围: {df.index.min()}年 - {df.index.max()}年")
    print(f"自变量维度: {X.shape}")
    print(f"因变量维度(GDP): {Y.shape}")
    print("\n11项PCA自变量列表:")
    for i, col in enumerate(x_cols, 1):
        print(f"  {i}. {col}")
    print("="*60)
    return X, Y, df.index.tolist(), x_cols
def standardize_data(X):
    """
    对自变量数据标准化处理，消除量纲影响
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\n数据标准化完成（仅自变量）")
    print(f"标准化后数据均值: {np.round(X_scaled.mean(axis=0), 4)}")
    print(f"标准化后数据标准差: {np.round(X_scaled.std(axis=0), 4)}")
    return X_scaled, scaler
def manual_pca(X, n_components=None):
    # 计算协方差矩阵（标准化数据等价于相关矩阵）
    cov_matrix = np.cov(X.T)
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # 消除复数误差
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors) 
    # 按特征值从大到小排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # 计算方差贡献率
    total_variance = np.sum(eigenvalues)
    variance_ratio = eigenvalues / total_variance
    cumulative_variance_ratio = np.cumsum(variance_ratio)
    
    #选取前2个主成分（累计贡献率97.29%）
    if n_components is None:
        n_components = 2
        print(f"选取前2个主成分，累计贡献率: {cumulative_variance_ratio[1]:.2%}")
    # 输出完整主成分结果
    print("\n特征值与方差贡献率:")
    print("-"*60)
    print(f"{'主成分':<8}{'特征值':<12}{'方差贡献率':<12}{'累计贡献率':<12}")
    print("-"*60)
    for i in range(len(eigenvalues)):
        print(f"PC{i+1:<7}{eigenvalues[i]:<12.4f}{variance_ratio[i]:<12.4f}{cumulative_variance_ratio[i]:<12.4f}")
    
    # 选取前n_components个特征向量
    components = eigenvectors[:, :n_components]
    # 投影到主成分空间
    X_pca = np.dot(X, components) 
    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'variance_ratio': variance_ratio,
        'cumulative_variance_ratio': cumulative_variance_ratio,
        'components': components,
        'X_pca': X_pca,
        'n_components': n_components,
        'cov_matrix': cov_matrix
    }
def pca_regression(pca_result, Y):
    X_pca = pca_result['X_pca']
    # 构建线性回归模型
    lr_model = LinearRegression()
    lr_model.fit(X_pca, Y)
    # 提取回归参数
    intercept = lr_model.intercept_[0]
    coef = lr_model.coef_[0]
    r2 = lr_model.score(X_pca, Y)
    # 输出回归结果
    print("\n" + "="*60)
    print("主成分回归实证结果")
    print("="*60)
    print(f"模型截距项: {intercept:.4f}")
    print(f"第一主成分(PC1)系数: {coef[0]:.4f}")
    print(f"第二主成分(PC2)系数: {coef[1]:.4f}")
    print(f"模型拟合优度R²: {r2:.4f}")
    print(f"\n最终回归方程：Y = {intercept:.2f} + {coef[0]:.2f}*PC1 + {coef[1]:.2f}*PC2")
    print("="*60)
    return lr_model, intercept, coef, r2
def plot_scree_plot(pca_result, save_path='scree_plot.png'):
    """
    绘制碎石图（方差贡献率+累计贡献率）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    n = len(pca_result['variance_ratio'])
    
    # 方差贡献率碎石图
    ax1.plot(range(1, n+1), pca_result['variance_ratio'], 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('主成分编号', fontsize=12)
    ax1.set_ylabel('方差贡献率', fontsize=12)
    ax1.set_title('方差贡献率碎石图', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, n+1))
    
    # 累计方差贡献率图
    ax2.plot(range(1, n+1), pca_result['cumulative_variance_ratio'], 'ro-', linewidth=2, markersize=8)
    ax2.axhline(y=0.85, color='gray', linestyle='--', label='85%阈值')
    ax2.axhline(y=0.97, color='orange', linestyle='--', label='97.29%阈值')
    ax2.set_xlabel('主成分编号', fontsize=12)
    ax2.set_ylabel('累计方差贡献率', fontsize=12)
    ax2.set_title('累计方差贡献率图', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(range(1, n+1))
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n碎石图已保存至: {save_path}")
    plt.close()
def calculate_loadings(pca_result, feature_names):
    """
    计算主成分载荷矩阵：载荷 = 特征向量 × sqrt(特征值)
    用于解释各指标对主成分的贡献程度
    """
    n = pca_result['n_components']
    loadings = np.zeros((len(feature_names), n))
    
    for i in range(n):
        loadings[:, i] = pca_result['eigenvectors'][:, i] * np.sqrt(pca_result['eigenvalues'][i])
    
    loadings_df = pd.DataFrame(
        loadings,
        index=feature_names,
        columns=[f'PC{i+1}' for i in range(n)]
    )
    
    print("\n" + "="*60)
    print("主成分载荷矩阵")
    print("="*60)
    print(loadings_df.round(4))
    
    return loadings_df
def calculate_comprehensive_scores(pca_result, years):
    """
    计算各年份综合得分与排名
    """
    n = pca_result['n_components']
    weights = pca_result['variance_ratio'][:n]
    weights = weights / weights.sum()  # 权重归一化
    
    scores = pca_result['X_pca']
    comprehensive_scores = np.dot(scores, weights)
    
    scores_df = pd.DataFrame(scores, index=years, columns=[f'PC{i+1}得分' for i in range(n)])
    scores_df['综合得分'] = comprehensive_scores
    scores_df['排名'] = scores_df['综合得分'].rank(ascending=False).astype(int)
    
    scores_ranked = scores_df.sort_values('综合得分', ascending=False)
    
    print("\n" + "="*60)
    print("各年份经济发展综合得分与排名")
    print("="*60)
    print(scores_ranked.round(4))
    
    return scores_df, scores_ranked
def plot_scores_trend(scores_df, save_path='scores_trend.png'):
    """
    绘制湖北省经济发展综合得分趋势图
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    years = scores_df.index
    ax.plot(years, scores_df['综合得分'], 'b-o', linewidth=2.5, markersize=6, label='综合得分')
    
    # 数值标签
    for x, y in zip(years, scores_df['综合得分']):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('综合得分', fontsize=12)
    ax.set_title('湖北省经济发展水平综合得分趋势图（2001-2021）', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n综合得分趋势图已保存至: {save_path}")
    plt.close()
def plot_pca_2d_visualization(pca_result, years, save_path='pca_2d.png'):
    """
    绘制PCA二维降维可视化图
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    X_pca = pca_result['X_pca']
    
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=range(len(years)), 
                        cmap='viridis', s=100, alpha=0.8, edgecolors='k')
    
    # 年份标签
    for i, year in enumerate(years):
        ax.annotate(str(year), (X_pca[i, 0], X_pca[i, 1]), 
                   textcoords="offset points", xytext=(5, 5), fontsize=9)
    
    ax.set_xlabel(f'PC1 ({pca_result["variance_ratio"][0]*100:.2f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca_result["variance_ratio"][1]*100:.2f}%)', fontsize=12)
    ax.set_title('湖北省经济发展PCA二维降维可视化（2001-2021）', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('年份顺序', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPCA二维可视化图已保存至: {save_path}")
    plt.close()
def plot_loadings_heatmap(loadings_df, save_path='loadings_heatmap.png'):
    """
    绘制主成分载荷热力图
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(loadings_df.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xticks(range(len(loadings_df.columns)))
    ax.set_xticklabels(loadings_df.columns, fontsize=11)
    ax.set_yticks(range(len(loadings_df.index)))
    ax.set_yticklabels(loadings_df.index, fontsize=10)
    
    # 显示载荷数值
    for i in range(len(loadings_df.index)):
        for j in range(len(loadings_df.columns)):
            text = ax.text(j, i, f'{loadings_df.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('主成分载荷热力图', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im)
    cbar.set_label('载荷系数', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n主成分载荷热力图已保存至: {save_path}")
    plt.close()
def main():
    """
    主函数：完整PCA+主成分回归分析流程
    """
    print("="*60)
    print("基于主成分分析的湖北省经济影响因素实证分析")
    print("="*60)
    
    # 1. 数据加载与变量拆分
    X, Y, years, feature_names = load_and_preprocess_data()
    
    # 2. 自变量标准化
    X_scaled, scaler = standardize_data(X)
    
    # 3. 手动PCA分析（固定选取前2个主成分，匹配论文结果）
    pca_result = manual_pca(X_scaled)
    
    # 4. 绘制碎石图
    plot_scree_plot(pca_result)
    
    # 5. 计算载荷矩阵
    loadings_df = calculate_loadings(pca_result, feature_names)
    
    # 6. 绘制载荷热力图
    plot_loadings_heatmap(loadings_df)
    
    # 7. 主成分回归分析（论文核心实证步骤）
    lr_model, intercept, coef, r2 = pca_regression(pca_result, Y)
    
    # 8. 计算综合得分与排名
    scores_df, scores_ranked = calculate_comprehensive_scores(pca_result, years)
    
    # 9. 绘制可视化图表
    plot_scores_trend(scores_df)
    plot_pca_2d_visualization(pca_result, years)
    
    # 10. 批量保存所有实证结果至Excel
    with pd.ExcelWriter('湖北省经济PCA实证结果.xlsx') as writer:
        # 原始标准化自变量数据
        pd.DataFrame(X_scaled, index=years, columns=feature_names).to_excel(writer, sheet_name='标准化自变量数据')
        
        # 主成分特征值、贡献率
        variance_df = pd.DataFrame({
            '特征值': pca_result['eigenvalues'],
            '方差贡献率': pca_result['variance_ratio'],
            '累计方差贡献率': pca_result['cumulative_variance_ratio']
        }, index=[f'PC{i+1}' for i in range(len(pca_result['eigenvalues']))])
        variance_df.to_excel(writer, sheet_name='主成分方差贡献率')
        
        # 载荷矩阵
        loadings_df.to_excel(writer, sheet_name='主成分载荷矩阵')
        
        # 主成分得分
        pca_scores_df = pd.DataFrame(
            pca_result['X_pca'],
            index=years,
            columns=[f'PC{i+1}' for i in range(pca_result['n_components'])]
        )
        pca_scores_df.to_excel(writer, sheet_name='主成分得分')
        
        # 综合得分与排名
        scores_df.to_excel(writer, sheet_name='年度综合得分')
        
        # 回归结果汇总
        reg_result = pd.DataFrame({
            '参数名称': ['截距项', 'PC1系数', 'PC2系数', '拟合优度R²'],
            '数值': [intercept, coef[0], coef[1], r2]
        })
        reg_result.to_excel(writer, sheet_name='主成分回归结果', index=False)
    
    print("\n" + "="*60)
    print("全部实证分析完成！")
    print("结果文件：湖北省经济PCA实证结果.xlsx")
    print("输出图表：碎石图、载荷热力图、得分趋势图、PCA二维可视化图")
    print("="*60)
    
    return {
        'X_scaled': X_scaled,
        'pca_result': pca_result,
        'loadings_df': loadings_df,
        'scores_df': scores_df,
        'reg_result': reg_result
    }

if __name__ == '__main__':
    results = main()