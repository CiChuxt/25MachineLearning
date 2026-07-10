# 实验结果汇总

表中指标含义：Silhouette Score 越大表示类别分离越明显；Trustworthiness 越大表示低维空间越能保持高维近邻关系；SSE 越小表示 KMeans 聚类内部误差越小；运行时间越短表示效率越高。

| method | dataset | n_samples | n_features | figure | silhouette_score | trustworthiness | sse | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PCA | sklearn digits | 1200 | 64 | experiment_results/figures/pca_result.png | 0.0571 | 0.8167 | 1857.9568 | 0.0016 |
| MDS | sklearn digits | 1200 | 64 | experiment_results/figures/mds_result.png | 0.0611 | 0.8861 | 11676.9420 | 14.9477 |
| KPCA | sklearn digits | 1200 | 64 | experiment_results/figures/kpca_result.png | 0.0742 | 0.7942 | 10.2336 | 0.1900 |
| Isomap | sklearn digits | 1200 | 64 | experiment_results/figures/isomap_result.png | 0.1691 | 0.8353 | 17738.1939 | 0.4353 |
| LLE | sklearn digits | 1200 | 64 | experiment_results/figures/lle_result.png | 0.1766 | 0.8279 | 0.0631 | 0.0983 |
| t-SNE | sklearn digits | 1200 | 64 | experiment_results/figures/t_sne_result.png | 0.3971 | 0.9811 | 56970.5430 | 1.9253 |
| UMAP | sklearn digits | 1200 | 64 | experiment_results/figures/umap_result.png | 0.5359 | 0.9718 | 1161.8651 | 8.6761 |
