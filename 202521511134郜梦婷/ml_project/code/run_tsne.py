from pathlib import Path
import time, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'processed'; z=np.load(OUT/'digits_28x28_stratified_200.npz'); X=StandardScaler().fit_transform(z['X']);y=z['y']
Xp=PCA(n_components=50,random_state=42).fit_transform(X)
t=time.perf_counter(); Y=TSNE(n_components=2,perplexity=30,learning_rate='auto',max_iter=750,init='pca',random_state=42,method='barnes_hut',angle=.5).fit_transform(Xp);sec=time.perf_counter()-t
idx=NearestNeighbors(n_neighbors=11).fit(Y).kneighbors(Y,return_distance=False)[:,1:];purity=(y[idx]==y[:,None]).mean()
rng=np.random.default_rng(42);a=rng.integers(0,len(y),12000);b=rng.integers(0,len(y),12000);m=a!=b;a=a[m];b=b[m]
rho=float(spearmanr(np.linalg.norm(X[a]-X[b],axis=1),np.linalg.norm(Y[a]-Y[b],axis=1)).statistic)
row={'Method':'t-SNE','Trustworthiness@10':trustworthiness(X,Y,n_neighbors=10),'2D label purity@10':purity,'Silhouette (labels)':silhouette_score(Y,y),'Global distance Spearman ρ':rho,'Runtime (s)':sec}
np.savez_compressed(OUT/'embeddings_tsne.npz',tsne=Y.astype(np.float32));pd.DataFrame([row]).to_csv(OUT/'metrics_tsne.csv',index=False);print(row)
