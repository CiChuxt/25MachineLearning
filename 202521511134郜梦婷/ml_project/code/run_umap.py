from pathlib import Path
import time, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import umap
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'processed'; z=np.load(OUT/'digits_28x28_stratified_200.npz'); X=StandardScaler().fit_transform(z['X']);y=z['y']
t=time.perf_counter();Y=umap.UMAP(n_neighbors=15,min_dist=.1,n_components=2,metric='euclidean',random_state=42,n_jobs=1).fit_transform(X);sec=time.perf_counter()-t
idx=NearestNeighbors(n_neighbors=11).fit(Y).kneighbors(Y,return_distance=False)[:,1:];purity=(y[idx]==y[:,None]).mean()
rng=np.random.default_rng(42);a=rng.integers(0,len(y),12000);b=rng.integers(0,len(y),12000);m=a!=b;a=a[m];b=b[m]
rho=float(spearmanr(np.linalg.norm(X[a]-X[b],axis=1),np.linalg.norm(Y[a]-Y[b],axis=1)).statistic)
row={'Method':'UMAP','Trustworthiness@10':trustworthiness(X,Y,n_neighbors=10),'2D label purity@10':purity,'Silhouette (labels)':silhouette_score(Y,y),'Global distance Spearman ρ':rho,'Runtime (s)':sec}
np.savez_compressed(OUT/'embeddings_umap.npz',umap=Y.astype(np.float32));pd.DataFrame([row]).to_csv(OUT/'metrics_umap.csv',index=False);print(row)
