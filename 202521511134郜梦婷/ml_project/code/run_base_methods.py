from pathlib import Path
import time, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'processed'
z=np.load(OUT/'digits_28x28_stratified_200.npz'); X_raw,y=z['X'],z['y']; X=StandardScaler().fit_transform(X_raw)
def purity(Y,k=10):
 idx=NearestNeighbors(n_neighbors=k+1).fit(Y).kneighbors(Y,return_distance=False)[:,1:]
 return (y[idx]==y[:,None]).mean()
def rho(Y):
 rng=np.random.default_rng(42); a=rng.integers(0,len(y),12000); b=rng.integers(0,len(y),12000); m=a!=b;a=a[m];b=b[m]
 return float(spearmanr(np.linalg.norm(X[a]-X[b],axis=1),np.linalg.norm(Y[a]-Y[b],axis=1)).statistic)
methods={
'PCA':PCA(n_components=2,svd_solver='full',random_state=42),
'MDA':LinearDiscriminantAnalysis(solver='svd',n_components=2),
'KPCA':KernelPCA(n_components=2,kernel='rbf',gamma=1/X.shape[1],eigen_solver='arpack',random_state=42),
'Isomap':Isomap(n_neighbors=12,n_components=2,eigen_solver='arpack',n_jobs=-1),
}
emb={}; rows=[]
for name,mod in methods.items():
 t=time.perf_counter(); Y=mod.fit_transform(X,y) if name=='MDA' else mod.fit_transform(X); sec=time.perf_counter()-t
 emb[name]=Y.astype(np.float32)
 row={'Method':name,'Trustworthiness@10':trustworthiness(X,Y,n_neighbors=10),'2D label purity@10':purity(Y),'Silhouette (labels)':silhouette_score(Y,y),'Global distance Spearman ρ':rho(Y),'Runtime (s)':sec}
 rows.append(row);print(row,flush=True)
np.savez_compressed(OUT/'embeddings_base.npz',**emb)
pd.DataFrame(rows).to_csv(OUT/'metrics_base.csv',index=False)
