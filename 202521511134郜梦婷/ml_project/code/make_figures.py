from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data'/'processed'
FIG = ROOT/'figures'
FIG.mkdir(exist_ok=True)
z = np.load(OUT/'digits_28x28_stratified_200.npz')
y = z['y']
emb = {}
for fn in ['embeddings_base.npz','embeddings_tsne.npz','embeddings_umap.npz']:
    path=OUT/fn
    if path.exists():
        emb.update({k:v for k,v in np.load(path).items()})
if 'tsne' in emb: emb['t-SNE']=emb.pop('tsne')
if 'umap' in emb: emb['UMAP']=emb.pop('umap')
order=['PCA','MDA','KPCA','Isomap','t-SNE','UMAP']
emb={k:emb[k] for k in order if k in emb}
rows=[]
for fn in ['metrics_base.csv','metrics_tsne.csv','metrics_umap.csv']:
    path=OUT/fn
    if path.exists(): rows.append(pd.read_csv(path))
metrics=pd.concat(rows,ignore_index=True).set_index('Method').loc[list(emb)].reset_index()
metrics.to_csv(OUT/'metrics_all.csv',index=False)
metrics.to_csv(FIG/'table4_metrics.csv',index=False)
np.savez_compressed(OUT/'embeddings_all_2d.npz',**{k:v.astype(np.float32) for k,v in emb.items()})

# Figure 2: 2D embeddings
fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.4))
for ax,(name,Y) in zip(axes.flat,emb.items()):
    sc=ax.scatter(Y[:,0],Y[:,1],c=y,cmap='tab10',s=28,alpha=.84,linewidths=.12,edgecolors='white')
    ax.set_title(name,fontsize=15,pad=8)
    ax.set_xlabel('Dimension 1',fontsize=11)
    ax.set_ylabel('Dimension 2',fontsize=11)
    ax.grid(alpha=.15,linewidth=.5)
for ax in axes.flat[len(emb):]: ax.axis('off')
fig.colorbar(sc,ax=axes.ravel().tolist(),ticks=list(range(10)),fraction=.025,pad=.02,label='Digit label')
fig.suptitle('Two-dimensional embeddings of 200 stratified 784-dimensional digit vectors',fontsize=16,y=.995)
fig.tight_layout(rect=[0,0,.94,.97])
fig.savefig(FIG/'fig2_embedding_comparison.png',dpi=260,bbox_inches='tight')
plt.close(fig)

# Figure 3: the three primary score columns
fig, ax = plt.subplots(figsize=(10.5,5.1))
x=np.arange(len(metrics)); w=.24
ax.bar(x-w,metrics['Trustworthiness@10'],w,label='Trustworthiness@10')
ax.bar(x,metrics['2D label purity@10'],w,label='2D label purity@10')
ax.bar(x+w,metrics['Silhouette (labels)'],w,label='Silhouette')
ax.set_xticks(x); ax.set_xticklabels(metrics['Method'],fontsize=11)
ax.set_ylabel('Score',fontsize=12); ax.set_ylim(-.08,1.05)
ax.set_title('Local structure preservation and label separability',fontsize=15)
ax.grid(axis='y',alpha=.2); ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.14),frameon=False)
fig.tight_layout()
fig.savefig(FIG/'fig3_metric_comparison.png',dpi=260,bbox_inches='tight')
plt.close(fig)

# Figure 4: runtime on log axis
fig, ax=plt.subplots(figsize=(9,4.6))
ax.bar(metrics['Method'],metrics['Runtime (s)'])
ax.set_yscale('log'); ax.set_ylabel('Runtime (s, log scale)',fontsize=12)
ax.set_title('Observed runtime under the same environment',fontsize=15)
ax.grid(axis='y',alpha=.2,which='both')
for i,v in enumerate(metrics['Runtime (s)']): ax.text(i,v*1.15,f'{v:.2f}',ha='center',va='bottom',fontsize=9)
fig.tight_layout(); fig.savefig(FIG/'fig4_runtime.png',dpi=260,bbox_inches='tight');plt.close(fig)

print(metrics.to_string(index=False))
