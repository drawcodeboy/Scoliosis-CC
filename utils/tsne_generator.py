from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap, MDS, SpectralEmbedding
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import umap

def tsne_generator(data, target, dataset_name, n_clusters):
    
    model = TSNE(n_components=2, init='random')
    # model = SpectralEmbedding(n_components=2)
    # model = MDS(n_components=2)
    # model = Isomap(n_components=2)
    # model = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
    # model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    labels = ["01st", "02nd", "03rd", "04th", "05th", "06th", "07th", "08th", "09th", "10th"]
    
    ### silhouette coefficient
    s_score = silhouette_score(data, target, sample_size=128)
    
    ### t-SNE ###
    
    # np.seterr(invalid='ignore') # during T-SNE if init is PCA
    embedded = model.fit_transform(data)
    # np.seterr(invalid='warn')
    
    target = target.reshape(-1, 1)
    
    data=np.concatenate((embedded, target), axis=1)
    
    target = [labels[int(cluster)] for cluster in target]
    
    df = pd.DataFrame({
        "Dim 1": embedded[:, 0],
        "Dim 2": embedded[:, 1],
        "Cluster": target
    })
    df_sorted = df.sort_values(by='Cluster')
    
    plt.figure(figsize=(8, 8))
    scatter = sns.scatterplot(data=df_sorted,
                              x="Dim 1",
                              y="Dim 2",
                              hue="Cluster",
                              legend='full',
                              palette=sns.color_palette('bright'))
    title_str = {
        'seq': 'Sequence',
        'image': 'Image',
        'mask': 'Mask'
    }
    
    scatter.set_title(f"{title_str[dataset_name]}: {n_clusters:02d} clusters t-SNE (Silhouette score: {s_score:.3f})")
    scatter.set_xlabel("")
    scatter.set_ylabel("")
    
    save_path = rf"result/_t-SNE_figures/{dataset_name}_{n_clusters:02d}_clusters_tSNE.jpg"
    
    # plt.show()
    plt.savefig(save_path, dpi=500)