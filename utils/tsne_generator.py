from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def tsne_generator(data, target, dataset_name, n_clusters):
    
    model = TSNE(n_components=2, init='random')
    # labels = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]
    
    # np.seterr(invalid='ignore') # during T-SNE
    embedded = model.fit_transform(data)
    # np.seterr(invalid='warn')
    
    target = target.reshape(-1, 1)
    
    data=np.concatenate((embedded, target), axis=1)
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=data[:, 0],
                    y=data[:, 1],
                    hue=data[:, 2],
                    legend=False,
                    palette=sns.color_palette('bright'))
    '''
    plt.legend(
        title='Clusters',
        labels=labels[:n_clusters]
    )
    '''
    save_path = rf"result/t-SNE_figures/{dataset_name}_{n_clusters:02d}_clusters_tSNE.jpg"
    
    # plt.show()
    plt.savefig(save_path, dpi=500)