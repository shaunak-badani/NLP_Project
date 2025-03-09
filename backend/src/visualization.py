import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def plot_embeddings_multi(embeddings_2d, title, x_axis_label, y_axis_label):
    plt.figure(figsize=(8, 6))
    
    plt.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1], color='red', marker='*', s=100, label="Query Embedding")
    
    output_points = embeddings_2d[1:]
    plt.scatter(output_points[:, 0], output_points[:, 1], color='blue', alpha=0.7, label="Output Embeddings")
    
    plt.text(embeddings_2d[0, 0], embeddings_2d[0, 1], "Query", fontsize=12, 
             verticalalignment='bottom', horizontalalignment='right')
    
    for i, point in enumerate(output_points):
        plt.text(point[0], point[1], f"Output {i+1}", fontsize=10, 
                 verticalalignment='bottom', horizontalalignment='right')
    
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def PCA_visualization(output_embs, query_emb):
    if isinstance(output_embs, list):
        output_embs = np.array(output_embs)
    
    query_emb = np.array(query_emb).reshape(1, -1)
    
    embeddings = np.vstack([query_emb, output_embs])
    
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    plot_embeddings_multi(embeddings_2d, "PCA Projection", "PCA Component 1", "PCA Component 2")

def tSNE_visualization(output_embs, query_emb):
    if isinstance(output_embs, list):
        output_embs = np.array(output_embs)
    
    query_emb = np.array(query_emb).reshape(1, -1)
    
    embeddings = np.vstack([query_emb, output_embs])
    
    perplexity = min(30, max(1, len(embeddings) // 3))
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    plot_embeddings_multi(embeddings_2d, "t-SNE Projection", "t-SNE Component 1", "t-SNE Component 2")

def UMAP_visualization(output_embs, query_emb):
    if isinstance(output_embs, list):
        output_embs = np.array(output_embs)
    
    query_emb = np.array(query_emb).reshape(1, -1)
    
    embeddings = np.vstack([query_emb, output_embs])
    
    n_samples = len(embeddings)
    
    if n_samples <= 3:
        # For very small datasets, we need to add synthetic points
        noise_level = 0.01
        synthetic_points = []
        for emb in embeddings:
            for _ in range(3):  # Add 3 synthetic points per embedding
                noise = np.random.normal(0, noise_level, size=emb.shape)
                synthetic_points.append(emb + noise)
        
        all_embeddings = np.vstack([embeddings] + synthetic_points)
        
        # Configure UMAP for the augmented dataset
        n_neighbors = min(5, len(all_embeddings) - 1)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )
        
        # Fit and transform all embeddings
        all_embeddings_2d = reducer.fit_transform(all_embeddings)
        
        # Extract only the original points for plotting
        embeddings_2d = all_embeddings_2d[:n_samples]
    else:
        # For larger datasets, UMAP works fine without augmentation
        n_neighbors = min(15, n_samples - 1)  # Default is 15, but we need fewer for small datasets
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )
        embeddings_2d = reducer.fit_transform(embeddings)
    
    plot_embeddings_multi(embeddings_2d, "UMAP Projection", "UMAP Component 1", "UMAP Component 2")

if __name__ == "__main__":
    query_embedding = np.random.rand(512)
    
    output_embeddings = [
        np.random.rand(512),
        np.random.rand(512),
        np.random.rand(512)
    ]
    
    PCA_visualization(output_embeddings, query_embedding)
    tSNE_visualization(output_embeddings, query_embedding)
    UMAP_visualization(output_embeddings, query_embedding)