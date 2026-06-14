import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def load_and_preprocess(filepath: str) -> tuple:
    """Loads dataset, selects numeric columns, and scales the features."""
    data = pd.read_csv(filepath)
    X = data.select_dtypes(include=['float64', 'int64'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def find_best_k(X_scaled, min_k: int = 2, max_k: int = 7) -> int:
    """Finds the optimal number of clusters using the Silhouette Score."""
    print("Finding best K...")
    best_k = min_k
    best_score = -1

    for k in range(min_k, max_k + 1):
        # n_init='auto' suppresses future warnings in newer sklearn versions
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"K = {k}, Silhouette Score = {round(score, 3)}")
        
        if score > best_score:
            best_score = score
            best_k = k

    print(f"\nBest K identified: {best_k}\n")
    return best_k

def run_clustering(X_scaled, k: int) -> tuple:
    """Runs K-Means and GMM clustering on the dataset."""
    # K-Means
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0)
    k_labels = kmeans.fit_predict(X_scaled)
    k_centers = kmeans.cluster_centers_

    # GMM
    gmm = GaussianMixture(n_components=k, random_state=0)
    g_labels = gmm.fit_predict(X_scaled)
    g_centers = gmm.means_

    return k_labels, k_centers, g_labels, g_centers

def plot_clusters(X_scaled, k: int, k_labels, k_centers, g_labels, g_centers):
    """Visualizes the clustering results side-by-side using the first two features."""
    plt.figure(figsize=(14, 5))

    # K-Means Plot
    plt.subplot(1, 2, 1)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=k_labels, cmap='viridis', alpha=0.6, edgecolors='k')
    plt.scatter(k_centers[:, 0], k_centers[:, 1], marker='X', s=200, c='red', label='Centroids')
    plt.title(f"K-Means Clustering (K={k})")
    plt.xlabel("Feature 1 (Scaled)")
    plt.ylabel("Feature 2 (Scaled)")
    plt.legend()

    # GMM Plot
    plt.subplot(1, 2, 2)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=g_labels, cmap='viridis', alpha=0.6, edgecolors='k')
    plt.scatter(g_centers[:, 0], g_centers[:, 1], marker='X', s=200, c='red', label='Means')
    plt.title(f"GMM Clustering (K={k})")
    plt.xlabel("Feature 1 (Scaled)")
    plt.ylabel("Feature 2 (Scaled)")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Path to your dataset
    dataset_path = 'wine-clustering.csv'
    
    # 1. Pipeline execution
    X_scaled = load_and_preprocess(dataset_path)
    best_k = find_best_k(X_scaled)
    
    # 2. Model Fitting
    k_labels, k_centers, g_labels, g_centers = run_clustering(X_scaled, best_k)
    
    # 3. Evaluation
    k_sil = silhouette_score(X_scaled, k_labels)
    g_sil = silhouette_score(X_scaled, g_labels)
    
    print("Final Comparison:")
    print(f"  K-Means Silhouette Score: {round(k_sil, 3)}")
    print(f"  GMM Silhouette Score:     {round(g_sil, 3)}\n")
    
    # 4. Visualization
    plot_clusters(X_scaled, best_k, k_labels, k_centers, g_labels, g_centers)