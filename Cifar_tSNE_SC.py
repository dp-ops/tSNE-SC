import os
import tensorflow as tf 
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import pairwise_distances, adjusted_rand_score
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from cluster.Spectral_V2 import SpectralClustering
from sklearn.metrics import silhouette_score
import time 
import csv

start_time = time.time()
number_of_clusters = 20
output_dir = "C:\\Users\\dimos\\PyProjecks\\t-SNE_Spectral_Clustering\\results\\Cifar-10"
os.makedirs(output_dir, exist_ok=True)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train_mini = x_train[0:5000]
y_train_mini = y_train[0:5000]
x_train_m = x_train_mini.reshape(x_train_mini.shape[0], -1)
y_train_m = y_train_mini.reshape(y_train_mini.shape[0])

def normal(X_mini):
    #dedomeno oti einai gray scale eikones, max(xi)=255, min(xi)=0
    X_norm = 2*(X_mini/255) - 1
    return X_norm

X_mini =normal(x_train_m)

# Perform t-SNE
X_embedded = TSNE(random_state = 0, n_components=2,verbose=0).fit_transform(X_mini)

results_file = os.path.join(output_dir, "clustering_results.csv")
with open(results_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Number of Clusters", "Adjusted Rand Score", "Silhouette Score"])

# Perform Spectral Clustering
silhouette_scores = []
ari_scores = []
inertia = []

fig, axes = plt.subplots(4, 5, figsize=(20, 16))   # 4 rows and 5 columns of subplots  (20, 8) # 2 rows and 5 columns of subplots
axes = axes.ravel()  

for i, n_clusters in enumerate(range(2, number_of_clusters + 2)):
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        n_neighbors=10
    ).fit(X_embedded)

    score = silhouette_score(X_embedded, clustering)
    silhouette_scores.append(score)

    ari = adjusted_rand_score(y_train_m, clustering)
    ari_scores.append(ari)

    cluster_centers = np.array([
        X_embedded[clustering == j].mean(axis=0)
        for j in range(n_clusters)
    ])
    distances = pairwise_distances(X_embedded, cluster_centers, metric='euclidean')
    inertia.append(np.sum(np.min(distances, axis=1)**2))

    with open(results_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([n_clusters, ari, score])

    axes[i].scatter(
        X_embedded[:, 0], X_embedded[:, 1],
        c=clustering,
        cmap='rainbow',
        s=1
    )
    axes[i].set_title(f"{n_clusters} Clusters")
    axes[i].axis('on')

# Save clustering visualization
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "Clustering.png"))
plt.close(fig)

# Plot silhouette scores
plt.figure(figsize=(16, 12))
plt.plot(range(2, number_of_clusters + 2), silhouette_scores, marker='o')
plt.title("Silhouette Score for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.savefig(os.path.join(output_dir, "Silhouette_Score.png"))
plt.close()

# Plot ARI scores
plt.figure(figsize=(16, 12))
plt.plot(range(2, number_of_clusters + 2), ari_scores, marker='o', color='red')
plt.title("Adjusted Rand Index for Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Adjusted Rand Index")
plt.savefig(os.path.join(output_dir, "ARI_Scores.png"))
plt.close()

# Plot inertia
plt.figure(figsize=(16, 12))
plt.plot(range(2, number_of_clusters + 2), inertia, marker='o')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.savefig(os.path.join(output_dir, "Elbow_Method.png"))
plt.close()

print(f"Execution time: {time.time() - start_time:.2f} seconds")
print(f"Figures saved in directory: {output_dir}")

if __name__ == "__main__":
    pass