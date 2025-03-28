# t-SNE and Spectral Clustering Project

This project demonstrates the use of dimensionality reduction and clustering techniques on image datasets for unsupervised learning and visualization.

---

## 📋 Overview

### Key Techniques:
- **t-SNE**: t-Distributed Stochastic Neighbor Embedding for dimensionality reduction.
- **Spectral Clustering**: Unsupervised learning to group data into clusters.
- **Graph Embedding**: Techniques like Laplacian Eigenmaps, Kernel PCA, Isomap, and LLE.

### Visualizations:
- High-dimensional data visualized in **2D** and **3D** spaces.
- Clustering results and performance metrics.

---

## 📂 Datasets

### Supported Datasets:
1. **MNIST**: Handwritten digits dataset.
2. **CIFAR-10**: Color images across 10 categories.

---

## ✨ Features

- **Data Preprocessing**: Normalization and preparation for analysis.
- **Dimensionality Reduction**: Using t-SNE and other embedding techniques.
- **Clustering**: Spectral Clustering with adjustable parameters.
- **Evaluation Metrics**:
  - Silhouette Score
  - Adjusted Rand Index (ARI)
- **Performance Analysis**: Optimal cluster number selection using:
  - Elbow Method
  - Silhouette Scores
- **Visualization**: Interactive and static plots for embeddings and clusters.

---

## 🚀 Usage

### Main Components:
1. **`Mnist_tSNE_SC.py`**: 
   - Runs t-SNE and Spectral Clustering on the MNIST dataset.
   - Saves results in `/results/Mnist`.

2. **`Cifar_tSNE_SC.py`**:
   - Runs t-SNE and Spectral Clustering on the CIFAR-10 dataset.
   - Saves results in `/results/Cifar-10`.

3. **Jupyter Notebooks**:
   - Interactive exploration and visualization of results.

---

## 📁 Results

- Results are saved in the `/results` directory.
- Separate folders for MNIST and CIFAR-10 datasets.
- Includes:
  - Cluster visualizations
  - Performance metrics (Silhouette Scores, ARI, etc.)
  - Embedding plots in 2D and 3D.

---

## 🛠️ Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `scikit-learn`
  - `tensorflow` (for CIFAR-10)

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## 📊 Example Visualizations

### t-SNE Visualization:
![t-SNE Example](results/example_tsne.png)

### Spectral Clustering:
![Clustering Example](results/example_clustering.png)

---

## 📜 License

This project is licensed under the MIT License.
