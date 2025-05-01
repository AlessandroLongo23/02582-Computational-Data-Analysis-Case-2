import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

# Load feature data
feature_path = 'Preprocessing/HR_features_scaled.csv'
feature_data = pd.read_csv(feature_path)
print(f"Successfully loaded data from {feature_path}")
print(f"Dataset shape: {feature_data.shape}")

feature_columns = feature_data.select_dtypes(include=np.number).columns.tolist()

print(f"\nUsing {len(feature_columns)} features for t-SNE:")
print(feature_columns)

X = feature_data[feature_columns].to_numpy()

indices = np.arange(X.shape[0])
sample_size = X.shape[0]

# Load target data
targets_path = 'Preprocessing/HR_targets_scaled.csv'
targets_data = pd.read_csv(targets_path)
print(f"Successfully loaded data from {targets_path}")
print(f"Dataset shape: {targets_data.shape}")

targets_columns = targets_data.select_dtypes(include=np.number).columns.tolist()

print(f"\nUsing {len(targets_columns)} features for t-SNE:")
print(targets_columns)

Z = targets_data[targets_columns].to_numpy()

print(f"\nRunning t-SNE on {X.shape[0]} samples with {X.shape[1]} features")

point_size = max(10, min(50, 1000 // sample_size))

reducer = umap.UMAP(n_neighbors=3, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(X)

for i in range(11):
    plt.figure(figsize=(10, 8))
    target_values = Z[:, i]
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=target_values, cmap='viridis', s=point_size, alpha=0.8)
    plt.colorbar(scatter, label=targets_columns[i])
    plt.title("UMAP projection")
    plt.xlabel('UMAP dimension 1')
    plt.ylabel('UMAP dimension 2')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'UMAP/plots/umap_{targets_columns[i]}.png', dpi=300)
    plt.show()