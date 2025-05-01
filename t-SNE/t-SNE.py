import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def compute_pairwise_affinities(X, perplexity=30.0, tol=1e-5):
    """Compute pairwise affinities with automatic sigma selection."""
    n = X.shape[0]
    P = np.zeros((n, n))
    
    # Compute squared Euclidean distances
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    
    # Add a small amount to the diagonal to prevent numerical issues
    np.fill_diagonal(D, 0.0)
    D = np.maximum(D, 0.0)  # Ensure no negative distances due to numerical errors
    
    print("Computing pairwise affinities...")
    for i in range(n):
        if i % 10 == 0:
            print(f"Processing point {i}/{n}")
            
        # Initialize variables for binary search
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0
        
        # Create indices array for all points except i
        indices = np.concatenate((np.arange(0, i), np.arange(i+1, n)))
        
        # Extract distances to all other points
        dist_i = D[i, indices]
        
        # Binary search for the right perplexity
        H_diff = 1.0  # Initialize difference from target perplexity
        tries = 0  # Count attempts
        
        while (H_diff > tol and tries < 50):
            # Compute conditional probabilities with current beta
            P_i = np.exp(-dist_i * beta)
            
            # Avoid numerical issues
            sum_P_i = np.maximum(np.sum(P_i), 1e-12)
            
            # Normalize
            P_i = P_i / sum_P_i
            
            # Compute entropy and perplexity
            H = -np.sum(P_i * np.log2(np.maximum(P_i, 1e-12)))
            H_diff = np.abs(2**H - perplexity)
            
            # Binary search update
            if 2**H > perplexity:
                # Increase precision (beta)
                beta_min = beta
                if beta_max == np.inf:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                # Decrease precision (beta)
                beta_max = beta
                if beta_min == -np.inf:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0
                    
            tries += 1
        
        # Store conditional probabilities
        P[i, indices] = P_i
    
    # Symmetrize and normalize the probability matrix
    P = (P + P.T) / (2.0 * n)
    
    # Add a small value to avoid numerical issues
    P = np.maximum(P, 1e-12)
    
    return P


def initialize_embedding(X, dim=2, method='pca'):
    """Initialize the low-dimensional embedding."""
    n = X.shape[0]
    
    if method == 'random':
        # Random initialization with small values
        np.random.seed(42)  # For reproducibility
        Y = np.random.normal(0, 0.001, size=(n, dim))
    elif method == 'pca':
        # PCA initialization (often leads to better results)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=dim)
        Y = pca.fit_transform(X)
        Y = Y * 0.001  # Scale down the values but not too much
    
    return Y


def compute_low_dimensional_probabilities(Y):
    """Compute the joint probability distribution Q in low dimensional space."""
    n = Y.shape[0]
    
    # Compute squared Euclidean distances
    sum_Y = np.sum(np.square(Y), axis=1)
    D = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)
    D = np.maximum(D, 0)  # Ensure distances are non-negative due to numerical errors
    
    # Compute Q matrix using Student t-distribution (degrees of freedom = 1)
    Q = 1.0 / (1.0 + D)
    np.fill_diagonal(Q, 0)  # Set diagonal to zero
    
    # Normalize Q to create a probability distribution
    Q_sum = np.maximum(np.sum(Q), 1e-12)  # Avoid division by zero
    Q = Q / Q_sum
    
    # Add small value to avoid numerical issues
    Q = np.maximum(Q, 1e-12)
    
    return Q, D


def compute_gradient(P, Q, Y, D):
    """Compute the gradient of KL divergence between P and Q."""
    n = Y.shape[0]
    
    # Calculate the difference between P and Q
    PQ_diff = P - Q
    
    # Compute the weights for the gradient
    weights = PQ_diff / (1.0 + D)
    
    # Compute the gradient for each point
    dY = np.zeros_like(Y)
    for i in range(n):
        dY[i] = 4.0 * np.sum(np.multiply(weights[i].reshape(-1, 1), (Y[i] - Y)), axis=0)
    
    return dY


def tsne(X, n_components=2, perplexity=30.0, n_iter=1000, learning_rate=200.0, momentum=0.8):
    """Perform t-SNE dimensionality reduction."""
    # Check input
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = X.astype(np.float64)
    
    # Handle NaN values more robustly
    if np.isnan(X).any():
        print("Warning: Input contains NaN values. Filling with column means...")
        # First check if entire columns are NaN
        col_means = np.nanmean(X, axis=0)
        # Replace NaN means with 0 if entire column is NaN
        col_means = np.where(np.isnan(col_means), 0, col_means)
        
        # Fill NaN values with column means
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            X[mask, i] = col_means[i]
            
        # Double-check that all NaNs are gone
        if np.isnan(X).any():
            print("Warning: Some NaN values could not be filled. Replacing with zeros...")
            X = np.nan_to_num(X, nan=0.0)
    
    # Initialize variables
    n_samples = X.shape[0]
    
    # Compute pairwise affinities
    P = compute_pairwise_affinities(X, perplexity)
    
    # Apply early exaggeration
    P = P * 12.0  # Standard early exaggeration factor
    
    # Initialize solution (PCA or random)
    try:
        if X.shape[1] > n_components:
            print("Initializing with PCA...")
            Y = initialize_embedding(X, dim=n_components, method='pca')
            print("PCA initialization complete")
        else:
            raise ValueError("Using random initialization instead")
    except Exception as e:
        print(f"PCA initialization failed: {e}")
        print("Using random initialization...")
        Y = initialize_embedding(X, dim=n_components, method='random')
    
    # Initialize optimization variables
    Y_prev = np.zeros_like(Y)
    gains = np.ones_like(Y)
    
    # Start gradient descent
    print("Starting gradient descent...")
    for iter in range(n_iter):
        # Compute Q distribution and distances
        Q, D = compute_low_dimensional_probabilities(Y)
        
        # Compute gradient
        dY = compute_gradient(P, Q, Y, D)
        
        # Update solution with gains and momentum
        gains = (gains + 0.2) * ((dY * (Y - Y_prev)) <= 0) + \
                gains * 0.8 * ((dY * (Y - Y_prev)) > 0)
        gains[gains < 0.01] = 0.01
        
        Y_prev = Y.copy()
        Y = Y - learning_rate * gains * dY + momentum * (Y - Y_prev)
        Y = Y - np.mean(Y, axis=0)  # Center
        
        # Print progress and check for convergence
        if iter == 250:
            P = P / 12.0  # Remove early exaggeration after 250 iterations
            print("Early exaggeration removed")
        
        if iter % 50 == 0 or iter == n_iter - 1:
            cost = np.sum(P * np.log(np.maximum(P, 1e-12) / np.maximum(Q, 1e-12)))
            print(f"Iteration {iter}: cost = {cost}")
            
            # Print some statistics for debugging
            print(f"Y stats - mean: {np.mean(Y):.6f}, std: {np.std(Y):.6f}")
            print(f"grad stats - mean: {np.mean(dY):.6f}, std: {np.std(dY):.6f}")
            print(f"gains stats - mean: {np.mean(gains):.6f}, std: {np.std(gains):.6f}")
    
    return Y


def find_best_perplexity(X, perplexity_range=None, n_components=2, n_iter=1000, learning_rate=200.0, momentum=0.8):
    """
    Try multiple perplexity values and plot the results to help find the optimal value.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    perplexity_range : list or None
        List of perplexity values to try. If None, uses [5, 15, 30, 50, 100].
    n_components : int
        Number of components for t-SNE.
    n_iter : int
        Number of iterations.
    learning_rate : float
        Learning rate for t-SNE.
    momentum : float
        Momentum for t-SNE.
    
    Returns:
    --------
    dict : Dictionary mapping perplexity values to resulting embeddings.
    """
    if perplexity_range is None:
        # Use a range of perplexity values, ensuring they don't exceed the dataset size
        max_perp = min(100, X.shape[0] // 3)
        perplexity_range = [5, 15, 30, min(50, max_perp), min(100, max_perp)]
        # Remove duplicates and sort
        perplexity_range = sorted(list(set(perplexity_range)))
    else:
        # Ensure all perplexity values are reasonable
        max_allowed_perp = X.shape[0] // 3
        perplexity_range = [min(p, max_allowed_perp) for p in perplexity_range]
        print(f"Note: Maximum allowed perplexity is {max_allowed_perp} for this dataset")
    
    results = {}
    
    # Calculate embeddings for each perplexity
    print("\nFinding optimal perplexity value...")
    for perplexity in perplexity_range:
        print(f"Computing t-SNE with perplexity = {perplexity}")
        start_time = time.time()
        Y = tsne(X, n_components=n_components, perplexity=perplexity, n_iter=n_iter, learning_rate=learning_rate, momentum=momentum)
        end_time = time.time()
        print(f"t-SNE execution time: {end_time - start_time:.2f} seconds")
        
        results[perplexity] = Y
    
    if n_components == 2:
        n_plots = len(perplexity_range)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        plt.figure(figsize=(n_cols * 5, n_rows * 4))
        
        for i, perplexity in enumerate(perplexity_range):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.scatter(results[perplexity][:, 0], results[perplexity][:, 1], s=30, alpha=0.7)
            plt.title(f'Perplexity = {perplexity}')
            plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('tsne_perplexity_comparison_2d.png', dpi=300)
        plt.show()
    
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        
        n_plots = len(perplexity_range)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(n_cols * 6, n_rows * 5))
        
        for i, perplexity in enumerate(perplexity_range):
            ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
            ax.scatter(results[perplexity][:, 0], results[perplexity][:, 1], results[perplexity][:, 2], s=30, alpha=0.7)
            ax.set_title(f'Perplexity = {perplexity}')
            ax.grid(alpha=0.3)
            ax.view_init(30, 45)
        
        plt.tight_layout()
        plt.savefig('tsne_perplexity_comparison_3d.png', dpi=300)
        plt.show()
    
    print("\nPerplexity optimization complete!")
    print("The 'best' perplexity value is subjective and depends on what structure you're trying to reveal.")
    print("Look for a visualization where:")
    print("- Similar data points are grouped together")
    print("- Dissimilar data points are far apart")
    print("- The structure matches your prior knowledge about the data")
    print("- There's a balance between local and global structure")

    return results


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

# Find the best perplexity value
n_components = 2  # Set to 2 or 3 based on your preference
perplexity_values = [5, 15, 30]  # Use reasonable perplexity values

# results = find_best_perplexity(
#     X, 
#     perplexity_range=perplexity_values,
#     n_components=n_components,
#     n_iter=1000,
#     learning_rate=200.0,  # Standard learning rate
#     momentum=0.8
# )

# # Get the embedding for the median perplexity
# Y = results[perplexity_values[1]]  # Using the middle perplexity (15)

# # Now visualize the final result with the chosen perplexity
# print(f"\nCreating final visualization with perplexity = {perplexity_values[1]}")

# # Set up the colormap
# colormap = plt.cm.get_cmap('viridis', 11)  # One color for each emotion

# for emotion_index in range(11):
#     plt.figure(figsize=(10, 8))
    
#     point_size = max(10, min(50, 1000 // sample_size))

#     # Color points by target if available
#     target_values = Z[:, emotion_index]
#     scatter = plt.scatter(Y[:, 0], Y[:, 1], c=target_values, cmap='viridis', 
#                           s=point_size, alpha=0.7)
#     plt.colorbar(scatter, label=targets_columns[emotion_index])
#     plt.title(f't-SNE Visualization - {targets_columns[emotion_index]} (Perplexity={perplexity_values[1]})')
    
#     plt.xlabel('t-SNE dimension 1')
#     plt.ylabel('t-SNE dimension 2')
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(f'tsne_{targets_columns[emotion_index]}.png', dpi=300)
#     plt.show()

# # Optional: Create a combined visualization
# plt.figure(figsize=(12, 10))
# # Use the first emotion for coloring in the combined view
# target_values = Z[:, 0]
# scatter = plt.scatter(Y[:, 0], Y[:, 1], c=target_values, cmap='viridis', 
#                       s=point_size, alpha=0.7)
# plt.colorbar(scatter, label=targets_columns[0])
# plt.title(f't-SNE Visualization - Overview (Perplexity={perplexity_values[1]})')
# plt.xlabel('t-SNE dimension 1')
# plt.ylabel('t-SNE dimension 2')
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig('tsne_overview.png', dpi=300)
# plt.show()

# Comparing with sklearn's implementation (uncomment if needed)
from sklearn.manifold import TSNE
point_size = max(10, min(50, 1000 // sample_size))

# Run sklearn's t-SNE
start_time = time.time()
tsne_sklearn = TSNE(n_components=2, perplexity=80, random_state=42)
Y_sklearn = tsne_sklearn.fit_transform(X)
end_time = time.time()
print(f"sklearn t-SNE execution time: {end_time - start_time:.2f} seconds")

# Visualize sklearn results
plt.figure(figsize=(10, 8))
target_values = Z[:, 0]  # Using first emotion for coloring
scatter = plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=target_values, cmap='viridis', s=point_size, alpha=0.8)
plt.colorbar(scatter, label=targets_columns[0])
plt.title('sklearn t-SNE Visualization')
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('tsne_sklearn.png', dpi=300)
plt.show()