import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from typing import Tuple, Optional, List, Union
from correlation_matrix import (
    compute_degree_matrix,
    normalize_by_degree,
    decompose_correlation_matrix
)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler


def spectral_clustering(
        correlation_matrix: pd.DataFrame,
        n_clusters: int,
        use_abs: bool = True
) -> np.ndarray:
    """
    Performs spectral clustering on the correlation matrix.

    Args:
        correlation_matrix: DataFrame with correlation matrix
        n_clusters: Number of clusters to find
        use_abs: Whether to use absolute values of correlations

    Returns:
        Array of cluster labels for each stock
    """
    # Step 1: Prepare the adjacency matrix
    if use_abs:
        # Use absolute values as we're interested in strength of correlation
        adjacency_matrix = correlation_matrix.abs()
    else:
        adjacency_matrix = correlation_matrix

    # Step 2: Compute the Laplacian matrix
    # First, compute the degree matrix
    D = compute_degree_matrix(adjacency_matrix)

    # Then, compute the Laplacian: L = D - A
    L = D - adjacency_matrix

    # Step 3: Find the k smallest eigenvectors of L
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Sort eigenvalues and corresponding eigenvectors
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select the k smallest non-zero eigenvectors
    # (first eigenvector with eigenvalue ≈ 0 is often uninformative)
    if eigenvalues[0] < 1e-10:
        embedding = eigenvectors[:, 1:n_clusters + 1]
    else:
        embedding = eigenvectors[:, :n_clusters]

    # Step 4: Cluster the points in the embedding space
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embedding)

    return labels


def signed_laplacian_clustering(
        correlation_matrix: pd.DataFrame,
        n_clusters: int,
        normalization: str = 'symmetric'
) -> np.ndarray:
    """
    Performs clustering using the Signed Laplacian.

    Args:
        correlation_matrix: DataFrame with correlation matrix
        n_clusters: Number of clusters to find
        normalization: Type of normalization ('symmetric' or 'random_walk')

    Returns:
        Array of cluster labels for each stock
    """
    # Step 1: Compute the signed degree matrix using absolute values
    D_signed = compute_degree_matrix(correlation_matrix, signed=True)

    # Step 2: Compute the Signed Laplacian
    # For signed Laplacian: L = D - A (where D uses |A_ij|)
    signed_laplacian = D_signed - correlation_matrix

    # Step 3: Normalize the Laplacian if requested
    if normalization == 'symmetric':
        # Symmetric normalization: L_sym = D^(-1/2) * L * D^(-1/2)
        normalized_laplacian = normalize_by_degree(
            signed_laplacian, D_signed, method='symmetric'
        )
    elif normalization == 'random_walk':
        # Random walk normalization: L_rw = D^(-1) * L
        normalized_laplacian = normalize_by_degree(
            signed_laplacian, D_signed, method='random_walk'
        )
    else:
        normalized_laplacian = signed_laplacian

    # Step 4: Find the k smallest eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(normalized_laplacian)

    # Sort eigenvalues and corresponding eigenvectors
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select the k smallest eigenvectors
    embedding = eigenvectors[:, :n_clusters]

    # Step 5: Cluster the points in the embedding space
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embedding)

    return labels


def sponge_clustering(
        correlation_matrix: pd.DataFrame,
        n_clusters: int,
        tau_p: float = 0.5,
        tau_n: float = 0.5,
        use_symmetric: bool = False
) -> np.ndarray:
    """
    Performs clustering using the SPONGE algorithm.

    Args:
        correlation_matrix: DataFrame with correlation matrix
        n_clusters: Number of clusters to find
        tau_p: Regularization parameter for positive matrix
        tau_n: Regularization parameter for negative matrix
        use_symmetric: Whether to use symmetric normalization

    Returns:
        Array of cluster labels for each stock
    """
    # Step 1: Decompose the correlation matrix into positive and negative parts
    A_pos, A_neg = decompose_correlation_matrix(correlation_matrix)

    # Step 2: Compute the Laplacian matrices for positive and negative parts
    D_pos = compute_degree_matrix(A_pos)
    D_neg = compute_degree_matrix(A_neg)

    L_pos = D_pos - A_pos
    L_neg = D_neg - A_neg

    if use_symmetric:
        # Use symmetric normalization for both Laplacians
        L_pos_sym = normalize_by_degree(L_pos, D_pos, method='symmetric')
        L_neg_sym = normalize_by_degree(L_neg, D_neg, method='symmetric')

        # Prepare matrices for generalized eigenvalue problem
        A = L_pos_sym + tau_n * np.eye(len(correlation_matrix))
        B = L_neg_sym + tau_p * np.eye(len(correlation_matrix))
    else:
        # Prepare matrices for generalized eigenvalue problem
        A = L_pos + tau_n * D_neg
        B = L_neg + tau_p * D_pos

    # Step 3: Solve the generalized eigenvalue problem A*v = lambda*B*v
    # Find the k smallest generalized eigenvalues/eigenvectors
    eigenvalues, eigenvectors = sparse.linalg.eigsh(
        A=A, M=B, k=n_clusters, sigma=0, which='LM'
    )

    # Step 4: Cluster the points in the embedding space
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(eigenvectors)

    return labels


def determine_clusters_mp(
        correlation_matrix: pd.DataFrame,
        T: int,
        significance_level: float = 0.01
) -> int:
    """
    Determines the number of clusters using the Marchenko-Pastur distribution.

    Args:
        correlation_matrix: DataFrame with correlation matrix
        T: Number of time periods used to estimate the correlation
        significance_level: Statistical significance level

    Returns:
        Number of clusters to use
    """
    # Get the eigenvalues of the correlation matrix
    eigenvalues = np.linalg.eigvalsh(correlation_matrix)

    # Sort eigenvalues in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Calculate the ratio of dimensions
    N = len(correlation_matrix)
    rho = N / T

    # Calculate the upper bound of the MP distribution
    lambda_max = (1 + np.sqrt(rho)) ** 2

    # Count eigenvalues above the threshold
    n_clusters = np.sum(eigenvalues > lambda_max)

    # Ensure we have at least 2 clusters
    return max(2, n_clusters)


def determine_clusters_variance(
        correlation_matrix: pd.DataFrame,
        variance_explained: float = 0.9
) -> int:
    """
    Determines the number of clusters based on explained variance.

    Args:
        correlation_matrix: DataFrame with correlation matrix
        variance_explained: Proportion of variance to explain

    Returns:
        Number of clusters to use
    """
    # Get the eigenvalues of the correlation matrix
    eigenvalues = np.linalg.eigvalsh(correlation_matrix)

    # Sort eigenvalues in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Calculate cumulative explained variance ratio
    total_variance = np.sum(eigenvalues)
    cumulative_variance_ratio = np.cumsum(eigenvalues) / total_variance

    # Find the number of components needed to explain the desired variance
    n_clusters = np.argmax(cumulative_variance_ratio >= variance_explained) + 1

    # Ensure we have at least 2 clusters
    return max(2, n_clusters)


# ==================== NEW CLUSTERING APPROACHES ====================

def hierarchical_clustering(
        correlation_matrix: pd.DataFrame,
        n_clusters: int,
        method: str = 'ward'
) -> np.ndarray:
    """
    Performs hierarchical clustering on the correlation matrix.

    Args:
        correlation_matrix: DataFrame with correlation matrix
        n_clusters: Number of clusters to find
        method: Linkage method ('ward', 'complete', 'average', 'single')

    Returns:
        Array of cluster labels for each stock
    """
    # Convert correlation to distance
    distance_matrix = 1 - np.abs(correlation_matrix)

    # Perform hierarchical clustering
    condensed_dist = squareform(distance_matrix)
    Z = linkage(condensed_dist, method=method)

    # Cut the dendrogram to get n_clusters
    labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # zero-based

    return labels, Z


def dynamic_tree_cut(
        linkage_matrix: np.ndarray,
        correlation_matrix: pd.DataFrame,
        deepSplit: int = 2,
        minClusterSize: int = 2
) -> np.ndarray:
    """
    Cuts dendrogram using dynamic tree cut algorithm.

    Args:
        linkage_matrix: Hierarchical clustering linkage matrix
        correlation_matrix: Correlation matrix
        deepSplit: Parameter controlling the sensitivity of the splitting (0-4)
        minClusterSize: Minimum cluster size

    Returns:
        Array of cluster labels for each stock
    """
    try:
        from dynamicTreeCut import cutreeHybrid

        # Convert correlation to similarity
        similarity_matrix = np.abs(correlation_matrix.values)

        # Apply dynamic cutting
        result = cutreeHybrid(
            dendro=linkage_matrix,
            distM=1 - similarity_matrix,
            deepSplit=deepSplit,
            minClusterSize=minClusterSize
        )

        labels = result['labels'] - 1  # Convert to zero-based indexing

        return labels
    except ImportError:
        print("dynamicTreeCut package not found. Install with: pip install dynamicTreeCut")
        # Fall back to regular hierarchical clustering
        n_clusters = max(int(len(correlation_matrix) / 10), 2)  # Estimate reasonable number
        return fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1


def density_based_clustering(
        correlation_matrix: pd.DataFrame,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, object]:
    """
    Applies HDBSCAN clustering to correlation matrix.

    Args:
        correlation_matrix: DataFrame with correlation matrix
        min_cluster_size: Minimum size of clusters
        min_samples: Number of samples in a neighborhood for a point to be a core point

    Returns:
        Tuple of (cluster_labels, probabilities, clusterer)
    """
    try:
        import hdbscan

        # Convert correlation to distance
        distance_matrix = 1 - np.abs(correlation_matrix)

        # Set min_samples as a percentage of min_cluster_size if not provided
        if min_samples is None:
            min_samples = max(2, min_cluster_size // 2)

        # Apply HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='precomputed',
            cluster_selection_method='eom'  # Excess of Mass
        )

        labels = clusterer.fit_predict(distance_matrix)

        # Handle noise points (labeled as -1)
        # Assign noise points to the nearest cluster
        if -1 in labels:
            for i, label in enumerate(labels):
                if label == -1:
                    # Find nearest non-noise point
                    distances = distance_matrix.iloc[i].values
                    # Exclude noise points and self when finding nearest
                    valid_indices = [j for j, l in enumerate(labels) if l != -1 and j != i]
                    if valid_indices:
                        nearest_idx = valid_indices[np.argmin(distances[valid_indices])]
                        labels[i] = labels[nearest_idx]
                    else:
                        # If all points are noise, assign to a new cluster
                        labels[i] = np.max(labels) + 1

        # Get cluster probabilities if available (not for noise points)
        try:
            probabilities = clusterer.probabilities_
            # Assign probability 0 for noise points
            probabilities = np.array([0.0 if label == -1 else prob for label, prob in zip(labels, probabilities)])
        except:
            probabilities = np.ones(len(labels))

        return labels, probabilities, clusterer

    except ImportError:
        print("HDBSCAN package not found. Install with: pip install hdbscan")
        # Fall back to spectral clustering
        labels = spectral_clustering(
            correlation_matrix,
            n_clusters=max(2, min_cluster_size)
        )
        return labels, np.ones(len(labels)), None


def deep_embedding_clustering(
        residual_returns: pd.DataFrame,
        n_clusters: int = 10,
        embedding_dim: int = 10,
        epochs: int = 100,
        batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray, object]:
    """
    Deep embedding clustering for returns data.

    Args:
        residual_returns: DataFrame with residual returns
        n_clusters: Number of clusters to find
        embedding_dim: Dimension of the embedding space
        epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        Tuple of (cluster_labels, embeddings, encoder_model)
    """
    tf.keras.backend.clear_session()

    # Fill missing values
    returns_filled = residual_returns.fillna(0)

    # 1. Create autoencoder for dimensionality reduction
    input_dim = returns_filled.shape[1]

    # Encoder
    inputs = Input(shape=(input_dim,))
    encoded = Dense(embedding_dim * 2, activation='relu')(inputs)
    encoded = Dense(embedding_dim, activation='relu')(encoded)

    # Decoder
    decoded = Dense(embedding_dim * 2, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)

    # Autoencoder model
    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    # Compile and train
    autoencoder.compile(optimizer='adam', loss='mse')

    # Standardize input data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(returns_filled.values)

    # Train the autoencoder
    try:
        autoencoder.fit(
            scaled_data,
            scaled_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=0
        )
    except Exception as e:
        print(f"Error training autoencoder: {e}")
        # Fall back to spectral clustering
        labels = spectral_clustering(returns_filled.corr(), n_clusters)
        return labels, np.zeros((len(labels), embedding_dim)), None

    # 2. Get embeddings
    embeddings = encoder.predict(scaled_data)

    # 3. Apply clustering to embeddings
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    return labels, embeddings, encoder


def cluster_stocks(
        correlation_matrix: pd.DataFrame,
        method: str = 'spectral',
        n_clusters: Optional[int] = None,
        lookback_periods: Optional[int] = None,
        **kwargs
) -> Tuple[np.ndarray, int]:
    """
    Clusters stocks based on the correlation matrix.

    Args:
        correlation_matrix: DataFrame with correlation matrix
        method: Clustering method to use
        n_clusters: Number of clusters (if None, determined automatically)
        lookback_periods: Number of periods used to estimate correlation
        **kwargs: Additional arguments for clustering methods

    Returns:
        Tuple of (cluster_labels, n_clusters)
    """
    # Determine number of clusters if not provided
    if n_clusters is None:
        if lookback_periods is not None:
            # Use Marchenko-Pastur distribution
            n_clusters = determine_clusters_mp(
                correlation_matrix, lookback_periods
            )
        else:
            # Use variance explained method
            n_clusters = determine_clusters_variance(
                correlation_matrix
            )

    # Apply the appropriate clustering method
    if method == 'spectral':
        labels = spectral_clustering(
            correlation_matrix, n_clusters, **kwargs
        )
    elif method == 'signed_laplacian_sym':
        labels = signed_laplacian_clustering(
            correlation_matrix, n_clusters, normalization='symmetric', **kwargs
        )
    elif method == 'signed_laplacian_rw':
        labels = signed_laplacian_clustering(
            correlation_matrix, n_clusters, normalization='random_walk', **kwargs
        )
    elif method == 'sponge':
        labels = sponge_clustering(
            correlation_matrix, n_clusters, use_symmetric=False, **kwargs
        )
    elif method == 'sponge_sym':
        labels = sponge_clustering(
            correlation_matrix, n_clusters, use_symmetric=True, **kwargs
        )
    # New clustering methods
    elif method == 'hierarchical':
        labels, _ = hierarchical_clustering(
            correlation_matrix, n_clusters, **kwargs
        )
    elif method == 'dynamic_tree':
        Z = hierarchical_clustering(correlation_matrix, n_clusters)[1]
        labels = dynamic_tree_cut(Z, correlation_matrix, **kwargs)
        # Update n_clusters based on actual number of clusters
        n_clusters = len(np.unique(labels))
    elif method == 'density':
        labels, _, _ = density_based_clustering(
            correlation_matrix, **kwargs
        )
        # Update n_clusters based on actual number of clusters (excluding noise)
        n_clusters = len(np.unique([x for x in labels if x >= 0]))
    elif method == 'deep_embedding':
        # Convert correlation matrix to returns if needed
        if isinstance(correlation_matrix, pd.DataFrame) and correlation_matrix.shape[0] == correlation_matrix.shape[1]:
            # We need actual returns, not correlation matrix
            # This is a fallback but not recommended - caller should pass residual_returns directly
            from warnings import warn
            warn("Deep embedding clustering should be called with residual returns, not correlation matrix")
            labels = spectral_clustering(correlation_matrix, n_clusters, **kwargs)
        else:
            labels, _, _ = deep_embedding_clustering(
                correlation_matrix, n_clusters, **kwargs
            )
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return labels, n_clusters


def get_clusters_dict(
        labels: np.ndarray,
        tickers: List[str]
) -> dict:
    """
    Creates a dictionary mapping cluster labels to lists of tickers.

    Args:
        labels: Array of cluster assignments
        tickers: List of stock tickers

    Returns:
        Dictionary mapping cluster IDs to lists of tickers
    """
    clusters = {}

    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(tickers[i])

    return clusters


def calculate_silhouette_score(correlation_matrix: pd.DataFrame, labels: np.ndarray):
    """
    Calculates the silhouette score for a clustering.

    Args:
        correlation_matrix: DataFrame with correlation matrix
        labels: Array of cluster assignments

    Returns:
        Silhouette score
    """
    from sklearn.metrics import silhouette_score

    # Convert correlation to distance
    distance_matrix = 1 - np.abs(correlation_matrix)

    # Calculate silhouette score
    score = silhouette_score(distance_matrix, labels, metric='precomputed')

    return score


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import prepare_stock_data
    from correlation_matrix import compute_correlation_matrix

    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'PG']
    start_date = '2020-01-01'
    end_date = '2022-12-31'

    returns, betas, residual_returns = prepare_stock_data(tickers, start_date, end_date)

    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(residual_returns)

    # Test all clustering methods
    for method in ['spectral', 'signed_laplacian_sym', 'signed_laplacian_rw', 'sponge', 'sponge_sym',
                   'hierarchical', 'density']:
        labels, n_clusters = cluster_stocks(
            corr_matrix, method=method, lookback_periods=60
        )

        clusters = get_clusters_dict(labels, residual_returns.columns.tolist())

        print(f"\n{method.upper()} clustering with {n_clusters} clusters:")
        for cluster_id, cluster_tickers in clusters.items():
            print(f"Cluster {cluster_id}: {', '.join(cluster_tickers)}")

        # Calculate silhouette score
        sil_score = calculate_silhouette_score(corr_matrix, labels)
        print(f"Silhouette score: {sil_score:.4f}")

    # Test deep embedding clustering with returns data
    try:
        print("\nDEEP_EMBEDDING clustering:")
        deep_labels, n_deep_clusters = cluster_stocks(
            residual_returns, method='deep_embedding', n_clusters=3,
            embedding_dim=5, epochs=50, batch_size=2
        )

        deep_clusters = get_clusters_dict(deep_labels, residual_returns.columns.tolist())

        for cluster_id, cluster_tickers in deep_clusters.items():
            print(f"Cluster {cluster_id}: {', '.join(cluster_tickers)}")
    except Exception as e:
        print(f"Error with deep embedding clustering: {e}")