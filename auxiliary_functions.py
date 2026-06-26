"""
Auxiliary functions for graph analysis and visualization.

This module provides utilities for analyzing social network graphs, including:
- Null model generation
- Balance metrics calculation
- Triangle analysis
- Weight distribution visualization
- Kolmogorov-Smirnov statistical tests
"""

import numpy as np
import pandas as pd
import networkx as nx
import random
import scipy.sparse.linalg as spla
import scipy.linalg as la
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import gaussian_kde, percentileofscore
import json


def generate_null_model(graph: nx.Graph, seed: int) -> nx.Graph:
    """
    Generate a null model by randomly shuffling edge weights.
    
    Creates a new graph with the same structure as the input but with
    edge weights randomly reassigned to different edges.
    
    Args:
        graph: Input NetworkX graph with weighted edges.
        
    Returns:
        A new graph with shuffled edge weights.
    """

    random.seed(seed)

    weights = [graph[a][b]["weight"] for a, b in graph.edges()]
    random.shuffle(weights)
    
    null_model = nx.Graph()
    for i, (a, b) in enumerate(graph.edges()):
        null_model.add_edge(a, b, weight=weights[i])
    
    return null_model


def absolute_graph(graph: nx.Graph) -> nx.Graph:
    """
    Convert edge weights to their signs (-1, 0, or +1).
    
    Args:
        graph: Input NetworkX graph with weighted edges.
        
    Returns:
        A new graph where each edge weight is replaced by its sign.
    """
    new_graph = nx.Graph()
    for a, b, data in graph.edges(data=True):
        new_graph.add_edge(a, b, weight=np.sign(data["weight"]))
    
    return new_graph


def calculate_bw(graph: nx.Graph, z: int = 3) -> float:
    """
    Calculate the balance metric BW(α) for a signed graph.
    
    The balance metric is computed as:
    BW(α) = 1/2 Tr[N((αλI - P)^(-1))]
    where N is the negative adjacency matrix, P is the positive adjacency matrix,
    and λ is the maximum eigenvalue of P.
    
    Args:
        graph: Input NetworkX graph with edge weights of -1 or +1.
        z: Scaling parameter (currently unused; α is hardcoded to 2).
        
    Returns:
        The balance metric value.
    """

    n = graph.number_of_nodes()
    if n == 0:
        return 0.0
        
    # 1. Fast sparse matrix construction
    nodes = list(graph.nodes())
    A = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight="weight", format="csr")
    
    # Extract Positive (P) and Negative (N) adjacency matrices
    P = A.copy()
    P.data = np.where(P.data > 0, P.data, 0)
    P.eliminate_zeros()
    
    N = A.copy()
    N.data = np.where(N.data < 0, -N.data, 0) # Convert -1 to 1 to match original logic
    N.eliminate_zeros()
    
    # 2. Fast Maximum Eigenvalue computation
    if P.nnz == 0:
        max_eigenvalue = 0.0
    else:
        if n < 10:
            # Fallback for very small graphs
            max_eigenvalue = max(la.eigvalsh(P.toarray())) 
        else:
            # eigsh finds only the top k eigenvalues, dramatically faster than all of them
            max_eigenvalue = spla.eigsh(P.astype(float), k=1, which='LA', return_eigenvectors=False)[0]
            
    alfa = 2.0
    
    # 3. Dense inversion using Cholesky
    matrix_to_invert = np.eye(n) * (alfa * max_eigenvalue) - P.toarray()
    
    try:
        # Since (αλI - P) is Symmetric Positive Definite, Cholesky factorization 
        # is roughly 2x faster than a standard inverse.
        c, lower = la.cho_factor(matrix_to_invert, check_finite=False)
        inv_matrix = la.cho_solve((c, lower), np.eye(n), check_finite=False)
    except la.LinAlgError:
        # Fallback to standard inverse if matrix acts singular
        inv_matrix = la.inv(matrix_to_invert, check_finite=False)
        
    # 4. Eliminate O(n^3) matrix multiplication for the trace
    N_coo = N.tocoo()
    
    # We only sum the specific elements of inv_matrix where N has non-zero entries
    trace_val = np.sum(N_coo.data * inv_matrix[N_coo.row, N_coo.col])
    
    return trace_val / 2.0



def geo_abs(triangle: list) -> float:
    """
    Calculate the signed geometric mean of triangle edge weights.
    
    Computes the geometric mean of absolute values and applies the sign
    based on the product of signs (negative if odd number of negative edges).
    
    Args:
        triangle: List of three edge weights.
        
    Returns:
        Signed geometric mean of the triangle weights.
    """
    product = (abs(triangle[0]) * abs(triangle[1]) * abs(triangle[2])) ** (1 / 3)
    signs = [np.sign(triangle[0]), np.sign(triangle[1]), np.sign(triangle[2])]
    
    # Apply negative sign if odd number of negative edges
    if signs.count(-1) % 2:
        product *= -1
    
    return product


def in_balance(triangle: list) -> int:
    """
    Determine if a triangle is balanced according to structural balance theory.
    
    A triangle is balanced if it has an even number of negative edges.
    
    Args:
        triangle: List of three edge weights.
        
    Returns:
        1 if balanced (even number of negative edges), 0 otherwise.
    """
    triangle = list(np.sign(np.array(triangle)))
    for el in triangle:
        assert(el)
    
    return 1 - (triangle.count(-1) % 2)


def kolmogorov(vals_a, cum_a, vals_b, cum_b, normalize=False):
    vals_a = np.asarray(vals_a)
    cum_a = np.asarray(cum_a, dtype=float)
    vals_b = np.asarray(vals_b)
    cum_b = np.asarray(cum_b, dtype=float)

    all_vals = np.array(sorted(set(vals_a.tolist()) | set(vals_b.tolist())))
    if all_vals.size == 0:
        return 0.0

    def ff(vals, cum):
        if vals.size == 0:
            return np.zeros_like(all_vals, dtype=float)
        idx = np.searchsorted(vals, all_vals, side='right') - 1
        return np.where(idx >= 0, cum[idx], 0.0)

    y_a = ff(vals_a, cum_a)
    y_b = ff(vals_b, cum_b)

    if normalize:
        if y_a[-1] > 0:
            y_a = y_a / y_a[-1]
        if y_b[-1] > 0:
            y_b = y_b / y_b[-1]

    return float(np.max(np.abs(y_a - y_b)))


def find_alfa(vals_a: np.ndarray, cum_a: np.ndarray,
              vals_b: np.ndarray, cum_b: np.ndarray) -> float:
    """
    Calculate the p-value for the Kolmogorov-Smirnov test.
    
    Uses the asymptotic distribution of the KS statistic to compute
    a p-value indicating the probability that two distributions are identical.
    
    Args:
        vals_a: Sorted x-values for distribution A.
        cum_a: Cumulative counts for distribution A.
        vals_b: Sorted x-values for distribution B.
        cum_b: Cumulative counts for distribution B.
        
    Returns:
        P-value for the two-sample KS test.
    """
    D = kolmogorov(vals_a, cum_a, vals_b, cum_b, normalize=True)
    n_a = len(vals_a)
    n_b = len(vals_b)
    
    return 2 * np.exp(-2 * D * D * n_a * n_b / (n_b + n_a))


def plot_weight_distribution(graph: nx.Graph) -> None:
    """
    Visualize edge weight and sign distributions for a single graph.
    
    Creates a two-column plot:
    - Left: Histogram of edge weights
    - Right: Histogram of edge signs (negative/neutral/positive)
    
    Args:
        graph: Input NetworkX graph with weighted edges.
    """
    # Configuration for weight distribution histograms
    WEIGHT_BIN_CENTERS = np.arange(-1.0, 1.01, 0.1)
    WEIGHT_BIN_EDGES = np.append(WEIGHT_BIN_CENTERS - 0.05, WEIGHT_BIN_CENTERS[-1] + 0.05)

    # Configuration for sign distribution histograms
    SIGN_BIN_EDGES = [-1.5, -0.5, 0.5, 1.5]
    SIGN_LABELS = ["negative", "neutral", "positive"]
    SIGN_TICK_POSITIONS = [-1, 0, 1]

    # Create subplots: one row, two columns
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    def add_percentage_labels(ax, counts, bins, fontsize=14):
        """
        Add percentage labels on top of histogram bars.
        
        Args:
            ax: Matplotlib axes object.
            counts: Histogram bin counts.
            bins: Histogram bin edges.
            fontsize: Font size for labels.
        """
        total = counts.sum()
        
        for count, bin_left, bin_right in zip(counts, bins[:-1], bins[1:]):
            percent = 100 * count / total if total > 0 else 0
            x_position = (bin_left + bin_right) / 2
            ax.text(x_position, count, f"{percent:.1f}%", 
                    ha='center', va='bottom', fontsize=fontsize)

    # Generate histograms for the single graph
    weights = np.array([data["weight"] for _, _, data in graph.edges(data=True)])
    
    # Left panel: Weight distribution
    ax_weight = axes[0]
    ax_weight.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7)
    ax_weight.set_axisbelow(True)
    
    counts, bins, patches = ax_weight.hist(weights, bins=WEIGHT_BIN_EDGES, edgecolor="black", alpha=0.85)
    
    # Color patches dynamically based on weight values
    for patch, bin_left, bin_right in zip(patches, WEIGHT_BIN_EDGES[:-1], WEIGHT_BIN_EDGES[1:]):
        center = (bin_left + bin_right) / 2
        if center < -0.05:
            patch.set_facecolor(plt.cm.Reds(0.3 + 0.7 * abs(center)))
        elif center > 0.05:
            patch.set_facecolor(plt.cm.Greens(0.3 + 0.7 * center))
        else:
            patch.set_facecolor('#bdc3c7')
            
    ax_weight.set_xticks(WEIGHT_BIN_CENTERS)
    ax_weight.set_xticklabels([f"{x:.1f}" for x in WEIGHT_BIN_CENTERS], rotation=60)
    add_percentage_labels(ax_weight, counts, WEIGHT_BIN_EDGES)
    ax_weight.set_title("Weight Distribution", fontsize=20, fontweight='bold')
    ax_weight.set_xlabel("Weight", fontsize=18)
    ax_weight.set_ylabel("Count", fontsize=18)
    ax_weight.tick_params(axis='both', labelsize=16)
    ax_weight.spines['top'].set_visible(False)
    ax_weight.spines['right'].set_visible(False)
    
    # Right panel: Sign distribution
    ax_sign = axes[1]
    ax_sign.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7)
    ax_sign.set_axisbelow(True)
    
    sign_counts, bins, patches = ax_sign.hist(np.sign(weights), bins=SIGN_BIN_EDGES, 
                                    edgecolor="black", rwidth=0.8, alpha=0.85)
                                    
    # Color coding the 3 bars: negative (red), neutral (gray), positive (green)
    colors = ['#e74c3c', '#bdc3c7', '#2ecc71']
    for idx, patch in enumerate(patches):
        if idx < len(colors):
            patch.set_facecolor(colors[idx])
            
    ax_sign.set_xticks(SIGN_TICK_POSITIONS)
    ax_sign.set_xticklabels(SIGN_LABELS, fontsize=16)
    add_percentage_labels(ax_sign, sign_counts, SIGN_BIN_EDGES)
    ax_sign.set_title("Sign Distribution", fontsize=20, fontweight='bold')
    ax_sign.set_xlabel("Sign", fontsize=18)
    ax_sign.tick_params(axis='both', labelsize=16)
    ax_sign.spines['top'].set_visible(False)
    ax_sign.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


def simplify_graph(graph: nx.Graph, std_threshold: float, continuous:bool) -> nx.Graph:
    """
    Simplify a multi-edge graph by consolidating parallel edges.
    
    For each pair of nodes:
    1. Collects all edge weights (treating graph as undirected)
    2. Filters out edges with high variance (std >= threshold)
    3. Replaces multiple edges with a single edge having mean weight
    4. Removes zero-weight edges and self-loops
    
    Args:
        graph: MultiDiGraph with 'subreddit' and 'weight' edge attributes.
        std_threshold: Maximum allowed standard deviation for edge weights.
        
    Returns:
        Dictionary mapping subreddit names to simplified undirected graphs.
    """
    edge_weights = defaultdict(lambda: defaultdict(list))
    
    for node_a, node_b, edge_data in graph.edges(data=True):
        edge_weights[node_a][node_b].append(edge_data["weight"])
    
    # Create simplified undirected graph
    simplified_graph = nx.Graph()

    excluded = 0
        
    for node in edge_weights.keys():
        for neighbor in edge_weights[node].keys():
            if node == neighbor:  # Skip self-loops
                continue
            
            # Collect weights in both directions
            forward_weights = edge_weights[node][neighbor]
            backward_weights = (edge_weights[neighbor][node] 
                            if neighbor in edge_weights and node in edge_weights[neighbor] 
                            else [])
            all_weights = np.array(forward_weights + backward_weights)
            
            # Add edge with mean weight (skip zero weights)
            mean_weight = np.mean(all_weights)
            if mean_weight == 0:
                continue
            
            # Filter by variance threshold
            if all_weights.std() >= std_threshold:
                excluded += 1
                continue
            
            
            simplified_graph.add_edge(node, neighbor, weight=mean_weight if continuous else np.sign(mean_weight))
    
    print(excluded)
    
    return simplified_graph


def calculate_triangles_graph(graph: nx.Graph) -> list:
    """
    Find all triangles in each graph and extract their edge weights.
    
    A triangle is a set of three nodes where each pair is connected by an edge.
    
    Args:
        graphs: Dictionary mapping subreddit names to NetworkX graphs.
        
    Returns:
        Dictionary mapping subreddit names to lists of triangles,
        where each triangle is a tuple of three edge weights.
    """
    found = {}

    # Find all triangles using three nested loops
    for node1 in graph.nodes:
        for node2 in graph.neighbors(node1):
            for node3 in graph.neighbors(node2):
                if graph.has_edge(node1, node3):
                    cur_triangle = tuple(sorted([node1, node2, node3]))
                    if cur_triangle not in found:
                        found[cur_triangle] = (graph[node1][node2]["weight"], graph[node1][node3]["weight"], graph[node2][node3]["weight"])
        
    return list(found.values())


def plot_triangle_distribution(triangles_graph: list) -> None:
    """
    Visualize triangle mean and product-sign distributions.

    Creates two histograms:
    - Left: Mean of edge weights within each triangle
    - Right: Sign of the product of the three edge weights
      (negative / neutral / positive)

    Args:
        triangles_graph: List of triangle weight tuples.
    """
    # Configuration for mean distribution histograms (same as weights)
    WEIGHT_BIN_CENTERS = np.arange(-1.0, 1.01, 0.1)
    WEIGHT_BIN_EDGES = np.append(WEIGHT_BIN_CENTERS - 0.05, WEIGHT_BIN_CENTERS[-1] + 0.05)

    # Configuration for sign distribution histograms
    SIGN_BIN_EDGES = [-1.5, -0.5, 0.5, 1.5]
    SIGN_LABELS = ["negative", "neutral", "positive"]
    SIGN_TICK_POSITIONS = [-1, 0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    def add_percentage_labels(ax, counts, bins, fontsize=14):
        """
        Add percentage labels on top of histogram bars.
        """
        total = counts.sum()
        for count, bin_left, bin_right in zip(counts, bins[:-1], bins[1:]):
            percent = 100 * count / total if total > 0 else 0
            x_position = (bin_left + bin_right) / 2
            ax.text(x_position, count, f"{percent:.1f}%", ha='center', va='bottom', fontsize=fontsize)

    # Prepare data
    if triangles_graph is None or len(triangles_graph) == 0:
        means = np.array([])
        products = np.array([])
        signs = np.array([])
    else:
        # Ensure triangles are sequences of three numbers
        tri_arr = [np.array(tri, dtype=float) for tri in triangles_graph if len(tri) == 3]
        if len(tri_arr) == 0:
            means = np.array([])
            products = np.array([])
            signs = np.array([])
        else:
            means = np.array([tri.mean() for tri in tri_arr])
            products = np.array([tri.prod() for tri in tri_arr])
            signs = np.sign(products)

    # Left panel: Mean of triangle weights
    ax_mean = axes[0]
    ax_mean.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7)
    ax_mean.set_axisbelow(True)
    
    if means.size > 0:
        counts, bins, patches = ax_mean.hist(means, bins=WEIGHT_BIN_EDGES, edgecolor="black", alpha=0.85)
        # Color patches dynamically
        for patch, bin_left, bin_right in zip(patches, WEIGHT_BIN_EDGES[:-1], WEIGHT_BIN_EDGES[1:]):
            center = (bin_left + bin_right) / 2
            if center < -0.05:
                patch.set_facecolor(plt.cm.Reds(0.3 + 0.7 * abs(center)))
            elif center > 0.05:
                patch.set_facecolor(plt.cm.Greens(0.3 + 0.7 * center))
            else:
                patch.set_facecolor('#bdc3c7')
        ax_mean.set_xticks(WEIGHT_BIN_CENTERS)
        ax_mean.set_xticklabels([f"{x:.1f}" for x in WEIGHT_BIN_CENTERS], rotation=60)
        add_percentage_labels(ax_mean, counts, WEIGHT_BIN_EDGES)
    else:
        ax_mean.text(0.5, 0.5, 'No triangles', ha='center', va='center', fontsize=14, transform=ax_mean.transAxes)

    ax_mean.set_title("Triangle Mean Weights", fontsize=20, fontweight='bold')
    ax_mean.set_xlabel("Mean weight", fontsize=18)
    ax_mean.set_ylabel("Count", fontsize=18)
    ax_mean.tick_params(axis='both', labelsize=16)
    ax_mean.spines['top'].set_visible(False)
    ax_mean.spines['right'].set_visible(False)

    # Right panel: Sign of product of edges
    ax_sign = axes[1]
    ax_sign.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7)
    ax_sign.set_axisbelow(True)
    
    if signs.size > 0:
        sign_counts, bins, patches = ax_sign.hist(signs, bins=SIGN_BIN_EDGES, edgecolor="black", rwidth=0.8, alpha=0.85)
        colors = ['#e74c3c', '#bdc3c7', '#2ecc71']
        for idx, patch in enumerate(patches):
            if idx < len(colors):
                patch.set_facecolor(colors[idx])
        ax_sign.set_xticks(SIGN_TICK_POSITIONS)
        ax_sign.set_xticklabels(SIGN_LABELS, fontsize=16)
        add_percentage_labels(ax_sign, sign_counts, SIGN_BIN_EDGES)
    else:
        ax_sign.text(0.5, 0.5, 'No triangles', ha='center', va='center', fontsize=14, transform=ax_sign.transAxes)

    ax_sign.set_title("Sign of Triangle Product", fontsize=20, fontweight='bold')
    ax_sign.set_xlabel("Sign", fontsize=18)
    ax_sign.tick_params(axis='both', labelsize=16)
    ax_sign.spines['top'].set_visible(False)
    ax_sign.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()



def number_of_triangles_per_type(triangles_graph: list) -> pd.DataFrame:
    """
    Count triangles by the number of positive edges they contain.
    
    Categorizes each triangle into four types based on how many of its
    three edges have positive weight.
    
    Args:
        graphs: Dictionary mapping subreddit names to NetworkX graphs.
        triangles_graph: Dictionary mapping subreddit names to triangle weight tuples.
        
    Returns:
        DataFrame with columns for 0, 1, 2, and 3 positive edges,
        and rows for each subreddit.
    """
    series_dict = {}

    qnt_pos = [0, 0, 0, 0]  # Count for 0, 1, 2, 3 positive edges

    for triangle in triangles_graph:
        qnt_pos[list(np.sign(triangle)).count(1)] += 1

    qnt_pos_series = pd.Series(qnt_pos)
    series_dict = qnt_pos_series

    df = pd.DataFrame(series_dict).T
    df.columns = ['0 pos edges', '1 pos edge', '2 pos edges', '3 pos edges']
    return df


def calculate_balance_metrics(graph, null_models: list, 
                              NumberOfRandoms: int) -> dict:
    """
    Calculate balance metrics comparing real graphs to null models.
    
    Computes the BW balance metric for each graph and compares it to
    the average BW of randomly shuffled null models.
    
    Args:
        graphs: Dictionary mapping subreddit names to NetworkX graphs.
        null_models: Dictionary mapping subreddit names to lists of null model graphs.
        NumberOfRandoms: Number of null model realizations to average.
        
    Returns:
        DataFrame with columns 'B_w' (real), 'Standard_B_w' (null average),
        and 'nu_w' (ratio of real to null).
    """
    results = {}
    distributions = []
    
    # Convert to absolute (signed) graphs
    simplified_original_graph = absolute_graph(graph)
    simplified_null_model = [absolute_graph(null_models[i]) 
                            for i in range(NumberOfRandoms)]
    
    # Calculate balance metrics
    b_w = calculate_bw(simplified_original_graph)

    null_model_distribution = []

    for i in range(NumberOfRandoms):
        null_model_distribution.append(calculate_bw(simplified_null_model[i]))
        print(i, end=' ')

    null_model_distribution = np.array(null_model_distribution)
    
    distributions.append(null_model_distribution)
    
    mean = np.mean(null_model_distribution)
    
    std = np.std(null_model_distribution)
    
    percentile = (np.sum(np.array(null_model_distribution) < b_w) / len(null_model_distribution)) * 100
    
    results = {
        'bw': b_w, 
        "mean": mean, 
        "std": std,
        "z-score": (b_w - mean) / std,
        "percentile": percentile
    }

    return results, distributions


def calculate_triangles_null_graph(null_models: list) -> list:
    """
    Calculate triangles for all null model realizations.
    
    Finds triangles in each null model graph and extracts their edge weights.
    
    Args:
        graphs: Dictionary mapping subreddit names to NetworkX graphs.
        null_models: Dictionary mapping subreddit names to lists of null model graphs.
        
    Returns:
        Dictionary mapping subreddit names to lists of triangle weight lists,
        where each inner list corresponds to one null model realization.
    """
    null_triangles = []
        
    for null_graph in null_models:
        found = {}

        # Find all triangles using three nested loops
        for node1 in null_graph.nodes:
            for node2 in null_graph.neighbors(node1):
                for node3 in null_graph.neighbors(node2):
                    if null_graph.has_edge(node1, node3):
                        cur_triangle = tuple(sorted([node1, node2, node3]))
                        if cur_triangle not in found:
                            found[cur_triangle] = (null_graph[node1][node2]["weight"], null_graph[node1][node3]["weight"], null_graph[node2][node3]["weight"])
            
        
        null_triangles.append(list(found.values()))
    
    return null_triangles


def non_binary_metric(triangles_graph: list, null_triangles: list):
    """
    Calculate a non-binary balance metric for triangles.
    
    Computes the average signed geometric mean of triangle weights for real data
    and compares it to the average across null models.
    
    Args:
        triangles_graph: List of triangle weight tuples.
        null_triangles: List of lists of null triangle weights.
        
    Returns:
        DataFrame with columns 'prod' (real metric),
        'avg_null' (null average), and 'ratio' (real/null).
    """
    # Calculate metric for real data
    prod = 0
    for triangle in triangles_graph:
        temp = triangle[0] * triangle[1] * triangle[2]
        temp *= abs(temp) ** (1/3)
        prod += temp
    prod /= len(triangles_graph)
    
    # Calculate metric for null models
    null_model_distribution = []
    for null_triangle_list in null_triangles:
        null_prod = 0
        for triangle in null_triangle_list:
            temp = triangle[0] * triangle[1] * triangle[2]
            temp *= abs(temp) ** (1/3)
            null_prod += temp
        null_prod /= len(null_triangle_list)
        null_model_distribution.append(null_prod)
    
    null_model_distribution = np.array(null_model_distribution)

    mean = np.mean(null_model_distribution)
    
    std = np.std(null_model_distribution)
    
    percentile = (np.sum(null_model_distribution < prod) / len(null_model_distribution)) * 100

    
    results = {
        'prod': prod,
        "mean": mean, 
        "std": std,
        "z-score": (prod - mean) / std,
        "percentile": percentile
    }

    results_df = pd.DataFrame([results])

    return results_df, null_model_distribution

def prepare_cumulative(d):
    """Convert dictionary to sorted values and cumulative counts."""
    if len(d) == 0:
        return np.array([]), np.array([])
    vals = np.array(sorted(d.keys()))
    counts = np.array([d[v] for v in vals], dtype=int)
    return vals, np.cumsum(counts)

def average_null_models(dict_list):
    """Average cumulative distributions across null models using forward-fill."""
    if not dict_list:
        return np.array([]), np.array([])
    
    # Collect all unique values
    all_vals = set()
    for d in dict_list:
        all_vals.update(d.keys())
    
    if not all_vals:
        return np.array([]), np.array([])
    
    all_vals = np.array(sorted(all_vals))
    
    # Build cumulative curves with forward-fill
    cumulative_curves = []
    for d in dict_list:
        vals, cum = prepare_cumulative(d)
        if len(vals) == 0:
            cumulative_curves.append(np.zeros_like(all_vals))
        else:
            cum_interp = np.searchsorted(vals, all_vals, side='right') - 1
            cum_interp = np.where(cum_interp >= 0, cum[cum_interp], 0)
            cumulative_curves.append(cum_interp)
    
    avg_cumulative = np.mean(cumulative_curves, axis=0)
    return all_vals, avg_cumulative

def kolmogorov_smirnov(triangles_graph: dict, null_triangles: dict) -> None:
    """
    Perform Kolmogorov-Smirnov test comparing real and null triangle distributions.
    
    For each subreddit:
    1. Computes cumulative distribution of signed geometric means
    2. Compares real data to average of null models
    3. Calculates KS statistic and p-value
    4. Visualizes distributions with step plots
    
    Args:
        triangles_graph: Dictionary mapping subreddit names to triangle weight tuples.
        null_triangles: Dictionary mapping subreddit names to lists of null triangle weights.
    """
    n_subreddits = len(triangles_graph)
    fig, axes = plt.subplots(n_subreddits, 1, figsize=(10, 5 * n_subreddits))

    # Ensure axes is iterable for single subplot
    if n_subreddits == 1:
        axes = [axes]

    kolmogorov_results = {}

    for idx, (subreddit, triangle_list) in enumerate(triangles_graph.items()):
        # Process real data: compute signed geometric means
        all_triangles_dict = defaultdict(int)
        for triangle in triangle_list:
            val = geo_abs(triangle)
            all_triangles_dict[val] += 1
        
        # Process null models
        null_all_dicts = []
        for null_model in null_triangles[subreddit]:
            null_all = defaultdict(int)
            for triangle in null_model:
                val = geo_abs(triangle)
                null_all[val] += 1
            null_all_dicts.append(null_all)
        
       
        # Get cumulative curves
        vals_all, cum_all = prepare_cumulative(all_triangles_dict)
        vals_all_null, cum_all_null = average_null_models(null_all_dicts)

        # Calculate Kolmogorov statistics
        if vals_all.size > 0 and vals_all_null.size > 0:
            D = int(round(kolmogorov(vals_all, cum_all, vals_all_null, cum_all_null)))
            alfa = find_alfa(vals_all, cum_all, vals_all_null, cum_all_null)
            kolmogorov_results[subreddit] = {'D': D, 'p-value': alfa}
        else:
            kolmogorov_results[subreddit] = {'D': None, 'p-value': None}

        # Plot cumulative distributions
        x_min, x_max = -1, 1
        ax = axes[idx]

        # Plot real data
        if vals_all.size > 0:
            ax.step(vals_all, cum_all, where='post', color='C0', 
                   linewidth=2, label='Real data')
            ax.plot(vals_all, cum_all, 'o', color='C0', markersize=6)
        
        # Plot null model average
        if vals_all_null.size > 0:
            ax.step(vals_all_null, cum_all_null, where='post', color='C2', 
                   linewidth=2, label='Null model avg')
            ax.plot(vals_all_null, cum_all_null, 's', color='C2', 
                   markersize=5, alpha=0.7)
        
        # Handle empty data
        if vals_all.size == 0 and vals_all_null.size == 0:
            ax.text(0.5, 0.5, 'No triangles', ha='center', va='center', 
                   fontsize=12, transform=ax.transAxes)
        
        # Add title with statistics
        if kolmogorov_results[subreddit]['D'] is not None:
            title_text = (f"{subreddit} (D={kolmogorov_results[subreddit]['D']}, "
                         f"p={kolmogorov_results[subreddit]['p-value']:.4f})")
        else:
            title_text = f"{subreddit}"
        
        ax.set_xlabel('Geometric mean weight', fontsize=12)
        ax.set_ylabel('Accumulated number of triangles', fontsize=12)
        ax.set_title(title_text, fontsize=14)
        ax.grid(alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.legend(fontsize=10)

    plt.suptitle("Accumulated triangles: Real vs Null Model", fontsize=16, y=0.998)
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nKolmogorov-Smirnov Test Results:")
    print("=" * 60)
    for subreddit, stats in kolmogorov_results.items():
        if stats['D'] is not None:
            print(f"{subreddit:30s} | D = {stats['D']:8.4f} | "
                  f"p-value = {stats['p-value']:8.4f}")
        else:
            print(f"{subreddit:30s} | No data available")

def top_nodes_stats(graph: nx.Graph, k: int):
    # one-column + triangle-product column + triangle-sign-count column; per-node plots for top-k nodes
    # ensure degrees are sorted in descending order
    top_k = sorted(graph.degree(), key=lambda x: x[1], reverse=True)[:min(k, graph.number_of_nodes())]
    top_nodes = [n for n, _ in top_k]

    print("Nodes plotted in descending degree order:")
    for n, d in top_k:
        print(f"  {n}: {d}")

    # helper to read edge weight
    def _get_weight(g, u, v):
        attr = g[u][v]
        if isinstance(attr, dict):
            return attr.get("weight", 1.0)
        try:
            return float(attr)
        except Exception:
            return 1.0

    # Build triangles with node identities and weights (avoid duplicates)
    found = {}
    for n1 in graph.nodes():
        for n2 in graph.neighbors(n1):
            for n3 in graph.neighbors(n2):
                if graph.has_edge(n1, n3):
                    tri = tuple(sorted([n1, n2, n3]))
                    if tri not in found:
                        w12 = _get_weight(graph, tri[0], tri[1])
                        w13 = _get_weight(graph, tri[0], tri[2])
                        w23 = _get_weight(graph, tri[1], tri[2])
                        found[tri] = (tri, (w12, w13, w23))

    triangles_with_nodes = list(found.values())

    # Compute global triangle products (same KDE for all nodes)
    global_products = np.array([float(ws[0] * ws[1] * ws[2]) for _, ws in triangles_with_nodes], dtype=float)

    # Precompute KDE for the entire-graph triangle-product distribution
    global_kde = None
    x_vals_global = None
    y_vals_global = None
    if global_products.size > 1:
        try:
            global_kde = gaussian_kde(global_products)
            x_min_g, x_max_g = global_products.min(), global_products.max()
            pad_g = max(0.05 * (x_max_g - x_min_g), 1e-12)
            x_vals_global = np.linspace(x_min_g - pad_g, x_max_g + pad_g, 300)
            # density values (no scaling) so KDE is identical across subplots
            y_vals_global = global_kde(x_vals_global)
        except Exception:
            global_kde = None

    # Map node -> list of triangle product weights it participates in
    node_triangle_products = defaultdict(list)
    # Map node -> list of counts of positive edges (0..3) for each triangle it participates in
    node_triangle_pos_counts = defaultdict(list)
    for nodes, weights in triangles_with_nodes:
        prod = float(weights[0] * weights[1] * weights[2])
        pos_count = int(sum(1 for w in weights if w > 0))
        for n in nodes:
            node_triangle_products[n].append(prod)
            node_triangle_pos_counts[n].append(pos_count)

    # Define fixed 21 bins centered on -1.0, -0.9, ... , 1.0
    centers = np.linspace(-1.0, 1.0, 21)
    # edges are centers +/- 0.05 -> from -1.05 to 1.05 inclusive (22 edges)
    bins_edges = np.linspace(centers[0] - 0.05, centers[-1] + 0.05, len(centers) + 1)

    # compute global mean/std for z-score (robust if variables not present)
    if global_products.size > 0:
        global_mean = np.mean(global_products)
        global_std = np.std(global_products, ddof=0)
    else:
        global_mean = 0.0
        global_std = 0.0

    # Prepare subplots: three columns (left = incident edge weights, middle = global KDE + per-node mean + stats, right = triangle sign-count histogram)
    rows = len(top_nodes)
    fig, axes = plt.subplots(rows, 3, figsize=(18, 2.5 * rows))
    axes = np.atleast_2d(axes)

    for i, (node, deg) in enumerate(top_k):
        ax_w = axes[i, 0]  # edge-weight histogram for node
        ax_p = axes[i, 1]  # triangle-product KDE (global) + per-node stats
        ax_t = axes[i, 2]  # triangle sign-count histogram for node

        # collect weights of edges incident to this node
        weights = []
        for nbr, attr in graph[node].items():
            if nbr == node:
                continue  # skip self-loops
            if isinstance(attr, dict):
                w = attr.get("weight", 1.0)
            else:
                try:
                    w = float(attr)
                except Exception:
                    w = 1.0
            weights.append(w)

        # Left: incident edge weight histogram (fixed 21 bins centered -1..1)
        if len(weights) == 0:
            ax_w.text(0.5, 0.5, "No edges", ha="center", va="center")
            ax_w.set_xticks([])
            ax_w.set_yticks([])
        else:
            ax_w.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7)
            ax_w.set_axisbelow(True)
            counts, bins, patches = ax_w.hist(weights, bins=bins_edges, edgecolor="k", alpha=0.85)
            # Color patches dynamically based on weight
            for patch, bin_left, bin_right in zip(patches, bins_edges[:-1], bins_edges[1:]):
                center = (bin_left + bin_right) / 2
                if center < -0.05:
                    patch.set_facecolor(plt.cm.Reds(0.3 + 0.7 * abs(center)))
                elif center > 0.05:
                    patch.set_facecolor(plt.cm.Greens(0.3 + 0.7 * center))
                else:
                    patch.set_facecolor('#bdc3c7')
            mean_w = np.mean(weights)
            ax_w.axvline(mean_w, color="red", linestyle="--", linewidth=1.5, label=f"mean={mean_w:.4f}")
            ax_w.set_xlabel("Edge weight")
            ax_w.set_ylabel("Frequency")
            ax_w.legend(fontsize="small", loc="upper left")
            ax_w.set_xticks(centers)
            ax_w.set_xlim(bins_edges[0], bins_edges[-1])
            ax_w.spines['top'].set_visible(False)
            ax_w.spines['right'].set_visible(False)

        ax_w.set_title(f"{node} (deg={deg})", loc="left", fontweight="bold")

        # Middle: only global KDE (same for all nodes) + per-node mean vertical and stats (z-score + percentile)
        tri_prods = node_triangle_products.get(node, [])
        if len(tri_prods) == 0:
            ax_p.text(0.5, 0.5, "No triangles", ha="center", va="center")
            ax_p.set_xticks([])
            ax_p.set_yticks([])
        else:
            ax_p.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7)
            ax_p.set_axisbelow(True)
            # plot global KDE (density) if available
            if global_kde is not None:
                ax_p.plot(x_vals_global, y_vals_global, color="darkgreen", linewidth=1.5, label="Global KDE")
                ax_p.fill_between(x_vals_global, y_vals_global, color="darkgreen", alpha=0.15)
                ax_p.set_xlabel("Triangle edge-weight product")
                ax_p.set_ylabel("Density")
                # keep same y-limits for all nodes based on global KDE
                ax_p.set_ylim(0, y_vals_global.max() * 1.05)
                ax_p.set_xlim(x_vals_global[0], x_vals_global[-1])
            else:
                # fallback: show a simple histogram of global products if KDE failed
                ax_p.hist(global_products, bins=bins_edges, color="lightgreen", edgecolor="k", alpha=0.7, density=True)
                ax_p.set_xlabel("Triangle edge-weight product")
                ax_p.set_ylabel("Density")
                ax_p.set_xticks(centers)
                ax_p.set_xlim(bins_edges[0], bins_edges[-1])

            # per-node mean (computed only on node's triangle products)
            arr = np.asarray(tri_prods, dtype=float)
            mean_p = np.mean(arr)
            ax_p.axvline(mean_p, color="red", linestyle="--", linewidth=1.5, label=f"mean={mean_p:.4e}")

            # compute z-score and percentile
            try:
                if global_std == 0:
                    z_score = np.nan
                else:
                    z_score = (mean_p - global_mean) / global_std
                if global_products.size > 0:
                    pct = float(percentileofscore(global_products, mean_p, kind="mean"))
                else:
                    pct = np.nan
            except Exception:
                z_score = np.nan
                pct = np.nan

            # annotate the plot
            stat_text = f"z = {z_score:.3f}\npercentile = {pct:.1f}%"
            ax_p.text(
                0.98, 0.95, stat_text,
                transform=ax_p.transAxes,
                ha="right", va="top",
                fontsize="small",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray")
            )

            ax_p.legend(fontsize="small", loc="upper left")
            ax_p.set_title(f"Triangle products (n={len(arr)})", loc="left", fontweight="bold")
            ax_p.spines['top'].set_visible(False)
            ax_p.spines['right'].set_visible(False)

        # Right: histogram of number of triangles as a function of number of positive edges (0..3)
        pos_counts = node_triangle_pos_counts.get(node, [])
        if len(pos_counts) == 0:
            ax_t.text(0.5, 0.5, "No triangles", ha="center", va="center")
            ax_t.set_xticks([])
            ax_t.set_yticks([])
        else:
            ax_t.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.5)
            ax_t.set_axisbelow(True)
            counts = np.bincount(np.array(pos_counts, dtype=int), minlength=4)[:4]
            labels = ["---", "+--", "++-", "+++"]
            x = np.arange(len(labels))
            bar_colors = ["#d73027", "#fc8d59", "#fee090", "#91bfdb"]  # color ramp
            ax_t.bar(x, counts, color=bar_colors, edgecolor="k", alpha=0.85)
            ax_t.set_xticks(x)
            ax_t.set_xticklabels(labels, fontsize="small")
            ax_t.set_ylabel("Triangle count")
            ax_t.set_xlabel("Triangle edge configuration")
            # annotate counts on bars
            for xi, c in zip(x, counts):
                ax_t.text(xi, c + max(1, 0.01 * sum(counts)), str(int(c)), ha="center", va="bottom", fontsize="small")
            ax_t.set_title(f"Triangles by edges (n={len(pos_counts)})", loc="left", fontweight="bold")
            ax_t.spines['top'].set_visible(False)
            ax_t.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

def calculate_balance_given_distribution(distributions_b, bw):
    mean = np.mean(distributions_b)
    std = np.std(distributions_b)
    z_score = (bw - mean) / std if std != 0 else 0
    percentile = (np.sum(np.array(distributions_b) < bw) / len(distributions_b)) * 100

    results_b = {
        'bw': bw,
        'mean': mean,
        'std': std,
        'z-score': z_score,
        'percentile': percentile
    }

    return results_b

def plot_bw_distribution(distributions_b, results_b):
    distribution = np.asarray(distributions_b)
    title = "Graph Bw Metric Distribution"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7)
    ax.set_axisbelow(True)

    # Try to extract the single graph's Bw value from results_b
    b_w = results_b["bw"]

    if distribution.size == 0:
        ax.text(0.5, 0.5, "Empty distribution", ha='center')
    else:
        # KDE with automatic x-range based on data
        kde = gaussian_kde(distribution)
        x_min, x_max = min(distribution.min(), b_w), max(distribution.max(), b_w)
        pad = max(0.05 * (x_max - x_min), 1e-6)
        x_vals = np.linspace(x_min - pad, x_max + pad, 300)
        y_vals = kde(x_vals)
        ax.plot(x_vals, y_vals, color='#3498db', linewidth=2, label='KDE (Null Models)')
        ax.fill_between(x_vals, y_vals, color='#3498db', alpha=0.3)

        if b_w is not None:
            ax.axvline(b_w, color='#e74c3c', linestyle='--', linewidth=2, label=f'Real Bw: {b_w:.4f}')

    ax.set_title(title, loc='left', fontweight='bold', fontsize=12)
    ax.set_xlim(x_vals.min() if distribution.size else -0.01, x_vals.max() if distribution.size else 0.06)
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

def calculate_kolmogorov_stats(triangles_graph, null_triangles):
    D_distribution = []
    results = []

    all_triangles_dict = defaultdict(int)
    for triangle in triangles_graph:
        val = geo_abs(triangle)
        all_triangles_dict[val] += 1
    vals_all, cum_all = prepare_cumulative(all_triangles_dict)

    null_all_dicts = []
    for null_model in null_triangles:
        null_all = defaultdict(int)
        for triangle in null_model:
            val = geo_abs(triangle)
            null_all[val] += 1
        null_all_dicts.append(null_all)

    vals_all_null, cum_all_null = average_null_models(null_all_dicts)

    for null_all in null_all_dicts:
        vals_null, cum_null = prepare_cumulative(null_all)
        D_null_vs_avg = kolmogorov(vals_null, cum_null, vals_all_null, cum_all_null)
        D_distribution.append(D_null_vs_avg)

    D = kolmogorov(vals_all, cum_all, vals_all_null, cum_all_null)

    mean = np.mean(D_distribution)
    std = np.std(D_distribution)
    percentile = (np.sum(np.array(D_distribution) < D) / len(D_distribution)) * 100

    results = {
        'D': D,
        "mean": mean,
        "std": std,
        "z-score": (D - mean) / std if std != 0 else 0,
        "percentile": percentile
    }

    return results, D_distribution

def plot_kolmogorov(D_distribution, results):
    # Adaptation for single-graph (dimensions reduced by one).
    # Use existing D_distribution and results_df (or fallback to D) without re-importing.

    # Extract single distribution and a sensible title
    if isinstance(D_distribution, dict):
        key = next(iter(D_distribution))
        distribution = np.asarray(D_distribution[key])
        title = f"KS Statistic Distribution: {key}"
    else:
        distribution = np.asarray(D_distribution)
        title = "KS Statistic Distribution"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7)
    ax.set_axisbelow(True)

    if distribution.size == 0:
        ax.text(0.5, 0.5, "Empty distribution", ha='center')
    else:
        # Estimate KDE if there are enough points, otherwise show a rug/hist
        if distribution.size > 1:
            kde = gaussian_kde(distribution)
            x_min, x_max = distribution.min(), distribution.max()
            pad = max(0.05 * (x_max - x_min), 1e-6)
            x_vals = np.linspace(x_min - pad, x_max + pad, 300)
            y_vals = kde(x_vals)
            ax.plot(x_vals, y_vals, color='#2ecc71', linewidth=2, label='KDE (Null Models)')
            ax.fill_between(x_vals, y_vals, color='#2ecc71', alpha=0.3)
        else:
            # single point -> plot a vertical marker and small histogram fallback
            x_vals = np.array([distribution[0]])
            ax.plot(x_vals, np.array([1.0]), marker='o', linestyle='', color='#2ecc71', label='Value')

        d_val = results["D"]
        if d_val is not None:
            ax.axvline(d_val, color='#e74c3c', linestyle='--', linewidth=2, label=f'Real D: {d_val:.4f}')

        # Set x-limits based on data and optional D value
        data_min = distribution.min()
        data_max = distribution.max()
        if d_val is not None:
            data_min = min(data_min, d_val)
            data_max = max(data_max, d_val)
        pad = max(0.05 * (data_max - data_min if data_max != data_min else abs(data_max) + 1e-6), 1e-6)
        ax.set_xlim(data_min - pad, data_max + pad)

    ax.set_title(title, loc='left', fontweight='bold', fontsize=12)
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


def detect_fraudster_groups(graph: nx.Graph, min_neg_degree: int = 2, min_pos_density: float = 0.7, min_balance_ratio: float = 0.9) -> list:
    """
    Detect groups of potential fraudsters in a signed graph.
    
    A group is defined as a set of nodes S where:
    - Each node in S has at least min_neg_degree negative incident edges in the global graph G.
    - S forms a dense positive subgraph (density of positive edges inside S >= min_pos_density).
    - S is in almost perfect structural balance (triangle balance ratio >= min_balance_ratio).
    
    Args:
        graph: Undirected signed NetworkX graph with 'weight' attribute on edges (+1, -1, or continuous).
        min_neg_degree: Minimum number of negative incident edges in G for each node in a group.
        min_pos_density: Minimum positive edge density within the group.
        min_balance_ratio: Minimum ratio of balanced triangles to total triangles within the group.
        
    Returns:
        List of dictionaries, each containing:
        - 'nodes': Set of node IDs in the group.
        - 'size': Number of nodes in the group.
        - 'pos_density': Density of positive edges within the group.
        - 'neg_density': Density of negative edges within the group.
        - 'total_triangles': Total triangles inside the induced subgraph on S.
        - 'balanced_triangles': Number of balanced triangles inside S.
        - 'balance_ratio': Ratio of balanced triangles to total triangles.
    """
    # 1. Filter nodes by global negative degree
    candidate_nodes = []
    for node in graph.nodes():
        neg_deg = sum(1 for nbr in graph.neighbors(node) if graph[node][nbr].get('weight', 1.0) < 0)
        if neg_deg >= min_neg_degree:
            candidate_nodes.append(node)
            
    if not candidate_nodes:
        return []
        
    # 2. Build the positive edge induced subgraph on candidate nodes
    pos_graph = nx.Graph()
    pos_graph.add_nodes_from(candidate_nodes)
    for u, v in graph.edges():
        if u in pos_graph and v in pos_graph:
            w = graph[u][v].get('weight', 1.0)
            if w > 0:
                pos_graph.add_edge(u, v, weight=w)
                
    # 3. Perform community detection using Louvain on the positive candidate graph
    from networkx.algorithms.community import louvain_communities
    try:
        communities = louvain_communities(pos_graph, seed=42)
    except Exception as e:
        # Fallback to connected components if Louvain fails for some reason
        communities = list(nx.connected_components(pos_graph))
        
    detected_groups = []
    
    # 4. Filter and compute metrics for each community/group
    for comm in communities:
        if len(comm) < 3: # Need at least 3 nodes to form triangles and densities
            continue
            
        nodes_list = list(comm)
        n = len(comm)
        possible_edges = n * (n - 1) / 2
        
        # Count positive and negative edges in the induced subgraph G[comm]
        pos_edges_count = 0
        neg_edges_count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                u, v = nodes_list[i], nodes_list[j]
                if graph.has_edge(u, v):
                    w = graph[u][v].get('weight', 1.0)
                    if w > 0:
                        pos_edges_count += 1
                    elif w < 0:
                        neg_edges_count += 1
                        
        pos_density = pos_edges_count / possible_edges
        neg_density = neg_edges_count / possible_edges
        
        # Filter by positive density
        if pos_density < min_pos_density:
            continue
            
        # Calculate triangles and structural balance inside the group
        triangles_inside = 0
        balanced_triangles = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    u, v, w_node = nodes_list[i], nodes_list[j], nodes_list[k]
                    if graph.has_edge(u, v) and graph.has_edge(u, w_node) and graph.has_edge(v, w_node):
                        triangles_inside += 1
                        # Get edge signs
                        s1 = np.sign(graph[u][v].get('weight', 1.0))
                        s2 = np.sign(graph[u][w_node].get('weight', 1.0))
                        s3 = np.sign(graph[v][w_node].get('weight', 1.0))
                        
                        # A triangle is balanced if the product of signs is positive
                        neg_count = sum(1 for s in (s1, s2, s3) if s < 0)
                        if neg_count % 2 == 0:
                            balanced_triangles += 1
                            
        if triangles_inside > 0:
            balance_ratio = balanced_triangles / triangles_inside
        else:
            balance_ratio = 1.0
            
        # Filter by balance ratio
        if balance_ratio < min_balance_ratio:
            continue
            
        detected_groups.append({
            'nodes': comm,
            'size': n,
            'pos_density': pos_density,
            'neg_density': neg_density,
            'total_triangles': triangles_inside,
            'balanced_triangles': balanced_triangles,
            'balance_ratio': balance_ratio
        })
        
    # Sort groups by size descending
    detected_groups.sort(key=lambda x: x['size'], reverse=True)
    return detected_groups


def plot_fraudster_group(graph: nx.Graph, group_nodes: set, title: str = "Fraudster Group", save_path: str = None, ax=None, stats: dict = None):
    """
    Visualize a detected fraudster group with force-directed gravity.
    
    Nodes in the group are plotted, with positive edges colored green and negative edges colored red.
    Nodes are colored based on their global negative degree.
    """
    # Create the induced subgraph
    subgraph = graph.subgraph(group_nodes)
    
    # Define layout weights (absolute weight for spring tension / gravity)
    # NetworkX spring_layout will use this attribute to pull strongly connected nodes closer
    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(subgraph.nodes())
    for u, v in subgraph.edges():
        w = abs(subgraph[u][v].get('weight', 1.0))
        layout_graph.add_edge(u, v, weight=max(w, 0.1))
        
    n_nodes = len(group_nodes)
    k_distance = 1.8 / np.sqrt(n_nodes) if n_nodes > 0 else 1.0
    pos = nx.spring_layout(layout_graph, weight='weight', k=k_distance, iterations=120, seed=42)
    
    # Calculate global negative degrees for node coloring and sizing
    node_neg_degs = []
    for node in group_nodes:
        neg_deg = sum(1 for nbr in graph.neighbors(node) if graph[node][nbr].get('weight', 1.0) < 0)
        node_neg_degs.append(neg_deg)
        
    # Scale node size based on suspicion (gravity)
    node_sizes = [250 + 120 * deg for deg in node_neg_degs]
        
    # Split edges into positive and negative
    pos_edges = []
    neg_edges = []
    pos_widths = []
    neg_widths = []
    
    for u, v in subgraph.edges():
        w = subgraph[u][v].get('weight', 1.0)
        if w > 0:
            pos_edges.append((u, v))
            pos_widths.append(1.0 + 3.0 * abs(w))
        elif w < 0:
            neg_edges.append((u, v))
            neg_widths.append(1.0 + 3.0 * abs(w))
            
    # Handle axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        show_colorbar = True
        show_plot = True
    else:
        show_colorbar = False
        show_plot = False
        
    # Draw nodes with YlOrRd colormap
    cmap = plt.cm.YlOrRd
    nodes = nx.draw_networkx_nodes(
        subgraph, pos, ax=ax,
        node_color=node_neg_degs,
        node_size=node_sizes,
        cmap=cmap,
        edgecolors='#2c3e50',
        linewidths=1.5,
        alpha=0.9
    )
    
    # Draw positive edges in vibrant teal/green
    if pos_edges:
        nx.draw_networkx_edges(
            subgraph, pos, ax=ax,
            edgelist=pos_edges,
            edge_color='#2ecc71',
            width=pos_widths,
            alpha=0.8
        )
    
    # Draw negative edges in coral red
    if neg_edges:
        nx.draw_networkx_edges(
            subgraph, pos, ax=ax,
            edgelist=neg_edges,
            edge_color='#e74c3c',
            width=neg_widths,
            style='dashed',
            alpha=0.8
        )
    
    # Draw labels
    nx.draw_networkx_labels(subgraph, pos, ax=ax, font_size=8, font_weight='bold', font_color='#2c3e50')
    
    # Draw stats if provided inside a textbox
    if stats is not None:
        stats_text = (
            f"Size: {stats['size']}\n"
            f"Pos Density: {stats['pos_density']:.2f}\n"
            f"Neg Density: {stats['neg_density']:.2f}\n"
            f"Balance Ratio: {stats['balance_ratio']:.2f}"
        )
        ax.text(
            0.02, 0.02, stats_text,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="lightgray")
        )
        
    # Colorbar for global negative degree
    if show_colorbar:
        cbar = fig.colorbar(nodes, ax=ax, orientation='horizontal', pad=0.08, shrink=0.7)
        cbar.set_label("Global Negative Degree (Suspicion Level)", fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        
    ax.set_title(title, fontsize=12, fontweight='bold', color='#2c3e50')
    ax.axis('off')
    
    if show_plot:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_all_fraudster_groups(graph: nx.Graph, groups: list, title: str = "Detected Fraudster Groups", save_path: str = None):
    """
    Plot all detected groups on a grid of subplots within a single Matplotlib figure.
    Includes a single shared figure-level legend at the bottom.
    """
    n_groups = len(groups)
    if n_groups == 0:
        print("No groups to plot.")
        return
        
    # Determine grid dimensions (2 columns)
    n_cols = 2
    n_rows = (n_groups + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, (group, ax) in enumerate(zip(groups, axes)):
        plot_fraudster_group(
            graph, 
            group['nodes'], 
            title=f"Group {idx+1} (Size {group['size']})", 
            ax=ax, 
            stats=group
        )
        
    # Hide unused axes
    for ax in axes[n_groups:]:
        ax.axis('off')
        
    # Add a single global legend at the bottom of the figure
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2ecc71', lw=3, label='Positive (Friendship Link)'),
        Line2D([0], [0], color='#e74c3c', lw=3, linestyle='--', label='Negative (Conflict Link)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12, frameon=True, edgecolor='lightgray')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.99, color='#2c3e50')
    
    # Adjust layout to prevent overlap and leave space for the bottom legend
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

