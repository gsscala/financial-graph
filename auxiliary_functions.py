"""
Auxiliary functions for graph analysis, structural balance calculation, and network visualization.

This module provides production-grade utilities for analyzing social network graphs, including:
- Null model generation and network simplification
- Structural balance metrics calculation (B_w metric using Cholesky factorization)
- Triangle extraction, classification, and statistical analysis
- Kolmogorov-Smirnov tests for distribution validation
- Modern, grid-styled plotting utilities for distributions and local node statistics
- Cohesive fraudster group detection and community visualization
"""

import os
import json
import random
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, percentileofscore


# ==============================================================================
# 1. HELPERS & DRY UTILITIES
# ==============================================================================

def _apply_plot_style(ax: plt.Axes, title: str) -> None:
    """Apply unified modern layout styling to a matplotlib axis.
    
    Args:
        ax: Matplotlib axes object.
        title: Title string of the plot.
    """
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, fontsize=12, fontweight='bold', color='#2c3e50', pad=10)
    ax.tick_params(axis='both', labelsize=10, colors='#2c3e50')
    ax.xaxis.label.set_color('#2c3e50')
    ax.yaxis.label.set_color('#2c3e50')


def _find_triangles(graph: nx.Graph) -> list[tuple[float, float, float]]:
    """Find all unique triangles in a graph and extract their edge weights.
    
    Args:
        graph: Input signed graph.
        
    Returns:
        List of tuples, where each tuple contains three edge weights.
    """
    found = {}
    for node1 in graph.nodes():
        for node2 in graph.neighbors(node1):
            for node3 in graph.neighbors(node2):
                if graph.has_edge(node1, node3):
                    cur_triangle = tuple(sorted([node1, node2, node3]))
                    if cur_triangle not in found:
                        found[cur_triangle] = (
                            graph[node1][node2]["weight"], 
                            graph[node1][node3]["weight"], 
                            graph[node2][node3]["weight"]
                        )
    return list(found.values())


# ==============================================================================
# 2. GRAPH SIMPLIFICATION & NULL MODELS
# ==============================================================================

def generate_null_model(graph: nx.Graph, seed: int) -> nx.Graph:
    """Generate a null model by randomly shuffling edge weights.
    
    Args:
        graph: Input NetworkX graph with weighted edges.
        seed: Random seed for shuffling.
        
    Returns:
        A new graph with identical structure but shuffled edge weights.
    """
    random.seed(seed)
    weights = [graph[a][b]["weight"] for a, b in graph.edges()]
    random.shuffle(weights)
    
    null_model = nx.Graph()
    for i, (a, b) in enumerate(graph.edges()):
        null_model.add_edge(a, b, weight=weights[i])
    return null_model


def absolute_graph(graph: nx.Graph) -> nx.Graph:
    """Convert edge weights to their signs (-1.0, 0.0, or +1.0).
    
    Args:
        graph: Input NetworkX graph with weighted edges.
        
    Returns:
        A new graph where each edge weight is replaced by its sign.
    """
    new_graph = nx.Graph()
    for a, b, data in graph.edges(data=True):
        new_graph.add_edge(a, b, weight=float(np.sign(data["weight"])))
    return new_graph


def simplify_graph(graph: nx.MultiDiGraph, std_threshold: float, continuous: bool) -> nx.Graph:
    """Consolidate parallel edges by taking mean weight and filtering by standard deviation.
    
    Args:
        graph: MultiDiGraph containing parallel edges.
        std_threshold: Maximum allowed standard deviation for weights between the same nodes.
        continuous: If True, keeps mean weights; if False, applies np.sign(mean).
        
    Returns:
        A simplified undirected Graph.
    """
    edge_weights = defaultdict(lambda: defaultdict(list))
    for node_a, node_b, edge_data in graph.edges(data=True):
        edge_weights[node_a][node_b].append(edge_data["weight"])
    
    simplified_graph = nx.Graph()
    for node in edge_weights.keys():
        for neighbor in edge_weights[node].keys():
            if node == neighbor:
                continue
            
            forward_weights = edge_weights[node][neighbor]
            backward_weights = (
                edge_weights[neighbor][node] 
                if neighbor in edge_weights and node in edge_weights[neighbor] 
                else []
            )
            all_weights = np.array(forward_weights + backward_weights)
            mean_weight = float(np.mean(all_weights))
            
            if mean_weight == 0.0 or all_weights.std() >= std_threshold:
                continue
            
            val = mean_weight if continuous else float(np.sign(mean_weight))
            simplified_graph.add_edge(node, neighbor, weight=val)
            
    return simplified_graph


# ==============================================================================
# 3. BALANCE METRIC CALCULATION (B_w)
# ==============================================================================

def calculate_bw(graph: nx.Graph, z: int = 3) -> float:
    """Calculate the Bw balance metric for a signed graph using Cholesky factor solver.
    
    Args:
        graph: Input signed graph.
        z: Scaling parameter (unused, maintained for signature compatibility).
        
    Returns:
        The calculated Bw balance metric.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return 0.0
        
    nodes = list(graph.nodes())
    A = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight="weight", format="csr")
    
    P = A.copy()
    P.data = np.where(P.data > 0, P.data, 0)
    P.eliminate_zeros()
    
    N = A.copy()
    N.data = np.where(N.data < 0, -N.data, 0)
    N.eliminate_zeros()
    
    if P.nnz == 0:
        max_eigenvalue = 0.0
    else:
        if n < 10:
            max_eigenvalue = max(la.eigvalsh(P.toarray())) 
        else:
            max_eigenvalue = spla.eigsh(P.astype(float), k=1, which='LA', return_eigenvectors=False)[0]
            
    alfa = 2.0
    diag_coeff = max(alfa * max_eigenvalue, 1e-5)
    matrix_to_invert = np.eye(n) * diag_coeff - P.toarray()
    
    try:
        c, lower = la.cho_factor(matrix_to_invert, check_finite=False)
        inv_matrix = la.cho_solve((c, lower), np.eye(n), check_finite=False)
    except la.LinAlgError:
        try:
            inv_matrix = la.inv(matrix_to_invert, check_finite=False)
        except la.LinAlgError:
            inv_matrix = np.zeros((n, n))
        
    N_coo = N.tocoo()
    trace_val = np.sum(N_coo.data * inv_matrix[N_coo.row, N_coo.col])
    return float(trace_val / 2.0)


def calculate_balance_metrics(graph: nx.Graph, null_models: list[nx.Graph], NumberOfRandoms: int) -> tuple[dict, list[list[float]]]:
    """Calculate Bw metrics comparing the original graph to randomized null models.
    
    Args:
        graph: Real graph.
        null_models: List of null models.
        NumberOfRandoms: Number of random models.
        
    Returns:
        Tuple of (results_dict, [null_model_distribution_list]).
    """
    simplified_original_graph = absolute_graph(graph)
    simplified_null_model = [absolute_graph(null_models[i]) for i in range(NumberOfRandoms)]
    
    b_w = calculate_bw(simplified_original_graph)
    null_model_distribution = []
    
    for i in range(NumberOfRandoms):
        null_model_distribution.append(calculate_bw(simplified_null_model[i]))
        
    null_model_distribution = np.array(null_model_distribution)
    mean = np.mean(null_model_distribution)
    std = np.std(null_model_distribution)
    percentile = (np.sum(null_model_distribution < b_w) / len(null_model_distribution)) * 100
    
    results = {
        'bw': float(b_w), 
        "mean": float(mean), 
        "std": float(std),
        "z-score": float((b_w - mean) / std) if std != 0 else 0.0,
        "percentile": float(percentile)
    }
    return results, [null_model_distribution.tolist()]


def calculate_balance_given_distribution(distributions_b: list[float], bw: float) -> dict:
    """Compute balance metrics given a pre-computed Bw distribution.
    
    Args:
        distributions_b: List of null model Bw values.
        bw: Real graph Bw value.
        
    Returns:
        Dict containing stats (bw, mean, std, z-score, percentile).
    """
    mean = np.mean(distributions_b)
    std = np.std(distributions_b)
    z_score = (bw - mean) / std if std != 0 else 0.0
    percentile = (np.sum(np.array(distributions_b) < bw) / len(distributions_b)) * 100
    
    return {
        'bw': float(bw),
        'mean': float(mean),
        'std': float(std),
        'z-score': float(z_score),
        'percentile': float(percentile)
    }


def get_balance_analysis_results(
    graph: nx.Graph, 
    null_models: list[nx.Graph], 
    NumberOfRandoms: int, 
    recompute: bool = False, 
    cache_path: str = "distribution.json"
) -> tuple[dict, list[float]]:
    """Get Bw metrics and null distribution, dynamically loaded from cache or computed.
    
    Args:
        graph: Undirected signed Graph.
        null_models: List of null model graphs.
        NumberOfRandoms: Number of random realizations to run.
        recompute: If True, bypasses cache and recomputes the simulation.
        cache_path: Path to the cached JSON distribution file.
        
    Returns:
        A tuple of (results_dict, null_distribution_list).
    """
    simplified_original_graph = absolute_graph(graph)
    b_w = calculate_bw(simplified_original_graph)
    
    if not recompute and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                distributions_b = json.load(f)
            results = calculate_balance_given_distribution(distributions_b, b_w)
            return results, distributions_b
        except Exception:
            pass
            
    results, distributions = calculate_balance_metrics(graph, null_models, NumberOfRandoms)
    try:
        with open(cache_path, 'w') as f:
            json.dump(distributions[0], f)
    except Exception:
        pass
        
    return results, distributions[0]


# ==============================================================================
# 4. TRIANGLE EXTRACTION & BALANCE ANALYSIS
# ==============================================================================

def calculate_triangles_graph(graph: nx.Graph) -> list[tuple[float, float, float]]:
    """Find all unique triangles in the graph.
    
    Args:
        graph: Signed graph.
        
    Returns:
        List of triangle edge weight tuples.
    """
    return _find_triangles(graph)


def calculate_triangles_null_graph(null_models: list[nx.Graph]) -> list[list[tuple[float, float, float]]]:
    """Find all triangles for all null model realizations.
    
    Args:
        null_models: List of null graphs.
        
    Returns:
        List of lists of triangle weight tuples.
    """
    return [_find_triangles(null_graph) for null_graph in null_models]


def geo_abs(triangle: list[float] | tuple[float, float, float]) -> float:
    """Calculate the signed geometric mean of triangle weights.
    
    Args:
        triangle: List/tuple of three edge weights.
        
    Returns:
        Signed geometric mean.
    """
    product = (abs(triangle[0]) * abs(triangle[1]) * abs(triangle[2])) ** (1 / 3)
    signs = [np.sign(triangle[0]), np.sign(triangle[1]), np.sign(triangle[2])]
    if signs.count(-1) % 2:
        product *= -1
    return float(product)


def in_balance(triangle: list[float] | tuple[float, float, float]) -> int:
    """Check if a triangle is balanced (has an even number of negative edges).
    
    Args:
        triangle: Three edge weights.
        
    Returns:
        1 if balanced, 0 otherwise.
    """
    signs = list(np.sign(np.array(triangle)))
    return 1 - (signs.count(-1) % 2)


def number_of_triangles_per_type(triangles_graph: list[tuple[float, float, float]]) -> pd.DataFrame:
    """Count triangles by positive edge count (0, 1, 2, or 3 positive edges).
    
    Args:
        triangles_graph: List of triangle weight tuples.
        
    Returns:
        DataFrame containing counts.
    """
    qnt_pos = [0, 0, 0, 0]
    for triangle in triangles_graph:
        qnt_pos[list(np.sign(triangle)).count(1)] += 1
    
    df = pd.DataFrame(pd.Series(qnt_pos)).T
    df.columns = ['0 pos edges', '1 pos edge', '2 pos edges', '3 pos edges']
    return df


def non_binary_metric(triangles_graph: list[tuple[float, float, float]], null_triangles: list[list[tuple[float, float, float]]]) -> pd.DataFrame:
    """Compute a non-binary metric based on signed geometric means.
    
    Args:
        triangles_graph: Triangles of real graph.
        null_triangles: Triangles of null graphs.
        
    Returns:
        DataFrame with metric details.
    """
    real_means = [geo_abs(tri) for tri in triangles_graph]
    real_metric = float(np.mean(real_means)) if real_means else 0.0
    
    null_metrics = []
    for null_model in null_triangles:
        null_means = [geo_abs(tri) for tri in null_model]
        null_metrics.append(np.mean(null_means) if null_means else 0.0)
        
    avg_null = float(np.mean(null_metrics)) if null_metrics else 0.0
    ratio = real_metric / avg_null if avg_null != 0 else 0.0
    
    df = pd.DataFrame({'prod': [real_metric], 'avg_null': [avg_null], 'ratio': [ratio]})
    return df


# ==============================================================================
# 5. KOLMOGOROV-SMIRNOV STATISTICAL TESTS
# ==============================================================================

def kolmogorov(vals_a: list | np.ndarray, cum_a: list | np.ndarray, vals_b: list | np.ndarray, cum_b: list | np.ndarray, normalize: bool = False) -> float:
    """Compute the Kolmogorov-Smirnov distance between two cumulative distributions.
    
    Args:
        vals_a: Sorted values for A.
        cum_a: Cumulative counts for A.
        vals_b: Sorted values for B.
        cum_b: Cumulative counts for B.
        normalize: If True, normalizes cumulative values to [0, 1].
        
    Returns:
        Kolmogorov distance.
    """
    v_a = np.asarray(vals_a)
    c_a = np.asarray(cum_a, dtype=float)
    v_b = np.asarray(vals_b)
    c_b = np.asarray(cum_b, dtype=float)

    all_vals = np.array(sorted(set(v_a.tolist()) | set(v_b.tolist())))
    if all_vals.size == 0:
        return 0.0

    def get_step_func(vals, cum):
        if vals.size == 0:
            return np.zeros_like(all_vals, dtype=float)
        idx = np.searchsorted(vals, all_vals, side='right') - 1
        return np.where(idx >= 0, cum[idx], 0.0)

    y_a = get_step_func(v_a, c_a)
    y_b = get_step_func(v_b, c_b)

    if normalize:
        if y_a[-1] > 0:
            y_a = y_a / y_a[-1]
        if y_b[-1] > 0:
            y_b = y_b / y_b[-1]

    return float(np.max(np.abs(y_a - y_b)))


def find_alfa(vals_a: np.ndarray, cum_a: np.ndarray, vals_b: np.ndarray, cum_b: np.ndarray) -> float:
    """Calculate p-value using asymptotic distribution of the KS statistic.
    
    Args:
        vals_a: Sorted values for A.
        cum_a: Cumulative values for A.
        vals_b: Sorted values for B.
        cum_b: Cumulative values for B.
        
    Returns:
        P-value.
    """
    D = kolmogorov(vals_a, cum_a, vals_b, cum_b, normalize=True)
    n_a = len(vals_a)
    n_b = len(vals_b)
    return float(2 * np.exp(-2 * D * D * n_a * n_b / (n_b + n_a)))


def prepare_cumulative(d: dict[float, int]) -> tuple[np.ndarray, np.ndarray]:
    """Prepare cumulative sums from dictionary frequency mappings.
    
    Args:
        d: Frequency map.
        
    Returns:
        Sorted values and cumulative count arrays.
    """
    sorted_keys = np.array(sorted(d.keys()))
    counts = np.array([d[k] for k in sorted_keys])
    cum_sum = np.cumsum(counts)
    return sorted_keys, cum_sum


def average_null_models(dict_list: list[dict[float, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Average frequency dictionaries across all null models.
    
    Args:
        dict_list: List of null model frequency maps.
        
    Returns:
        Merged unique values and averaged cumulative values.
    """
    all_vals = set()
    for d in dict_list:
        all_vals.update(d.keys())
    sorted_vals = np.array(sorted(all_vals))
    
    cum_sums = []
    for d in dict_list:
        v, c = prepare_cumulative(d)
        if v.size == 0:
            cum_sums.append(np.zeros_like(sorted_vals, dtype=float))
            continue
        idx = np.searchsorted(v, sorted_vals, side='right') - 1
        cum_sums.append(np.where(idx >= 0, c[idx], 0.0))
        
    avg_cum = np.mean(cum_sums, axis=0)
    return sorted_vals, avg_cum


def calculate_kolmogorov_stats(triangles_graph: list[tuple[float, float, float]], null_triangles: list[list[tuple[float, float, float]]]) -> tuple[dict, list[float]]:
    """Calculate KS test stats comparing real triangles to randomized null models.
    
    Args:
        triangles_graph: Real triangles list.
        null_triangles: Null model triangles.
        
    Returns:
        Tuple of (results_dict, D_null_vs_avg_distribution).
    """
    D_distribution = []
    
    all_triangles_dict = defaultdict(int)
    for triangle in triangles_graph:
        all_triangles_dict[geo_abs(triangle)] += 1
    vals_all, cum_all = prepare_cumulative(all_triangles_dict)

    null_all_dicts = []
    for null_model in null_triangles:
        null_all = defaultdict(int)
        for triangle in null_model:
            null_all[geo_abs(triangle)] += 1
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
        'D': float(D),
        "mean": float(mean),
        "std": float(std),
        "z-score": float((D - mean) / std) if std != 0 else 0.0,
        "percentile": float(percentile)
    }
    return results, D_distribution


def kolmogorov_smirnov(triangles_graph: list[tuple[float, float, float]], null_triangles: list[list[tuple[float, float, float]]]) -> None:
    """Print the KS statistic comparing real vs null models (maintained for compatibility).
    
    Args:
        triangles_graph: Real triangles list.
        null_triangles: Null model triangles.
    """
    results, _ = calculate_kolmogorov_stats(triangles_graph, null_triangles)
    print(f"KS Statistic comparison: D = {results['D']:.4f} | p-value = {results['percentile']/100:.4f}")


# ==============================================================================
# 6. COMPREHENSIVE NETWORK VISUALIZATIONS
# ==============================================================================

def plot_weight_distribution(graph: nx.Graph) -> None:
    """Visualize edge weight and sign distributions dynamically color-coded by value.
    
    Args:
        graph: Signed graph with edge weights.
    """
    WEIGHT_BIN_CENTERS = np.arange(-1.0, 1.01, 0.1)
    WEIGHT_BIN_EDGES = np.append(WEIGHT_BIN_CENTERS - 0.05, WEIGHT_BIN_CENTERS[-1] + 0.05)
    SIGN_BIN_EDGES = [-1.5, -0.5, 0.5, 1.5]
    SIGN_LABELS = ["negative", "neutral", "positive"]
    SIGN_TICK_POSITIONS = [-1, 0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def add_percentage_labels(ax, counts, bins, fontsize=11):
        total = counts.sum()
        for count, bin_left, bin_right in zip(counts, bins[:-1], bins[1:]):
            percent = 100 * count / total if total > 0 else 0
            x_position = (bin_left + bin_right) / 2
            if percent > 0.1:
                ax.text(x_position, count, f"{percent:.1f}%", ha='center', va='bottom', fontsize=fontsize, color='#2c3e50')

    weights = np.array([data["weight"] for _, _, data in graph.edges(data=True)])
    
    # Left Panel: Continuous distribution
    ax_weight = axes[0]
    _apply_plot_style(ax_weight, "Consolidated Edge Weight Distribution")
    counts, bins, patches = ax_weight.hist(weights, bins=WEIGHT_BIN_EDGES, edgecolor="black", alpha=0.85)
    
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
    ax_weight.set_xlabel("Weight")
    ax_weight.set_ylabel("Count")
    
    # Right Panel: Categorized sign distribution
    ax_sign = axes[1]
    _apply_plot_style(ax_sign, "Aggregated Edge Sign Distribution")
    sign_counts, bins, patches = ax_sign.hist(np.sign(weights), bins=SIGN_BIN_EDGES, edgecolor="black", rwidth=0.8, alpha=0.85)
    
    colors = ['#e74c3c', '#bdc3c7', '#2ecc71']
    for idx, patch in enumerate(patches):
        if idx < len(colors):
            patch.set_facecolor(colors[idx])
            
    ax_sign.set_xticks(SIGN_TICK_POSITIONS)
    ax_sign.set_xticklabels(SIGN_LABELS, fontsize=11)
    add_percentage_labels(ax_sign, sign_counts, SIGN_BIN_EDGES)
    ax_sign.set_xlabel("Sign Category")

    plt.tight_layout()
    plt.show()


def plot_triangle_distribution(triangles_graph: list[tuple[float, float, float]]) -> None:
    """Visualize triangle mean and product-sign distributions dynamically color-coded by sign.
    
    Args:
        triangles_graph: List of triangle edge weight tuples.
    """
    WEIGHT_BIN_CENTERS = np.arange(-1.0, 1.01, 0.1)
    WEIGHT_BIN_EDGES = np.append(WEIGHT_BIN_CENTERS - 0.05, WEIGHT_BIN_CENTERS[-1] + 0.05)
    SIGN_BIN_EDGES = [-1.5, -0.5, 0.5, 1.5]
    SIGN_LABELS = ["unbalanced", "neutral", "balanced"]
    SIGN_TICK_POSITIONS = [-1, 0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def add_percentage_labels(ax, counts, bins, fontsize=11):
        total = counts.sum()
        for count, bin_left, bin_right in zip(counts, bins[:-1], bins[1:]):
            percent = 100 * count / total if total > 0 else 0
            x_position = (bin_left + bin_right) / 2
            if percent > 0.1:
                ax.text(x_position, count, f"{percent:.1f}%", ha='center', va='bottom', fontsize=fontsize, color='#2c3e50')

    tri_arr = [np.array(tri, dtype=float) for tri in triangles_graph if len(tri) == 3]
    if len(tri_arr) == 0:
        means = np.array([])
        signs = np.array([])
    else:
        means = np.array([tri.mean() for tri in tri_arr])
        signs = np.sign(np.array([tri.prod() for tri in tri_arr]))

    # Left Panel: Triangle mean weights
    ax_mean = axes[0]
    _apply_plot_style(ax_mean, "Triangle Mean Weights Distribution")
    if means.size > 0:
        counts, bins, patches = ax_mean.hist(means, bins=WEIGHT_BIN_EDGES, edgecolor="black", alpha=0.85)
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
        ax_mean.text(0.5, 0.5, 'No triangles found', ha='center', va='center', fontsize=14, transform=ax_mean.transAxes)
    ax_mean.set_xlabel("Mean weight")
    ax_mean.set_ylabel("Count")

    # Right Panel: Triangle product sign (balance compliance)
    ax_sign = axes[1]
    _apply_plot_style(ax_sign, "Structural Balance Compliance Distribution")
    if signs.size > 0:
        sign_counts, bins, patches = ax_sign.hist(signs, bins=SIGN_BIN_EDGES, edgecolor="black", rwidth=0.8, alpha=0.85)
        colors = ['#e74c3c', '#bdc3c7', '#2ecc71']
        for idx, patch in enumerate(patches):
            if idx < len(colors):
                patch.set_facecolor(colors[idx])
        ax_sign.set_xticks(SIGN_TICK_POSITIONS)
        ax_sign.set_xticklabels(SIGN_LABELS, fontsize=11)
        add_percentage_labels(ax_sign, sign_counts, SIGN_BIN_EDGES)
    else:
        ax_sign.text(0.5, 0.5, 'No triangles found', ha='center', va='center', fontsize=14, transform=ax_sign.transAxes)
    ax_sign.set_xlabel("Balance Status")

    plt.tight_layout()
    plt.show()


def plot_bw_distribution(distributions_b: list[float] | np.ndarray, results_b: dict) -> None:
    """Visualize the Bw distribution of null models compared to the real graph's Bw.
    
    Args:
        distributions_b: Null model Bw values.
        results_b: Analysis stats dict.
    """
    distribution = np.asarray(distributions_b)
    fig, ax = plt.subplots(figsize=(8, 4))
    _apply_plot_style(ax, "Bw Metric vs Null Model Distribution")
    b_w = results_b["bw"]

    if distribution.size == 0:
        ax.text(0.5, 0.5, "Empty distribution", ha='center')
    else:
        kde = gaussian_kde(distribution)
        x_min, x_max = min(distribution.min(), b_w), max(distribution.max(), b_w)
        pad = max(0.05 * (x_max - x_min), 1e-6)
        x_vals = np.linspace(x_min - pad, x_max + pad, 300)
        y_vals = kde(x_vals)
        ax.plot(x_vals, y_vals, color='#3498db', linewidth=2, label='Null Model KDE')
        ax.fill_between(x_vals, y_vals, color='#3498db', alpha=0.3)

        if b_w is not None:
            ax.axvline(b_w, color='#e74c3c', linestyle='--', linewidth=2, label=f'Real Bw = {b_w:.4f}')

    ax.set_xlim(x_vals.min() if distribution.size else -0.01, x_vals.max() if distribution.size else 0.06)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_kolmogorov(D_distribution: list[float] | np.ndarray, results: dict) -> None:
    """Visualize the Kolmogorov-Smirnov distance null distribution vs real distance.
    
    Args:
        D_distribution: KS distances between null runs.
        results: Analysis stats dict.
    """
    distribution = np.asarray(D_distribution)
    fig, ax = plt.subplots(figsize=(8, 4))
    _apply_plot_style(ax, "KS Distance Distribution vs Real D-Value")

    if distribution.size == 0:
        ax.text(0.5, 0.5, "Empty distribution", ha='center')
    else:
        if distribution.size > 1:
            kde = gaussian_kde(distribution)
            x_min, x_max = distribution.min(), distribution.max()
            pad = max(0.05 * (x_max - x_min), 1e-6)
            x_vals = np.linspace(x_min - pad, x_max + pad, 300)
            y_vals = kde(x_vals)
            ax.plot(x_vals, y_vals, color='#2ecc71', linewidth=2, label='Null Run KS Distances')
            ax.fill_between(x_vals, y_vals, color='#2ecc71', alpha=0.3)
        else:
            x_vals = np.array([distribution[0]])
            ax.plot(x_vals, np.array([1.0]), marker='o', linestyle='', color='#2ecc71', label='Null KS Distance')

        d_val = results["D"]
        if d_val is not None:
            ax.axvline(d_val, color='#e74c3c', linestyle='--', linewidth=2, label=f'Real D = {d_val:.4f}')

        data_min = min(distribution.min(), d_val) if d_val is not None else distribution.min()
        data_max = max(distribution.max(), d_val) if d_val is not None else distribution.max()
        pad = max(0.05 * (data_max - data_min if data_max != data_min else abs(data_max) + 1e-6), 1e-6)
        ax.set_xlim(data_min - pad, data_max + pad)

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def top_nodes_stats(graph: nx.Graph, k: int) -> None:
    """Generate a detailed 3-panel profile for the top-k nodes by degree.
    
    Args:
        graph: Signed graph.
        k: Number of top nodes to plot.
    """
    top_k = sorted(graph.degree(), key=lambda x: x[1], reverse=True)[:min(k, graph.number_of_nodes())]
    
    def _get_weight(g, u, v):
        attr = g[u][v]
        return attr.get("weight", 1.0) if isinstance(attr, dict) else float(attr)

    triangles_with_nodes = []
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
                        found[tri] = True
                        triangles_with_nodes.append((tri, (w12, w13, w23)))

    global_products = np.array([float(ws[0] * ws[1] * ws[2]) for _, ws in triangles_with_nodes], dtype=float)
    global_mean = np.mean(global_products) if global_products.size > 0 else 0.0
    global_std = np.std(global_products, ddof=0) if global_products.size > 0 else 0.0

    global_kde = None
    x_vals_global = None
    y_vals_global = None
    if global_products.size > 1:
        try:
            global_kde = gaussian_kde(global_products)
            x_min_g, x_max_g = global_products.min(), global_products.max()
            pad_g = max(0.05 * (x_max_g - x_min_g), 1e-12)
            x_vals_global = np.linspace(x_min_g - pad_g, x_max_g + pad_g, 300)
            y_vals_global = global_kde(x_vals_global)
        except Exception:
            global_kde = None

    node_triangle_products = defaultdict(list)
    node_triangle_pos_counts = defaultdict(list)
    for nodes, weights in triangles_with_nodes:
        prod = float(weights[0] * weights[1] * weights[2])
        pos_count = int(sum(1 for w in weights if w > 0))
        for n in nodes:
            node_triangle_products[n].append(prod)
            node_triangle_pos_counts[n].append(pos_count)

    centers = np.linspace(-1.0, 1.0, 21)
    bins_edges = np.linspace(centers[0] - 0.05, centers[-1] + 0.05, len(centers) + 1)

    rows = len(top_k)
    fig, axes = plt.subplots(rows, 3, figsize=(18, 3.5 * rows))
    axes = np.atleast_2d(axes)

    for i, (node, deg) in enumerate(top_k):
        ax_w = axes[i, 0]
        ax_p = axes[i, 1]
        ax_t = axes[i, 2]

        weights = []
        for nbr, attr in graph[node].items():
            if nbr == node:
                continue
            w = attr.get("weight", 1.0) if isinstance(attr, dict) else float(attr)
            weights.append(w)

        # Panel 1: Weight histogram of incident edges
        if len(weights) == 0:
            ax_w.text(0.5, 0.5, "No edges", ha="center", va="center")
            ax_w.set_xticks([])
            ax_w.set_yticks([])
        else:
            _apply_plot_style(ax_w, f"Incident Edges Weight (deg={deg})")
            counts, bins, patches = ax_w.hist(weights, bins=bins_edges, edgecolor="k", alpha=0.85)
            for patch, bin_left, bin_right in zip(patches, bins_edges[:-1], bins_edges[1:]):
                center = (bin_left + bin_right) / 2
                if center < -0.05:
                    patch.set_facecolor(plt.cm.Reds(0.3 + 0.7 * abs(center)))
                elif center > 0.05:
                    patch.set_facecolor(plt.cm.Greens(0.3 + 0.7 * center))
                else:
                    patch.set_facecolor('#bdc3c7')
            mean_w = np.mean(weights)
            ax_w.axvline(mean_w, color="red", linestyle="--", linewidth=1.5, label=f"mean={mean_w:.3f}")
            ax_w.set_xlabel("Edge weight")
            ax_w.set_ylabel("Frequency")
            ax_w.legend(fontsize="small", loc="upper left")
            ax_w.set_xticks(centers)
            ax_w.set_xlim(bins_edges[0], bins_edges[-1])

        # Panel 2: Product distributions (KDE overlay)
        tri_prods = node_triangle_products.get(node, [])
        if len(tri_prods) == 0:
            ax_p.text(0.5, 0.5, "No triangles", ha="center", va="center")
            ax_p.set_xticks([])
            ax_p.set_yticks([])
        else:
            _apply_plot_style(ax_p, f"Triangle products (n={len(tri_prods)})")
            if global_kde is not None:
                ax_p.plot(x_vals_global, y_vals_global, color="darkgreen", linewidth=1.5, label="Global KDE")
                ax_p.fill_between(x_vals_global, y_vals_global, color="darkgreen", alpha=0.15)
                ax_p.set_xlabel("Triangle product")
                ax_p.set_ylabel("Density")
                ax_p.set_ylim(0, y_vals_global.max() * 1.05)
                ax_p.set_xlim(x_vals_global[0], x_vals_global[-1])
            else:
                ax_p.hist(global_products, bins=bins_edges, color="lightgreen", edgecolor="k", alpha=0.7, density=True)
                ax_p.set_xlabel("Triangle product")
                ax_p.set_ylabel("Density")
                ax_p.set_xticks(centers)
                ax_p.set_xlim(bins_edges[0], bins_edges[-1])

            arr = np.asarray(tri_prods, dtype=float)
            mean_p = np.mean(arr)
            ax_p.axvline(mean_p, color="red", linestyle="--", linewidth=1.5, label=f"mean={mean_p:.4e}")

            try:
                z_score = (mean_p - global_mean) / global_std if global_std != 0 else 0.0
                pct = float(percentileofscore(global_products, mean_p, kind="mean")) if global_products.size > 0 else np.nan
            except Exception:
                z_score = np.nan
                pct = np.nan

            stat_text = f"z = {z_score:.3f}\npercentile = {pct:.1f}%"
            ax_p.text(
                0.98, 0.95, stat_text,
                transform=ax_p.transAxes,
                ha="right", va="top",
                fontsize="small",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray")
            )
            ax_p.legend(fontsize="small", loc="upper left")

        # Panel 3: Triangle configurations count chart (---, +--, ++-, +++)
        pos_counts = node_triangle_pos_counts.get(node, [])
        if len(pos_counts) == 0:
            ax_t.text(0.5, 0.5, "No triangles", ha="center", va="center")
            ax_t.set_xticks([])
            ax_t.set_yticks([])
        else:
            _apply_plot_style(ax_t, f"Triangles by positive count (n={len(pos_counts)})")
            counts = np.bincount(np.array(pos_counts, dtype=int), minlength=4)[:4]
            labels = ["---", "+--", "++-", "+++"]
            x = np.arange(len(labels))
            bar_colors = ["#d73027", "#fc8d59", "#fee090", "#91bfdb"]
            ax_t.bar(x, counts, color=bar_colors, edgecolor="k", alpha=0.85)
            ax_t.set_xticks(x)
            ax_t.set_xticklabels(labels, fontsize="small")
            ax_t.set_ylabel("Triangle count")
            ax_t.set_xlabel("Triangle edge configuration")
            for xi, c in zip(x, counts):
                ax_t.text(xi, c + max(1, 0.01 * sum(counts)), str(int(c)), ha="center", va="bottom", fontsize="small")

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 7. COHESIVE FRAUDSTER GROUP DETECTION
# ==============================================================================

def detect_fraudster_groups(graph: nx.Graph, min_neg_degree: int = 2, min_pos_density: float = 0.7, min_balance_ratio: float = 0.9, NumberOfRandoms: int = 100) -> list[dict[str, Any]]:
    """Detect cohesive groups of potential fraudsters in a signed graph.
    
    Args:
        graph: Undirected signed Graph.
        min_neg_degree: Minimum negative degree in G for a node to be considered.
        min_pos_density: Minimum positive edge density within the group.
        min_balance_ratio: Minimum triangle balance compliance inside the group.
        NumberOfRandoms: Number of random realizations to generate for subgraph null models.
        
    Returns:
        List of dictionaries with group nodes, size, positive/negative densities,
        and structural balance metrics.
    """
    candidate_nodes = []
    for node in graph.nodes():
        neg_deg = sum(1 for nbr in graph.neighbors(node) if graph[node][nbr].get('weight', 1.0) < 0)
        if neg_deg >= min_neg_degree:
            candidate_nodes.append(node)
            
    if not candidate_nodes:
        return []
        
    candidate_graph = nx.Graph()
    candidate_graph.add_nodes_from(candidate_nodes)
    for u, v in graph.edges():
        if u in candidate_graph and v in candidate_graph:
            w = graph[u][v].get('weight', 1.0)
            candidate_graph.add_edge(u, v, weight=w, pos_weight=(w if w > 0 else 0.0))
                
    from networkx.algorithms.community import louvain_communities
    try:
        communities = louvain_communities(candidate_graph, weight='pos_weight', seed=42)
    except Exception:
        communities = list(nx.connected_components(candidate_graph))
        
    detected_groups = []
    
    for comm in communities:
        if len(comm) < 3:
            continue
            
        nodes_list = list(comm)
        n = len(comm)
        possible_edges = n * (n - 1) / 2
        
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
        
        if pos_density < min_pos_density:
            continue
            
        triangles_inside = 0
        balanced_triangles = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    u, v, w_node = nodes_list[i], nodes_list[j], nodes_list[k]
                    if graph.has_edge(u, v) and graph.has_edge(u, w_node) and graph.has_edge(v, w_node):
                        triangles_inside += 1
                        s1 = np.sign(graph[u][v].get('weight', 1.0))
                        s2 = np.sign(graph[u][w_node].get('weight', 1.0))
                        s3 = np.sign(graph[v][w_node].get('weight', 1.0))
                        neg_count = sum(1 for s in (s1, s2, s3) if s < 0)
                        if neg_count % 2 == 0:
                            balanced_triangles += 1
                            
        balance_ratio = balanced_triangles / triangles_inside if triangles_inside > 0 else 1.0
        
        if balance_ratio < min_balance_ratio:
            continue
            
        # Calculate Bw balance metrics for the subgraph (group)
        group_subgraph = graph.subgraph(comm)
        group_null_models = [generate_null_model(group_subgraph, (214013 * i + 2531011) % (1 << 31)) for i in range(NumberOfRandoms)]
        group_bw_stats, _ = calculate_balance_metrics(group_subgraph, group_null_models, NumberOfRandoms)
        
        detected_groups.append({
            'nodes': comm,
            'size': n,
            'pos_density': pos_density,
            'neg_density': neg_density,
            'total_triangles': triangles_inside,
            'balanced_triangles': balanced_triangles,
            'balance_ratio': balance_ratio,
            'bw': group_bw_stats['bw'],
            'bw_zscore': group_bw_stats['z-score'],
            'bw_mean': group_bw_stats['mean'],
            'bw_std': group_bw_stats['std'],
            'bw_percentile': group_bw_stats['percentile']
        })
        
    detected_groups.sort(key=lambda x: x['size'], reverse=True)
    return detected_groups


def extract_ego_subgraphs(graph: nx.Graph, k: int = 5) -> tuple[nx.Graph, nx.Graph]:
    """Identify top k nodes with most negative and most -1.0 weight incident edges,
    and compose their ego-subgraphs (radius=1).
    Saves the composed subgraphs to negative_graph.gexf and negative1_graph.gexf.
    Prints top nodes stats and subgraph size info.
    
    Args:
        graph: Input signed graph.
        k: Number of top nodes to select.
        
    Returns:
        Tuple of (composed_negative_graph, composed_negative1_graph).
    """
    negative_edges_count = defaultdict(lambda: [0, 0])
    for u, v, data in graph.edges(data=True):
        w = float(data["weight"])
        if w < 0:
            negative_edges_count[u][0] += 1
            negative_edges_count[v][0] += 1
        if w == -1.0:
            negative_edges_count[u][1] += 1
            negative_edges_count[v][1] += 1

    top_neg_nodes = sorted(negative_edges_count.keys(), key=lambda n: negative_edges_count[n][0], reverse=True)[:k]
    top_neg1_nodes = sorted(negative_edges_count.keys(), key=lambda n: negative_edges_count[n][1], reverse=True)[:k]

    print(f"Top nodes with most negative links: {top_neg_nodes}")
    print(f"Top nodes with most -1.0 links: {top_neg1_nodes}")

    negative_graph = nx.compose_all([nx.ego_graph(graph, node, radius=1) for node in top_neg_nodes])
    negative1_graph = nx.compose_all([nx.ego_graph(graph, node, radius=1) for node in top_neg1_nodes])

    nx.write_gexf(negative_graph, "negative_graph.gexf")
    nx.write_gexf(negative1_graph, "negative1_graph.gexf")

    print(f"\nNegative Subgraph: {negative_graph.number_of_nodes()} nodes, {negative_graph.number_of_edges()} edges")
    print(f"Negative1 Subgraph: {negative1_graph.number_of_nodes()} nodes, {negative1_graph.number_of_edges()} edges")
    
    return negative_graph, negative1_graph


def detect_fraudster_groups_by_ratio(
    graph: nx.Graph, 
    min_neg_degree: int = 5, 
    min_neg_ratio: float = 0.7, 
    min_pos_density: float = 0.5, 
    min_balance_ratio: float = 0.8, 
    NumberOfRandoms: int = 100
) -> list[dict[str, Any]]:
    """Detect cohesive groups of potential fraudsters in a signed graph
    by filtering candidate nodes on both minimum negative degree and ratio,
    and then performing a raw exponential combination search over subsets of S.
    
    Args:
        graph: Undirected signed Graph.
        min_neg_degree: Minimum negative degree for a node to be a candidate.
        min_neg_ratio: Minimum ratio of negative degree to total degree.
        min_pos_density: Minimum positive edge density within the group (Z%).
        min_balance_ratio: Ignored in this raw search.
        NumberOfRandoms: Number of random realizations to generate for subgraph null models.
        
    Returns:
        List of dictionaries with group nodes, size, positive/negative densities,
        coverage metrics, and structural balance stats.
    """
    import itertools
    
    # 1. Filter candidate nodes S
    S = []
    for node in graph.nodes():
        deg = graph.degree(node)
        if deg == 0:
            continue
        neg_deg = sum(1 for nbr in graph.neighbors(node) if graph[node][nbr].get('weight', 1.0) < 0)
        if neg_deg >= min_neg_degree and (neg_deg / deg) >= min_neg_ratio:
            S.append(node)
            
    if not S:
        return []
        
    print(f"Subset S filtered: {len(S)} candidate nodes.")
    print(f"Running raw exponential search for subsets of size 3, 4, 5 with Z={min_pos_density*100}%...")
    
    # 2. Raw exponential search for subsets of size 3, 4, and 5
    all_valid_groups = []
    
    for size in [3, 4, 5]:
        for comm in itertools.combinations(S, size):
            n = len(comm)
            possible_edges = n * (n - 1) / 2
            
            pos_edges_count = 0
            neg_edges_count = 0
            
            for i in range(n):
                for j in range(i + 1, n):
                    u, v = comm[i], comm[j]
                    if graph.has_edge(u, v):
                        w = graph[u][v].get('weight', 1.0)
                        if w > 0:
                            pos_edges_count += 1
                        elif w < 0:
                            neg_edges_count += 1
                            
            pos_density = pos_edges_count / possible_edges
            
            if pos_density >= min_pos_density:
                all_valid_groups.append({
                    'nodes': set(comm),
                    'size': n,
                    'pos_density': pos_density,
                    'neg_density': neg_edges_count / possible_edges
                })
                
    # 3. Filter to keep only maximal groups
    maximal_groups = []
    for g in all_valid_groups:
        is_sub = False
        for other in all_valid_groups:
            if g != other and g['nodes'].issubset(other['nodes']) and len(g['nodes']) < len(other['nodes']):
                is_sub = True
                break
        if not is_sub:
            if not any(g['nodes'] == existing['nodes'] for existing in maximal_groups):
                maximal_groups.append(g)
                
    # 4. Compute coverage metrics and Bw stats for the maximal groups
    total_neg_edges = sum(1 for u, v in graph.edges() if graph[u][v].get('weight', 1.0) < 0)
    detected_groups = []
    
    for g in maximal_groups:
        comm = g['nodes']
        n = g['size']
        
        # Coverage stats
        neg_edges_count = 0
        nodes_list = list(comm)
        for i in range(n):
            for j in range(i + 1, n):
                u, v = nodes_list[i], nodes_list[j]
                if graph.has_edge(u, v) and graph[u][v].get('weight', 1.0) < 0:
                    neg_edges_count += 1
                    
        internal_coverage = (neg_edges_count / total_neg_edges) * 100 if total_neg_edges > 0 else 0.0
        
        incident_neg_edges = set()
        for node in comm:
            for nbr in graph.neighbors(node):
                if graph[node][nbr].get('weight', 1.0) < 0:
                    edge = tuple(sorted([node, nbr]))
                    incident_neg_edges.add(edge)
        incident_coverage = (len(incident_neg_edges) / total_neg_edges) * 100 if total_neg_edges > 0 else 0.0
        
        # Balance metrics
        group_subgraph = graph.subgraph(comm)
        group_null_models = [generate_null_model(group_subgraph, (214013 * i + 2531011) % (1 << 31)) for i in range(NumberOfRandoms)]
        group_bw_stats, _ = calculate_balance_metrics(group_subgraph, group_null_models, NumberOfRandoms)
        
        detected_groups.append({
            'nodes': comm,
            'size': n,
            'pos_density': g['pos_density'],
            'neg_density': g['neg_density'],
            'total_triangles': 0, # not computed for raw search
            'balanced_triangles': 0, # not computed for raw search
            'balance_ratio': 1.0, # disabled
            'internal_coverage': internal_coverage,
            'incident_coverage': incident_coverage,
            'bw': group_bw_stats['bw'],
            'bw_zscore': group_bw_stats['z-score'],
            'bw_mean': group_bw_stats['mean'],
            'bw_std': group_bw_stats['std'],
            'bw_percentile': group_bw_stats['percentile']
        })
        
    detected_groups.sort(key=lambda x: x['size'], reverse=True)
    return detected_groups


def plot_fraudster_group(graph: nx.Graph, group_nodes: set, title: str = "Fraudster Group", save_path: str = None, ax: plt.Axes = None, stats: dict = None) -> None:
    """Visualize a detected fraudster group with spring force layout.
    
    Args:
        graph: Signed graph.
        group_nodes: Nodes belonging to the group.
        title: Title of plot.
        save_path: Optional path to save the generated image.
        ax: Optional subplot axes.
        stats: Optional stats to annotate inside.
    """
    subgraph = graph.subgraph(group_nodes)
    
    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(subgraph.nodes())
    for u, v in subgraph.edges():
        w = abs(subgraph[u][v].get('weight', 1.0))
        layout_graph.add_edge(u, v, weight=max(w, 0.1))
        
    n_nodes = len(group_nodes)
    k_distance = 2.5 / np.sqrt(n_nodes) if n_nodes > 0 else 1.0
    pos = nx.spring_layout(layout_graph, weight='weight', k=k_distance, iterations=150, seed=42)
    
    node_neg_degs = []
    for node in group_nodes:
        neg_deg = sum(1 for nbr in graph.neighbors(node) if graph[node][nbr].get('weight', 1.0) < 0)
        node_neg_degs.append(neg_deg)
        
    # Scale community node sizes dynamically and cap them between 300 and 800
    max_deg = max(node_neg_degs) if node_neg_degs else 1
    node_sizes = [250 + 550 * (deg / max_deg) for deg in node_neg_degs]
        
    pos_edges = []
    neg_edges = []
    pos_widths = []
    neg_widths = []
    for u, v in subgraph.edges():
        w = subgraph[u][v].get('weight', 1.0)
        if w > 0:
            pos_edges.append((u, v))
            pos_widths.append(1.5 + 2.0 * abs(w))
        elif w < 0:
            neg_edges.append((u, v))
            neg_widths.append(1.0)
            
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        show_colorbar = True
        show_plot = True
    else:
        show_colorbar = False
        show_plot = False
        
    cmap = plt.cm.YlOrRd
    
    # Draw community nodes
    nodes = nx.draw_networkx_nodes(
        subgraph, pos, ax=ax,
        node_color=node_neg_degs,
        node_size=node_sizes,
        cmap=cmap,
        edgecolors='#2c3e50',
        linewidths=1.2,
        alpha=0.95
    )
    
    if pos_edges:
        nx.draw_networkx_edges(
            subgraph, pos, ax=ax,
            edgelist=pos_edges,
            edge_color='#2ecc71',
            width=pos_widths,
            alpha=0.85
        )
    
    if neg_edges:
        nx.draw_networkx_edges(
            subgraph, pos, ax=ax,
            edgelist=neg_edges,
            edge_color='#e74c3c',
            width=neg_widths,
            style='dashed',
            alpha=0.8
        )
    
    # Render labels with white background boxes for readability
    labels = {n: str(n) for n in group_nodes}
    nx.draw_networkx_labels(
        subgraph, pos, ax=ax, labels=labels, 
        font_size=8, font_weight='bold', font_color='#2c3e50',
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="none")
    )
    
    if stats is not None:
        stats_text = (
            f"Size: {stats['size']}\n"
            f"Pos Density: {stats['pos_density']:.2f}\n"
            f"Neg Density: {stats['neg_density']:.2f}\n"
            f"Balance Ratio: {stats['balance_ratio']:.2f}"
        )
        if 'bw_zscore' in stats:
            stats_text += f"\nBw Z-score: {stats['bw_zscore']:.2f}"
        if 'internal_coverage' in stats:
            stats_text += f"\nInt Cover: {stats['internal_coverage']:.1f}%"
        if 'incident_coverage' in stats:
            stats_text += f"\nInc Cover: {stats['incident_coverage']:.1f}%"
        ax.text(
            0.02, 0.02, stats_text,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="lightgray")
        )
        
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
            
            


def plot_all_fraudster_groups(graph: nx.Graph, groups: list[dict[str, Any]], title: str = "Detected Fraudster Groups", save_path: str = None) -> None:
    """Plot all detected groups on a unified grid layout.
    
    Args:
        graph: Signed graph.
        groups: List of detected group stats dicts.
        title: Overall plot title.
        save_path: Optional path to save the generated image.
    """
    n_groups = len(groups)
    if n_groups == 0:
        return
        
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
        
    for ax in axes[n_groups:]:
        ax.axis('off')
        
    legend_elements = [
        Line2D([0], [0], color='#2ecc71', lw=3, label='Positive (Friendship Link)'),
        Line2D([0], [0], color='#e74c3c', lw=3, linestyle='--', label='Negative (Conflict Link)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12, frameon=True, edgecolor='lightgray')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.99, color='#2c3e50')
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
