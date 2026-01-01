import numpy as np
import math
import itertools
from functools import lru_cache
import networkx as nx
import os
import time

# Scientific libraries
from scipy.optimize import fsolve, brentq
from scipy.sparse.linalg import spsolve, cg, eigsh, lobpcg, LinearOperator
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky, solve_triangular, cho_factor, cho_solve
from scipy.linalg.lapack import dpotrf, dpotri
from scipy.sparse import csr_matrix, lil_matrix, csr_array
from scipy.sparse.csgraph import laplacian
from scipy.stats import bootstrap, linregress, gaussian_kde
from scipy.spatial import ConvexHull
from scipy.linalg import orthogonal_procrustes


# Optional: PyAMG for efficient multigrid solvers
try:
    import pyamg
except ImportError:
    print("Warning: pyamg not installed. Eigenmap optimization might fail.")
    pyamg = None

# Plotting libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import seaborn as sns
import pandas as pd
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
import plotly.graph_objs as go

# --- Global Constants ---
CORNERS = {0: (0.5, np.sqrt(3)/2), 1: (0, 0), 2: (1, 0)}

# --- Layout Functions ---

@lru_cache(maxsize=None)
def get_vertex_position(vertex_tuple: tuple) -> tuple:
    if not vertex_tuple: return (0.5, np.sqrt(3)/6) # Return center for robustness
    sub_problem_pos = get_vertex_position(vertex_tuple[1:])
    corner_offset = CORNERS[vertex_tuple[0]]
    return (0.5 * (sub_problem_pos[0] + corner_offset[0]),
            0.5 * (sub_problem_pos[1] + corner_offset[1]))

def get_pos(G, tail_extension_factor = 0.5):
    # Fix: Changed 3^(n) to 3**n
    num_nodes_seq = [(3**(n)) + 3 for n in range(1,16)]

    # initialize graph order variable incase it does not pass the check
    graphOrder = 1
    current_order = G.order()
    
    for i, n in enumerate(num_nodes_seq):
        if n == current_order:
            graphOrder = i + 1
            break
            
    # Determine n from the first tuple node found
    n = 1
    for node in G.nodes():
        if isinstance(node, tuple):
            n = len(node)
            break

    pos = {v: get_vertex_position(v) for v in G.nodes() if isinstance(v, tuple)}

    boundary_nodes = { 0: (0,) * n, 1: (1,) * n, 2: (2,) * n }
    
    # First, find the geometric center of the triangle
    center_x = sum(p[0] for p in CORNERS.values()) / 3
    center_y = sum(p[1] for p in CORNERS.values()) / 3

    for i, b_node in boundary_nodes.items():
        # Handle case where boundary node might not exist in a subgraph
        if b_node not in pos: 
            continue
            
        boundary_pos = pos[b_node]
        tail_node = f'tail_{i}'
        
        # Get the direction vector from the center to the boundary
        direction_vec_x = boundary_pos[0] - center_x
        direction_vec_y = boundary_pos[1] - center_y
        
        # Place the tail node further along this direction vector
        pos[tail_node] = (
            boundary_pos[0] + tail_extension_factor * direction_vec_x * (1/graphOrder),
            boundary_pos[1] + tail_extension_factor * direction_vec_y * (1/graphOrder)
        )

    # --- Add position for all "extra_tail_*" nodes ---
    for node in G.nodes:
        if isinstance(node, str) and node.startswith("extra_tail_"):
            neighbor = next(G.neighbors(node))
            if neighbor not in pos:
                continue

            direction_vec = np.array(pos[neighbor]) - np.array([center_x, center_y])
            norm = np.linalg.norm(direction_vec)
            if norm == 0:
                direction_vec = np.array([0, 1])
            else:
                direction_vec = direction_vec / norm

            pos[node] = tuple(np.array(pos[neighbor]) + 0.15 * direction_vec)

    return pos 

# --- Graph Generation Functions ---

def hanoi_t(n: int):
    """
    Generates the Hanoi graph with tails and pre-calculates layout positions.
    """
    elements = {0, 1, 2}
    G = nx.Graph()
    vertices = list(itertools.product(elements, repeat=n))
    G.add_nodes_from(vertices)

    m = n
    bridge_edges = [list(itertools.product(elements, repeat=M)) for M in range(1, m + 1)]
    for edge_list in bridge_edges:
        for I_m in edge_list:
            # prefix logic
            if len(I_m) > 1:
                prefix = I_m[:-1]
            else:
                prefix = ()
            
            i = I_m[-1]
            # Find the other two elements
            others = [e for e in elements if e != i]
            j, k = others[0], others[1]
            
            suffix_len = n - len(prefix)
            if suffix_len > 0:
                v1 = prefix + tuple([j] + [k] * (suffix_len - 1))
                v2 = prefix + tuple([k] + [j] * (suffix_len - 1))

                subscript = "".join(map(str, I_m))
                label = f'r{subscript}'
                G.add_edge(v1, v2, label=label)

    boundary_nodes = {i: (i,) * n for i in range(3)}
    for i, b_node in boundary_nodes.items():
        tail_node = f'tail_{i}'
        tail_label = fr'$\tilde{{r}}_{{{i}}}$'
        G.add_edge(b_node, tail_node, label=tail_label)
    return G

def create_sierpinski_gasket_nx(n: int):
    if n < 0:
        raise ValueError("Sierpinski Gasket level n must be >= 0.")
        
    q_corners = [np.array([0, 0]), np.array([1, 0]), np.array([0.5, np.sqrt(3)/2])]
    def F(i, pt): return (pt + q_corners[i]) / 2.0
    def apply_F_w(word, pt):
        for i in reversed(word): pt = F(i, pt)
        return pt

    vertex_coords = {tuple(np.round(apply_F_w(w, qi), 10)) 
                     for w in itertools.product(range(3), repeat=n) 
                     for qi in q_corners}
    
    G = nx.Graph()
    pos = {str(v_tuple): v_tuple for v_tuple in vertex_coords}
    G.add_nodes_from(pos.keys())
    
    for w in itertools.product(range(3), repeat=n):
        cell_coords = [tuple(np.round(apply_F_w(w, qi), 10)) for qi in q_corners]
        cell_node_names = [str(t) for t in cell_coords]
        edges_to_add = [
            (cell_node_names[0], cell_node_names[1]),
            (cell_node_names[1], cell_node_names[2]),
            (cell_node_names[2], cell_node_names[0])
        ]
        G.add_edges_from(edges_to_add)

    G.pos = pos
    return G

def _nx_build_gasket_skeleton(n: int):
    if n < 0: raise ValueError("Level n must be >= 0.")

    q = [np.array([0, 0]), np.array([1, 0]), np.array([0.5, np.sqrt(3)/2])]
    def F(i, pt): return (pt + q[i]) / 2.0
    def apply_F_w(word, pt):
        for i in reversed(word): pt = F(i, pt)
        return pt
        
    V_coords_tuples = {tuple(np.round(apply_F_w(w, qi), 10)) for w in itertools.product(range(3), repeat=n) for qi in q}
    G = nx.Graph()
    pos = {str(v_tuple): v_tuple for v_tuple in V_coords_tuples}
    base_vertices = list(pos.keys())
    G.add_nodes_from(base_vertices)
    
    original_cells, base_edges = [], []
    for w in itertools.product(range(3), repeat=n):
        cell_tuples = [tuple(np.round(apply_F_w(w, qi), 10)) for qi in q]
        cell_names = [str(t) for t in cell_tuples]
        original_cells.append(cell_names)
        edges_to_add = [(cell_names[0], cell_names[1]), (cell_names[1], cell_names[2]), (cell_names[2], cell_names[0])]
        G.add_edges_from(edges_to_add)
        base_edges.extend(edges_to_add)
        
    boundary_vertices = [v for v, d in G.degree() if d == 2]
    mega_cells, mega_vertices, mega_edges = [], [], []
    
    for i, corner_v in enumerate(boundary_vertices):
        local_neighbors = list(G.neighbors(corner_v))
        if len(local_neighbors) < 2: continue # Safety check
        
        pos_corner = np.array(pos[corner_v])
        pos_n1 = np.array(pos[local_neighbors[0]])
        pos_n2 = np.array(pos[local_neighbors[1]])
        
        new_pos_1 = 2 * pos_corner - pos_n1
        new_pos_2 = 2 * pos_corner - pos_n2
        
        new_v_1_name = f"mega_corner{i}_v1"
        new_v_2_name = f"mega_corner{i}_v2"
        
        G.add_node(new_v_1_name); mega_vertices.append(new_v_1_name)
        G.add_node(new_v_2_name); mega_vertices.append(new_v_2_name)
        pos[new_v_1_name], pos[new_v_2_name] = tuple(new_pos_1), tuple(new_pos_2)
        
        edges_to_add = [(corner_v, new_v_1_name), (corner_v, new_v_2_name), (new_v_1_name, new_v_2_name)]
        G.add_edges_from(edges_to_add)
        mega_edges.extend(edges_to_add)
        mega_cells.append([corner_v, new_v_1_name, new_v_2_name])
        
    return G, pos, original_cells, mega_cells, base_vertices, list(set(base_edges)), mega_vertices, list(set(mega_edges))

def nx_hanoi_from_sg(n: int, m: int, hanoi_color: str = 'blue', sg_color: str = 'black'):
    G, pos, original_cells, mega_cells, _, _, _, _ = _nx_build_gasket_skeleton(n)
    all_cells = original_cells + mega_cells
    adjacency_pairs = set()
    
    # Identify adjacent cells
    for i in range(len(all_cells)):
        for j in range(i + 1, len(all_cells)):
            if set(all_cells[i]).intersection(set(all_cells[j])):
                adjacency_pairs.add(tuple(sorted((i, j))))
                
    cell_to_new_vertices = {}
    new_point_nodes = []
    
    for i, cell_corners in enumerate(all_cells):
        p1, p2, p3 = (np.array(pos[c]) for c in cell_corners)
        centroid = (p1 + p2 + p3) / 3.0
        median_vector = p1 - (p2 + p3) / 2.0
        norm = np.linalg.norm(median_vector)
        if norm > 1e-9: median_vector /= norm
        
        line_length = np.linalg.norm(p1 - centroid) * 0.5
        start_point = centroid - median_vector * (line_length / 2)
        end_point = centroid + median_vector * (line_length / 2)
        
        new_vertices_for_this_cell = []
        for j in range(m):
            new_vertex_name = f"cell_{i}_pt_{j}"
            new_vertices_for_this_cell.append(new_vertex_name)
            G.add_node(new_vertex_name)
            new_point_nodes.append(new_vertex_name)
            if m == 1:
                pos[new_vertex_name] = tuple(centroid)
            else:
                pos[new_vertex_name] = tuple(start_point + (j / (m - 1.0)) * (end_point - start_point))
        cell_to_new_vertices[i] = new_vertices_for_this_cell
        
    new_point_edges = []
    for i, j in adjacency_pairs:
        verts_i, verts_j = cell_to_new_vertices[i], cell_to_new_vertices[j]
        edges_to_add = list(itertools.product(verts_i, verts_j))
        G.add_edges_from(edges_to_add)
        new_point_edges.extend(edges_to_add)
        
    nx.set_node_attributes(G, sg_color, 'color')
    nx.set_edge_attributes(G, sg_color, 'color')
    node_color_map = {node: hanoi_color for node in new_point_nodes}
    edge_color_map = {edge: hanoi_color for edge in new_point_edges}
    nx.set_node_attributes(G, node_color_map, 'color')
    nx.set_edge_attributes(G, edge_color_map, 'color')
    G.pos = pos
    return G

def nx_create_j_graph(n: int):
    p_corners = [np.array([0, 0]), np.array([1, 0]), np.array([0.5, np.sqrt(3)/2])]
    def F(j, point): return (point + p_corners[j]) / 2.0

    # === Base Case (n=0) ===
    G_prev = nx.Graph()
    pos_prev = {}
    p0_coord = (p_corners[0] + p_corners[1] + p_corners[2]) / 3.0
    v_center_name = "p0"
    v_corner_names = [f"p{i+1}" for i in range(3)]
    
    G_prev.add_node(v_center_name)
    pos_prev[v_center_name] = p0_coord
    for i in range(3):
        G_prev.add_node(v_corner_names[i])
        pos_prev[v_corner_names[i]] = p_corners[i]

    for i in range(3):
        G_prev.add_edge(v_center_name, v_corner_names[i])

    if n == 0:
        G_prev.pos = {name: tuple(coord) for name, coord in pos_prev.items()}
        return G_prev

    # === Iterative Construction for n > 0 ===
    for _ in range(n):
        G_current = nx.Graph()
        pos_current = {}
        for j in range(3):
            for v_name_prev, v_pos_prev in pos_prev.items():
                new_pos = F(j, v_pos_prev)
                new_name = str(tuple(np.round(new_pos, 10)))
                G_current.add_node(new_name)
                pos_current[new_name] = new_pos
            for u_prev, v_prev in G_prev.edges():
                new_u_pos = F(j, pos_prev[u_prev])
                new_v_pos = F(j, pos_prev[v_prev])
                new_u_name = str(tuple(np.round(new_u_pos, 10)))
                new_v_name = str(tuple(np.round(new_v_pos, 10)))
                G_current.add_edge(new_u_name, new_v_name)
        G_prev, pos_prev = G_current, pos_current

    G_prev.pos = {name: tuple(coord) for name, coord in pos_prev.items()}
    return G_prev

# --- Resistance Calculation Setup ---

# REPLACEMENT for SageMath's find_root and symbolic variables
# Equation: ((1 - exp(-l))^2)/l^2 - (3/5)^n = 0
def solve_lambda_for_level(i):
    target = (0.6) ** i
    
    def equation(l):
        # Handle division by zero near 0 (limit is 1 - target)
        if abs(l) < 1e-9:
            return 1.0 - target
        return ((1 - np.exp(-l))**2) / (l**2) - target

    # Use Brent's method (robust root finding)
    # The function decreases from approx (1 - target) > 0 at l=0 
    # to -target < 0 as l -> infinity. 
    # [1e-5, 600] covers the range used in the original code.
    try:
        sol = brentq(equation, 1e-5, 1000)
        return sol
    except ValueError:
        print(f'no lambda value found for n = {i}')
        return None

# Pre-calculate lambda values
ll_vals = [solve_lambda_for_level(i) for i in range(1, 26)]


# --- Weighting Functions ---

def add_deterministic_weights(G: nx.Graph, n: int):
    # (3/5)**n
    r_xy = 0.6**n
    r_tail = r_xy / 2
    
    tail_nodes = {v for v, d in G.degree() if d == 1}
    
    for u, v in G.edges():
        if u in tail_nodes or v in tail_nodes:
            conductance = 1 / r_tail
        else:
            conductance = 1 / r_xy
        G.edges[u, v]['weight'] = float(conductance)
    return G

def add_random_weights(G: nx.Graph, n: int):
    # Retrieve pre-calculated lambda
    if n > len(ll_vals):
        lambda_n = solve_lambda_for_level(n)
    else:
        lambda_n = ll_vals[n-1]
    
    if lambda_n is None:
        raise ValueError(f"Could not solve lambda for n={n}")

    cell_populations = {v: np.random.poisson(lambda_n) + 1 for v in G.nodes()}
    tail_nodes = {v for v, d in G.degree() if d == 1}
    
    for u, v in G.edges():
        pop_u = cell_populations[u]
        pop_v = cell_populations[v]
        
        # Calculate resistance
        r_uv = 1.0 / (pop_u * pop_v)
        if u in tail_nodes or v in tail_nodes:
            r_uv /= 2.0
            
        G.edges[u, v]['weight'] = float(1.0 / r_uv)
    return G

# --- Resistance Computation Functions ---

def compute_resistance_matrix(G: nx.Graph):
    L = nx.laplacian_matrix(G, nodelist=list(G.nodes()), weight='weight')
    m = L.shape[0]

    # Convert to dense for dpotrf (Cholesky)
    # Note: For very large graphs, you might want sparse solvers, 
    # but dpotrf/dpotri requires dense arrays.
    gamma = L.toarray() + (np.ones((m,m)) / m)

    chol, info = dpotrf(gamma, lower=True)
    gi, info = dpotri(chol, lower=True)
    
    # Fill upper triangle
    gi = gi + gi.T - np.diag(np.diag(gi))

    diag_gi = np.diag(gi)
    res = diag_gi[:, None] + diag_gi[None, :] - 2 * gi
    
    return res

def compute_single_resistance1(G: nx.Graph, v1, v2):
    if v1 not in G or v2 not in G:
        raise ValueError(f"Vertices {v1}, {v2} not found in graph.")
    if v1 == v2: return 0.0

    nodelist = list(G.nodes())
    L = nx.laplacian_matrix(G, nodelist=nodelist, weight='weight')
    L = L.asformat('csr')
    
    n_verts = G.number_of_nodes()
    node_to_idx = {node: i for i, node in enumerate(nodelist)}
    i1, i2 = node_to_idx[v1], node_to_idx[v2]

    b = np.zeros(n_verts)
    b[i1] = 1.0
    b[i2] = -1.0

    # Pin the last node
    pinned_idx = n_verts - 1
    mask = np.arange(n_verts) != pinned_idx
    
    L_reduced = L[mask, :][:, mask]
    b_reduced = b[mask]

    potentials_reduced = spsolve(L_reduced, b_reduced)

    full_potentials = np.zeros(n_verts)
    full_potentials[mask] = potentials_reduced

    resistance = full_potentials[i1] - full_potentials[i2]
    return abs(resistance)

# --- Visualization & Analysis ---

def visualize_resistance_methods(G: nx.Graph, resistance_matrix: np.ndarray, nodelist: list, level=None, save_path=None):
    """
    Visualizes resistance metric results.
    
    Args:
        save_path: The FULL file path (including extension) to save the plot. 
                   e.g., "figures/level_5_analysis.png"
    """
    n_verts = resistance_matrix.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(25, 12))
    ax = axes.ravel()
    
    # 1. Add the Single Main Title
    title_text = f"Resistance Metric Analysis for Random SG at Level {level}" if level is not None else "Resistance Metric Analysis for Random SG"
    fig.suptitle(title_text, fontsize=24, fontweight='bold', y=0.95)

    # --- Heatmap ---
    sns.heatmap(resistance_matrix, ax=ax[0], cmap='viridis', xticklabels=False, yticklabels=False)
    ax[0].set_title(f'Resistance Distance Heatmap ({n_verts}x{n_verts})', fontsize=16)
    ax[0].set_xlabel('Node Index')
    ax[0].set_ylabel('Node Index')

    # --- Network Layout ---
    try:
        # Assuming get_pos is defined in your notebook/script
        pos = get_pos(G)
        avg_resistance = np.mean(resistance_matrix, axis=1)
        
        # Ensure ordering matches
        node_colors = [avg_resistance[nodelist.index(node)] for node in G.nodes()]
        
        nx.draw(G, pos, ax=ax[1], node_size=10, node_color=node_colors, cmap='plasma', width=0.5)
        ax[1].set_title('Network Layout (Colored by Avg. Resistance)', fontsize=16)
        
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
        sm.set_array([])
        fig.colorbar(sm, ax=ax[1], orientation='vertical', fraction=0.046, pad=0.04).set_label('Average Resistance Distance')

    except Exception as e:
        ax[1].text(0.5, 0.5, f'Network visualization failed:\n{e}', ha='center', va='center')

    # Adjust layout so title fits
    plt.tight_layout(rect=[0, 0, 1, 0.93]) 

    # --- Simple Saving ---
    if save_path:
        # Helper: ensure the folder exists so Python doesn't throw an error
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    plt.show()

def resistance_analysis_summary(resistance_matrix: np.ndarray):
    print("\n" + "="*40)
    print("Resistance Distance Analysis Summary")
    print("="*40)
    print(f"Matrix size: {resistance_matrix.shape}")
    print(f"Min resistance: {np.min(resistance_matrix):.6f}")
    print(f"Max resistance: {np.max(resistance_matrix):.6f}")
    print(f"Mean resistance: {np.mean(resistance_matrix):.6f}")
    print(f"Std dev: {np.std(resistance_matrix):.6f}")
    
    upper_triangle = resistance_matrix[np.triu_indices_from(resistance_matrix, k=1)]
    print(f"Median: {np.median(upper_triangle):.6f}")
    
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"{p}th percentile: {np.percentile(upper_triangle, p):.6f}")
    print("="*40 + "\n")

def generate_power_law_fit(x_values, slope, intercept):
    C = np.exp(intercept)
    return C * (x_values**slope)

def generate_exponential_fit(x_values, slope, intercept):
    C = np.exp(intercept)
    return C * np.exp(slope * x_values)

def analyze_statistic_scaling(n_values, statistic_values, statistic_name="Statistic"):
    # Convert inputs to arrays to ensure array ops work
    n_filt = np.array(n_values)
    stat_filt = np.array(statistic_values)

    log_n = np.log(n_filt)
    log_stat = np.log(stat_filt)
    slope_alpha, icept_alpha, r_alpha, _, _ = linregress(log_n, log_stat)
    slope_beta, icept_beta, r_beta, _, _ = linregress(n_filt, log_stat)

    power_law_fit = generate_power_law_fit(n_filt, slope_alpha, icept_alpha)
    exponential_fit = generate_exponential_fit(n_filt, slope_beta, icept_beta)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'Scaling Analysis for {statistic_name}', fontsize=22)
    ax = axes.ravel()

    ax[0].plot(n_filt, stat_filt, 'bo-', label='Simulated Data')
    ax[0].plot(n_filt, exponential_fit, 'g--', label=f'Exponential Fit ($R^2={r_beta**2:.4f}$)')
    ax[0].set_title('Data vs. Exponential Model'); ax[0].legend(); ax[0].grid(True, linestyle='--')

    ax[1].plot(n_filt, stat_filt, 'bo-', label='Simulated Data')
    ax[1].plot(n_filt, power_law_fit, 'r--', label=f'Power-Law Fit ($R^2={r_alpha**2:.4f}$)')
    ax[1].set_title('Data vs. Power-Law Model'); ax[1].legend(); ax[1].grid(True, linestyle='--')

    ax[2].scatter(log_n, log_stat, label='log(Data)')
    ax[2].plot(log_n, slope_alpha * log_n + icept_alpha, 'r--', label=f'Fit (slope={slope_alpha:.3f})')
    ax[2].set_title(f'Log-Log Plot'); ax[2].legend(); ax[2].grid(True, linestyle='--')
    
    ax[3].scatter(n_filt, log_stat, c='purple', label='log(Data)')
    ax[3].plot(n_filt, slope_beta * n_filt + icept_beta, 'orange', linestyle='--', label=f'Fit (rate={slope_beta:.3f})')
    ax[3].set_title(f'Semilog Plot'); ax[3].legend(); ax[3].grid(True, linestyle='--')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def bootstrap_resistance_statistic(num_samples: int, level: int, stat):
    mid_points, lower_bounds, upper_bounds = [], [], []
    v_a, v_b = 'tail_1', 'tail_2'

    print("--- Starting Bootstrap Analysis ---")
    for n in range(2, level + 1):
        print(f"Analyzing graph level n={n}...")
        G = hanoi_t(n)
        res_list = []
        for _ in range(num_samples):
            add_random_weights(G, n)
            res_list.append(compute_single_resistance1(G, v_a, v_b))

        res_arr = np.array(res_list)
        boot = bootstrap((res_arr,), statistic=stat, confidence_level=0.95, random_state=42)
        ci = boot.confidence_interval

        lower_bounds.append(ci.low)
        upper_bounds.append(ci.high)
        mid_points.append(boot.bootstrap_distribution.mean())
        
        print(f"  -> 95% CI for {stat.__name__}(R) at n={n}: ({ci.low:.8f}, {ci.high:.8f})")

    return np.array(mid_points), np.array(lower_bounds), np.array(upper_bounds)

# --- Eigenmap Functions ---

def get_eigenmap_optimized(L, dim=2):
    """
    Computes smallest nontrivial eigenvectors of Laplacian using LOBPCG.
    """
    L = csr_matrix(L)
    L.indices = L.indices.astype(np.int32)
    L.indptr = L.indptr.astype(np.int32)

    num_nodes = L.shape[0]
    null_space_vector = np.ones((num_nodes, 1))

    if pyamg:
        ml = pyamg.smoothed_aggregation_solver(L, B=null_space_vector)
        def amg_preconditioner_solve(b):
            b_proj = b - null_space_vector @ (null_space_vector.T @ b) / num_nodes
            return ml.solve(b_proj, tol=1e-8, accel='gmres')
        M = LinearOperator(L.shape, matvec=amg_preconditioner_solve, dtype=np.float64)
    else:
        M = None # Fallback to no preconditioner if pyamg missing

    X = np.random.rand(num_nodes, dim + 1)
    _, Evecs = lobpcg(L, X, M=M, largest=False, tol=1e-7, maxiter=1000)

    # Return dim nontrivial eigenvectors
    return Evecs[:, 1:1 + dim]

def find_principal_axis(coords):
    hull = ConvexHull(coords)
    hull_points = coords[hull.vertices]
    distances = squareform(pdist(hull_points))
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    return hull_points[i], hull_points[j]

def align_to_x_axis(coords):
    p1, p2 = find_principal_axis(coords)
    axis_vector = p2 - p1
    current_angle = np.arctan2(axis_vector[1], axis_vector[0])
    rotation_angle = -current_angle
    
    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    R = np.array([[c, -s], [s, c]])
    
    rotated_coords = coords @ R.T 
    return rotated_coords, (p1, p2)


def compute_convergence_error(n: int, dim: int = 2):
    """
    Computes the Frobenius norm of the difference between the 
    deterministic and random eigenmaps after Procrustes alignment.
    
    Returns:
        float: The Frobenius error.
    """
    # 1. Create Topology and Fixed Node List
    G = hanoi_t(n)
    fixed_nodelist = list(G.nodes())
    
    # 2. Deterministic Eigenmap (Reference)
    G_det = add_deterministic_weights(G.copy(), n)
    L_det = nx.laplacian_matrix(G_det, nodelist=fixed_nodelist, weight='weight')
    emb_det = get_eigenmap_optimized(L_det, dim=dim)
    
    # 3. Random Eigenmap (Target)
    G_rand = add_random_weights(G.copy(), n)
    L_rand = nx.laplacian_matrix(G_rand, nodelist=fixed_nodelist, weight='weight')
    emb_rand = get_eigenmap_optimized(L_rand, dim=dim)
    
    # 4. Procrustes Alignment
    # Find rotation R that maps emb_rand -> emb_det
    R, _ = orthogonal_procrustes(emb_rand, emb_det)
    emb_rand_aligned = emb_rand @ R
    
    # 5. Compute Error
    # Frobenius norm: sqrt(sum of squared differences)
    diff_matrix = emb_det - emb_rand_aligned
    frobenius_error = np.linalg.norm(diff_matrix, ord='fro')
    
    return frobenius_error

def run_convergence_experiment(max_n: int):
    """
    Runs the convergence experiment for n = 1 to max_n and plots the results.
    """
    n_values = range(1, max_n + 1)
    errors = []
    
    print(f"{'n':<5} | {'Frobenius Error':<20} | {'Time (s)':<10}")
    print("-" * 40)
    
    for n in n_values:
        start_time = time.time()
        
        # Compute the error for this level
        err = compute_convergence_error(n)
        errors.append(err)
        
        elapsed = time.time() - start_time
        print(f"{n:<5} | {err:<20.6f} | {elapsed:<10.2f}")

    return list(n_values), errors
