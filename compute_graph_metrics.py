import math
import extras
import reader
import directories
import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm
from pathlib import Path
from numpy import ndarray
from itertools import product
from typing import Dict, List, Iterable, Tuple
from pandas import DataFrame as df
from joblib import Parallel, delayed



def threshold_mask(
        matrix: ndarray, 
        threshold_percent: float | None = None, 
        threshold_absolute: float | None = None
) -> ndarray:
    
    '''
    Assumes that matrix is symmetrical
    '''

    if threshold_percent is None and threshold_absolute is None:
        raise ValueError('No threshold parameter was provided!')
    
    if not threshold_percent is None and not threshold_absolute is None:
        raise ValueError('More than one threshold parameter was provided!')

    if threshold_percent is not None:

        if threshold_percent > 1.0 or threshold_percent <= 0.0:
            raise ValueError(f'Percentage must be in the interval (0, 1], not {threshold_percent}')
        
        values = matrix[np.triu_indices(matrix.shape[0])]
        values = np.sort(values[values >= 0])
        
        thresholded_n = math.ceil(len(values) * threshold_percent)
        threshold_absolute = values[-thresholded_n]

    return matrix >= threshold_absolute

def small_world_sigma(G: nx.Graph, nrand: int) -> float:

    L = nx.average_shortest_path_length(G)
    C = nx.average_clustering(G)
    n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()

    L_rs, C_rs = np.empty(nrand), np.empty(nrand)
    for i in range(nrand):

        # Random graph with the same number of nodes and edges
        G_r = nx.gnm_random_graph(n_nodes, n_edges)
        while not nx.is_connected(G_r):
            G_r = nx.gnm_random_graph(n_nodes, n_edges)

        L_rs[i] = nx.average_shortest_path_length(G_r)
        C_rs[i] = nx.average_clustering(G_r)
    
    L_r_mean = np.mean(L_rs)
    C_r_mean = np.mean(C_rs)
    sigma = (C / C_r_mean) / (L / L_r_mean)

    return sigma

def global_efficiency(G: nx.Graph, weight: str | None = None) -> float:

    '''
    NetworkX implementation modified to consider edge weights.
    About |V| times slower.
    If weight parameter the function simply wraps an original
    impementation (without weights)
    '''

    if weight is None:
        return nx.global_efficiency(G)
    
    n = len(G)
    
    num = 0.0
    den = n * (n - 1)

    if den != 0:

        lengths = nx.all_pairs_bellman_ford_path_length(G, weight=weight)

        # iterates over each pair of nodes
        for source, targets in lengths:
            for target, distance in targets.items():
                if distance > 0:
                    num += 1.0 / distance

        num /= den

    return num

def all_pairs_bellman_ford_path_length(adj_mat: ndarray, weighted: bool = False) -> ndarray:

    '''
    Returns ndarray of the same size with the corresponding shortest path lengths.
    Zeroes are interpreted as the absence of edge.
    '''

    G = nx.from_numpy_array(adj_mat)
    lengths = None

    if weighted:
        lengths = nx.all_pairs_bellman_ford_path_length(G, weight='weight')
    else:
        lengths = nx.all_pairs_shortest_path_length(G)

    all_pairs_length = np.zeros_like(adj_mat)

    # iterates over each pair of nodes
    for source, targets in lengths:
        for target, distance in targets.items():
            all_pairs_length[source, target] = distance

    return all_pairs_length

def pairwise_network_global_efficiencies(
        adj_mat: np.ndarray, 
        inverse_atlas: Dict[str, List[int]],
        weighted: bool = False, 
        inverse_weights: bool = False
) -> Dict[str, int | float]:
    
    lengths = None
    if inverse_weights:
        lengths = all_pairs_bellman_ford_path_length(extras.ndarray_reciprocal(adj_mat), weighted=weighted)
    else:
        lengths = all_pairs_bellman_ford_path_length(adj_mat, weighted=weighted)

    efficiencies = extras.ndarray_reciprocal(lengths)
    inverse_atlas_items = list(inverse_atlas.items())
    results = {}

    for i, (network1_name, network1_nodes) in enumerate(inverse_atlas_items):

        for network2_name, network2_nodes in inverse_atlas_items[(i + 1):]:

            num, den = 0.0, 0

            for node1, node2 in product(network1_nodes, network2_nodes):
                num += efficiencies[node1 - 1, node2 - 1]
                den += 1

            prefix = 'Unweighted'
            if weighted:
                prefix = 'Weighted'

            results[f'{network1_name}_{network2_name}_{prefix}_Efficiency'] = num / den

    return results

def compute_metrics(
        fc_matrix: ndarray,
        inverse_atlas: Dict[str, List[int]],
        threshold_percent: float
) -> Dict[str, int | float]:
    
    '''
    LIMIT MATRIX TO GORDON FOR NOW
    assumes indexing from 1 in atlas
    '''
    
    metrics = {}

    abs_fc_matrix = np.abs(fc_matrix)
    np.fill_diagonal(abs_fc_matrix, -1)

    # Intra-network efficiency
    metrics.update(pairwise_network_global_efficiencies(
        np.where(threshold_mask(abs_fc_matrix, threshold_percent), abs_fc_matrix, 0), # weighted adjacency matrix
        inverse_atlas,
        inverse_weights=True, 
        weighted=True
    ))

    metrics.update(pairwise_network_global_efficiencies(
        np.where(threshold_mask(abs_fc_matrix, threshold_percent), 1, 0), # unweighted adjacency matrix
        inverse_atlas, 
        inverse_weights=False, 
        weighted=False
    ))

    inverse_atlas[''] = list(range(1, fc_matrix.shape[0] + 1))

    # Inter-network metrics
    for network_name, network_nodes in inverse_atlas.items():

        network_size = len(network_nodes)
        fc_matrix_network = abs_fc_matrix.copy()
        nodes_to_remove = []

        for i in range(fc_matrix_network.shape[0]):

            if (i + 1) not in network_nodes:

                fc_matrix_network[i] = -1
                fc_matrix_network[:, i] = -1
                nodes_to_remove.append(i)
        
        thresh_mask = threshold_mask(fc_matrix_network, threshold_percent)
        adj_matrix = np.where(thresh_mask, 1, 0)
        wgt_matrix = np.where(thresh_mask, fc_matrix_network, 0)

        unweighted_graph = nx.from_numpy_array(adj_matrix)
        weighted_graph = nx.from_numpy_array(wgt_matrix)
        inverse_weighted_graph = nx.from_numpy_array(extras.ndarray_reciprocal(wgt_matrix))

        unweighted_graph.remove_nodes_from(nodes_to_remove)
        weighted_graph.remove_nodes_from(nodes_to_remove)
        inverse_weighted_graph.remove_nodes_from(nodes_to_remove)
        
        prefix = network_name + '_'
        if network_name == '':
            prefix = '' 
        
        # Inter-network efficiency
        metrics[prefix + 'Unweighted_Efficiency'] = nx.global_efficiency(unweighted_graph)
        metrics[prefix + 'Weighted_Efficiency'] = global_efficiency(inverse_weighted_graph, weight='weight')

        # Degree
        degree = np.sum(adj_matrix, axis=0)
        degree = degree[degree != 0]
        degree = np.pad(degree, (0, network_size - len(degree)), constant_values=0)
        metrics[prefix + 'DegreeMean'] = np.mean(degree)
        metrics[prefix + 'DegreeStd'] = np.std(degree)

        # Strength
        strength = np.sum(wgt_matrix, axis=0)
        strength = strength[strength != 0]
        strength = np.pad(strength, (0, network_size - len(strength)), constant_values=0)
        metrics[prefix + 'StrengthMean'] = np.mean(strength)
        metrics[prefix + 'StrengthStd'] = np.std(strength)     
        
        # Transitivity
        metrics[prefix + 'Transitivity'] = nx.transitivity(unweighted_graph)

        # Clustering
        metrics[prefix + 'Unweighted_Clustering'] = nx.average_clustering(unweighted_graph)     
        metrics[prefix + 'Weighted_Clustering'] = nx.average_clustering(weighted_graph, weight='weight')

        # Assortativity
        if network_size > 9:
            metrics[prefix + 'Unweighted_Assortativity'] = nx.degree_assortativity_coefficient(unweighted_graph)
            metrics[prefix + 'Weighted_Assortativity'] = nx.degree_assortativity_coefficient(weighted_graph, weight='weight')

        # Modularity
        metrics[prefix + 'Unweighted_Modularity'] = nx.community.modularity(
            unweighted_graph, nx.community.louvain_communities(unweighted_graph))
 
        metrics[prefix + 'Weighted_Modularity'] = nx.community.modularity(
            weighted_graph, nx.community.louvain_communities(weighted_graph, weight='weight'), weight='weight')
        
        if network_size == fc_matrix.shape[0]:

            lcc_nodes = max(nx.connected_components(unweighted_graph), key=len)
            unweighted_lcc_graph = unweighted_graph.subgraph(lcc_nodes).copy()
            weighted_lcc_graph = weighted_graph.subgraph(lcc_nodes).copy()
            
            metrics[prefix + 'LCCsize'] = len(lcc_nodes)
            metrics[prefix + 'Components'] = nx.number_connected_components(unweighted_graph)

            # Small-world (Sigma)
            metrics[prefix + 'Smallworld'] = small_world_sigma(unweighted_lcc_graph, 10)

            # Characteristic path length
            metrics[prefix + 'Unweighted_AvgShortestPath'] = nx.average_shortest_path_length(unweighted_lcc_graph)
            metrics[prefix + 'Weighted_AvgShortestPath'] = nx.average_shortest_path_length(weighted_lcc_graph, weight='weight')
        
    return metrics



if __name__ == '__main__':

    inverse_atlas = extras.inverse_gordon_atlas()
    fcs, roi_vec, fc_ids = reader.readAdjustedFcMatrices(roi_names_key=None)
    screentime_ids = reader.readScreentimeData()['participant_id'].to_numpy()

    id_mask = np.isin(fc_ids, screentime_ids) # Compute only for those with screentime data
    fcs, fc_ids = fcs[id_mask], fc_ids[id_mask]

    metrics = df(Parallel(n_jobs=-1) (
        delayed(compute_metrics)(
            extras.reconstruct_fc_matrix(fc, roi_vec), # 2D matrix from a flat upper tri
            inverse_atlas.copy(),
            threshold_percent=0.1
        ) for fc in tqdm(fcs)
    ))

    metrics.insert(0, 'participant_id', fc_ids)
    metrics.to_csv(directories.dataDirectory.joinpath('graph_metrics.csv'), index=False)