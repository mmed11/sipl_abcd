import extras
import reader
import directories
import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm
from pathlib import Path
from numpy import ndarray
from typing import Dict, List
from pandas import DataFrame as df
from cdlib.algorithms import leiden
from joblib import Parallel, delayed


def threshold_mask(
        matrix: ndarray, 
        threshold_percent: int | float | None = None, 
        threshold_absolute: int | float | None = None
    ) -> ndarray:
    
    '''
    :param matrix: Must be symmetrical
    :param threshold_percent: Values between 0 and 100
    '''

    if threshold_percent is None and threshold_absolute is None:
        raise ValueError('No threshold parameter was provided!')
    
    if not threshold_percent is None and not threshold_absolute is None:
        raise ValueError('More than one threshold parameter was provided!')

    if threshold_percent is not None:
        values = matrix[np.triu_indices(matrix.shape[0], k=1)]
        threshold_absolute = np.percentile(values, 100 - threshold_percent)

    return matrix >= threshold_absolute


def small_world_sigma(G: nx.Graph, nrand: int) -> float:

    L = nx.average_shortest_path_length(G)
    C = nx.transitivity(G)
    n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()

    L_rs, C_rs = np.empty(nrand), np.empty(nrand)
    for i in range(nrand):

        # Random graph with the same number of nodes and edges
        G_r = nx.gnm_random_graph(n_nodes, n_edges)
        while not nx.is_connected(G_r):
            G_r = nx.gnm_random_graph(n_nodes, n_edges)

        L_rs[i] = nx.average_shortest_path_length(G_r)
        C_rs[i] = nx.transitivity(G_r)
    
    L_r_mean = np.mean(L_rs)
    C_r_mean = np.mean(C_rs)
    sigma = (C / C_r_mean) / (L / L_r_mean)

    return sigma


def global_efficiency(G: nx.Graph, weight: str | None = None) -> float:

    '''
    NetworkX implementation modified to consider edge weights.
    If weight parameter is not provided the function simply 
    wraps an original impementation (without weights)
    '''

    if weight is None:
        return nx.global_efficiency(G)
    
    n = len(G)
    num = 0.0
    den = n * (n - 1)

    if den != 0:
        lengths = nx.floyd_warshall_numpy(G, weight=weight)
        efficiencies = extras.ndarray_reciprocal(lengths)
        np.fill_diagonal(efficiencies, 0)
        num = np.sum(efficiencies)
        num /= den

    return num


def pairwise_network_global_efficiencies(
        adj_mat: np.ndarray, 
        inverse_atlas: Dict[str, List[int]],
        inverse_weights: bool = False,
        infix: str | None = None
    ) -> Dict[str, float]:

    lengths = None
    if inverse_weights:
        lengths = nx.floyd_warshall_numpy(nx.from_numpy_array(extras.ndarray_reciprocal(adj_mat)))
    else:
        lengths = nx.floyd_warshall_numpy(nx.from_numpy_array(adj_mat))

    efficiencies = extras.ndarray_reciprocal(lengths)
    inverse_atlas_items = list(inverse_atlas.items())
    results = {}

    for i, (network1_name, network1_nodes) in enumerate(inverse_atlas_items):
        network1_indices = np.array(network1_nodes) - 1  # Convert to 0-indexed
        for network2_name, network2_nodes in inverse_atlas_items[(i + 1):]:
            network2_indices = np.array(network2_nodes) - 1  # Convert to 0-indexed
            
            eff_submatrix = efficiencies[np.ix_(network1_indices, network2_indices)]
            num = np.sum(eff_submatrix)
            den = len(network1_indices) * len(network2_indices)

            title = f'{network1_name}_{network2_name}_'
            if infix is not None:
                title += infix + '_'
            title += 'Efficiency'

            results[title] = num / den

    return results
    

def modularity(G: nx.Graph, weight: str | None = None, niter: int = 100) -> float:
    modularities = np.empty(niter)
    for i in range(niter):
        partition = leiden(G).communities
        modularities[i] = nx.community.modularity(G, partition, weight=weight)
    return modularities.max()


def compute_metrics(
        fc_matrix: ndarray,
        inverse_atlas: Dict[str, List[int]],
        threshold_percent: float
    ) -> Dict[str, int | float]:
    
    '''Assumes indexing from 1 in the atlas'''
    
    metrics = {}
    abs_fc_mat = np.abs(fc_matrix)
    np.fill_diagonal(abs_fc_mat, 0)

    adj_mat = np.where(threshold_mask(abs_fc_mat, threshold_percent), 1, 0)
    wgt_mat = np.where(threshold_mask(abs_fc_mat, threshold_percent), abs_fc_mat, 0)

    # Between-network efficiency
    metrics.update(pairwise_network_global_efficiencies(
        adj_mat,
        inverse_atlas, 
        inverse_weights=False,
        prefix='Unweighted'
    ))
    metrics.update(pairwise_network_global_efficiencies(
        wgt_mat,
        inverse_atlas,
        inverse_weights=True,
        prefix='Weighted'
    ))
    
    inverse_atlas[''] = np.arange(fc_matrix.shape[0]) + 1 # To compute metrics on a whole matrix
    net_sizes = extras.gordon_community_sizes()

    # Inside-network metrics
    for net_name, net_nodes in inverse_atlas.items():

        net_size = len(net_nodes)
        net_idx = np.array(net_nodes) - 1

        net_fc_mat = abs_fc_mat[np.ix_(net_idx, net_idx)]
        net_adj_mat = adj_mat[np.ix_(net_idx, net_idx)]
        net_wgt_mat = wgt_mat[np.ix_(net_idx, net_idx)]

        bin_graph = nx.from_numpy_array(net_adj_mat)
        wgt_graph = nx.from_numpy_array(net_wgt_mat)
        nothresh_graph = nx.from_numpy_array(net_fc_mat)
        inv_wgt_graph = nx.from_numpy_array(extras.ndarray_reciprocal(net_wgt_mat))

        isolates = list(nx.isolates(nothresh_graph)) # Alien nodes from other networks
        bin_graph.remove_nodes_from(isolates)
        wgt_graph.remove_nodes_from(isolates)
        inv_wgt_graph.remove_nodes_from(isolates)
        assert bin_graph.number_of_nodes() == net_sizes.get(net_name, net_size) == net_size

        prefix = None
        if net_name is None or net_name == '':
            prefix = '' 
        else:
            prefix = net_name + '_'    
            
        # Degree
        degrees = np.array([d for _, d in bin_graph.degree()])
        metrics[prefix + 'AvgDegree'] = degrees.mean()
        metrics[prefix + 'SdDegree'] = degrees.std(ddof=1)

        # Strength
        strengths = np.array([s for _, s in wgt_graph.degree(weight='weight')])
        metrics[prefix + 'AvgStrength'] = strengths.mean()
        metrics[prefix + 'SdStrength'] = strengths.std(ddof=1)   

        # Inside-network efficiency
        metrics[prefix + 'Unweighted_Efficiency'] = nx.global_efficiency(bin_graph)
        metrics[prefix + 'Weighted_Efficiency'] = global_efficiency(inv_wgt_graph, weight='weight')
        
        # Transitivity
        metrics[prefix + 'Transitivity'] = nx.transitivity(bin_graph)

        # Clustering
        metrics[prefix + 'Unweighted_Clustering'] = nx.average_clustering(bin_graph)     
        metrics[prefix + 'Weighted_Clustering'] = nx.average_clustering(wgt_graph, weight='weight')
        
        # Assortativity
        metrics[prefix + 'Unweighted_Assortativity'] = nx.degree_assortativity_coefficient(bin_graph)
        metrics[prefix + 'Weighted_Assortativity'] = nx.degree_assortativity_coefficient(wgt_graph, weight='weight')
            
        # Modularity
        metrics[prefix + 'Unweighted_Modularity'] = modularity(bin_graph)
        metrics[prefix + 'Weighted_Modularity'] = modularity(wgt_graph, weight='weight')

        # Substitution with a LCC for metrics requiring a connected graph
        lcc_nodes = max(nx.connected_components(bin_graph), key=len)
        lcc_graph = bin_graph.subgraph(lcc_nodes).copy()

        # Small-world (Sigma)
        metrics[prefix + 'Smallworld'] = small_world_sigma(lcc_graph, nrand=100)

    return metrics


if __name__ == '__main__':

    fcs, roi_vec, fc_ids = reader.readAdjustedFcMatrices(roi_names_key=None)
    screentime_ids = reader.readScreentimeData()['participant_id'].to_numpy()
    inverse_atlas = extras.inverse_gordon_atlas()

    # Avoid computing metrics on small networks and on None or Subcortical networks
    for net_name, net_size in extras.gordon_community_sizes().items():
        if net_size <= 10 or net_name == 'None':
            inverse_atlas.pop(net_name)
    inverse_atlas.pop('Subcortical')
    print(list(inverse_atlas.keys()))
    
    id_mask = np.isin(fc_ids, screentime_ids) # Compute only for participants with screentime data
    fcs, fc_ids = fcs[id_mask], fc_ids[id_mask]

    metrics = df(Parallel(n_jobs=-1) (
        delayed(compute_metrics)(
            extras.reconstruct_fc_matrix(fc, roi_vec, 366)[:333, :333], # Limit to Gordon's cortical
            inverse_atlas.copy(),
            threshold_percent=10
        ) for fc in tqdm(fcs)
    ))

    metrics.insert(0, 'participant_id', fc_ids)
    metrics.to_csv(directories.dataDirectory.joinpath('graph_metrics.csv'), index=False)

