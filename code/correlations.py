import reader
import extras
import directories
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy import stats
from pathlib import Path
from numpy import ndarray
from itertools import product
from typing import Tuple, Callable
from pandas import DataFrame as df
from abc import ABC, abstractmethod
from joblib import Parallel, delayed


class descriptionGenerator(ABC):
    @abstractmethod
    def description(self, feature):
        pass
    

class descriptionGeneratorGraph(descriptionGenerator):
    
    '''Parses the metrics' titles'''

    def description(self, feature: str) -> str:
        
        parts = feature.split('_')
        
        if len(parts) > 4:
            raise ValueError("Metric's name is too long!")
        
        if len(parts) == 2:  

            # e.g., Unweighted_Efficiency, 'NoThresh_Efficiency'
            if 'weighted' in parts[0].lower() or 'thresh' in parts[0].lower(): 
                return feature.replace('_', ' ')
            
            # e.g., Default_DegreeMean
            else: 
                return f'{parts[1]} of {parts[0]}'
            
        # e.g., Default_Weighted_Efficiency
        if len(parts) == 3: 
            return f'{parts[1]} {parts[2]} of {parts[0]}'
        
        # e.g., Default_SMhand_Weighted_Efficiency
        if len(parts) == 4: 
            return f'{parts[2]} {parts[3]} between {parts[0]} and {parts[1]}'

        # e.g., Smallworld
        return feature 
    

def target_corr(
        features: df, 
        target: pd.Series, 
        corr_coef: stats.pearsonr | stats.spearmanr,
        fdr: bool = False,
        description_gen: descriptionGenerator | None = None,
    ) -> df:
    
    '''Computes correlations between every feature variable and a single target variable'''

    results = {
        'feature': [],
        'description': [],
        'target': None,
        'r_val': [],
        'p_val': [],
        'n_samples': [],
    }

    for feature in features:

        temp = features.copy()[['participant_id', feature]]
        temp[target.name] = target.copy()
        temp.dropna(inplace=True)
        X, y = temp[feature], temp[target.name]

        if 'id' in feature.lower():
            continue

        results['feature'].append(feature)
        results['description'].append(
            description_gen.description(feature) if description_gen is not None else None
        )
        
        corr = None
        results['n_samples'].append(len(y))
        corr = corr_coef(X, y)
        results['r_val'].append(corr.statistic)
        results['p_val'].append(corr.pvalue)

    if fdr:
       results['p_val'] = stats.false_discovery_control(results['p_val'], method='bh')

    results['target'] = [target.name.title()] * len(results['feature'])
    return df(results)


def compute_correlations(
        X: df, 
        y: df, 
        coef_name: str, 
        fdr: bool = False,
        description_gen: descriptionGenerator | None = None
    ) -> df:

    '''Computes correlations between every variable from X and variables in y'''
    
    corr_coef = None
    if coef_name.lower() == 'pearson':
        corr_coef = stats.pearsonr
    elif coef_name.lower() == 'spearman':
        corr_coef = stats.spearmanr
    else:
        raise ValueError('Unknown correlation coefficient!')

    targets = y.columns.to_list()
    targets = [target for target in targets if not ('id' in target.lower())]

    corrs = pd.concat(Parallel(n_jobs=-1, verbose=0)(
        delayed(target_corr)(X, y[target], corr_coef, fdr, description_gen) for target in tqdm(targets)
    ))

    return corrs


if __name__ == '__main__':

    screentime = reader.readScreentimeData()
    screentime.drop(columns=['session_id'], inplace=True)
    screentime = pd.concat(
        [screentime[['participant_id'] + extras.get_categories(screentime, 100, 1)]] +
        [extras.generalize_logs_data(screentime)], 
        axis='columns'
    )
    
    log_screentime = screentime.copy()
    num_cols = log_screentime.select_dtypes(include='number').columns
    log_screentime[num_cols] = np.log1p(log_screentime[num_cols])

    graph_metrics = reader.readGraphMetrics()    
    graph_metrics = graph_metrics.loc[:, ~graph_metrics.columns.str.contains('Degree|Strength')]
    graph_metrics = graph_metrics.loc[:, graph_metrics.nunique(dropna=True) > 1]

    graph_metrics, screentime = extras.ensure_common_ids(graph_metrics, screentime)
    graph_metrics, log_screentime = extras.ensure_common_ids(graph_metrics, log_screentime)
    assert graph_metrics['participant_id'].equals(screentime['participant_id'])
    assert screentime['participant_id'].equals(log_screentime['participant_id'])
    print(f'{graph_metrics.shape[0]} x {graph_metrics.shape[1]}')

    min_p_val = 0.05
    fdrs = [False, True]
    targets = [screentime, log_screentime]

    for fdr, target in product(fdrs, targets):

        pearson = compute_correlations(graph_metrics, target, 'pearson', fdr=fdr, description_gen=descriptionGeneratorGraph())
        pearson['method'] = 'pearson'
        spearman = compute_correlations(graph_metrics, target, 'spearman', fdr=fdr, description_gen=descriptionGeneratorGraph()) 
        spearman['method'] = 'spearman'

        corrs = pd.concat([pearson, spearman], ignore_index=True)
        corrs = corrs[corrs['p_val'] < min_p_val]
        
        filename = 'graph' + ('_log' if target is log_screentime else '') + ('_fdr' if fdr else '') + '.csv'
        path = directories.resultsDirectory.joinpath(f'corrs\\{filename}')
        path.parent.mkdir(exist_ok=True, parents=True)
        corrs.to_csv(path, index=False)

