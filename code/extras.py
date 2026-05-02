import reader
import numpy as np
import pandas as pd

from pathlib import Path
from numpy import ndarray
from collections import Counter
from pandas import DataFrame as df
from typing import Dict, List, Tuple


def get_categories(
        screentime: df | None = None, 
        min_users: int | None = None, 
        min_usage: int | float | None = None
)    -> List[str]:

    '''
    Returns the names of the categories used by at least min_users 
    participants for at least min_usage minutes a day on average.
    If either of parameters is None - returns all of the categories.
    '''

    if screentime is None:
        screentime = reader.readScreentimeData()

    categories = screentime.columns.difference(['participant_id', 'session_id'])

    if min_users is None or min_usage is None:
        return categories.to_list()
    
    above_threshold = screentime[categories].ge(min_usage) # Greater or equal
    return categories[above_threshold.sum(axis=0) >= min_users].to_list()


def generalize_logs_data(screentime: df | None = None, include_ids: bool = False) -> df:

    mapping = {
        'g_networking': [
            'communication', 'social', 
            'dating',
        ],
        'g_recreation': [
            'entertainment', 'music', 
            'comics', 'sports'
        ],
        'g_commerce': [
            'shopping', 'events', 
            'travel', 'food'
        ],
        'g_interests': [
            'art', 'auto', 'beauty',
            'business', 'health', 'house',
            'lifestyle', 'parenting',
        ],
        'g_modern': [
            'gameaction', 'gameadventure', 
            'gamearcade', 'gamecasual', 
            'gamemusic', 'gameracing', 
            'gameroleplay', 'gamestrategy', 
            'gamesimulation', 'gamesports'
        ],        
        'g_classic': [
            'gameboard', 'gamecard',
            'gametrivia', 'gameword',
            'gamepuzzle', 'gamecasino'
        ],             
        'g_erudition': [
            'books', 'education',
            'gameeducation', 'news'
        ],   
        'g_everyday': [
            'maps', 'medical', 'libraries',
            'personalization', 'photography',
            'productivity', 'weather',
            'tools', 'video', 'finance'
        ]
    }

    if screentime is None:
        screentime = reader.readScreentimeData()
    
    g_screentime = pd.df(index=screentime.index)

    for group, categories in mapping.items():
        g_screentime[group] = screentime[categories].sum(axis=1)

    if include_ids:
        g_screentime.insert(0, 'participant_id', screentime['participant_id'])

    return g_screentime


def reconstruct_fc_matrix(fcs: ndarray, roi_vec: ndarray, orig_size: int) -> ndarray:

    '''Assumes indexing from 1'''

    if roi_vec.dtype is not int:
        roi_vec = np.asarray(roi_vec, dtype=int)
    fc_matrix = np.empty((orig_size, orig_size))
    for fc, (ind1, ind2) in zip(fcs, roi_vec):
        fc_matrix[ind1 - 1, ind2 - 1] = fc
    fc_matrix += np.triu(fc_matrix, k=1).T
    return fc_matrix


def ensure_common_ids(df1: df, df2: df, on: str | None = 'participant_id') -> Tuple[df, df]:

    '''
    Keeps only the rows in df1 and df2 that share common IDs and aligns them in the same order.
    on : Common column name to use as ID. If None, uses the index
    '''

    common_mask = None
    if on is None:
        common_mask = df1.index.isin(df2.index)
    else:
        common_mask = df1[on].isin(df2[on])
    
    df1_common, df2_common = df1[common_mask].copy(), None

    if on is None:
        df2_common = df2.loc[df1_common.index].copy()
    else:
        df2_common = df2.set_index(on).loc[df1_common[on]].reset_index().copy()
    
    if on is not None:
        df1_common.reset_index(drop=True, inplace=True)

    return df1_common, df2_common    


def ndarray_reciprocal(arr: ndarray) -> ndarray:

    '''Multiplicative inverse, ignores zeroes'''

    arr_inv = np.zeros_like(arr)
    mask = (arr != 0)
    arr_inv[mask] = 1 / arr[mask]
    return arr_inv


def inverse_gordon_atlas(include_subcortical: bool = True) -> Dict[str, List[int]]:

    '''
    Indexing from 1
    Maps networks to a list of ROIs
    '''

    # Cortical
    names = reader.readGordonCommunityNames()
    affil = reader.readGordonCommunityAffiliation()
    inverse_atlas = {name: [] for name in names}
    
    for i, name_ind in enumerate(affil):
        inverse_atlas[names[name_ind - 1]].append(i + 1)

    # Subcortical
    if include_subcortical:
        names = reader.readGpAsegCorr(corrmat_key=None, varmat_key=None, roi_vec_key=None)[0]
        inverse_atlas['Subcortical'] = list(range(len(affil) + 1, len(names) + 1))

    return inverse_atlas


def gordon_community_sizes() -> Dict[str, int]:

    names = reader.readGordonCommunityNames()
    affils = reader.readGordonCommunityAffiliation()
    cnt = Counter()
    for affil in affils:
        cnt[names[affil - 1]] += 1
    return dict(cnt)

