import h5py
import scipy
import numpy as np
import pandas as pd
import nibabel as nib

from pathlib import Path
from directories import *
from numpy import ndarray
from pandas import DataFrame as df
from typing import Sequence, Tuple, List, Dict, Iterable



session_id = 'ses-04A'



def readScreentimeData(path: Path | str | None = None) -> df:

    if path is None:
        path = dataDirectory.joinpath('screentime.csv')

    screentime = pd.read_csv(path)
    screentime = screentime[screentime['session_id'] == session_id]
    screentime = screentime[screentime['nt_y_earsapp__day_count'] > 6] # At least a week of observations
    screentime = screentime.dropna().reset_index(drop=True)

    new_labels = {}
    drop_cols = []
    skip = ('participant_id', 'session_id')
    mods = (('nt_y_earsapp__mins_', ''), 
            ('_andr', ''),
            ('___', '_'), 
            ('__', '_'))
    
    for col_label in screentime.columns:
        
        if col_label in skip:
            continue

        if (
            col_label.endswith('mean') 
            and ('mins' in col_label) 
            and not ('screenon' in col_label)
            and not ('wknd' in col_label or 'wkdy' in col_label)
        ):
            
            new_label = col_label
            for mod in mods:
                new_label = new_label.replace(mod[0], mod[1])
            if new_label[0] == '_':
                new_label = new_label[1:]
            new_labels[col_label] = new_label

        else:
            drop_cols.append(col_label)

    screentime.drop(columns=drop_cols, inplace=True)
    screentime.rename(columns=new_labels, inplace=True)
    screentime.rename(columns={'mean': 'total_mean'}, inplace=True)

    return screentime

def readGraphMetrics(path):
    raise NotImplementedError()

def readGordonAtlas(path: Path | str | None = None) -> Dict[int, str]:

    '''Indexing from 1'''

    if path is None:
        path = gordonDirectory.joinpath('gordon333NodeNames.txt')
    
    atlas = {}

    with open(path, 'r') as f:

        for line in f:

            line = line.rstrip()
            index = int(line.split('_')[-1])
            atlas[index] = line

    return atlas

def readGordonCommunityNames(path: Path | str | None = None) -> List[str]:

    if path is None:
        path = gordonDirectory.joinpath('gordon333CommunityNames.txt')
    
    community_names = []
    
    with open(path, 'r') as f:
        for line in f:
            community_names.append(line.rstrip())
    
    return community_names

def readGordonCommunityAffiliation(path: Path | str | None = None) -> List[int]:

    '''Indexing from 1'''
    
    if path is None:
        path = gordonDirectory.joinpath('gordon333CommunityAffiliation.txt')

    community_affiliation = []

    with open(path, 'r') as f:
        for line in f:
            community_affiliation.append(int(line.rstrip()))
    
    return community_affiliation

def readAdjustedFcMatrices(path):
    raise NotImplementedError()

def readVolInfo(path: Path | str | None = None, filter_session: bool = False) -> df:

    if path is None:
        path = imagingDirectory.joinpath(r'source\vol_info.mat')

    vol_info_dict = scipy.io.loadmat(path, 
                                     chars_as_strings=True, 
                                     struct_as_record=True, 
                                     verify_compressed_data_integrity=True, 
                                     simplify_cells=True)
    
    remove_columns = ['__header__', '__version__', '__globals__', 'vol_mask']
    for rmv in remove_columns:
        vol_info_dict.pop(rmv, 0)

    vol_info_df = df(vol_info_dict)

    if filter_session:
        vol_info_df = vol_info_df[vol_info_df['session_id'] == session_id].reset_index(drop=True)

    #vol_info_df['participant_id'] = vol_info_df['participant_id'].apply(lambda s: s[4:]) # sub-005V6D2C -> 005V6D2C
    vol_info_df['event_id'] = vol_info_df['visitidvec'].apply(lambda s: s.split('_')[-1]) # S042_INV003RTV85_baseline -> baseline
    vol_info_df['site_id'] = vol_info_df['visitidvec'].apply(lambda s: int(s.split('_')[0][1:])) # S042_INV003RTV85_baseline -> 42 (int)
    #vol_info_df['session_id'] = vol_info_df['session_id'].apply(lambda s: s[4:]) # ses-00A -> 00A
    vol_info_df = vol_info_df[['participant_id', 'event_id', 'session_id', 'site_id']]

    return vol_info_df

def __readOrganizedImagingData(
        path: Path | str,
        keys: Iterable[str],
        is_str: Iterable[bool] | None = None,
        is_filterable: Iterable[bool] | None = None,
        mask: ndarray[bool] = None) -> List[ndarray]:
    
    '''
    :param keys: Names of datasets to read from an HDF5 file
    :param is_str: Whether a dataset is comprised of strings
    :param is_filterable: Whether a mask should be applied to a dataset
    :param mask: Mask to be applied to datasets (i.e., to return a specific session). 
    POINTLESS for now, as the data is filtered when saved

    NOTE: is_str and is_filterable can be set to None if not applicable, otherwise must be the same length as keys
    '''
    
    if (not is_str is None) and (not is_filterable is None):
        if (len(keys) != len(is_str)) or (len(keys) != len(is_filterable)):
            raise Exception('keys, is_str, is_filterable should all be the same length!')
        
    else:

        if is_str is None:
            is_str = [False] * len(keys)
        
        if is_filterable is None:
            is_filterable = [False] * len(keys)
        
    data = []
    with h5py.File(path, 'r') as f:

        for key, key_is_str, key_is_filterable in zip(keys, is_str, is_filterable):
            
            dataset = None

            if key_is_str:
                dataset = np.asarray(f[key].asstr(), dtype=str)
            else:
                dataset = np.asarray(f[key])

            if not mask is None and key_is_filterable:
                dataset = dataset[mask]
            
            data.append(dataset)
    
    return data
    
def readGpnetCorr(path: Path | str | None = None,
                  corrmat_key: str | None = 'corrmat',
                  net_vec_key: str | None = 'net_vec',
                  network_names_key: str | None = 'network_names') -> List[ndarray]:
    
    '''
    Pass None instead of key names to avoid including
    '''

    if path is None:
        path = imagingDirectory.joinpath(r'organized\gpnet_corr.h5')

    # (key, is_str, is_filterable)
    params = [
        (corrmat_key, False, True), 
        (net_vec_key, False, False), 
        (network_names_key, True, False)
    ]

    params = [param for param in params if param[0] is not None]
    keys, is_str, is_filterable = zip(*params)

    return __readOrganizedImagingData(path, keys, is_str, is_filterable)

def readGpnetAsegCorr(path: Path | str | None = None, 
                      corrmat_key: str | None = 'corrmat',
                      net_vec_key: str | None = 'net_vec',
                      network_names_key: str | None = 'network_names',
                      roi_names_key: str | None = 'roi_names') -> List[ndarray]:

    '''
    Pass None instead of key names to avoid including
    '''

    if path is None:
        path = imagingDirectory.joinpath(r'organized\gpnet_aseg_corr.h5')

    # (key, is_str, is_filterable)
    params = [
        (corrmat_key, False, True), 
        (net_vec_key, False, False), 
        (network_names_key, True, False),
        (roi_names_key, True, False)
    ]

    params = [param for param in params if param[0] is not None]
    keys, is_str, is_filterable = zip(*params)

    return __readOrganizedImagingData(path, keys, is_str, is_filterable)

def readGpAsegCorr(path: Path | str | None = None,
                   corrmat_key: str | None = 'corrmat',
                   varmat_key: str | None = 'varmat',
                   roi_vec_key: str | None = 'roi_vec',
                   roi_names_key: str | None = 'roi_names') -> List[ndarray]:
    
    '''
    Pass None instead of key names to avoid including
    '''

    if path is None:
        path = imagingDirectory.joinpath(r'organized\gp_aseg_corr.h5')

    # (key, is_str, is_filterable)
    params = [
        (corrmat_key, False, True), 
        (varmat_key, False, True), 
        (roi_vec_key, False, False),
        (roi_names_key, True, False)
    ]

    params = [param for param in params if param[0] is not None]
    keys, is_str, is_filterable = zip(*params)

    return __readOrganizedImagingData(path, keys, is_str, is_filterable)


