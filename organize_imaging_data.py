import h5py
import reader
import directories
import numpy as np
import pandas as pd

from pathlib import Path
from numpy import ndarray
from pandas import DataFrame as df
from typing import Callable, Tuple, List



session_id = 'ses-04A'

dir_imaging = directories.dataDirectory.joinpath(r'imaging')

dir_source = dir_imaging.joinpath(r'source')
dir_source.mkdir(parents=True, exist_ok=True)

dir_organized = dir_imaging.joinpath(r'organized')
dir_organized.mkdir(parents=True, exist_ok=True)

file_vol_info = 'vol_info'
file_gpnet_corr = 'gpnet_corr'
file_gpnet_aseg_corr = 'gpnet_aseg_corr'
file_gp_aseg_corr = 'gp_aseg_corr'

dtype_utf8 = h5py.string_dtype(encoding='utf-8')



def translateGpnetCorr(mask: ndarray | None = None
                       ) -> Tuple[ndarray, ndarray, ndarray]:

    corrmat, net_vec, network_names = None, None, None

    with h5py.File(dir_source.joinpath('gpnet_corr.mat'), 'r') as f:

        corrmat = np.asarray(f['corrmat']).T

        net1vec = np.asarray(f['net1vec'][0])
        net2vec = np.asarray(f['net2vec'][0])
        net_vec = np.stack((net1vec, net2vec), axis=-1)

        network_names_list = []

        for ref in f['network_names']:
            network_names_list.append(f[ref[0]][()].tobytes()[::2].decode())

        network_names = np.asarray(network_names_list, dtype=dtype_utf8)

    if not mask is None:
        corrmat = corrmat[mask]   

    return (corrmat, net_vec, network_names)

def translateGpnetAsegCorr(mask: ndarray | None = None
                           ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    
    corrmat, net_vec, network_names, roi_names = None, None, None, None
    
    with h5py.File(dir_source.joinpath('gpnet_aseg_corr.mat'), 'r') as f:
        
        corrmat = np.asarray(f['corrmat']).T

        net1vec = np.asarray(f['net1vec'][0])
        roi2vec = np.asarray(f['roi2vec'][0])
        net_vec = np.stack((net1vec, roi2vec), axis=-1)
        
        network_names_list, roi_names_list = [], []

        for ref in f['network_names']:
            network_names_list.append(f[ref[0]][()].tobytes()[::2].decode())
        
        for ref in f['roinames']:
            roi = f[ref[0]][()].tobytes()[::2].decode()
            roi = roi.replace('-', '_')
            roi_names_list.append(roi)    
            
        network_names = np.asarray(network_names_list, dtype=dtype_utf8)
        roi_names = np.asarray(roi_names_list, dtype=dtype_utf8)

    if not mask is None:
        corrmat = corrmat[mask]
        
    return (corrmat, net_vec, network_names, roi_names)

def translateGpAsegCorr(mask: ndarray | None = None
                        ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:

    corrmat, varmat, roi_vec, roi_names = None, None, None, None

    with h5py.File(dir_source.joinpath('gp_aseg_corr.mat'), 'r') as f:

        corrmat = np.asarray(f['corrmat']).T
        varmat = np.asarray(f['varmat']).T

        roi1vec = np.asarray(f['roi1vec'][0])
        roi2vec = np.asarray(f['roi2vec'][0])
        roi_vec = np.stack((roi1vec, roi2vec), axis=-1)

        roi_names_list = []

        for ref in f['roinames']:
            roi_name = f[ref[0]][()].tobytes()[::2].decode()
            roi_names_list.append(roi_name.replace('-', '_'))

        roi_names = np.asarray(roi_names_list, dtype=dtype_utf8)

    if not mask is None:
        corrmat = corrmat[mask]
        varmat = varmat[mask]

    return (corrmat, varmat, roi_vec, roi_names)

def establishDataset(translator: Callable,
                     dataset_names: list[str],
                     path: Path | str,
                     mask: ndarray | None = None,
                     force: bool = False) -> None:
    
    exists_valid = False

    if path.exists() and (not force):
        with h5py.File(path, 'r') as f:
            if set(dataset_names) == set(f.keys()): # if datasets differ - delete
                exists_valid = True
    
    if exists_valid:
        return
    
    path.unlink(missing_ok=True)
    datasets = translator(mask=mask)
    
    with h5py.File(path, 'w') as f:
        for i in range(len(dataset_names)):
            f.create_dataset(dataset_names[i], data=datasets[i])

def organizeDatasets(datasets_to_establish: List[str]) -> None:

    vol_info = reader.readVolInfo()
    session_mask = (vol_info['session_id'] == session_id).to_numpy()

    datasets_params = {
        
        'gpnet_corr': (
            translateGpnetCorr,
            ('corrmat', 'net_vec', 'network_names'),
            dir_organized.joinpath(file_gpnet_corr + '.h5')),

        'gpnet_aseg_corr': (
            translateGpnetAsegCorr,
            ('corrmat', 'net_vec', 'network_names', 'roi_names'),
            dir_organized.joinpath(file_gpnet_aseg_corr + '.h5')),

        'gp_aseg_corr': (
            translateGpAsegCorr,
            ('corrmat', 'varmat', 'roi_vec', 'roi_names'),
            dir_organized.joinpath(file_gp_aseg_corr + '.h5'))
    }
    
    for dataset in datasets_to_establish:
        if dataset in datasets_params:
            establishDataset(*datasets_params[dataset], session_mask, True)
        else:
            print(f'{dataset} is an unknown dataset!')



if __name__ == '__main__':

    datasets_to_establish = [] # ['gpnet_corr', 'gp_aseg_corr', 'gp_aseg_corr']
    organizeDatasets(datasets_to_establish)


    
'''    
def establishGpnetCorr(force: bool = False) -> None:

exists_valid = False
dataset_names = ('corrmat', 'net_vec', 'network_names')
path_gpnet_corr_h5 = dir_organized.joinpath(file_gpnet_corr + '.h5')

if path_gpnet_corr_h5.exists() and (not force):
    with h5py.File(path_gpnet_corr_h5, 'r') as f:
        if set(dataset_names) == set(f.keys()): # if datasets differ - delete
            exists_valid = True

if exists_valid:
    return

path_gpnet_corr_h5.unlink(missing_ok=True)
datasets = translateGpnetCorr()

with h5py.File(path_gpnet_corr_h5, 'w') as f:
    for i in range(len(dataset_names)):
        f.create_dataset(dataset_names[i], data=datasets[i])



def establishGpnetAsegCorr(force: bool = False) -> None:

    exists_valid = False
    dataset_names = ('corrmat', 'net_vec', 'network_names', 'roi_names')
    path_gpnet_aseg_corr_h5 = dir_organized.joinpath(file_gpnet_aseg_corr + '.h5')

    if path_gpnet_aseg_corr_h5.exists() and (not force):
        with h5py.File(path_gpnet_aseg_corr_h5, 'r') as f:
            if set(dataset_names) == set(f.keys()): # if datasets differ - delete
                exists_valid = True
    
    if exists_valid:
        return
    
    path_gpnet_aseg_corr_h5.unlink(missing_ok=True)
    datasets = translateGpnetAsegCorr()
    
    with h5py.File(path_gpnet_aseg_corr_h5, 'w') as f:
        for i in range(len(dataset_names)):
            f.create_dataset(dataset_names[i], data=datasets[i], compression='gzip')

            

def establishGpAsegCorr(force: bool = False) -> None:

    exists_valid = False
    dataset_names = ('corrmat', 'varmat', 'roi_vec', 'roi_names')
    path_gp_aseg_corr_h5 = dir_organized.joinpath(file_gp_aseg_corr + '.h5')

    if path_gp_aseg_corr_h5.exists() and (not force):
        with h5py.File(path_gp_aseg_corr_h5, 'r') as f:
            if set(dataset_names) == set(f.keys()): # if datasets differ - delete
                exists_valid = True
    
    if exists_valid:
        return
    
    path_gp_aseg_corr_h5.unlink(missing_ok=True)
    datasets = translateGpAsegCorr()
    
    with h5py.File(path_gp_aseg_corr_h5, 'w') as f:
        for i in range(len(dataset_names)):
            f.create_dataset(dataset_names[i], data=datasets[i], compression='gzip')
'''
