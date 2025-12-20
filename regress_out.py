import pandas as pd
import numpy as np
import directories
import reader
import extras
import h5py
from tqdm import tqdm
from pathlib import Path
from numpy import ndarray
from pandas import DataFrame as df
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler



dtype_utf8 = h5py.string_dtype(encoding='utf-8')



def regress_out(X: df, y: ndarray) -> ndarray:

    '''
    X - predictors (confounders)
    y - target (FCs)
    '''

    reg = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=-1).fit(X, y)
    y1 = y - reg.predict(X)

    return y1



if __name__ == '__main__':

    dataDyn = reader.readDynamicVars(filter_session=True)
    dataDyn = dataDyn[['participant_id', 'age', 'scanner_serial', 'scanner_software']]

    dataStc = reader.readStaticVars()[['participant_id', 'sex']]

    data = pd.merge(dataDyn, dataStc, on='participant_id')

    corrmat, varmat, roi_vec, roi_names = reader.readGpAsegCorr()
    vol_info = reader.readVolInfo(filter_session=True)


    vol_mask = vol_info['participant_id'].isin(data['participant_id'])
    data_mask = data['participant_id'].isin(vol_info['participant_id'])

    vol_info = vol_info[vol_mask]
    corrmat = corrmat[vol_mask]
    varmat = varmat[vol_mask]
    data = data[data_mask]
    
    nan_mask = ~(np.isnan(varmat).any(axis=1))

    vol_info = vol_info[nan_mask]
    corrmat = corrmat[nan_mask]
    varmat = varmat[nan_mask]
    data = data[nan_mask]


    continuos_vars = ['age']
    categorical_vars = ['sex', 'scanner_serial', 'scanner_software']
    confound_vars = continuos_vars + categorical_vars

    confounds = pd.get_dummies(data[confound_vars], columns=categorical_vars, drop_first=True)

    path = directories.imagingDirectory.joinpath(r'organized\fc_adjusted.h5')
    path.unlink(missing_ok=True)

    with h5py.File(path, 'w') as f:

        ids = data['participant_id'].to_numpy()
        np.save(str(path).split('.')[0] + '_participant_id.npy', ids)

        f.create_dataset('varmat', data=regress_out(confounds, varmat))
        f.create_dataset('corrmat', data=regress_out(confounds, corrmat))
    
