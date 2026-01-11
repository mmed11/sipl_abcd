import os
import extras
import reader
import numpy as np
import directories
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from math import log10, nan
from sklearn.svm import SVC
from pandas import DataFrame as df
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay, get_scorer
from sklearn.model_selection import cross_val_predict, GridSearchCV, StratifiedKFold, KFold, RandomizedSearchCV, HalvingGridSearchCV


SLICE_FC_COLUMNS = None
SLICE_GRAPH_COLUMNS = None


def train_svm(X, y, scoring='balanced_accuracy', conf_mat_path=None, conf_mat_label=None):

    preprocess = ColumnTransformer(
        transformers=[
            ('fc', Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA())
            ]), SLICE_FC_COLUMNS),
            ('graph', Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA())
            ]), SLICE_GRAPH_COLUMNS)
        ],
        remainder='drop' # NOTE: Needs modification if other data types are added!
    )
    pipe = Pipeline([
        ('preprocess', preprocess),
        ('svm', SVC())
    ])
    param_grid = {
        'preprocess__fc__pca__n_components': [10, 20, 50, 100],
        'preprocess__graph__pca__n_components': [5, 10, 20, 40],
        'svm__C': np.logspace(-2, 3, 6),  # Exponentially spaced values between 0.01 and 1000
        'svm__gamma': np.logspace(-4, 1, 6), # Exponentially spaced values between 0.0001 and 10
        'svm__class_weight': [None, 'balanced'],
        'svm__kernel': ['rbf']
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=11_01_2026)
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)

    gs.fit(X, y)
    best_params = gs.best_params_
    pipe.set_params(**best_params)
    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)

    scoring_map = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score
    }
    accuracy = scoring_map[scoring](y, y_pred)

    if conf_mat_path is not None:
        conf_matrix = confusion_matrix(y, y_pred, labels=gs.best_estimator_.classes_)
        plt.figure(figsize=(9, 7))
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        disp.plot()
        plt.title(f'{conf_mat_label}\nFC PCs: {best_params["preprocess__fc__pca__n_components"]} | Graph PCs: {best_params["preprocess__graph__pca__n_components"]} | Class Weight: {best_params["svm__class_weight"]}\n{scoring.replace("_", " ")}: {(accuracy * 100):.1f}%')
        plt.tight_layout()
        plt.savefig(conf_mat_path, dpi=200)
        
    plt.close()
    return accuracy



if __name__ == '__main__':
    
    fcs, roi_vec, _, ids = reader.readAdjustedFcMatrices()
    graph_metrics = reader.readGraphMetrics()
    screentime = reader.readScreentimeData()
    screentime.drop(columns=['session_id'], inplace=True)
    screentime = pd.concat([screentime] + [extras.generalize_logs_data(screentime)], axis='columns')

    id_mask = np.isin(ids, screentime['participant_id']) # Limit to those with screentime data
    fcs, ids = fcs[id_mask], ids[id_mask]
    fcs = fcs[:, [(max(parcel1, parcel2) < 334 and parcel1 != parcel2) for parcel1, parcel2 in roi_vec]] # Limit to cortical ROIs not on diagonal

    SLICE_FC_COLUMNS = slice(0, fcs.shape[1])
    SLICE_GRAPH_COLUMNS = slice(fcs.shape[1], fcs.shape[1] + graph_metrics.shape[1])

    data = pd.concat(
        [
            pd.DataFrame(fcs, index=ids, columns=[f'fc_{i+1}' for i in range(fcs.shape[1])]),
            graph_metrics.set_index('participant_id'),
        ],
        axis='columns',
        join='inner',
        copy=False
    )
    data.dropna(inplace=True)
    data, screentime = extras.ensure_common_ids(data, screentime.set_index('participant_id'), on=None)

    scoring = 'balanced_accuracy'
    accuracies = {}
    for category in tqdm(extras.get_categories(screentime, 100, 1/7)):

        classes = np.where(
            screentime[category] >= np.percentile(screentime[category], 85),
            'In 85th Percentile',
            'Normal'
        )

        accuracy = train_svm(
            data,
            classes,
            scoring=scoring,
            conf_mat_path=directories.figuresDirectory.joinpath(f'confusion_matrices\\{category}_confusion_matrix.png'),
            conf_mat_label=category.capitalize()
        )
        accuracies[category] = accuracy

    accuracy_df = pd.DataFrame(accuracies, columns=['Category', scoring.replace('_', ' ').capitalize()])
    accuracy_df.to_csv(directories.resultsDirectory.joinpath(f'svm_{scoring}.csv'),index=False)