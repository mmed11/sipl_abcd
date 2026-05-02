import os
import extras
import reader
import numpy as np
import directories
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from itertools import product
from pandas import DataFrame as df
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold, GridSearchCV, BaseCrossValidator
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# Constants
os.environ['OMP_NUM_THREADS'] = '16'
RANDOM_STATE = 11_03_2026
SLICE_FC_COLUMNS = None         # \ 
SLICE_GRAPH_COLUMNS = None      #  > Indices of correspoding variables in a feature matrix
SLICE_REMAINING_COLUMNS = None  # /


def svc(X, y, scoring='balanced_accuracy', n_jobs=-1):

    preprocess = ColumnTransformer(
        transformers=[
            ('fc', Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA())
            ]), SLICE_FC_COLUMNS),
            ('graph', Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA())
            ]), SLICE_GRAPH_COLUMNS),
            ('rest', StandardScaler(), SLICE_REMAINING_COLUMNS)
        ],
        remainder='drop' # NOTE: Needs modification if other data types are added!
    )
    pipe = Pipeline([
        ('preprocess', preprocess),
        ('svm', SVC())
    ])
    param_grid = {
        'preprocess__fc__pca__n_components': [100, 150],
        'preprocess__graph__pca__n_components': [50],
        #'svm__C': [0.1, 1, 10, 100, 1000],
        #'svm__gamma': ['scale'],
        'svm__class_weight': ['balanced'],
        #'svm__kernel': ['rbf']
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=1)
    gs.fit(X, y)

    best_params = gs.best_params_
    pipe.set_params(**best_params)
    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=n_jobs)

    scoring_map = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score
    }
    accuracy = scoring_map[scoring](y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)

    return accuracy, conf_matrix, best_params


class NoveltyCV(BaseCrossValidator):

    '''
    NOTE: Not a true CV, but a workaround!
    Test on equal amount of inliers and outliers, train only on the remaining inliers
    '''

    def __init__(self, n_splits=None, random_state=None):
        self.n_splits = 1
        self.random_state = random_state

    def split(self, X, y, groups=None):
        rng = np.random.default_rng(self.random_state)

        inliers_idx = np.where(y == 1)[0]
        outliers_idx = np.where(y == -1)[0]
        test_inliers_idx = rng.choice(inliers_idx, size=len(outliers_idx), replace=False)
        test_idx = np.hstack([test_inliers_idx, outliers_idx])
        train_idx = inliers_idx[~np.isin(inliers_idx, test_inliers_idx)]

        yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    

def one_class_svm(X, y, scoring='balanced_accuracy', n_jobs=-1): 
    
    '''1 - normal, -1 - anomaly'''

    preprocess = ColumnTransformer(
        transformers=[
            ('fc', Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA())
            ]), SLICE_FC_COLUMNS),
            ('graph', Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA())
            ]), SLICE_GRAPH_COLUMNS),
            ('rest', StandardScaler(), SLICE_REMAINING_COLUMNS)
        ],
        remainder='drop' # NOTE: Needs modification if other data types are added!
    )
    pipe = Pipeline([
        ('preprocess', preprocess),
        ('one_class_svm', OneClassSVM())
    ])
    param_grid = {
        'preprocess__fc__pca__n_components': [100, 150],
        'preprocess__graph__pca__n_components': [50],
        #'one_class_svm__nu': np.linspace(0.01, 0.15, 10),
        #'one_class_svm__gamma': ['scale'],
        #'one_class_svm__kernel': ['rbf']
    }
    
    cv = NoveltyCV(random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=1)
    
    gs.fit(X, y)
    best_params = gs.best_params_
    pipe.set_params(**best_params)
    train_idx, test_idx = next(cv.split(X=X, y=y))
    y_pred = pipe.fit(X.iloc[train_idx]).predict(X.iloc[test_idx])

    scoring_map = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score
    }
    accuracy = scoring_map[scoring](y[test_idx], y_pred)
    conf_matrix = confusion_matrix(np.where(y[test_idx] == 1, 0, 1), np.where(y_pred == 1, 0, 1))

    return accuracy, conf_matrix, best_params


def isolation_forest(X, y, scoring='balanced_accuracy', n_jobs=-1):

    '''1 - normal, -1 - anomaly'''

    preprocess = ColumnTransformer(
        transformers=[
            ('fc', Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA())
            ]), SLICE_FC_COLUMNS),
            ('graph', Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA())
            ]), SLICE_GRAPH_COLUMNS),
            ('rest', StandardScaler(), SLICE_REMAINING_COLUMNS)
        ],
        remainder='drop' # NOTE: Needs modification if other data types are added!
    )
    pipe = Pipeline([
        ('preprocess', preprocess),
        ('isolation_forest', IsolationForest())
    ])
    contamination = (y == -1).sum() / y.shape[0]
    param_grid = {
        'preprocess__fc__pca__n_components': [100, 150],
        'preprocess__graph__pca__n_components': [50],
        #'isolation_forest__n_estimators': [100],
        #'isolation_forest__max_samples': ['auto'],
        'isolation_forest__contamination': [contamination, 'auto'],
        'isolation_forest__random_state': [RANDOM_STATE]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=1)
    gs.fit(X, y)

    best_params = gs.best_params_
    pipe.set_params(**best_params)
    y_pred = pipe.fit_predict(X)

    scoring_map = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score
    }
    accuracy = scoring_map[scoring](y, y_pred)
    conf_matrix = confusion_matrix(np.where(y == 1, 0, 1), np.where(y_pred == 1, 0, 1))

    return accuracy, conf_matrix, best_params


def plot_confusion_matrix(conf_matrix, conf_mat_label, conf_mat_path):

    fig = plt.figure(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.title(conf_mat_label)
    plt.tight_layout()
    Path.mkdir(conf_mat_path.parent, exist_ok=True, parents=True)
    plt.savefig(conf_mat_path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    
    fcs, roi_vec, _, ids = reader.readAdjustedFcMatrices()
    graph_metrics = reader.readGraphMetrics()

    screentime = reader.readScreentimeData()
    screentime.drop(columns=['session_id'], inplace=True)
    screentime = pd.concat(
        [screentime[['participant_id'] + extras.get_categories(screentime, 100, 1)]] +
        [extras.generalize_logs_data(screentime, mapping='michael')], 
        axis='columns'
    )
    # Limit to those with screentime data
    id_mask = np.isin(ids, screentime['participant_id']) 
    fcs, ids = fcs[id_mask], ids[id_mask]
    # Limit to cortical ROIs not on diagonal (Indexing from 1)
    fcs = fcs[:, [(max(parcel1, parcel2) < 334 and parcel1 != parcel2) for parcel1, parcel2 in roi_vec]] 
    
    graph_metrics.drop(columns=['AvgDegree'], inplace=True)
    pca_exclude = graph_metrics.loc[:, ['SdDegree', 'AvgStrength', 'SdStrength']].copy()
    graph_metrics.drop(columns=['SdDegree', 'AvgStrength', 'SdStrength'], inplace=True)

    SLICE_FC_COLUMNS = slice(0, fcs.shape[1])
    SLICE_GRAPH_COLUMNS = slice(fcs.shape[1], fcs.shape[1] + graph_metrics.shape[1])
    data = pd.concat(
        [
            pd.df(fcs, index=ids, columns=[f'fc_{i+1}' for i in range(fcs.shape[1])]),
            graph_metrics.set_index('participant_id'),
        ],
        axis='columns',
        join='inner',
        copy=False
    )
    data['degree_std'] = pca_exclude['SdDegree'].to_numpy()
    data['strength_mean'] = pca_exclude['AvgStrength'].to_numpy()
    data['strength_std'] = pca_exclude['SdStrength'].to_numpy()
    SLICE_REMAINING_COLUMNS = slice(SLICE_GRAPH_COLUMNS.stop, SLICE_GRAPH_COLUMNS.stop + 3)

    data.dropna(inplace=True)
    data, screentime = extras.ensure_common_ids(data, screentime.set_index('participant_id'), on=None)

    scoring = 'balanced_accuracy'
    percentiles = [85]
    categories = screentime.columns # .difference(['participant_id', 'session_id'])
    methods = [svc, one_class_svm, isolation_forest]
    results = {'category': [], 'percentile': [], 'method': [], scoring: [], 'best_params': []}
    
    for category, percentile, method in tqdm(product(categories, percentiles, methods)):

            classes = np.where(screentime[category] >= np.percentile(screentime[category], percentile), -1, 1)
            accuracy, conf_mat, best_params = method(
                data,
                classes,
                scoring=scoring,
                n_jobs=10
            )

            results['category'].append(category)
            results['percentile'].append(percentile)
            results['method'].append(method.__name__)
            results[scoring].append(accuracy)
            results['best_params'].append(best_params)

            conf_mat_path=directories.figuresDirectory.joinpath(f'confusion_matrices/{percentile}/{method.__name__}/{category}_confusion_matrix.png')
            conf_mat_label=f'{percentile}th percentile in {category.capitalize()},\nusing {method.__name__}; {scoring}: {accuracy * 100:.1f}%'
            plot_confusion_matrix(conf_mat, conf_mat_label, conf_mat_path)

    accuracy_df = pd.df(results)
    accuracy_df.to_csv(directories.resultsDirectory.joinpath(f'ml_{scoring}.csv'), index=False)

