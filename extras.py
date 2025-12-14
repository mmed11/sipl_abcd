import reader
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from numpy import ndarray
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def apply_pca(data, n_pcs, pre_scale=True, post_scale=True):
    raise NotImplementedError()

def apply_scaling(data):
    raise NotImplementedError()

def get_categories(min_users=None, min_usage=None):
    raise NotImplementedError()

__categories_default_mapping = {'G_Social': ['Communication', 'Social'],
                                'G_Passive': ['Entertainment', 'Music'],
                                'G_Information': ['Art', 'Auto', 'Beauty',
                                                  'Business', 'Comics', 'Finance',
                                                  'Food', 'Health', 'House',
                                                  'Lifestyle', 'News', 'Parenting',
                                                  'Sports', 'Travel'],
                                'G_Games': ['Gameaction', 'Gameadventure', 
                                            'Gamearcade', 'Gameboard', 'Gamecard', 
                                            'Gamecasino', 'Gamecasual', 'Gamemusic',
                                            'Gamepuzzle', 'Gameracing', 'Gameroleplay', 
                                            'Gamesimulation', 'Gamesports', 
                                            'Gamestrategy', 'Gametrivia'],
                                'G_Education': ['Books', 'Education',
                                                'Gameeducation', 'Gameword'],              
                                'G_Everyday': ['Events', 'Maps', 'Medical',
                                               'Personalization', 'Photography',
                                               'Productivity', 'Shopping', 'Weather']}

def generalize_logs_data(logs_df, keep_columns=False, mapping='default'):
    raise NotImplementedError()

def ndarray_reciprocal(arr: ndarray) -> ndarray:

    '''Multiplicative inverse, ignores zeroes'''

    arr_inv = np.zeros_like(arr)
    mask = (arr != 0)
    arr_inv[mask] = 1 / arr[mask]

    return arr_inv