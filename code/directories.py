from pathlib import Path

# Central directory that everything circles around
abcdDirectory = Path.home().joinpath(r'Documents\abcd\ABCD_6')
abcdDirectory.mkdir(parents=True, exist_ok=True)

dataDirectory = abcdDirectory.joinpath('data')
dataDirectory.mkdir(exist_ok=True)

imagingDirectory = dataDirectory.joinpath('imaging')
imagingDirectory.mkdir(exist_ok=True)

gordonDirectory = dataDirectory.joinpath('gordon333')
gordonDirectory.mkdir(exist_ok=True)

generalDirectory = dataDirectory.joinpath('general')
generalDirectory.mkdir(exist_ok=True)

figuresDirectory = abcdDirectory.joinpath('figures')
figuresDirectory.mkdir(exist_ok=True)

resultsDirectory = abcdDirectory.joinpath('results')
resultsDirectory.mkdir(exist_ok=True)


'''
ABCD_6:.
├───code
│       correlations.py
│       ...
│       reader.py
│
├───data
│   │   graph_metrics.csv
│   │   ...
│   │   screentime.csv
│   │
│   ├───general
│   │       ab_g_dyn.tsv
│   │       ...
│   │       ab_g_stc.tsv
│   │
│   ├───gordon333
│   │       gordon333CommunityAffiliation.txt
│   │       gordon333CommunityNames.txt
│   │       gordon333NodeNames.txt
│   │
│   └───imaging
│       ├───organized
│       │       fc_adjusted.h5
│       │       ...
│       │       vol_info.h5
│       │
│       └───source
│               gp_aseg_corr.mat
│               ...
│               vol_info.mat
│
├───figures
│   │   figure_1.png
│   │   ...
│   │   figure_N.png
│   │
│   └───confusion_matrices
│       └───85
│           ├───isolation_forest
│           │       g_games_confusion_matrix.png
│           │       ...
│           │       personalization_confusion_matrix.png
│           │
│           ├───isolation_forest
│           │
│           └───one_class_svm
│
└───results
    │   ml_balanced_accuracy.csv
    │
    └───corrs
            graph.csv
            ...
            graph_log_fdr.csv
'''