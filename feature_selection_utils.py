# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:27:54 2023

@author: charles
"""

from ReliefF import ReliefF
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle


def relieff_fs(features, targets, regu, test_ratio, step, save_dir=''):
    """
    This function computes classification performance based on subdata selected
    by feature ranking given by ReliefF algorithm.

    Parameters
    ----------
    features : numpy array of size (num_elements, num_features)
        Array of features.
    targets : numpy array of size (num_elements,)
        Array of labels.
    params : dict
        Stores model parameters.
    test_ratio : float
        Ratio of test data.
    step : int
        Number of features to add at each step.

    Returns
    -------
    fs_recap : dict
        Stores results of feature selection, as:
        fs_recap['n_features'] --> list of number of features.
        fs_recap['features'] --> corresponding features.
        fs_recap['scores'] --> test scores.

    """
    n_features_to_keep = [i for i in range(10, features.shape[1], step)]
    n_features_to_keep.append(features.shape[1])
    scores = []
    selected_features = []
    fs = ReliefF(n_neighbors=10, n_features_to_keep=features.shape[1])
    fs.fit(features, targets)
    rankings = fs.top_features
    x_train, x_test, y_train, y_test = train_test_split(features, targets,
                                                        test_size=test_ratio,
                                                        stratify=targets,
                                                        random_state=42
                                                        )
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    for num in n_features_to_keep:
        x_train_fs = x_train[:, rankings[:num]]
        x_test_fs = x_test[:, rankings[:num]]
        model = SVC(C=regu, probability=True, random_state=42)
        model.fit(x_train_fs, y_train)
        scores.append(model.score(x_test_fs, y_test))
        selected_features.append(rankings[:num])
    fs_recap = {'n_features': n_features_to_keep,
                'features': selected_features,
                'scores': scores}
    if len(save_dir) > 0:
        with open(save_dir+'/fs_results.pkl', 'wb') as f:
            pickle.dump(fs_recap, f)
    return fs_recap


def plot_fs_scores(fs_recap, save_dir=''):
    baseline_score = fs_recap['scores'][-1]
    plt.figure()
    plt.scatter(fs_recap['n_features'][:-1], fs_recap['scores'][:-1],
                marker='o', s=32, c='k')
    plt.plot([fs_recap['n_features'][0], fs_recap['n_features'][-1]],
             [baseline_score, baseline_score], '--', lw=2,
             label='Baseline accuracy')
    plt.xlabel('Number of features', fontsize=16, fontweight='bold')
    plt.ylabel('Test accuracy', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.legend()
    if len(save_dir) > 0:
        plt.savefig(save_dir+'/fs_results.png',  dpi=300)
        plt.close()


def do_fs(features, targets, regu, test_ratio, step, save_dir=''):
    fs_recap = relieff_fs(features, targets, regu, test_ratio, step, save_dir)
    plot_fs_scores(fs_recap, save_dir)
