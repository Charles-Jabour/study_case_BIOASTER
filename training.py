# -*- coding: utf-8 -*-
"""
@author: charles
"""

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import eval_utils as eu
import pickle
from tqdm import tqdm


def nested_cross_val(features, targets, test_ratio, param_grid, inner_n,
                     outer_n, save_dir=''):
    """
    This function performs nested cross-validation on the whole dataset.
    At each outer step, classification model is selected so as to
    maximize the validation accuracy.

    Parameters
    ----------
    features : numpy array of size (num_samples, num_features)
        Array of features.
    targets : nupy array of size(num_samples,)
        Array of labels.
    test_ratio : float
        The ratio of data to hold out of the inner loop. Must be between 0 & 1.
    param_grid : dict
        Same param_grid as in GridSearchCV.
    inner_n : int
        Number of folds for the inner cross-validation.
    outer_n : TYPE
        Number of folds for the outer cross-validation.
    save_dir : str, optional
        Path to save the nested cross-validation results. Without a path,
        the results will not be saved. The default is ''

    Returns
    -------
    recap : dict
        Stores the nested cross-validation results. For i in [1, outer_n],
        it is organized as:
        recap['Model i']
            |_________['train_data'] --> tuple of train features & labels
            |_________['test_data'] --> tuple of test features & labels
            |_________['model'] --> the selected model
            |_________['best_params'] --> selected hyperparameters
            |_________['val_acc'] --> validation accuracy

    """
    outer_cv = StratifiedKFold(n_splits=outer_n)
    recap = {}
    for outer_incr, (train_idx, test_idx) in enumerate(
            tqdm(outer_cv.split(features, targets)), start=1):
        x_train, y_train = features[train_idx, :], targets[train_idx]
        x_test, y_test = features[test_idx, :], targets[test_idx]

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        model = SVC(probability=True, random_state=42)

        clf = GridSearchCV(model, param_grid, scoring='accuracy',
                           refit='accuracy', n_jobs=-1, cv=inner_n)
        clf.fit(x_train, y_train)
        recap_inner = {'train_data': (x_train, y_train),
                       'test_data': (x_test, y_test)}
        recap_inner['model'] = clf.best_estimator_
        recap_inner['val_acc'] = clf.best_score_
        recap_inner['best_params'] = clf.best_params_
        recap_inner = eu.eval_model(recap_inner)
        recap['Model '+str(outer_incr)] = recap_inner
    if len(save_dir) > 0:
        with open(save_dir+'/recap_results.pkl', 'wb') as f:
            pickle.dump(recap, f)
    return recap
