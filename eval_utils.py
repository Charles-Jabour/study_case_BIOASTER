# -*- coding: utf-8 -*-
"""
@author: charles
"""

from table_utils import plot_table
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.inspection import permutation_importance
import numpy as np

true_labels = ['A1', 'A2', 'A3', 'A4']
n_classes = 4


def eval_model(recap_inner):
    """
    This function evaluates the model selected in the inner loop on the
    held-out test data. Computed metrics are: confusion matrix, precision,
    recall and AUC. These metrics are saved in the input dictionary.

    Parameters
    ----------
    recap_inner : dict
        Stores inner loop results.

    Returns
    -------
    recap_inner : dict
        Same as input, with test results.

    """
    x_test, y_test = recap_inner['test_data']
    model = recap_inner['model']
    preds = model.predict(x_test)
    probas = model.predict_proba(x_test)
    recap_inner['test_preds'] = preds
    recap_inner['test_probas'] = probas
    recap_inner['test_accuracy'] = model.score(x_test, y_test)
    recap_inner['test_confusion_matrix'] = metrics.confusion_matrix(y_test,
                                                                    preds)
    recap_inner['test_precision'] = metrics.precision_score(y_test, preds,
                                                            average='macro')
    recap_inner['test_recall'] = metrics.recall_score(y_test, preds,
                                                      average='macro')
    recap_inner['test_auc'] = metrics.roc_auc_score(y_test, probas,
                                                    multi_class='ovr')
    return recap_inner


def plot_avg_roc_curve(recap, save_dir=''):
    """
    This function displays the average roc curves for each labels.
    Based on the probability predictions of the best model from each outer
    loop, FPRs & TPRs are computed for each labels. TPR are interporlated then
    averaged to make sure they hold the same amount of data. Mean AUCs for each
    class are also displayed and calculated from the average TPRs & FPRs

    Parameters
    ----------
    recap : dict
        Output of the nested cross validation, stores its results.
    save_dir : str, optional
        Path to save the plot. Without a path, the plot will not be saved.
        The default is ''.

    Returns
    -------
    None.

    """
    probas = [recap[key]['test_probas'] for key in recap.keys()]
    test_tgts = [recap[key]['test_data'][1] for key in recap.keys()]
    tpr_per_class = {}
    auc_per_class = {}
    fpr_grid = np.linspace(0.0, 1.0, 1000)

    colors = ['blue', 'red', 'green', 'black']
    plt.figure()

    for i in range(n_classes):
        fpr = {}
        tpr = {}
        for j in range(len(probas)):
            curr_tgts = test_tgts[j]
            curr_proba = probas[j]
            fpr[j], tpr[j], _ = metrics.roc_curve(curr_tgts,
                                                  curr_proba[:, i],
                                                  pos_label=i)

        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(fpr_grid)

        for key in fpr.keys():
            mean_tpr += np.interp(fpr_grid, fpr[key], tpr[key])
        mean_tpr[0] = 0.0

        # Average it and compute AUC
        mean_tpr /= len(fpr.keys())
        tpr_per_class[i] = mean_tpr
        auc_per_class[i] = metrics.auc(fpr_grid, mean_tpr)

        label = "Class %s, avg AUC = %.2f" % (true_labels[i], auc_per_class[i])
        plt.plot(fpr_grid, mean_tpr, color=colors[i], label=label, lw=2)

    plt.plot([0, 1], [0, 1], '--', color='gray', label='Luck')
    plt.xlabel('False positive rate', fontsize=16, fontweight='bold')
    plt.ylabel('True positive rate', fontsize=16, fontweight='bold')
    plt.xticks([0, 0.5, 1], labels=[0, 0.5, 1], fontsize=14)
    plt.yticks([0, 0.5, 1], labels=[0, 0.5, 1], fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if len(save_dir) > 0:
        plt.savefig(save_dir+'/average_roc_curve.png', dpi=300)
        plt.close()


def show_avg_cf_matrix(recap, save_dir=''):
    """
    This function displays the average confusion matrix.
    Based on the predictions of the best models from each outer loop, a
    confusion matrix is calculated. The average confusion matrix is the
    element-wise average of these matrices.

    Parameters
    ----------
    recap : dict
        Output of the nested cross validation, stores its results.
    save_dir : str, optional
        Path to save the matrix. Without a path, the matrix will not be saved.
        The default is ''.

    Returns
    -------
    None.

    """
    cf_matrices = [recap[key]['test_confusion_matrix'] for key in recap.keys()]
    avg_cf_matrix = np.mean(cf_matrices, axis=0)
    std_cf_matrix = np.std(cf_matrices, axis=0)
    figcf, ax = plt.subplots()

    ax.imshow(np.ones((4, 4)), cmap=cm.Blues)

    for (i, j), z in np.ndenumerate(avg_cf_matrix):
        ax.text(j, i, '%d ± %d' % (round(z), round(std_cf_matrix[i, j])),
                ha='center', va='center', fontsize=14)
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(avg_cf_matrix.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(avg_cf_matrix.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="k", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks(np.arange(len(true_labels)), labels=true_labels)
    ax.set_yticks(np.arange(len(true_labels)), labels=true_labels)
    ax.tick_params(labelsize=16)
    ax.set_xlabel('Predicted label', fontdict=dict(weight='bold'),
                  fontsize=14)
    ax.set_ylabel('True label', fontdict=dict(weight='bold'),
                  fontsize=14)
    figcf.tight_layout()
    if len(save_dir) > 0:
        plt.savefig(save_dir+'/average_cf_matrix.png', dpi=300)
        plt.close()


def show_metrics(recap, save_dir=''):
    """
    This function displays a table of metrics computed from the predictions
    of the best models selected in the inner loops. Metrics are:
    validation accuracy, test accuracy, test precision, test recall & test AUC.
    Their averages through the outer loop are also computed.

    Parameters
    ----------
    recap : dict
        Output of the nested cross validation, stores its results.
    save_dir : str, optional
        Path to save results. Without a path, results will not be saved.
        The default is ''.

    Returns
    -------
    None.

    """
    val_df = {'Model': [],
              'Val accuracy': [],
              'Accuracy': [],
              'Precision': [],
              'Recall': [],
              'AUC': []}
    for model_name in recap.keys():
        val_df['Model'].append(model_name)
        val_df['Val accuracy'].append(round(recap[model_name]['val_acc'], 2))
        val_df['Accuracy'].append(round(recap[model_name]['test_accuracy'], 2))
        val_df['Precision'].append(round(recap[model_name]['test_precision'],
                                         2))
        val_df['Recall'].append(round(recap[model_name]['test_recall'], 2))
        val_df['AUC'].append(round(recap[model_name]['test_auc'], 2))
    val_df['Model'].append('Mean')
    val_df['Val accuracy'].append(round(np.mean(val_df['Val accuracy']), 2))
    val_df['Accuracy'].append(round(np.mean(val_df['Accuracy']), 2))
    val_df['Precision'].append(round(np.mean(val_df['Precision']), 2))
    val_df['Recall'].append(round(np.mean(val_df['Recall']), 2))
    val_df['AUC'].append(round(np.mean(val_df['AUC']), 2))
    val_df = pd.DataFrame(val_df)
    data = np.vstack((val_df.columns.values, val_df.values.astype(str)))
    plot_table(data, fontsize=8, edge_color='k')
    if len(save_dir) > 0:
        plt.savefig(save_dir+'/table_results.png',  dpi=300)
        plt.close()


def plot_feature_importance(recap, num_features=10, save_dir=''):
    """
    This function displays the feature importance. Feature
    importance is the difference in accuracy provoked by shuffling this
    feature.

    Parameters
    ----------
    recap : dict
        Output of the nested cross validation, stores its results.
    num_features : int, optional
        The number of most important features to display. The default is 10.
    save_dir : str, optional
        Path to save the plots. Without a path, the plots will not be saved.
        The default is ''.

    Returns
    -------
    None.

    """
    for model_name in recap.keys():
        x_test, y_test = recap[model_name]['test_data']
        model = recap[model_name]['model']
        perm_importance = permutation_importance(model, x_test, y_test,
                                                 n_repeats=1,
                                                 scoring='accuracy', n_jobs=-1)
        sorted_idx = perm_importance.importances_mean.argsort()[-num_features:]
        feature_names = np.arange(1, x_test.shape[1]+1, dtype=int)
        plt.figure()
        plt.barh(range(num_features),
                 perm_importance.importances_mean[sorted_idx],
                 color='k')
        plt.yticks(range(num_features), feature_names[sorted_idx])
        plt.xlabel("Permutation Importance", fontsize=16, fontweight='bold')
        plt.ylabel('%d most important features' % num_features, fontsize=16,
                   fontweight='bold')
        if len(save_dir) > 0:
            plt.savefig(save_dir+'/'+model_name+'feature importance.png',
                        dpi=300)
            plt.close()


def show_test_results(recap, num_features=10, save_dir=''):
    """
    This function calls other functions to display multiple test results :
    average ROC curves, average confusion matrix, metrics table and feature
    importance. Check corresponding functions for more details.

    Parameters
    ----------
    recap : dict
        Output of the nested cross validation, stores its results.
    num_features : int, optional
        The number of most important features to display. The default is 10.
    save_dir : str, optional
        Path to save results. Without a path, results will not be save.
        The default is ''.

    Returns
    -------
    None.

    """
    plot_avg_roc_curve(recap, save_dir)
    show_avg_cf_matrix(recap, save_dir)
    show_metrics(recap, save_dir)
    plot_feature_importance(recap, num_features, save_dir)
