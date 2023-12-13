# -*- coding: utf-8 -*-
"""
@author: charles
"""

from scipy.stats import shapiro, kruskal, mannwhitneyu, f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


true_labels = ['A1', 'A2', 'A3', 'A4']


def load_data():
    """
    This function loads the xlsx file, remove incomplete data and encode labels
    to numerical data.

    Returns
    -------
    df : DataFrame
        The dataset.
    features : numpy array
        The array of features.
    targets : numpy array
        The encoded labels.

    """
    df = pd.read_excel('Dataset.xlsx').dropna(axis=0)
    tgt_encoder = LabelEncoder()
    targets = tgt_encoder.fit_transform(df['Class'])
    features = df.drop('Class', axis=1).to_numpy()
    return df, features, targets


def plot_label_distribution(df, save_dir=''):
    """
    This function displays the distribution of labels in the dataset.

    Parameters
    ----------
    df : DataFrame
        The dataset.
    save_dir : str, optional
        Path to save the plot. Without a path, the plot will not be saved.
        The default is ''.

    Returns
    -------
    None.

    """
    labels, counts = np.unique(df['Class'], return_counts=True)
    plt.bar(labels, counts, align='center')

    plt.xticks(ticks=np.arange(len(labels)), labels=true_labels, fontsize=12)
    plt.xlabel('Label', fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=14, fontweight='bold')
    plt.title('Target distribution', fontsize=16)
    plt.tight_layout()
    if len(save_dir) > 0:
        plt.savefig(save_dir+'/label_distribution.png', dpi=300)
        plt.close()


def plot_feature_corr(df, save_dir=''):
    """
    This function shows the absolute Pearson correlation between features.

    Parameters
    ----------
    df : DataFrame
        The dataset.
    save_dir : str, optional
        Path to save the correlation image. Without a path, the correlation
        image will not be saved. The default is ''.

    Returns
    -------
    None.

    """
    features = df.drop('Class', axis=1)
    corr = features.corr().abs()
    plt.imshow(corr, cmap='gray')
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.set_label(label='Absolute Pearson correlation', size=14,
                   weight='bold')
    plt.tight_layout()
    if len(save_dir) > 0:
        plt.savefig(save_dir+'/features_corr.png', dpi=300)
        plt.close()


def plot_discr_features(df, save_dir=''):
    features = df.drop('Class', axis=1)
    df['Class'] = LabelEncoder().fit_transform(df['Class'])
    n_classes = len(df['Class'].unique())
    class_pairs = [true_labels[i] + ' & ' + true_labels[j]
                   for i in range(n_classes-1) for j in range(i+1, n_classes)]
    img = np.zeros((features.shape[1], len(class_pairs)), dtype=bool)
    for incr, feature_num in enumerate(features.columns):
        curr_feature = df[feature_num]
        split_features = [curr_feature[df['Class'] == _]
                          for _ in range(n_classes)]
        pnorms = [shapiro(_)[1] <= 0.05 for _ in split_features]
        if any(pnorms):
            _, ptest = kruskal(*split_features)
            is_norm = False
        else:
            _, ptest = f_oneway(*split_features)
            is_norm = True
        if ptest <= 0.05:
            if is_norm:
                paired_p = [
                    ttest_ind(split_features[i], split_features[j])[1]
                    for i in range(n_classes-1) for j in range(i+1, n_classes)
                    ]
            else:
                paired_p = [
                    mannwhitneyu(split_features[i], split_features[j])[1]
                    for i in range(n_classes-1) for j in range(i+1, n_classes)
                    ]
            corrected_p = multipletests(paired_p)[1]
            check = [_ <= 0.05 for _ in corrected_p]
            img[incr, :] = check
    plt.imshow(img, cmap='gray')
    plt.yticks([], [])
    plt.xticks(ticks=np.arange(len(class_pairs)), labels=class_pairs)
    plt.ylabel('Features', fontsize=16)
    plt.tight_layout()
    if len(save_dir) > 0:
        plt.savefig(save_dir+'pvals.png', dpi=300)
        plt.close()


def show_stats(df, save_dir=''):
    """
    This function displays all available statistical analyses on the dataset:
    label distribution, feature correlation & feature discriminative power
    between classes.

    Parameters
    ----------
    df : DataFrame
        The dataset.
    save_dir : str, optional
        Path to save the correlation image. Without a path, the correlation
        image will not be saved. The default is ''.

    Returns
    -------
    None.

    """
    plt.close('all')
    plot_label_distribution(df, save_dir)
    plot_feature_corr(df, save_dir)
    plot_discr_features(df, save_dir)
