# -*- coding: utf-8 -*-
"""
@author: charles
"""

import argparse

parser = argparse.ArgumentParser()

# %% General inputs
parser.add_argument("--exp_type", type=str, choices=['stats', 'ncv', 'fs'],
                    default='ncv',
                    help='Set to stats for statiscal analysis, ncv for' +
                    ' nested cross validation, fs for feature selection.')
parser.add_argument("--exp_name", type=str, default='exp1',
                    help='Give it a name to save prediction results.')

parser.add_argument("--test_ratio", type=float, default=.2,
                    choices=range(0, 1),
                    help='Ratio of test data, must be between 0 and 1.')

# %% Nested cross validation inputs
parser.add_argument("--min_C_power", type=int, default=-3,
                    help='Sets lower SVM regularization bound, included, to ' +
                    '2**min_C_power.')
parser.add_argument("--max_C_power", type=int, default=4,
                    help='Sets lower SVM regularization bound, excluded, to ' +
                    '2**max_C_power.')
parser.add_argument('--inner_n', type=int, default=5,
                    help='Number of folds for the inner nested CV loop.')
parser.add_argument('--outer_n', type=int, default=5,
                    help='Number of folds for the outer nested CV loop.')
parser.add_argument('--num_features_imp', type=int, default=10,
                    help='Number of features of highest importance' +
                    ' to display.')

# %% Feature selection inputs
parser.add_argument("--regu", type=float, default=4,
                    help='Sets SVM regularization parameter.')
parser.add_argument("--step", type=int, default=10,
                    help='Sets the number of features to add' +
                    ' at each iteration.')


args = parser.parse_args()
