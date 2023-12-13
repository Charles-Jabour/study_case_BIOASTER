# -*- coding: utf-8 -*-
"""
@author: charles
"""

from data_utils import load_data, show_stats
from training import nested_cross_val
from eval_utils import show_test_results
from feature_selection_utils import do_fs
from options import args
import os


exp_type = args.exp_type

df, features, targets = load_data()

if args.exp_type == 'stats':
    stats_save_dir = 'stats_results/'
    os.makedirs(stats_save_dir, exist_ok=True)
    show_stats(df, stats_save_dir)

elif args.exp_type == 'ncv':
    test_save_dir = args.exp_name
    os.makedirs(test_save_dir)
    test_ratio = args.test_ratio
    param_grid = {'C': [2**i for i in range(args.min_C_power,
                                            args.max_C_power)]}
    inner_n = args.inner_n
    outer_n = args.outer_n
    num_features_imp = args.num_features_imp
    recap = nested_cross_val(features, targets, test_ratio, param_grid,
                             inner_n, outer_n, test_save_dir)
    show_test_results(recap, num_features_imp, test_save_dir)
elif args.exp_type == 'fs':
    fs_save_dir = args.exp_name
    os.makedirs(fs_save_dir)
    test_ratio = args.test_ratio
    regu = args.regu
    step = args.step
    do_fs(features, targets, regu, test_ratio, step, fs_save_dir)
