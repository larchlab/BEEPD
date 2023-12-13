""" 
Utility class for calculating and writing metrics for assessing models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, matthews_corrcoef, cohen_kappa_score
from typing import Dict, List, Any, Optional

# Map metric names to fns for metrics and whether they satisfy:
# 1) Require thresholded predictions
# 2) Require binarization of outcomes in multiclass case
METRICS = {
    "f1": {
        "function": f1_score,
        "thresholded": True,
        "multiclass": "binarized",
    },
    "recall": {
        "function": recall_score,
        "thresholded": True,
        "multiclass": "binarized",
    },
    "precision": {
        "function": precision_score,
        "thresholded": True,
        "multiclass": "binarized",
    },
    "auroc": {
        "function": roc_auc_score,
        "thresholded": False,
        "multiclass": "no",
    },
    "auprc": {
        "function": average_precision_score,
        "thresholded": False,
        "multiclass": "binarized",
    },
    "mcc": {
        "function": matthews_corrcoef,
        "thresholded": True,
        "multiclass": "true"
    }
}


def calc_all_metrics(df: pd.DataFrame, truth_col, prob_col, pred_col, class_list=[]):
    """ Figure out what metrics to calculate, and do it
    
    """
    y_true = df[truth_col]
    y_prob: pd.Series = df[prob_col]
    y_prob = np.stack(y_prob.values)[:,1]
    y_pred = df[pred_col]
    class_list = class_list or sorted(set(y_true))
    is_binary = len(class_list) == 2
    
    metric_values_dict = {}

    for metric_name, metric_info in METRICS.items():
        metric_fn = metric_info["function"]
        if metric_info["thresholded"]:
            y_to_use = y_pred
        else:
            y_to_use = y_prob
        
        multiclass = metric_info["multiclass"]

        if multiclass == "binarized":
            if is_binary:
                for i in [0,1]:
                    metric_values_dict[f"{metric_name}_wrt_{class_list[i]}"] = metric_fn(y_true, y_to_use, pos_label=class_list[i])
            for average_technique in ["micro", "macro", "weighted"]:
                metric_values_dict[f"{metric_name}_{average_technique}"] = metric_fn(y_true, y_to_use, average=average_technique)
        elif multiclass == "no":
            # currently this is just AUROC, so this is a bit hard-coded :/
            metric_values_dict[metric_name] = metric_fn(y_true, y_to_use, average="macro", multi_class="ovo")
        elif multiclass == "true":
            metric_values_dict[metric_name] = metric_fn(y_true, y_to_use)
        else:
            raise ValueError(f"Metric '{metric_name}' has bad 'multiclass' value {multiclass}.")

    return metric_values_dict

def calc_cross_group_metrics(df: pd.DataFrame, truth_col, prob_col, pred_col, demographic_col, demographic_group, pos_class=1):
    """ Calculate cross-group fairness metrics. This needs to be a different function because the basic calc_all_metrics doesn't do demographic filtering
    Right now, this function only calculates cross-group rankings

    """
    curr_group_pos = ((df[demographic_col] == demographic_group) & (df[truth_col] == pos_class))
    other_group_neg = ((df[demographic_col] != demographic_group) & (df[truth_col] != pos_class))
    sub_df = df[curr_group_pos | other_group_neg]
    return roc_auc_score(sub_df[truth_col], np.stack(sub_df[prob_col].values)[:,1])