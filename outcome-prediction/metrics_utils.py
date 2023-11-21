""" 
Utility class for calculating and writing metrics for assessing models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score
from typing import List, Union, Any
# Want to write code to create DF containing HADM_ID, true label, predicted label/probabilities array, then write functions
# over this DF, rather than doing infinite ad hoc things. Maybe make a PredictionsCalculator class that takes in
# this info, as well as list of classes, threshold, etc.
# for each fn, if false negs, false pos, true negs, true pos, etc. not in the DF, calculate and add that column so
# we're nto reproducing work.


class PredictionCalculator:
    """ Utility class for calculating and writing metrics for assessing models.
    """
    def __init__(self, results_df: pd.DataFrame, true_label_col: str, pred_probs_col: str, demographics_cols: List[str] = None, outcome_classes: List[int] = None, pos_label: Any = 1, random_seed: int=998) -> None:
        """ Populate the fields used to compute metrics
        
        """
        self.results_df: pd.DataFrame = results_df
        self.true_label_col: str = true_label_col
        self.pred_probs_col: str = pred_probs_col
        self.demographics_cols: List[str] = demographics_cols
        if outcome_classes is not None:
            self.outcome_classes = outcome_classes
        else:
            self.outcome_classes = list(set(results_df[self.true_label_col]).union(set(results_df[self.pred_probs_col])))
        self.pos_label = pos_label
        # DF to calculate metrics over. When bootstrapping, this takes on the value of the sample mtx
        self.curr_df = results_df
        self.random_state = np.random.RandomState(seed=random_seed)
        self.decision_threshold = None
        # TODO: use demographics cols to populate this with the columns  
        self.metrics_df: pd.DataFrame = None

    def calc_metric_conf_ints(self, num_samples: int, conf_int: float=0.95):
        """ Calculate confidence intervals for each metric by bootstrapping 
        
        """
        for demographics_col in self.demographics_cols:
            
            for demographic_val in set(self.results_df[demographics_col]):
                sub_df = self.results_df[demographics_col == demographic_val]
                

        # Reset to the initial curr_df at the end.
        self.curr_df = self.results_df


    def calc_metrics(self):
        """ Calculate metrics over the result_df, store them in self.metrics_df
        
        """
        # calculate over self.curr_df

    def create_bootstrap_sample(self, df: pd.DataFrame):
        """ create an N x k array of indices, where N is the number of samples, k is the number of bootstrap
        datasets to create. The resulting DF is stored in the self.curr_df field

        Arguments:
            df (pd.DataFrame): DataFrame to get bootstrap sample over 

        """
        if not all:
            # exp
            groups = self.results_df.groupby(by=self.demographics_cols+[self.true_label_col])
        else:
            groups = self.results_df.groupby(by=self.true_label_col)
        self.curr_df = groups.apply(lambda group: group.sample(frac=1.0, replace=True, random_state=self.random_state)).reset_index(drop=True)

    def bin_column(self, col_to_bin: str, binning_fn: callable):
        """ Bin a column of the results DF. Performs an in-place binning that modifies self.results_df

        Arguments:
            col_to_bin (str): the column to bin using binning_fn
            binning_fn (str): the function used to bin col_to_bin
        """
        self.results_df[col_to_bin].apply(binning_fn)

    def write_dataframe(self, outfile: str):
        """ Write the core results df to a CSV file

        Arguments:
            outfile (str): filepath to write the results to
        
        """

    
    def write_metrics_file(self, outfile: str):
        """ Write the stored metrics to a CSV file

        Arguments:
            outfile (str): filepath to write the metrics to
        
        """

    def print_metrics(self):
        """ Print all the metrics that have been calculated
        
        """

    def calc_decision_threshold(self, metric: str = None):
        """ Calculate an "optimal" threshold for metrics requiring a threshold,
        and store it in the self.decision_threshold field

        Arguments:
            metric (str): either f1 or mcc, use best value to compute threshold

        """
        metric_name_to_metric_fn = {
            "f1": self.calc_f1,
            "mcc": self.calc_mcc,
        }
        # Use f1 as default metric
        metric_fn = metric_name_to_metric_fn.get(metric) or self.calc_f1
        opt_threshold = 0
        # Should we use mcc instead?
        opt_metric_val = 0
        self.decision_threshold = opt_threshold

    def calc_confusion_matrix(self):
        """ 
        
        """

    def calc_precision(self):
        """
        
        """

    def calc_recall(self):
        """
        
        """
    
    def calc_f1(self):
        """ Calc f1 over self.curr_df. I don't like how I'm doing this rn, I want something more generalizable
        so I don't need to call different functions with only slightly variable arguments. The main thing is that
        I want to have a good way of separating by the non/binary cases, and non/thresholded cases
        
        """
        return {
            f"f1_wrt_{self.pos_label}": f1_score(self.curr_df[self.true_label_col], self.curr_df[self.pred_probs_col], pos_label=self.pos_label),
            "f1_micro": f1_score(self.curr_df[self.true_label_col], self.curr_df[self.pred_probs_col], average="micro"),
            "f1_macro": f1_score(self.curr_df[self.true_label_col], self.curr_df[self.pred_probs_col], average="micro"),
        }

    def calc_auc(self):
        """
        
        """

    def calc_mcc(self):
        """ Calculate the Matthews Correlation Coefficient
        
        """

HYPOTHETICAL_METRIC_FN_DATA_STRUCTURE = {
    "f1_wrt_{}": {
        "binary": True,
        "thresholded": True,
        "requires": "pos_label", # figure out a way to put the actual self.pos_label here
        "fn": f1_score, # unholy mixture of object-oriented and functional design
        "kwargs": {"pos_label": "self.pos_label"}
    }
}
