""" 
Utility class for calculating and writing metrics for assessing models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, matthews_corrcoef
from typing import Dict, List, Any, Optional
# Want to write code to create DF containing HADM_ID, true label, predicted label/probabilities array, then write functions
# over this DF, rather than doing infinite ad hoc things. Maybe make a PredictionsCalculator class that takes in
# this info, as well as list of classes, threshold, etc.
# for each fn, if false negs, false pos, true negs, true pos, etc. not in the DF, calculate and add that column so
# we're nto reproducing work.


class PredictionCalculator:
    """ Utility class for calculating and writing metrics for assessing models.
    """
    def __init__(self, results_df: pd.DataFrame, true_label_col: str, pred_probs_col: str, pred_label_col: Optional[str] = None, decision_threshold: float | str = None, demographics_cols: List[str] = [], outcome_classes: List[int] = None, pos_label: Any = 1, random_seed: int=998) -> None:
        """ Populate the fields used to compute metrics
        
        """
        self.results_df: pd.DataFrame = results_df
        self.true_label_col: str = true_label_col
        self.pred_probs_col: str = pred_probs_col
        self.pred_label_col: Optional[str] = pred_label_col
        self.demographics_cols: List[str] = demographics_cols
        self.decision_threshold = decision_threshold
        if outcome_classes is not None:
            self.outcome_classes = outcome_classes
        else:
            self.outcome_classes = list(set(results_df[self.true_label_col]).union(set(results_df[self.pred_probs_col])))
        self.pos_label = pos_label
        if self.pos_label is not None or self.decision_threshold == "argmax":
            self.calc_decision_threshold()
        # DF to calculate metrics over. When bootstrapping, this takes on the value of the sample mtx
        self.curr_df = results_df
        self.random_state = np.random.RandomState(seed=random_seed)
        # This is where metrics we calculate get stored
        self.metrics_df = pd.DataFrame()

    def calc_metric_conf_ints(self, num_samples: int, conf_level: float=0.95):
        """ Calculate confidence intervals for each metric by bootstrapping 
        
        Arguments:
            num_samples (int): Number of bootstrap samples to take
            conf_level (float): Confidence level for computing statistical significance
        """
        lower_percentile = (1 - conf_level) / 2
        upper_percentile = 1 - lower_percentile
        # Compute "Actual" metrics, then 
        for demographics_col in self.demographics_cols:
            for demographic_val in set(self.results_df[demographics_col]):
                sub_df = self.results_df[self.results_df[demographics_col] == self.results_df[demographic_val]]
                self.curr_df = sub_df
                self.calc_metrics(sub_df)
                results_dict = {}
                for i in range(num_samples):
                    self.curr_df = self.create_bootstrap_sample(sub_df, demographics_col, demographic_val)
                    if not results_dict:
                        results_dict = self.calc_metrics()

        
        # Then take metrics over everything...

        # Reset to the initial curr_df at the end.
        self.curr_df = self.results_df


    def calc_metrics(self, df: pd.DataFrame):
        """ Calculate metrics over self.curr_df, store them in self.metrics_df
        
        Arguments:
            name (str): Name used to store results in the metrics_df
        """
        # calculate over self.curr_df


    def create_bootstrap_sample(self, df: pd.DataFrame):
        """ create an N x k array of indices, where N is the number of samples, k is the number of bootstrap
        datasets to create. The resulting DF is stored in the self.curr_df field

        Arguments:
            df (pd.DataFrame): DataFrame to get bootstrap sample over 

        """
        groups = df.groupby(by=self.true_label_col)
        return groups.apply(lambda group: group.sample(frac=1.0, replace=True, random_state=self.random_state)).reset_index(drop=True)

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
            "f1": f1_score,
            "mcc": matthews_corrcoef,
        }
        # Use f1 as default metric
        metric_fn = metric_name_to_metric_fn.get(metric, self.calc_f1)
        opt_threshold = 0
        opt_metric_val = 0
        for thresh in range(0.05, 1.0, 0.05):
            metric_fn()
        self.decision_threshold = opt_threshold

    def _check_and_set_threshold(self):
        """ Check whether a threshold column or value exists; set them if they don't 
        
        """
        if not self.pred_label_col:
            # Get the col name
            pred_label_col = "THRESHOLDED_PREDICTIONS"
            while pred_label_col in self.results_df.columns:
                pred_label_col = "_" + pred_label_col
            # Let the user just argmax the preds if none provided
            if self.decision_threshold == "argmax":
                self.results_df[pred_label_col] = np.argmax(self.results_df[self.pred_label_col])
            else:
                if not self.decision_threshold:
                    self.calc_decision_threshold()
                self.results_df[pred_label_col] = self.results_df[self.pred_probs_col].apply(lambda probs: probs[self.pos_label]) > self.decision_threshold
