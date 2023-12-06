""" 
Utility class for calculating and writing metrics for assessing models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef
from typing import Dict, List, Any, Optional

from metrics_utils import calc_all_metrics, calc_cross_group_metrics
# Want to write code to create DF containing HADM_ID, true label, predicted label/probabilities array, then write functions
# over this DF, rather than doing infinite ad hoc things. Maybe make a PredictionsCalculator class that takes in
# this info, as well as list of classes, threshold, etc.
# for each fn, if false negs, false pos, true negs, true pos, etc. not in the DF, calculate and add that column so
# we're nto reproducing work.


class PredictionCalculator:
    """ Utility class for calculating and writing metrics for assessing models.
    """
    def __init__(self, results_df: pd.DataFrame, true_label_col: str, pred_probs_col: str, demographics_df: pd.DataFrame = None, join_on: str = None, pred_label_col: Optional[str] = None, decision_threshold: float | str = None, pos_label_ind: int = 1, demographics_cols: List[str] = [], outcome_classes: List[int] = None, random_seed: int=998) -> None:
        """ Populate the fields used to compute metrics
        
        """
        self.results_df: pd.DataFrame = results_df
        if demographics_df is not None:
            if join_on is None:
                raise ValueError("Argument 'join_on' cannot be None if demographics_df is defined")
            self.results_df = pd.merge(self.results_df, demographics_df, how='left', on=join_on)
        self.true_label_col: str = true_label_col
        self.pred_probs_col: str = pred_probs_col
        if self.results_df[pred_probs_col].dtype != 'o':
            self.results_df[pred_probs_col] = self.results_df[pred_probs_col].map(lambda prob: [1-prob, prob])
        self.pred_label_col: Optional[str] = pred_label_col
        self.demographics_cols: List[str] = demographics_cols
        self.decision_threshold = decision_threshold
        self.pos_label_ind = pos_label_ind
        self.outcome_classes = outcome_classes or sorted(results_df[self.true_label_col].unique())
        
        # For multiclass take argmax of probabilities as prediction
        if len(self.outcome_classes) > 2:
            self.decision_threshold = 'argmax'
        
        # Get actual predictions for probabilities if none given
        if self.pred_label_col is None:
            self._threshold_predictions()
        
        self.random_state = np.random.RandomState(seed=random_seed)
        
        # This is where metrics we calculate get stored, indexed by subgroup calculated over (or "all") and metric name
        self.metrics_df = pd.DataFrame( 
            columns=["true", "boot_mean", "boot_std", "lower_percentile", "upper_percentile", "boot_conf_level", "fold_n", "threshold_prob"]
        )
        

    def calc_metric_conf_ints(self, num_samples: int, conf_level: float=0.95):
        """ Calculate confidence intervals for each metric by bootstrapping 
        
        Arguments:
            num_samples (int): Number of bootstrap samples to take
            conf_level (float): Confidence level for computing statistical significance
        """
        lower_percentile = ((1 - conf_level) / 2) * 100
        upper_percentile = 100 - lower_percentile
        
        # Compute metrics over everything 
        results_dict = self.calc_metrics(self.results_df)
        for metric_name, metric_value in results_dict.items():
            self.metrics_df.loc[f'all::{metric_name}', 'true'] = metric_value
        
        for demographics_col in self.demographics_cols:
            for demographic_val in self.results_df[demographics_col].unique():
                demographic_df = self.results_df[self.results_df[demographics_col] == demographic_val]
                results_dict = self.calc_metrics(demographic_df)
                for metric_name, metric_value in results_dict.items():
                    self.metrics_df.loc[f'{demographics_col}={demographic_val}::{metric_name}', 'true'] = metric_value
                if num_samples > 0:
                    results_dict = {}
                    for i in range(num_samples):
                        if not i % 100:
                            print(f"fold {i} for demographic {demographics_col}={demographic_val}")
                        bootstrap_sample = self.create_bootstrap_sample(demographic_df)
                        if not results_dict:
                            results_dict = {metric_name: [metric_val] for metric_name, metric_val in self.calc_metrics(bootstrap_sample).items()}
                            whole_df_bootstrap = self.create_bootstrap_sample(self.results_df)
                            results_dict["xgroup_auroc_wrt_1"] = [calc_cross_group_metrics(whole_df_bootstrap, self.true_label_col, self.pred_probs_col, self.pred_label_col, demographics_col, demographic_val)]
                        else:
                            for metric_name, metric_val in self.calc_metrics(bootstrap_sample).items():
                                results_dict[metric_name].append(metric_val)
                                results_dict["xgroup_auroc_wrt_1"].append(calc_cross_group_metrics(whole_df_bootstrap, self.true_label_col, self.pred_probs_col, self.pred_label_col, demographics_col, demographic_val))
                    # compute statistics for confidence intervals
                    for metric_name, metric_values in results_dict.items():
                        d = {
                            'boot_mean': np.mean(metric_values),
                            'boot_std': np.std(metric_values, ddof=1), # dont assume normality, so use ddof=1
                            'lower_percentile': np.percentile(metric_values, lower_percentile),
                            'upper_percentile': np.percentile(metric_values, upper_percentile),
                            'boot_conf_level': conf_level,
                            'fold_n': demographic_df.shape[0],
                            'threshold_prob': self.decision_threshold
                        }
                        # loc syntax to set values of specific columns in row of specific index is df.loc[index, column names...] = column values...
                        self.metrics_df.loc[f'{demographics_col}={demographic_val}::{metric_name}', d.keys()] = list(d.values())

        
        # Then take metrics over everything...

        # Reset to the initial curr_df at the end.
        self.curr_df = self.results_df


    def calc_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """ Calculate metrics over self.df, return dict mapping metric names to their values
        
        Arguments:
            df (pd.DataFrame): Dataframe to calculate metrics over

        Returns:
            Dict[str, float]: Dict mapping metric names to their values
        """
        return calc_all_metrics(df, self.true_label_col, self.pred_probs_col, self.pred_label_col, class_list=self.outcome_classes)


    def create_bootstrap_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Create a new df with the same shape as df by sampling with replacement, stratified by the true label col.

        Arguments:
            df (pd.DataFrame): DataFrame to get bootstrap sample over 

        Returns:
            pd.DataFrame: dataframe containing the sample
        """
        # stratifying by true_label_col gives us less variance on the metrics calculated using this sample
        groups = df.groupby(by=self.true_label_col)
        return groups.apply(lambda group: group.sample(frac=1.0, replace=True, random_state=self.random_state)).reset_index(drop=True)

    def bin_column(self, col_to_bin: str, binning_fn: callable):
        """ Bin a column of the results DF. Performs an in-place binning that modifies self.results_df

        Arguments:
            col_to_bin (str): the column to bin using binning_fn
            binning_fn (str): the function used to bin col_to_bin
        """
        self.results_df[col_to_bin] = self.results_df[col_to_bin].apply(binning_fn)

    def write_csv(self, outfile: str):
        """ Write the core results df to a CSV file

        Arguments:
            outfile (str): filepath to write the results to
        
        """
        self.metrics_df.to_csv(outfile)

    
    def write_metrics_file(self, outfile: str):
        """ Write the stored metrics to a CSV file

        Arguments:
            outfile (str): filepath to write the metrics to
        
        """

    def print_metrics(self):
        """ Print all the metrics that have been calculated
        
        """

    def calc_confusion_matrix(self, df: pd.DataFrame):
        """
        
        """

    def _threshold_predictions(self):
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
                if self.decision_threshold is None:
                    self._calc_decision_threshold()
                self.results_df[pred_label_col] = self.results_df[self.pred_probs_col].apply(lambda probs: probs[self.pos_label_ind]) > self.decision_threshold
            self.pred_label_col = pred_label_col

    def _calc_decision_threshold(self, metric: str = None):
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
        metric_fn = metric_name_to_metric_fn.get(metric, f1_score)
        opt_threshold = 0
        opt_metric_val = 0
        for thresh in np.arange(0.05, 1.0, 0.05):
            metric_val = metric_fn(self.results_df[self.true_label_col], self.results_df[self.pred_probs_col].apply(lambda probs: probs[self.pos_label_ind]) > thresh)
            if metric_val > opt_metric_val:
                opt_metric_val = metric_val
                opt_threshold = thresh
        self.decision_threshold = opt_threshold

