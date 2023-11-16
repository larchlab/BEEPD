""" 
Utility class for calculating and writing metrics for assessing models.
"""

#NOTE: Need to cite sklearn!
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score
# Want to write code to create DF containing HADM_ID, true label, predicted label/probabilities array, then write functions
# over this DF, rather than doing infinite ad hoc things. Maybe make a PredictionsCalculator class that takes in
# this info, as well as list of classes, threshold, etc.
# for each fn, if false negs, false pos, true negs, true pos, etc. not in the DF, calculate and add that column so
# we're nto reproducing work.


class PredictionCalculator:
    """ Utility class for calculating and writing metrics for assessing models.
    """
    def __init__(self) -> None:
        self.decision_threshold = None
        self.classes = None
        self.result_df: pd.DataFrame = None
        self.outcome_col: str = None
        self.prediction_col: str = None
        self.bootstrap_samples = None
        self.metrics_dict = None
        self.random_seed = None

    def create_bootstrap_samples(self, k: int):
        """ create an N x k array of indices, where N is the number of samples, k is the number of bootstrap
        datasets to create. 
        """
        

    def calc_metrics(self):
        """ Calculate metrics over the result_df, store them in self.metrics_df
        
        """
    
    def write_metrics_file(self, outfile: str):
        """ Write the stored metrics to a file
        
        """

    def print_metrics(self):
        """ Print all the metrics that have been calculated
        
        """
        for k,v in self.metrics_dict.items():
            print(k,v)

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
        """
        
        """

    def calc_auc(self):
        """
        
        """

    def calc_mcc(self):
        """ Calculate the Matthews Correlation Coefficient
        
        """

    def calc_kappa(self):
        """ Calculate the Kappa Statistic (Kappa for Bias and Prevalence?)
        
        """