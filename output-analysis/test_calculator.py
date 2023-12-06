import pandas as pd
from PredictionCalculator import PredictionCalculator
from pathlib import Path

intermediate_products_dir = Path('../intermediate_products')

df = pd.read_pickle(intermediate_products_dir / "outcome_prediction_res.pkl")
demographics_df = pd.read_csv(intermediate_products_dir / "demographics_sheet.csv")

df.HADM_ID = df['HADM_ID'].map(int)

calc = PredictionCalculator(df, "true_labels", "pred_probs", demographics_df, "HADM_ID", demographics_cols=["GENDER"])
calc.calc_metric_conf_ints(num_samples=1000)
calc.write_csv("/Users/chaiken/research/data/testing_inputs/basic_metrics.csv")