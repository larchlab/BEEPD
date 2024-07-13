# just test some results :)

import pandas as pd
from PredictionCalculator import PredictionCalculator
from pathlib import Path
import numpy as np
from df_utils import MIMIC_3_DIR

intermediate_products_dir = Path('intermediate_products')

# df = pd.read_pickle(intermediate_products_dir / "outcome_prediction_res.pkl")
df = pd.read_pickle(intermediate_products_dir / "umlsbert_avg_mor_results.pkl")
demographics_df = pd.read_csv(MIMIC_3_DIR / "demographics_sheet.csv")
df.HADM_ID = df['HADM_ID'].map(int)

calc = PredictionCalculator(df, "true_labels", "pred_probs", demographics_df, "HADM_ID", demographics_cols=["GENDER"])
# calc.test_normality(plot=True)
conf_mtx_thresh = calc.calc_confusion_matrix(calc.results_df)
print(f"Threshold: {calc.decision_threshold}\n", conf_mtx_thresh, '\n', (np.round(conf_mtx_thresh/np.sum(conf_mtx_thresh), 2)*100).astype(int))
conf_mtx_0_5 = calc.calc_confusion_matrix(calc.results_df, threshold=0.5)
print(f"Threshold: {0.5}\n", conf_mtx_0_5,'\n', (np.round(conf_mtx_0_5/np.sum(conf_mtx_0_5), 2)*100).astype(int))
# calc.calc_metric_conf_ints(num_samples=1000)
# calc.write_csv("/Users/chaiken/research/data/testing_inputs/basic_metrics.csv")