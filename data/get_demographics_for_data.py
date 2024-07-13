"""
Associate rows in the BEEP Outcome Prediction Datasets (keyed by MIMIC-III HADM_ID), to the demographic info for the
given patient.
"""

import argparse
import pandas as pd
import pathlib

from os import listdir

from mimic_iii_utils import get_age

def get_args() -> argparse.Namespace:
    """ Parse provided args
    
    Returns: 
        argparse.Namespace: namespace containing the parsed args
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data-dir", type=str, required=True, help="Path to generated outcome prediction datasets")
    parser.add_argument("--mimic-dir", type=str, required=True, help="Path to directory containing MIMIC-III files")
    parser.add_argument("--outfile", type=str, default=None, help="Path to CSV File to write, if desired")
    return parser.parse_args()


def get_hadm_ids(data_dir: pathlib.Path) -> pd.Series:
    """ Get all the mimic HADM_IDs associated with the dataset. This should be data processed into outcome prediction
    datasets using the dataset creation scripts from https://github.com/bvanaken/clinical-outcome-prediction/, in which
    the ID column corresponds to MIMIC-III HADM_IDs 

    Arguments:
        data_dir (pathlib.Path): path to the directory containing pre-generated outcome prediction datasets

    Returns:
        pd.Series: Series of the HADM_IDs present
    """
    # get the files from data_dir that look like CSVs
    data_files = [fname for fname in listdir(data_dir) if fname.endswith(".csv")]
    dfs = [pd.read_csv(data_dir/data_file) for data_file in data_files]
    concatted = pd.concat(dfs)
    return concatted["id"].drop_duplicates()


def load_mimic_info(mimic_dir: pathlib.Path, tables: list = ["ADMISSIONS.csv", "PATIENTS.csv"], merge_on: str = "SUBJECT_ID") -> pd.DataFrame:
    """ Load the ADMISSIONS and PATIENTS table from mimic_dir
    
    Arguments:
        mimic_dir (pathlib.Path): path to the directory containing mimic data
        tables (List): list of subpaths to the tables to load
        merge_on (str): name of column to merge on
    
    Returns:
        pd.DataFrame: merged MIMIC-III ADMISSIONS and PATIENTS tables, which together contain the info we need.
    """
    tables = [pd.read_csv(mimic_dir/fname) for fname in tables]
    merged = tables[0]
    for right_table in tables[1:]:
        merged = pd.merge(merged, right_table, on=merge_on, how="inner")
    return merged


def create_hadm_id_to_demographics_sheet(hadm_ids: pd.Series, admission_info_table: pd.DataFrame) -> pd.DataFrame:
    """ Create a table with the following demographics associated with each HADM_ID in the given list:
    * Age (binned for age >=89)
    * Binarized Gender
    * Ethnicity
    * Language
    * Insurance Type

    Arguments:
        hadm_ids (List[int]): List of HADM_IDS to extract info for
        admission_info_table (pd.DataFrame): DataFrame containing joined data from MIMIC-III Admissions and Patients tables

    Returns:
        pd.DataFrame: DF containing columns corresponding to HADM_ID and the demographic info
    
    """
    filtered = pd.merge(hadm_ids, admission_info_table, left_on="HADM_ID", right_on="HADM_ID", how="inner")
    filtered["AGE"] = filtered.apply(get_age, axis=1)
    return filtered[["HADM_ID", "SUBJECT_ID", "AGE", "GENDER", "ETHNICITY", "LANGUAGE", "INSURANCE", "HOSPITAL_EXPIRE_FLAG"]]


if __name__ == "__main__":
    # Get the location of the processed dataset and MIMIC data, write them to a file
    args = get_args()
    # data_dir = pathlib.Path(args.data_dir)
    mimic_dir = pathlib.Path(args.mimic_dir)
    outfile = pathlib.Path(args.outfile)
    # hadm_ids = get_hadm_ids(data_dir)
    admission_info_table = load_mimic_info(mimic_dir)
    hadm_ids = admission_info_table["HADM_ID"]
    demographics_sheet = create_hadm_id_to_demographics_sheet(hadm_ids, admission_info_table)
    if outfile is not None:
        demographics_sheet.to_csv(outfile, index=False)
