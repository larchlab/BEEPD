"""
Utility functions for working with MIMIC-III
"""

import pandas as pd

def get_age(row: pd.Series) -> int:
    """ Get the age of MIMIC-III Patients from a row containing both the patient's DOB (from PATIENTS.csv)
    and ADMITTIME (from ADMISSIONS.csv). Meant to be applied as a DF mapping function with axis=1

    Arguments:
        row (pd.Series): row of the DF with DOB and ADMITTIME cols

    returns:
        int: age of patient in the given row
    
    """
    admittime = pd.Timestamp(row.ADMITTIME)
    dob = pd.Timestamp(row.DOB)
    # Ages >=89 are listed as like 300, which causes an overflow error that we need to catch
    try:
        return min(int((admittime - dob).days / 365.25), 89)
    except:
        return 89