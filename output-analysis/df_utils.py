""" Utility functions for doing common things with pandas dataframes

"""
import pandas as pd
from typing import Dict

def stratify_df(df: pd.DataFrame, stratify_col: str) -> Dict[str, pd.DataFrame]:
    """ Create a dictionary mapping col=val strings to the input dataframe filtered as such
    
    Arguments:
        df (pd.DataFrame): dataframe to stratify
        stratify_col (str): column to stratify on

    Returns:
        Dict[str, pd.DataFrame]: dict mapping "COL_NAME=VAL" strings to respective filterings of the DF

    """
    return {
        f"{stratify_col}={unique_val}": df[df[stratify_col] == unique_val] for unique_val in set(df[stratify_col])
    }

def bin_ethnicity(ethnicity):
    """ Bin ethnicities in MIMIC
    
    """
    white_ethnicities = [
        'WHITE', 
        'WHITE - RUSSIAN', 
        'WHITE - OTHER EUROPEAN', 
        'WHITE - BRAZILIAN', 
        'WHITE - EASTERN EUROPEAN',
        ]
    black_ethnicities = [
        'BLACK/AFRICAN AMERICAN',
        'BLACK/CAPE VERDEAN',
        'BLACK/HAITIAN',
        'BLACK/AFRICAN',
        'CARIBBEAN ISLAND',
    ]
    hispanic_ethnicities = [
        'HISPANIC OR LATINO',
        'HISPANIC/LATINO - PUERTO RICAN',
        'HISPANIC/LATINO - DOMINICAN',
        'HISPANIC/LATINO - GUATEMALAN',
        'HISPANIC/LATINO - CUBAN',
        'HISPANIC/LATINO - SALVADORAN',
        'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)',
        'HISPANIC/LATINO - MEXICAN',
        'HISPANIC/LATINO - COLOMBIAN',
        'HISPANIC/LATINO - HONDURAN',
    ]
    asian_ethnicities = [
        'ASIAN',
        'ASIAN - CHINESE',
        'ASIAN - ASIAN INDIAN',
        'ASIAN - VIETNAMESE',
        'ASIAN - FILIPINO',
        'ASIAN - CAMBODIAN',
        'ASIAN - OTHER',
        'ASIAN - KOREAN',
        'ASIAN - JAPANESE',
        'ASIAN - THAI',
    ]
    native_ethnicities = [
        'AMERICAN INDIAN/ALASKA NATIVE',
        'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE'
    ]
    unknown_ethnicities = [
        'UNKNOWN/NOT SPECIFIED',
        'UNABLE TO OBTAIN',
    ]
    declined_ethnicities = [
        'PATIENT DECLINED TO ANSWER',
    ]
    if ethnicity in white_ethnicities:
        return 'white'
    elif ethnicity in black_ethnicities:
        return 'black'
    elif ethnicity in hispanic_ethnicities:
        return 'hispanic'
    elif ethnicity in asian_ethnicities:
        return 'asian'
    elif ethnicity in native_ethnicities:
        return 'native'
    elif ethnicity in unknown_ethnicities:
        return 'unknown'
    elif ethnicity in declined_ethnicities:
        return 'declined'
    else:
        return 'other'
