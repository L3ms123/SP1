# cleaning_utils.py
import pandas as pd

def drop_invalid_dates(df, col):
    """
    Detects invalid dates (Na) in a column of a DataFrame.
    Drops invalid rows with invalid dates.
    
    :param df: DataFrame containing the dates
    :param col: Name of the column to check for invalid dates
    :param valid_date: Valid date to replace invalid dates with
    :return: DataFrame with valid dates and list of invalid dates
    """
    
    # Convert the column to datetime
    parsed_dates = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
    
    # Find invalid dates
    invalid_dates = df[col][parsed_dates.isna()]
    
    # Find rows with invalid dates (NaT)
    invalid_dates_mask = parsed_dates.isna()
    
    if invalid_dates.empty:
        print(f"No invalid dates found in column '{col}'.")
    else:
        print(f"Dropping invalid rows in column '{col}'...")

    # Drop rows with invalid dates
    df_cleaned = df[~invalid_dates_mask]
    
    return df_cleaned, invalid_dates


def drop_invalid_rows(df):
    """
    Drops rows with missing values (containing NaN values) 
    Drop rows with invalid time; START == END.

    :param df: DataFrame to clean
    :return: Cleaned DataFrame
    """
    initial_len = len(df)

    # Drop rows where START == END
    if 'START' in df.columns and 'END' in df.columns:
        df = df[df['START'] != df['END']]

    # Drop rows with any NaN values
    df = df.dropna()

    print(f"Dropped {initial_len - len(df)} rows with missing values or START == END.")
    return df