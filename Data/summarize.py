import numpy as np
import pandas as pd

def count_unique(my_data):
    """
    Prints out some cursory information regarding the data.
    """
    rows, cols = my_data.shape
    rows_unique = np.sum( np.logical_not(my_data.duplicated()) )
    print(f'This data has {rows} rows and {cols} columns.')
    print(f'This includes {rows_unique} unique rows ({round(rows_unique / rows * 100,2)}%).')
    print('\n')
    cols = my_data.columns
    n_unique = []
    for col in cols:
        n_unique.append(len(my_data[col].unique()))
    unique_df = pd.DataFrame({'column':cols, 'n_unique':n_unique})
    print(unique_df[['column', 'n_unique']].sort_values(by='n_unique', ascending=True))