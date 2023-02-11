import pandas as pd

# Ch1 Functions

# Normalises using a rolling median
def norm_df_med(df, window = 30, stocks = ['CBA.AX']):
    new_df = pd.DataFrame()
    for stock in stocks:
        new_df[stock] = (df[stock] / (df[stock].rolling(window, min_periods = 0).median()) - 1)
    return new_df

# Normalises using rolling mean
def norm_df_mean(df, window = 30,  stocks = ['CBA.AX']):
    new_df = pd.DataFrame()
    for stock in stocks:
        new_df[stock] = df[stock] / (df[stock].rolling(window, min_periods = 0).mean()-1)
    return new_df
