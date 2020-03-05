import pandas as pd
import math


# Function to determine how many times NaN occurs in Enron Data set
def how_many_NaN(enron_df):
    dict_ = {}
    features = enron_df.columns
    for j in range(len(features)):
        dict_[features[j]] = 0
        for i in range(len(enron_df)):
            try:
                value = float(enron_df[features[j]].iloc[i])
                if math.isnan(value):
                    dict_[features[j]] += 1
            except ValueError:
                continue

    df = pd.DataFrame(dict_.items(), columns=['feature', 'NaNs'])
    df = df.sort_values('NaNs', ascending=False).set_index('feature')
    return df
