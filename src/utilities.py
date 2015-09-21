import pandas as pd
import numpy as np


def describe_features(dataframe):
    features = []
    for i in range(0, dataframe.shape[1]):
        features.append(describe_series(dataframe.iloc[:, i]))

    df = pd.DataFrame(features, columns=['Name', 'Type', 'Cardinality', 'Top1', 'Top3', 'Top5', 'Top50'])
    return df


def describe_series(data_series):
    feature_type = data_series.dtype.name
    unique_values = data_series.value_counts()
    data_length = float(data_series.shape[0])

    return data_series.name, feature_type, unique_values.shape[0], unique_values.iloc[0:1].sum() / data_length, unique_values.iloc[0:3].sum() / data_length, unique_values.iloc[0:5].sum() / data_length, unique_values.iloc[0:50].sum() / data_length


def remove_string_columns(data):
    columns_to_remove = []
    for i in range(0, data.shape[1]):
        if data.iloc[:, i].dtype == 'object':
            columns_to_remove.append(i)

    data = data.drop(data.columns[columns_to_remove], axis=1)
    return data


def remove_high_cardinality_features(data, threshold=1000):
    columns_to_remove = []
    for i in range(0, data.shape[1]):
        if data.iloc[:, i].value_counts().shape[0] >= threshold:
            columns_to_remove.append(i)

    data = data.drop(data.columns[columns_to_remove], axis=1)
    return data