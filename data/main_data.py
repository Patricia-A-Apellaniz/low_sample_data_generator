# Author: Juan Parras & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 24/01/2024


# Packages to import
import os
import pickle

import numpy as np
import pandas as pd

from data_manager import DataManager
from sdv.datasets.demo import download_demo

# Number of data samples
DATA_N = 10000
DATA_M = 7500
DATA_L = 2 * 1000
DATA_ALL = DATA_N + DATA_M + DATA_L

# Convert continuous variables to categorical, zero is a class and the rest of the elements are divided into 3
# classes
def cont2cat(df, cols, n=4):
    # Compute the bins for each column
    for col in cols:
        # Compute quartiles
        bins_aux = [df[col].quantile(i / n) for i in range(n + 1)]
        if len(bins_aux) == len(set(bins_aux)):
            df[col], bins = pd.qcut(df[col], q=n, retbins=True, duplicates='drop', labels=False)
        else:
            for i in range(1, len(bins_aux)):
                if bins_aux[i] < bins_aux[i - 1]:
                    bins_aux[i] = bins_aux[i - 1] + 0.0001
                elif bins_aux[i] == bins_aux[i - 1]:
                    bins_aux[i] = bins_aux[i] + 0.0001

            # Create interval index
            df[col] = pd.cut(df[col], bins_aux, labels=False, include_lowest=True, right=True)
            df[col] = df[col].astype('Int64')
            # column should have consecutive integers as categories
            # unique values
            df[col] = df[col].astype('category')
            # label encoding
            df[col] = df[col].astype('category').cat.codes
    return df


def preprocess_adult(dataset_name):
    # Load data
    raw_df, metadata = download_demo(modality='single_table', dataset_name='adult')

    # Transform '?' values to nan values
    raw_df = raw_df.replace('?', np.nan)

    # Transform continuous variables to categorical
    raw_df = cont2cat(raw_df, ['hours-per-week', 'capital-gain', 'capital-loss'])

    # Take just 10.000 samples
    raw_df = raw_df.sample(n=DATA_ALL, random_state=0).reset_index(drop=True)

    # Drop irrelevant columns
    raw_df = raw_df.drop(labels=['education-num'], axis=1)

    # Remove columns also in metadata
    metadata_cols = metadata.columns
    del metadata_cols['education-num']
    metadata.columns = metadata_cols.copy()

    # Transform covariates and create df
    df = raw_df.copy()
    mapping_info = {}
    df['workclass'], classes = df['workclass'].factorize()
    mapping_info['workclass'] = np.array(classes.values)
    df['workclass'] = df['workclass'].replace(-1, np.nan)
    df['education'], classes = df['education'].factorize()
    mapping_info['education'] = np.array(classes.values)
    df['marital-status'], classes = df['marital-status'].factorize()
    mapping_info['marital-status'] = np.array(classes.values)
    df['occupation'], classes = df['occupation'].factorize()
    mapping_info['occupation'] = np.array(classes.values)
    df['occupation'] = df['occupation'].replace(-1, np.nan)
    df['relationship'], classes = df['relationship'].factorize()
    mapping_info['relationship'] = np.array(classes.values)
    df['race'], classes = df['race'].factorize()
    mapping_info['race'] = np.array(classes.values)
    df['sex'] = df['sex'].apply(lambda x: 0 if x == 'Male' else 1)
    mapping_info['sex'] = np.array(['Male', 'Female'])
    df['native-country'], classes = df['native-country'].factorize()
    mapping_info['native-country'] = np.array(classes.values)
    df['native-country'] = df['native-country'].replace(-1, np.nan)
    df['label'] = df['label'].apply(lambda x: 0 if x == '<=50K' else 1)
    mapping_info['label'] = np.array(['<=50K', '>50K'])

    # Create data manager object
    data_manager = DataManager(dataset_name, raw_df, df, mapping_info)

    # Obtain feature distributions
    feat_distributions = []
    for i in range(df.shape[1]):
        values = df.iloc[:, i].unique()
        no_nan_values = values[~pd.isnull(values)]
        if no_nan_values.size <= 2 and np.all(np.sort(no_nan_values).astype(int) ==
                                              np.array(range(no_nan_values.min().astype(int),
                                                             no_nan_values.min().astype(int) + len(no_nan_values)))):
            feat_distributions.append(('bernoulli', 1))
        elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
            # Check if values are floats but don't have decimals and transform to int. They are floats because of NaNs
            if no_nan_values.dtype == 'float64':
                no_nan_values = no_nan_values.astype(int)
            if np.unique(no_nan_values).size < 50 and np.amin(no_nan_values) == 0:
                feat_distributions.append(('categorical', np.max(no_nan_values) + 1))
            else:
                feat_distributions.append(('gaussian', 2))
        else:
            feat_distributions.append(('gaussian', 2))
    data_manager.set_feat_distributions(feat_distributions)

    # Normalize, impute data
    # Necessary to impute before normalization because of the categorical variables treated as gaussian.
    data_manager.norm_df = data_manager.transform_data(df)
    data_manager.imp_norm_df = data_manager.impute_data(data_manager.norm_df)
    data_manager.processed_df = data_manager.imp_norm_df
    # Create metadata for ctgan and tvae
    data_manager.get_metadata()

    return data_manager

def preprocess_news(dataset_name):
    # Load data
    raw_df, metadata = download_demo(modality='single_table', dataset_name='news')
    raw_df = raw_df.rename(columns={'shares': 'label'})
    raw_df = raw_df.loc[:, raw_df.nunique() != 1]
    # Removing Space Character from Feature names
    raw_df.columns = raw_df.columns.str.replace(" ", "")
    # Remove n_unique_tokens since there is just one 701 and the rest is 0
    raw_df = raw_df.drop('n_unique_tokens', axis=1)
    raw_df = raw_df.drop('n_non_stop_words', axis=1)
    raw_df = raw_df.drop('n_non_stop_unique_tokens', axis=1)  # One 650, 10 1s and the rest is 0
    raw_df = raw_df.drop('global_subjectivity', axis=1)
    raw_df = cont2cat(raw_df,
                      ['kw_avg_min', 'kw_avg_max', 'kw_min_min', 'kw_max_min', 'kw_avg_avg', 'kw_min_avg', 'kw_max_avg',
                       'kw_avg_max', 'kw_min_max', 'kw_max_max'], n=4)
    raw_df = cont2cat(raw_df,
                      ['self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_sharess'], n=4)

    # Weekday is onehot-encoded
    wd_df = raw_df[
        ['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday',
         'weekday_is_saturday', 'weekday_is_sunday']]
    category_column = wd_df.idxmax(axis=1)
    category_column = category_column.str.replace('weekday_is_', '', regex=True)
    raw_df = pd.concat((raw_df, category_column.rename('weekday')), axis=1)
    raw_df = raw_df.drop(
        ['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday',
         'weekday_is_saturday', 'weekday_is_sunday'], axis=1)
    # Datachannel is onehot-encoded
    dc = raw_df[
        ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus', 'data_channel_is_socmed',
         'data_channel_is_tech', 'data_channel_is_world']]
    category_column = dc.idxmax(axis=1)
    category_column = category_column.str.replace('data_channel_is_', '', regex=True)
    raw_df = pd.concat((raw_df, category_column.rename('data_channel')), axis=1)
    raw_df = raw_df.drop(
        ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus', 'data_channel_is_socmed',
         'data_channel_is_tech', 'data_channel_is_world'], axis=1)
    # Move label column to the end
    label = raw_df['label']
    raw_df = raw_df.drop(['label'], axis=1)
    raw_df = pd.concat((raw_df, label), axis=1)

    # Take just the needed samples
    raw_df = raw_df.sample(n=DATA_ALL, random_state=0).reset_index(drop=True)

    # Transform covariates and create df
    df = raw_df.copy()
    df['label'] = df['label'].apply(lambda x: 0 if x < 1400 else 1)
    mapping_info = {}
    df['title_sentiment_polarity'], classes = df['title_sentiment_polarity'].factorize()
    mapping_info['title_sentiment_polarity'] = np.array(classes.values)
    df['max_negative_polarity'], classes = df['max_negative_polarity'].factorize()
    mapping_info['max_negative_polarity'] = np.array(classes.values)
    df['min_negative_polarity'], classes = df['min_negative_polarity'].factorize()
    mapping_info['min_negative_polarity'] = np.array(classes.values)
    df['avg_negative_polarity'], classes = df['avg_negative_polarity'].factorize()
    mapping_info['avg_negative_polarity'] = np.array(classes.values)
    df['weekday'], classes = df['weekday'].factorize()
    mapping_info['weekday'] = np.array(classes.values)
    df['data_channel'], classes = df['data_channel'].factorize()
    mapping_info['data_channel'] = np.array(classes.values)

    # Create data manager object
    data_manager = DataManager(dataset_name, raw_df, df, mapping_info)

    # Obtain feature distributions
    feat_distributions = []
    for i in range(df.shape[1]):
        values = df.iloc[:, i].unique()
        no_nan_values = values[~pd.isnull(values)]
        if no_nan_values.size <= 2 and np.all(np.sort(no_nan_values).astype(int) ==
                                              np.array(range(no_nan_values.min().astype(int),
                                                             no_nan_values.min().astype(int) + len(no_nan_values)))):
            feat_distributions.append(('bernoulli', 1))
        elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
            # Check if values are floats but don't have decimals and transform to int. They are floats because of NaNs
            if no_nan_values.dtype == 'float64':
                no_nan_values = no_nan_values.astype(int)
            if np.unique(no_nan_values).size < 50 and np.amin(no_nan_values) == 0:
                feat_distributions.append(('categorical', np.max(no_nan_values) + 1))
            else:
                feat_distributions.append(('gaussian', 2))
        else:
            feat_distributions.append(('gaussian', 2))
    data_manager.set_feat_distributions(feat_distributions)

    # Normalize, impute data
    # Necessary to impute before normalization because of the categorical variables treated as gaussian.
    data_manager.norm_df = data_manager.transform_data(df)
    data_manager.imp_norm_df = data_manager.impute_data(data_manager.norm_df)
    data_manager.processed_df = data_manager.imp_norm_df
    # Create metadata for ctgan and tvae
    data_manager.get_metadata()

    return data_manager


def preprocess_king(dataset_name):
    # Load data
    data_filename = './raw_data/king/king.csv'
    raw_df = pd.read_csv(data_filename, sep=',')
    raw_df = raw_df.rename(columns={'price': 'label'})
    # raw_df = raw_df.drop(columns=['zipcode'])
    cols2cat = ['sqft_basement', 'yr_renovated', 'sqft_lot', 'sqft_lot15']
    raw_df = cont2cat(raw_df, cols2cat)

    # Take just 10.000 samples
    raw_df = raw_df.sample(n=DATA_ALL, random_state=0).reset_index(drop=True)

    # Transform covariates and create df
    df = raw_df.copy()
    mapping_info = {}
    df['grade'], classes = df['grade'].factorize()
    mapping_info['grade'] = np.array(classes.values)
    df['condition'], classes = df['condition'].factorize()
    mapping_info['condition'] = np.array(classes.values)
    df['floors'], classes = df['floors'].factorize()
    mapping_info['floors'] = np.array(classes.values)
    df['bathrooms'] = df['bathrooms'] * 4
    df['bathrooms'], classes = df['bathrooms'].factorize()
    mapping_info['bathrooms'] = np.array(classes.values)
    df['zipcode'], classes = df['zipcode'].factorize()
    mapping_info['zipcode'] = np.array(classes.values)

    # Create data manager object
    data_manager = DataManager(dataset_name, raw_df, df, mapping_info)

    # Obtain feature distributions
    feat_distributions = []
    for i in range(df.shape[1]):
        values = df.iloc[:, i].unique()
        no_nan_values = values[~pd.isnull(values)]
        if no_nan_values.size <= 2 and np.all(np.sort(no_nan_values).astype(int) ==
                                              np.array(range(no_nan_values.min().astype(int),
                                                             no_nan_values.min().astype(int) + len(no_nan_values)))):
            feat_distributions.append(('bernoulli', 1))
        elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
            # Check if values are floats but don't have decimals and transform to int. They are floats because of NaNs
            if no_nan_values.dtype == 'float64':
                no_nan_values = no_nan_values.astype(int)
            if np.unique(no_nan_values).size < 80 and np.amin(no_nan_values) == 0:
                feat_distributions.append(('categorical', np.max(no_nan_values) + 1))
            else:
                feat_distributions.append(('gaussian', 2))
        else:
            feat_distributions.append(('gaussian', 2))
    data_manager.set_feat_distributions(feat_distributions)

    # Normalize, impute data
    # Necessary to impute before normalization because of the categorical variables treated as gaussian.
    data_manager.norm_df = data_manager.transform_data(df)
    data_manager.imp_norm_df = data_manager.impute_data(data_manager.norm_df)
    data_manager.processed_df = data_manager.imp_norm_df
    # Create metadata for ctgan and tvae
    data_manager.get_metadata()

    # Remove pii key from metadata columns sqft_living, zipcode and sqft_living15
    #del data_manager.metadata.columns['sqft_living']['pii']
    del data_manager.metadata.columns['zipcode']['pii']
    #del data_manager.metadata.columns['sqft_living15']['pii']
    # Change type of columns
    data_manager.metadata.columns['sqft_living']['sdtype'] = 'numerical'
    data_manager.metadata.columns['zipcode']['sdtype'] = 'numerical'
    data_manager.metadata.columns['sqft_living15']['sdtype'] = 'numerical'
    # Change data_manager metadata primary key to None
    data_manager.metadata.primary_key = None

    return data_manager

def preprocess_intrusion(dataset_name):
    # Load data
    raw_df, metadata = download_demo(modality='single_table', dataset_name='intrusion')
    raw_df = raw_df.loc[:, raw_df.nunique() != 1]
    # raw_df = raw_df.drop(columns=['wrong_fragment'])
    # cols2cat = ['duration', 'hot', 'num_root', 'dst_host_srv_count']
    cols2cat = ['src_bytes', 'dst_bytes', 'dst_host_count', 'dst_host_srv_count', 'count', 'srv_count']
    raw_df = cont2cat(raw_df, cols2cat)

    # Take just 10.000 samples
    raw_df = raw_df.sample(n=DATA_ALL, random_state=0).reset_index(drop=True)

    # Transform covariates and create df
    df = raw_df.copy()
    mapping_info = {}
    df['protocol_type'], classes = df['protocol_type'].factorize()
    mapping_info['protocol_type'] = np.array(classes.values)
    df['service'], classes = df['service'].factorize()
    mapping_info['service'] = np.array(classes.values)
    df['flag'], classes = df['flag'].factorize()
    mapping_info['flag'] = np.array(classes.values)
    df['label'], classes = df['label'].factorize()
    mapping_info['label'] = np.array(classes.values)

    # Create data manager object
    data_manager = DataManager(dataset_name, raw_df, df, mapping_info)

    # Obtain feature distributions
    feat_distributions = []
    for i in range(df.shape[1]):
        values = df.iloc[:, i].unique()
        no_nan_values = values[~pd.isnull(values)]
        if no_nan_values.size <= 2 and np.all(np.sort(no_nan_values).astype(int) ==
                                              np.array(range(no_nan_values.min().astype(int),
                                                             no_nan_values.min().astype(int) + len(no_nan_values)))):
            feat_distributions.append(('bernoulli', 1))
        elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
            # Check if values are floats but don't have decimals and transform to int. They are floats because of NaNs
            if no_nan_values.dtype == 'float64':
                no_nan_values = no_nan_values.astype(int)
            if np.unique(no_nan_values).size < 50 and np.amin(no_nan_values) == 0:
                feat_distributions.append(('categorical', (np.max(no_nan_values) + 1).astype(int)))
            else:
                feat_distributions.append(('gaussian', 2))
        else:
            feat_distributions.append(('gaussian', 2))
    data_manager.set_feat_distributions(feat_distributions)

    # Normalize, impute data
    # Necessary to impute before normalization because of the categorical variables treated as gaussian.
    data_manager.norm_df = data_manager.transform_data(df)
    data_manager.imp_norm_df = data_manager.impute_data(data_manager.norm_df)

    # Create metadata for ctgan and tvae
    data_manager.get_metadata()

    return data_manager


def preprocess_by_name(d_name):
    if d_name == 'adult':
        return preprocess_adult(d_name)
    elif d_name == 'news':
        return preprocess_news(d_name)
    elif d_name == 'king':
        return preprocess_king(d_name)
    elif d_name == 'intrusion':
        return preprocess_intrusion(d_name)
    else:
        raise ValueError('Dataset not found')


# Preprocess and save each dataset in the corresponding folder; run this code to obtain the preprocessed data
if __name__ == '__main__':
    for dataset_name in ['adult', 'news', 'king', 'intrusion']:
        data_manager = preprocess_by_name(dataset_name)
        save_path = './processed_data_old/' + str(dataset_name) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save dictionary with metadata needed later for CTGAN
        with open(save_path + 'metadata.pkl', 'wb') as f:
            pickle.dump({'feat_distributions': data_manager.feat_distributions, 'mask': data_manager.mask, 'metadata': data_manager.metadata}, f)

        # Save data
        data_manager.imp_norm_df.to_csv(save_path + 'preprocessed_data_all.csv', index=False)
        data_manager.imp_norm_df[0: DATA_N].to_csv(save_path + 'preprocessed_data_n.csv', index=False)
        data_manager.imp_norm_df[DATA_N: DATA_N + DATA_M].to_csv(save_path + 'preprocessed_data_m.csv', index=False)
        data_manager.imp_norm_df[DATA_N + DATA_M: DATA_ALL].to_csv(save_path + 'preprocessed_data_l.csv', index=False)
        print(f'{dataset_name} saved')