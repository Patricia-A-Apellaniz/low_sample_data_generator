# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 06/09/2023


# Packages to import
import math
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
# from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer as IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVR
import matplotlib.pyplot as plt
# -----------------------------------------------------------
#                   DATA IMPUTATION
# -----------------------------------------------------------
def hist_per_column(df):
    # For each column, chech if it is categorical or numerical and plot distribution as barplot or histogram
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].unique().size < 20:
            type = 'categorical'
            df[col].value_counts().plot.bar()
        else:
            type = 'numerical'
            # set variable number of bins depending on the data
            bins = 100 if df[col].nunique() > 50 else df[col].nunique()
            df[col].plot.hist(bins=50)
        plt.title(col + ' ' + type)
        # plt.show()


def zero_imputation(data):
    imp_data = data.copy()
    imp_data = imp_data.fillna(0)
    return imp_data


def mice_imputation(data, model='bayesian'):
    imp_data = data.copy()
    if model == 'bayesian':
        clf = BayesianRidge()
    elif model == 'svr':
        clf = SVR()
    else:
        raise RuntimeError('MICE imputation base_model not recognized')
    imp = IterativeImputer(estimator=clf, verbose=2, max_iter=30, tol=1e-10, imputation_order='roman')
    imp_data.iloc[:, :] = imp.fit_transform(imp_data)
    return imp_data


def statistics_imputation(data):
    imp_data = data.copy()
    n_samp, n_feat = imp_data.shape
    for i in range(n_feat):
        values = data.iloc[:, i].values
        if any(pd.isnull(values)):
            if len(pd.isnull(values)) == 1:
                values = np.zeros((1,))
            no_nan_values = values[~pd.isnull(values)]
            if no_nan_values.size <= 2 or no_nan_values.dtype in [object, str] or np.amin(
                    np.equal(np.mod(no_nan_values, 1), 0)):
                stats_value = stats.mode(no_nan_values, keepdims=True)[0][0]
            else:
                mean_value = no_nan_values.mean()
                # TODO: FIXXXXX!
                # Round to number of decimals of the original data
                # max_dec = 0
                # if data.dtypes[i] == 'float64':
                #     for val in data.iloc[:, i]:
                #         if not np.isnan(val):
                #             dec = len(str(val).split('.')[1])
                #             if dec > max_dec:
                #                 max_dec = dec
                # stats_value = round(mean_value, max_dec)
                stats_value = mean_value
            imp_data.iloc[:, i] = [stats_value if pd.isnull(x) else x for x in imp_data.iloc[:, i]]

    return imp_data


def impute_data(df, imp_mask=True, gen_mask=False, feat_distributions=None, mode='stats'):
    # If missing data exists, impute it
    if df.isna().any().any():
        # Data imputation
        if mode == 'zero':
            imp_df = zero_imputation(df)
        elif mode == 'stats':
            imp_df = statistics_imputation(df)
        else:
            imp_df = mice_imputation(df)

        # Generate missing data mask. Our model uses it to ignore missing data, although it has been imputed
        if imp_mask:
            nans = df.isna()
            mask = nans.replace([True, False], [0, 1])
        else:
            mask = np.ones((df.shape[0], df.shape[1]))
            mask = pd.DataFrame(mask, columns=imp_df.columns)

        # Concatenate mask to data to generate synthetic missing positions too
        if gen_mask:
            mask_names = ['imp_mask_' + col for col in df.columns]
            mask.columns = mask_names
            imp_df = pd.concat([imp_df, mask], axis=1)
            tr_mask_df = mask.copy()
            tr_mask_df.columns = ['tr_mask_' + col for col in df.columns]
            mask = pd.concat([mask, tr_mask_df.replace(0, 1)], axis=1)  # It is necessary to concatenate mask to mask
            # with all ones for training purposes
            # Get new data distributions. Mask features should be bernoulli!
            feat_distributions.extend([('bernoulli', 1) for _ in range(mask.shape[1] - len(feat_distributions))])
    else:
        imp_df = df.copy()
        mask = np.ones((df.shape[0], df.shape[1]))
        mask = pd.DataFrame(mask, columns=imp_df.columns)
        mask = mask.astype(int)
    return imp_df, mask, feat_distributions


def force_missing_at_random(df, p_missing):
    np.random.seed(1234)
    # check if time and event columns are present in df
    if 'time' not in df.columns and 'event' not in df.columns:
        arr = np.array(df)
        cols = df.columns
    else:
        arr = np.array(df.iloc[:, :-2])
        cols = df.columns[:-2]
    n_nan_values = int(arr.size * p_missing)
    for i in range(n_nan_values):
        while True:
            random_row = np.random.randint(arr.shape[0])
            random_col = np.random.randint(arr.shape[1])
            if not np.isnan(arr[random_row, random_col]):
                arr[random_row, random_col] = np.nan
                break

    nans_df = pd.DataFrame(arr, columns=cols)
    if 'time' in df.columns and 'event' in df.columns:
        tot_nans_df = pd.concat((nans_df.reset_index(drop=True), df.iloc[:, -2:].reset_index(drop=True)), axis=1)
    else:
        tot_nans_df = nans_df.copy()
    return tot_nans_df


# -----------------------------------------------------------
#                   DATA SPLITTING
# -----------------------------------------------------------
def split_data(data, mask):
    train_data, val_data, train_mask, val_mask = train_test_split(data, mask, test_size=0.2, random_state=0)
    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    train_mask.reset_index(drop=True, inplace=True)
    val_mask.reset_index(drop=True, inplace=True)
    return train_data, train_mask, val_data, val_mask


# Function that receives data and performs a split of a specific number of folds for cross-validatiom.
# It returns a list of tuples with the train and test data and the mask for each fold
# def split_data_cv(data, mask, real_df, n_folds):
def split_data_cv(data, mask, real_df, args, norm_time=True):
    cv_data = []
    if args['n_folds'] < 2:
        cv_data.append(split_data(data, mask))
        raise RuntimeError('Number of folds must be greater than 1')
    else:
        kf = KFold(n_splits=args['n_folds'], random_state=123, shuffle=True)
        for train_index, test_index in kf.split(data):
            train_data = data.iloc[train_index]
            test_data = data.iloc[test_index]
            train_mask = mask.iloc[train_index]
            test_mask = mask.iloc[test_index]

            if norm_time:
                tr_raw_time = denormalize_data(real_df[['time']], train_data[['time']], [('gaussian', 2)])
                test_raw_time = denormalize_data(real_df[['time']], test_data[['time']], [('gaussian', 2)])
            else:
                tr_raw_time = train_data[['time']]
                test_raw_time = test_data[['time']]
            cv_data.append((train_data.reset_index(drop=True), train_mask.reset_index(drop=True),
                            test_data.reset_index(drop=True), test_mask.reset_index(drop=True),
                            tr_raw_time.reset_index(drop=True), test_raw_time.reset_index(drop=True)))
    return cv_data


# TODO: refactor this function
def split_data_cv_no_test(real_df, missing_p, args, dataset_name=None, norm_time=True):
    cv_data = []
    if args['n_folds'] < 2:
        no_sa_data = real_df.drop(['time', 'event'], axis=1)
        feat_distributions_no_sa = get_feat_distributions(no_sa_data, dataset_name=dataset_name)
        if args['force_missing'] and missing_p > 0.0:
            no_sa_data = force_missing_at_random(no_sa_data, missing_p)
        # Second, impute all together
        no_sa_imp_data, mask, _ = impute_data(no_sa_data)
        # Add two last columns to mask
        mask['time'] = 1
        mask['event'] = 1
        # Third, normalize all together
        if norm_time:
            imp_data = pd.concat([no_sa_imp_data, real_df[['time', 'event']]], axis=1)
            feat_distributions = feat_distributions_no_sa + [args['time_distribution'], ('bernoulli', 1)]
            norm_imp_data = normalize_data(imp_data, feat_distributions)
        else:
            no_sa_norm_imp_data = normalize_data(no_sa_imp_data, feat_distributions_no_sa)
            norm_imp_data = pd.concat([no_sa_norm_imp_data, real_df[['time', 'event']]], axis=1)
        # Fourth, split again
        train_data, test_data, train_mask, test_mask = train_test_split(norm_imp_data, mask, test_size=0.2,
                                                                        random_state=1234)
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)
        train_mask = train_mask.reset_index(drop=True)
        test_mask = test_mask.reset_index(drop=True)

        if norm_time:
            tr_raw_time = denormalize_data(real_df[['time']], train_data[['time']], [('gaussian', 2)])
            test_raw_time = denormalize_data(real_df[['time']], test_data[['time']], [('gaussian', 2)])
        else:
            tr_raw_time = train_data[['time']]
            test_raw_time = test_data[['time']]
        cv_data.append((train_data.reset_index(drop=True), train_mask.reset_index(drop=True),
                        test_data.reset_index(drop=True), test_mask.reset_index(drop=True),
                        tr_raw_time.reset_index(drop=True), test_raw_time.reset_index(drop=True)))
        return cv_data, feat_distributions_no_sa
    else:
        kf = KFold(n_splits=args['n_folds'], random_state=1234, shuffle=True)
        for train_index, test_index in kf.split(real_df):
            # First, force nans
            # Force missing data at random if args['missing'] is True
            no_sa_data = real_df.drop(['time', 'event'], axis=1)
            feat_distributions_no_sa = get_feat_distributions(no_sa_data)
            if missing_p > 0.0:
                no_sa_data = force_missing_at_random(no_sa_data, missing_p)
            # Second, impute all together
            no_sa_imp_data, mask, _ = impute_data(no_sa_data)
            # Add two last columns to mask
            mask['time'] = 1
            mask['event'] = 1
            # Third, normalize all together
            if norm_time:
                imp_data = pd.concat([no_sa_imp_data, real_df[['time', 'event']]], axis=1)
                feat_distributions = feat_distributions_no_sa + [args['time_distribution'], ('bernoulli', 1)]
                norm_imp_data = normalize_data(imp_data, feat_distributions)
            else:
                no_sa_norm_imp_data = normalize_data(no_sa_imp_data, feat_distributions_no_sa)
                norm_imp_data = pd.concat([no_sa_norm_imp_data, real_df[['time', 'event']]], axis=1)
            # Fourth, split again
            train_data = norm_imp_data.iloc[train_index]
            test_data = norm_imp_data.iloc[test_index]
            train_mask = mask.iloc[train_index]
            test_mask = mask.iloc[test_index]

            if norm_time:
                tr_raw_time = denormalize_data(real_df[['time']], train_data[['time']], [('gaussian', 2)])
                test_raw_time = denormalize_data(real_df[['time']], test_data[['time']], [('gaussian', 2)])
            else:
                tr_raw_time = train_data[['time']]
                test_raw_time = test_data[['time']]
            cv_data.append((train_data.reset_index(drop=True), train_mask.reset_index(drop=True),
                            test_data.reset_index(drop=True), test_mask.reset_index(drop=True),
                            tr_raw_time.reset_index(drop=True), test_raw_time.reset_index(drop=True)))
        return cv_data, feat_distributions_no_sa


# -----------------------------------------------------------
#                   DATA NORMALIZATION
# -----------------------------------------------------------

def get_feat_distributions(df, cols_cat=[]):
    # TODO: revisar esto!!
    # hist_per_column(df)
    n_feat = df.shape[1]
    feat_dist = []

    for i in range(n_feat):
        col = df.columns[i]
        values = df.iloc[:, i].unique()
        if len(values) == 1 and math.isnan(values[0]):
            values = np.zeros((1,))
        # no_nan_values = values[~np.isnan(values)]
        no_nan_values = values[~pd.isnull(values)]
        if col in cols_cat:
            if col == 'sex':
                feat_dist.append(('bernoulli', 1))
                print('Bernoulli feature: ', col)
            else:
                feat_dist.append(('categorical', np.unique(no_nan_values).size))
                print('Categorical feature: ', col, ' with ', np.unique(no_nan_values).size, ' categories')
        else:
            if col == 'native-country':
                feat_dist.append(('categorical', np.unique(no_nan_values).size))
                print('Categorical feature: ', col)

            elif 'soil_type' in col and all(np.sort(no_nan_values) == np.array(
                    range(int(no_nan_values.min()), int(no_nan_values.min()) + len(no_nan_values)))):
                feat_dist.append(('categorical', np.unique(no_nan_values).size))
                print('Categorical feature: ', col)
            else:
                if no_nan_values.size <= 2 and all(np.sort(no_nan_values) == np.array(
                        range(int(no_nan_values.min()), int(no_nan_values.min()) + len(no_nan_values)))):
                    feat_dist.append(('bernoulli', 1))
                    print('Bernoulli feature: ', col)
                elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
                    # Check if values are floats but don't have decimals and transform to int. They are floats because of NaNs
                    if no_nan_values.dtype == 'float64':
                        no_nan_values = no_nan_values.astype(int)
                    # If number of categories is less than 20, categories start in 0 and categories are consecutive numbers, then it is categorical
                    if np.unique(no_nan_values).size < 20 and np.amin(no_nan_values) == 0 and all(
                            np.sort(no_nan_values) == np.array(
                                    range(int(no_nan_values.min()), int(no_nan_values.min()) + len(no_nan_values)))):
                        feat_dist.append(('categorical', np.unique(no_nan_values).size))
                        # feat_dist.append(('categorical', np.max(no_nan_values) + 1))
                        print('Categorical feature: ', col)
                    else:
                        feat_dist.append(('gaussian', 2))
                        print('Gaussian feature: ', col)
                else:
                    feat_dist.append(('gaussian', 2))
                    print('Gaussian feature: ', col)

    return feat_dist


def normalize_data(raw_df, feat_distributions, df_gen=None):
    num_patient, num_feature = raw_df.shape
    norm_df = raw_df.copy()
    if df_gen is not None:
        norm_gen = df_gen.copy()
    for i in range(num_feature):
        values = raw_df.iloc[:, i]
        if len(values) == 1 and math.isnan(values[0]):
            values = np.zeros((1,))
        no_nan_values = values[~pd.isnull(values)].values
        if feat_distributions[i][0] == 'gaussian':
            loc = np.mean(no_nan_values)
            scale = np.std(no_nan_values)
        elif feat_distributions[i][0] == 'bernoulli':
            if len(np.unique(no_nan_values)) == 1:
                continue
            loc = np.amin(no_nan_values)
            scale = np.amax(no_nan_values) - np.amin(no_nan_values)
        elif feat_distributions[i][0] == 'categorical':
            loc = np.amin(no_nan_values)
            scale = 1  # Do not scale
        elif feat_distributions[i][0] == 'weibull' or feat_distributions[i][0] == 'exponential':
            loc = -1 if 0 in no_nan_values else 0
            scale = 1
        elif feat_distributions[i][0] == 'log-normal':
            if raw_df.iloc[:, i].min() == 0:
                loc = 0
            else:
                loc = -1
            scale = 1
        else:
            print('Distribution ', feat_distributions[i][0], ' not normalized')
            param = np.array([0, 1])  # loc = 0, scale = 1, means that data is not modified!!
            loc = param[-2]
            scale = param[-1]
        if feat_distributions[i][0] == 'weibull' or feat_distributions[i][0] == 'exponential':
            if 0 in raw_df.iloc[:, i].values:
                raw_df.iloc[:, i] = raw_df.iloc[:, i] + 1
            else:
                # Mover datos para que empiecen en 1
                raw_df.iloc[:, i] = raw_df.iloc[:, i] - np.amin(raw_df.iloc[:, i]) + 1
            norm_df.iloc[:, i] = (raw_df.iloc[:, i]) / np.mean(raw_df.iloc[:, i])
        else:
            norm_df.iloc[:, i] = (raw_df.iloc[:, i] - loc) / scale if scale != 0 else raw_df.iloc[:, i] - loc
            if df_gen is not None:
                norm_gen.iloc[:, i] = (norm_gen.iloc[:, i] - loc) / scale if scale != 0 else norm_gen.iloc[:, i] - loc

    if df_gen is not None:
        return norm_df.reset_index(drop=True), norm_gen.reset_index(drop=True)

    return norm_df.reset_index(drop=True)


def denormalize_data(raw_df, norm_df, feat_distributions):
    num_feature = raw_df.shape[1]
    denorm_df = norm_df.copy()
    for i in range(num_feature):
        values = raw_df.iloc[:, i]
        no_nan_values = values[~np.isnan(values)].values
        if feat_distributions[i][0] == 'gaussian':
            loc = np.mean(no_nan_values)
            scale = np.std(no_nan_values)
        elif feat_distributions[i][0] == 'bernoulli':
            loc = np.amin(raw_df.iloc[:, i])
            scale = np.amax(no_nan_values) - np.amin(no_nan_values)
        elif feat_distributions[i][0] == 'categorical':
            loc = np.amin(no_nan_values)
            scale = 1  # Do not scale
        elif feat_distributions[i][0] == 'weibull' or feat_distributions[i][0] == 'exponential':
            loc = 1 if 0 in no_nan_values else 0
            scale = 1
        elif feat_distributions[i][0] == 'log-normal':
            loc = 1
            scale = 1
        else:
            print('Distribution ', feat_distributions[i][0], ' not normalized')
            param = np.array([0, 1])  # loc = 0, scale = 1, means that data is not modified!!
            loc = param[-2]
            scale = param[-1]
        if feat_distributions[i][0] == 'weibull' or feat_distributions[i][0] == 'exponential':
            # Multiplicar datos normalizados por la media de los datos en crudo
            denorm_df.iloc[:, i] = norm_df.iloc[:, i] * np.mean(no_nan_values)
            if 0 in no_nan_values:
                denorm_df.iloc[:, i] = denorm_df.iloc[:, i] - 1
            else:
                # Mover datos para que empiecen en 1
                denorm_df.iloc[:, i] = denorm_df.iloc[:, i] + np.amin(no_nan_values) - 1
            norm_df.iloc[:, i] = (raw_df.iloc[:, i]) / np.mean(raw_df.iloc[:, i])
        else:
            if scale != 0:
                denorm_df.iloc[:, i] = norm_df.iloc[:, i] * scale + loc
            else:
                denorm_df.iloc[:, i] = norm_df.iloc[:, i] + loc
    return denorm_df


# -----------------------------------------------------------
#                   String to numeric
# -----------------------------------------------------------
# def str2num(df, dataset_name=None):
#     df = df.copy()
#     for col in df.columns:
#         if df[col].dtype == 'object':
#             df[col] = df[col].astype('category').cat.codes
#     return df

def str2num(df, dataset_name=None, encoder=None):
    df_copy = df.copy()
    path = os.path.join(Path(sys.argv[0]).resolve().parent, 'labels')
    # create foldet if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, dataset_name)
    # create folder if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    cols = []
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            cols.append(col)
            # Initialize the LabelEncoder
    if encoder is None:
        label_encoder = OrdinalEncoder()
        label_encoder.fit(df_copy[cols])
    else:
        label_encoder = encoder
    # Fit and transform the data
    df_copy[cols] = label_encoder.transform(df_copy[cols])
    # Save the label encoder to a file
    joblib.dump(label_encoder, path + f'/label_encoder.joblib')
    # Now, you can use the encoded data in your machine learning model

    return df_copy, label_encoder, cols


def num2str(df, cols, encoder):
    if len(cols) > 0:
        df_copy = df.copy()
        df_copy = df_copy.fillna(0)
        df_copy[cols] = encoder.inverse_transform(df_copy[cols].astype(int))
        # Set nans again
        df_copy[df.isna()] = np.nan

        return df_copy
    else:
        return df


# -----------------------------------------------------------
#                   Sampling data
# -----------------------------------------------------------
def sample_group(group):
    return group.sample(1, random_state=1)


def sample_cat(df, cols, n):
    if n > df.shape[0]:
        print('There are not enough samples to sample ', n, ' samples. Returning all samples available.')
        return df
    elif n == 0 or n == len(df):  # Note: the last condition means that if n is the actual number of samples, do nothing
        return df
    sampled_df = pd.DataFrame()
    for feat in cols:
        sampled_dfi = (df.groupby([feat]).apply(sample_group))
        sampled_dfi = sampled_dfi.reset_index(drop=True)
        sampled_df = pd.concat([sampled_df, sampled_dfi])
    # samples that are in df but not in sampled_df
    df_dif = pd.merge(df, sampled_df, how='outer', indicator=True).query('_merge=="left_only"').drop('_merge', axis=1)
    if n - len(sampled_df) > 0 and n - len(sampled_df) <= len(df_dif):
        df_dif_sample = df_dif.sample(n=(n - len(sampled_df)), random_state=1)
        df_sample = pd.concat([sampled_df, df_dif_sample])
    elif n - len(sampled_df) > len(df_dif):
        print('WARNING: sampling must take repeated samples to achieve minimum number of samples required')
        df_dif_sample = df_dif.sample(n=(n - len(sampled_df)), random_state=1, replace=True)
        df_sample = pd.concat([sampled_df, df_dif_sample])
    else:
        df_sample = sampled_df.sample(n=n, random_state=1).reset_index(drop=True)
        print('WARNING: not enough samples to sample ', n, ' samples with all categories, minimum would be ', len(sampled_df),'. Returning ', n, ' samples.')
    return df_sample
