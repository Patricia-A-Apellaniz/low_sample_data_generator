import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from divergence_estimation.gmm_model import GMM
from divergence_estimation.utils import set_seed
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVR
from torch import distributions as D


def plot_histogram_per_class(df):
    """
    Plots histogram of each feature in df assuming it includes a feature
    called 'target' to identify real and generated data
    Args:
        df: dataframe with the features and a column indicating if the sample is real or synthetic

    Returns:

    """
    # Function to plot histograms of each feature per class. Identify problems in marginal distributions.
    # Feature 'target' identifies if the sample is real or synthetic.
    # Get unique class labels
    class_labels = df['target'].unique()

    # Define a list of colors for each class
    colors = sns.color_palette("husl", len(class_labels))

    # Iterate over columns and create a single figure for each column
    columns = df.columns
    columns = columns.drop('target')
    for column in columns:  # Exclude the 'Class' column
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x=column, hue="target", kde=False, bins=30, palette='bright')
        # sns.boxplot(x=df['target'], y=df[column])
        plt.title(f'{column} Histogram by Class')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        # plt.legend()
        plt.show()


def create_independent_gm(component_probs=torch.tensor([0.7, 0.3]), loc=torch.tensor([[1., 1.], [-1., -1.]]),
                          scale=torch.tensor([.5, .5])):
    """Creates an independent (isotropic) mixture of Gaussians and returns it

    Args:
        component_probs (torch.Tensor, optional): The categorical probabilities. Defaults to torch.tensor([0.7, 0.3]).
        loc (torch.Tensor, optional): The location of each Gaussian component. Defaults to torch.tensor([[1., 1.], [-1., -1.]]).
        scale (torch.Tensor, optional): The scale of each Gaussian component. Defaults to torch.tensor([.5, .5]).

    Returns:
        torch.distributions.Distribution: The Gaussian mixture model
    """
    categorical = D.Categorical(probs=component_probs)
    components = D.Independent(
        D.Normal(loc=loc, scale=scale),
        1
    )
    gm = D.MixtureSameFamily(categorical, components)
    return gm


def create_corr_bimodal_gm():
    """Creates a correlated bimodal mixture of Gaussians and returns it
    Args:
        component_probs (torch.Tensor, optional): The categorical probabilities. Defaults to torch.tensor([0.7, 0.3]).
        loc (torch.Tensor, optional): The location of each Gaussian component. Defaults to torch.tensor([[1., 1.], [-1., -1.]]).
        scale (torch.Tensor, optional): The scale of each Gaussian component. Defaults to torch.tensor([.5, .5]).
    Returns:
        torch.distributions.Distribution: The Gaussian mixture model
    """
    categorical = D.Categorical(probs=torch.tensor([0.7, 0.3]))
    components = D.MultivariateNormal(
        loc=torch.Tensor([[1., 1.], [-1., -1.]]),
        covariance_matrix=torch.Tensor([
            [[1., .2],
             [.2, 1.]],
            [[1., .2],
             [.2, 1.]]
        ])
    )
    gm = D.MixtureSameFamily(categorical, components)
    return gm


def load_mvn(n_dims=5, dist=1):
    """
    Generate a multivariate normal distribution with n_dims dimensions and a distance of dist between the means of
    each dimension.
    Args:
        n_dims: number of dimensions
        dist:  distance between the means of each dimension

    Returns:

    """
    cov = np.identity(n_dims).astype("float32")
    choices = [0.3, 0.5, 0.7]
    for row in range(n_dims):
        for col in range(n_dims):
            if row == col:
                cov[row, col] = 10.  # queda como 1. después de la división de más abajo
            else:
                cov[row, col] = np.random.choice(choices)
    cov /= 10.
    cov = cov @ cov.T
    locs = torch.rand((n_dims,)) * dist  # To generate random means.
    mvn = torch.distributions.MultivariateNormal(loc=locs, covariance_matrix=torch.tensor(cov))
    return mvn


def str2num(df):
    df_copy = df.copy()
    cols = []
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            cols.append(col)
    # Initialize the LabelEncoder
    label_encoder = OrdinalEncoder()
    label_encoder.fit(df_copy[cols])

    # Fit and transform the data
    df_copy[cols] = label_encoder.transform(df_copy[cols])

    return df_copy, cols

def get_feat_distributions(df, cols_cat=[]):
    # TODO: mirar esto!!
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
def normalize_data(raw_df, feat_distributions, gen_df=None, cfg=None):
    num_patient, num_feature = raw_df.shape
    norm_df = raw_df.copy()
    if gen_df is not None:
        gen_norm_df = gen_df.copy()
    else:
        gen_norm_df = None
    for i in range(num_feature):
        col = (raw_df.columns[i])
        values = raw_df.iloc[:, i]
        if gen_df is not None:
            values_gen = gen_df.iloc[:, i]
        if len(values) == 1 and math.isnan(values[0]):
            values = np.zeros((1,))
        no_nan_values = values[~np.isnan(values)].values
        if gen_df is not None:
            no_nan_values_gen = values_gen[~np.isnan(values_gen)].values
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
        elif feat_distributions[i][0] == 'weibull':
            loc = -1 if 0 in no_nan_values else 0
            scale = 1
        else:
            print('Distribution ', feat_distributions[i][0], ' not normalized')
            param = np.array([0, 1])  # loc = 0, scale = 1, means that data is not modified!!
            loc = param[-2]
            scale = param[-1]
        norm_df.iloc[:, i] = (raw_df.iloc[:, i] - loc) / scale if scale != 0 else raw_df.iloc[:, i] - loc
        if gen_df is not None:
            gen_norm_df.iloc[:, i] = (gen_df.iloc[:, i] - loc) / scale if scale != 0 else gen_df.iloc[:, i] - loc
            gen_norm_df = gen_norm_df.reset_index(drop=True)

    return norm_df.reset_index(drop=True), gen_norm_df


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
            if len(values) == 1:
                values = np.zeros((1,))
            no_nan_values = values[~pd.isnull(values)]
            if no_nan_values.size <= 2 or no_nan_values.dtype in [object, str] or np.amin(
                    np.equal(np.mod(no_nan_values, 1), 0)):
                stats_value = stats.mode(no_nan_values, keepdims=True)[0][0]
            else:
                mean_value = no_nan_values.mean()
                # # Round to number of decimals of the original data
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


def preprocess_data(data, df_gen, cfg):
    # Strings to numbers
    data, cols = str2num(data)
    if df_gen is not None:
        df_gen = str2num(df_gen)
    # plt_tsne(data, df_gen)
    # Normalise data
    feat_distributions = get_feat_distributions(data, cols)
    if cfg.normalise:
        norm_df, gen_norm_df = normalize_data(data, feat_distributions, df_gen, cfg=cfg)
        # plt_tsne(norm_df, gen_norm_df)
    else:
        norm_df = data.copy()
        gen_norm_df = df_gen.copy()

    # Remove categorical features
    if cfg.remove_cat:
        # check feat dristribution and remove categorical features
        for i, col in enumerate(norm_df.columns):
            if feat_distributions[i][0] == 'categorical':
                norm_df = norm_df.drop(columns=[col])
                if df_gen is not None:
                    gen_norm_df = gen_norm_df.drop(columns=[col])

    # Impute missing data
    imp_norm_df, mask, _ = impute_data(norm_df, gen_mask=False,
                                       feat_distributions=feat_distributions)
    # One hot encoding
    if cfg.ohe:
        # One hot encoding
        feat_distributions = get_feat_distributions(imp_norm_df, cols=cols)
        for i, col in enumerate(imp_norm_df.columns):
            if feat_distributions[i][0] == 'categorical':
                # One hot encoding
                index1 = imp_norm_df.index
                enc = OneHotEncoder(handle_unknown='ignore')
                encoded1 = enc.fit_transform(imp_norm_df[col].values.reshape(-1, 1)).toarray()
                encoded1 = encoded1.astype('int64')
                columns = [col + '_' + str(i) for i in range(encoded1.shape[1])]
                df1 = pd.DataFrame(encoded1, columns=columns, index=index1)
                imp_norm_df = pd.concat([imp_norm_df, df1], axis=1)
                imp_norm_df = imp_norm_df.drop(columns=[col])
                if df_gen is not None:
                    index2 = gen_norm_df.index
                    encoded2 = enc.transform(gen_norm_df[col].values.reshape(-1, 1)).toarray()
                    encoded2 = encoded2.astype('int64')
                    columns = [col + '_' + str(i) for i in range(encoded2.shape[1])]
                    df2 = pd.DataFrame(encoded2, columns=columns, index=index2)
                    gen_norm_df = pd.concat([gen_norm_df, df2], axis=1)
                    gen_norm_df = gen_norm_df.drop(columns=[col])

    return imp_norm_df, gen_norm_df


def plot_scatter(df, df_gen, results_path):
    df = df.sample(100)
    df_gen = df_gen.sample(100)
    df_aux = pd.concat([df, df_gen], axis=0)
    df_aux['target'] = np.concatenate([np.ones((df.shape[0],)), np.zeros((df_gen.shape[0],))])
    # drop columns with only one value
    df_aux = df_aux.loc[:, df_aux.apply(pd.Series.nunique) != 1]
    print('plotting pairplot')
    # sample
    pair_plot = sns.pairplot(df_aux, hue="target", diag_kind='kde')
    # Save the figure
    pair_plot.savefig(results_path + "/pairplot.png")
    # tikzplotlib.save(results_path + 'pairplot.tex')
    print('pairplot saved')


def get_data(case='mvn', folder=None, seed=42, m=100, l=100, n=1000, cfg=None, results_path=None, current_gen_model=None):
    if case == 'mvn':
        set_seed(42)
        pr = load_mvn(10, dist=cfg.dist)
        ps = load_mvn(10, dist=0)
        set_seed(int(seed * 2))
        x_real = pr.sample(torch.Size((m + (2 * l),))).clone().detach()
        x_real_train = x_real[0:m, :]
        x_real_eval = x_real[m:, :]
        # x_real_train = pr.sample(torch.Size((m,))).clone().detach()
        # x_real_eval = pr.sample(torch.Size((l * 2,))).clone().detach()
        # x_real = torch.cat((x_real_train, x_real_eval), dim=0)
        x_gen = ps.sample(torch.Size((m + (2 * l),))).clone().detach()
        x_gen_train = x_gen[0:m, :]
        x_gen_eval = x_gen[m:, :]
        # x_gen_train = ps.sample(torch.Size((m,))).clone().detach()
        # x_gen_eval = ps.sample(torch.Size((l * 2,))).clone().detach()
        # x_gen = torch.cat((x_gen_train, x_gen_eval), dim=0)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        # Tensor to device
        x_real = x_real.to(device)
        x_gen = x_gen.to(device)
        col_names = []
        return x_real, x_gen, pr, ps, col_names

    elif case == 'gm_dif':
        set_seed(42)
        # Generate data
        pr = create_independent_gm()
        ps = create_independent_gm(component_probs=torch.tensor([0.5, 0.5]), loc=torch.tensor([[0., 0.], [-1., -1.]]),
                                   scale=torch.tensor([.5, .5]))

        # Sample to generate m samples
        set_seed(seed)
        # x_real = pr.sample(torch.Size((m + (2 * l),))).clone().detach()
        # x_real_train = x_real[0:m, :]
        # x_real_eval = x_real[m:, :]
        x_real_train = pr.sample(torch.Size((m,))).clone().detach()
        x_real_eval = pr.sample(torch.Size((l * 2,))).clone().detach()
        x_real = torch.cat((x_real_train, x_real_eval), dim=0)

        # x_gen = ps.sample(torch.Size((m + (2 * l),))).clone().detach()
        # x_gen_train = x_gen[0:m, :]
        # x_gen_eval = x_gen[m:, :]

        x_gen_train = ps.sample(torch.Size((m,))).clone().detach()
        x_gen_eval = ps.sample(torch.Size((l * 2,))).clone().detach()
        x_gen = torch.cat((x_gen_train, x_gen_eval), dim=0)

        # to gpu
        # set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        # Dataframe to tensor
        x_real = x_real.to(device)
        x_gen = x_gen.to(device)
        col_names = []
        return x_real, x_gen, pr, ps, col_names

    elif case == 'gmm':
        set_seed(42)
        # Generate data
        pr = create_corr_bimodal_gm()
        # x = torch.tensor(pr.sample(torch.Size(((n+m+(2*l)),))))
        # # x_n are the first n samples of x
        # x_n = x[0:n, :]
        # # x_m are the next m samples of x
        # x_m = x[n:(n+m), :]
        # # x_l are the rest of the samples of x
        # x_l = x[(n+m):, :]
        x_n = torch.tensor(pr.sample(torch.Size(((n,)))))
        # Fit GMM
        ps = GMM(n_components=2, random_state=23)
        ps.fit(x_n)
        # Sample to generate m samples
        set_seed(seed)
        x_real_train = pr.sample(torch.Size((m,))).clone().detach()
        x_real_eval = pr.sample(torch.Size((l * 2,))).clone().detach()
        x_real = torch.cat((x_real_train, x_real_eval), dim=0)
        # x_real_train = x_m.clone().detach()
        # x_real_eval = x_l.clone().detach()
        # x_real = torch.cat((x_real_train, x_real_eval), dim=0)

        # set_seed(42)
        x_gen, _ = ps.sample((m))
        x_gen1 = x_gen[0:m, :]
        x_gen1 = torch.tensor(x_gen1, dtype=torch.float32)
        # set_seed(42)
        x_gen2, _ = ps.sample(2 * l)
        # x_gen2 = x_gen[m:, :]
        x_gen2 = torch.tensor(x_gen2, dtype=torch.float32)
        x_gen = torch.cat((x_gen1, x_gen2), dim=0)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        # Tensor to device
        x_real = x_real.to(device)
        x_gen = x_gen.to(device)
        col_names = []
        return x_real, x_gen, pr, ps, col_names

    elif case == 'data':
        # Check if folder exists or is empty
        if folder is None or not os.path.exists(folder) or os.listdir(folder) == []:
            print('Error: folder is not valid.')
            return None

        if current_gen_model is not None: # Added because cfg.gen_model may be a list
            if current_gen_model == 'ctgan' or current_gen_model == 'tvae':
                folder = os.path.join(folder, current_gen_model)
        elif cfg.gen_model == 'ctgan' or cfg.gen_model == 'tvae':
            folder = os.path.join(folder, cfg.gen_model)

        # Load data
        if cfg.separate_sets:    # Data used to validate is different from data used to generate
            dataset_name = '_'.join(folder.split('output')[1].split('/')[1].split('_')[:-1])
            # Check if data_full.csv exists
            if os.path.exists(os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'data', 'processed_data',
                              dataset_name, 'data_full.csv')):  # TODO: check this path, see if we can get rid of it
                full_file = os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'data', 'processed_data',
                                         dataset_name, 'data_full.csv')
                print('IMPORTANTE, ESTOY LEYENDO EL CSV CON TODOS LOS DATOS')
            else:
                full_file = os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'data', 'processed_data',
                                         dataset_name, 'preprocessed_data.csv')
            df_data = pd.read_csv(full_file, nrows=50000)
            print('IMPORTANTE: shape of df_data: ', df_data.shape)
            df_data = df_data.dropna()  # TODO: check why there are Nans in here!
            if 'is_real' in df_data.columns:  # TODO: check this also
                df_data = df_data[
                    df_data['is_real'] == 0]  # En la fase 2, los datos generados son los que se tratan como reales.
                df_data = df_data.drop(columns=['is_real'])
            df_train = pd.read_csv(os.path.abspath(os.path.join(folder, 'real_data.csv')))
            df = pd.merge(df_data, df_train, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge',
                                                                                                              axis=1)  # TODO: check the warning here
            print('shape of df: ', df.shape)
        else: # Data used to validate is the same as data used to generate
            real_file = os.path.abspath(os.path.join(folder, 'real_data.csv'))
            df = pd.read_csv(real_file)

        if cfg.evaluate_reconstructed:
            gen_file = os.path.join(folder, 'rec_data.csv')
        else:
            gen_file = os.path.join(folder, 'gen_data.csv')

        # remove nans
        df = df.dropna()

        print('shape of df after droping nan: ', df.shape)
        df_gen = pd.read_csv(gen_file)
        col_names = df.columns
        # Drop from df any column that is not in df_gen
        df = df[df_gen.columns]
        n_real, n_syn = df.shape[0], df_gen.shape[0]
        print('n_syn: ', n_syn)
        print('n_real: ', n_real)

        n_samples_available = min((n_syn, n_real))
        n_samples = m + (2 * l)
        if n_samples_available < n_samples:
            print(f'Warning: only {n_samples_available} samples available.')
            n_samples = n_samples_available

        real_df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        synthetic_df = df_gen.sample(frac=1, random_state=0).reset_index(drop=True)
        real_df = real_df[0: n_samples]
        synthetic_df = synthetic_df[0: n_samples]

        # TODO: muy importante que esto se esté haciendo bien.
        '''
        if cfg.normalise:
            if cfg.combined_scaling:
                x = pd.concat([real_df, synthetic_df], axis=0)
                y = np.concatenate([np.ones((real_df.shape[0],)), np.zeros((synthetic_df.shape[0],))])
                x, _ = preprocess_data(x, None, cfg)
                real_df = x[y == 1]
                synthetic_df = x[y == 0]
            else:
                real_df, synthetic_df = preprocess_data(df, df_gen, cfg)

        col_names = [col_names, real_df.columns]
        
        if cfg.plot_scatter:
            plot_scatter(real_df, synthetic_df, results_path)
        
        if cfg.remove_bad_features:
            if 'news' in folder:
                real_df = real_df.drop(columns=['title_subjectivity', 'title_sentiment_polarity',
                                                'abs_title_subjectivity', 'abs_title_sentiment_polarity'])
                synthetic_df = synthetic_df.drop(columns=['title_subjectivity', 'title_sentiment_polarity',
                                                          'abs_title_subjectivity', 'abs_title_sentiment_polarity'])
            elif 'credit' in folder:
                real_df = real_df.drop(columns=['V23', 'V18', 'V4'])
                synthetic_df = synthetic_df.drop(columns=['V23', 'V18', 'V4'])

            elif 'intrusion' in folder:
                real_df = real_df.drop(columns=['srv_count'])
                synthetic_df = synthetic_df.drop(columns=['srv_count'])
                real_df = real_df.drop(columns=['dst_host_count'])
                synthetic_df = synthetic_df.drop(columns=['dst_host_count'])
            elif 'king' in folder:
                real_df = real_df.drop(columns=['sqft_living', 'lat'])
                synthetic_df = synthetic_df.drop(columns=['sqft_living', 'lat'])
            elif 'adult' in folder:
                real_df = real_df.drop(columns=['hours-per-week'])
                synthetic_df = synthetic_df.drop(columns=['hours-per-week'])
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        # Dataframe to tensor
        if isinstance(real_df, pd.DataFrame):
            real_df = real_df.values
            synthetic_df = synthetic_df.values
        df_ml = torch.tensor(real_df, dtype=torch.float32, device=device)
        df_gen = torch.tensor(synthetic_df, dtype=torch.float32, device=device)
        return df_ml, df_gen, None, None, col_names

    elif case == 'data_check_overfitting':
        # This case computes the divergence between two samples of the same distribution. It is used to check if the
        # divergence estimator is overfitting
        real_file = os.path.abspath(os.path.join(folder, 'data.csv'))
        df = pd.read_csv(real_file)
        df = df.sample(n=50000, random_state=0)
        df_aux = df.copy()
        df = df.sample(frac=0.5, random_state=0).reset_index(drop=True)
        df_gen = df_aux[~df_aux.isin(df)].dropna().reset_index(drop=True)
        col_names = df.columns
        df, df_gen = preprocess_data(df, df_gen, cfg)
        # plt_tsne((df), (df_gen))
        n_syn, n_real = df.shape[0], df_gen.shape[0]

        n_samples_available = min((n_syn, n_real))
        n_samples = m + (2 * l)
        if n_samples_available < n_samples:
            print(f'Warning: only {n_samples_available} samples available.')
            n_samples = n_samples_available

        real_df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        synthetic_df = df_gen.sample(frac=1, random_state=0).reset_index(drop=True)
        real_df = real_df[0: n_samples]
        synthetic_df = synthetic_df[0: n_samples]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        # Dataframe to tensor
        df_ml = torch.tensor(real_df.values, dtype=torch.float32, device=device)
        df_gen = torch.tensor(synthetic_df.values, dtype=torch.float32, device=device)
        return df_ml, df_gen, None, None, col_names
