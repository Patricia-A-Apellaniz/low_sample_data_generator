import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import argparse
import yaml
# import easydict
import pathlib
import os
import logging
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
import seaborn as sns

def plot_gt(df, df_ci, title='KL', results_path='../results'):
    df1 = compute_error(df)
    df_ci = df_ci.reset_index(level=[0, 1, 2])
    # Drop columns
    df_ci = df_ci.drop(columns=[title+' Discriminator', title+' p/q'])
    # rename column
    df_ci.rename(columns={title+' MC': 'ci'}, inplace=True)
    # select unique values of n
    N = df1['n'].unique()
    df2 = df1.merge(df_ci, on=['n', 'l', 'm'])
    # drom columns
    df2 = df2.drop(columns=['m'])
    plt.errorbar(df2['n'], df2['error'], yerr=df2['ci'], fmt='o', capsize=5)

    plt.semilogx(df2['n'], df2['error'])
    plt.xlabel('m')
    plt.ylabel('Divergence MC')
    plt.title(f'Divergence MC. 100 realizations. L=50000')
    # plt.show()

    # Print error and confidence interval.
    print(f'Value and confidence interval for {title} divergence. 100 realizations. L=50000')
    for n in N:
        print(f'Divergence {title} for n={n}: {df2[df2["n"]==n]["error"].mean()} +- {df2[df2["n"]==n]["ci"].mean()}')


def compute_error(df):
    df1 = df.reset_index(level=[0, 1, 2])
    l_max = df1['l'].max()
    real = df.xs(l_max, level='l')
    real = real.reset_index(level=[0, 1])
    # group by n and average KL MC
    if 'KL MC' in real.columns:
        real = real.groupby(['n']).mean()['KL MC'].reset_index()
        # rename column
        real.rename(columns={'KL MC': 'real'}, inplace=True)
    elif 'JS MC' in real.columns:
        real = real.groupby(['n']).mean()['JS MC'].reset_index()
        # rename column
        real.rename(columns={'JS MC': 'real'}, inplace=True)
    else:
        raise ValueError('No MC column found')

    # select index named 10

    # merge with real
    df1 = df1.merge(real, on='n', how='left')
    if 'KL Discriminator' in df1.columns:
        df1['error'] = np.abs(df1['KL Discriminator'] - df1['real'])
        # df1['error'] = (df1['KL Discriminator'] - df1['real'])

    elif 'JS MC' in df1.columns:
        df1['error'] = np.abs(df1['JS Discriminator'] - df1['real'])
    else:
        raise ValueError('No Discriminator column found')
    return df1

def plot_value(df, df_std, title, results_path=''):
    df = df.reset_index(level=[0, 1, 2])
    df_std = df_std.reset_index(level=[0, 1, 2])
    # Drop columns
    df_std = df_std.drop(columns=[title+' MC', title+' p/q', 'val_seed'])
    # rename column
    df_std.rename(columns={title+' Discriminator': 'ci'}, inplace=True)
    # select unique values of n
    N = df['n'].unique()

    for n in N:
        df2 = df[df['n'] == n]
        df2_std = df_std[df_std['n']==n]
        # Create a line plot for each value of 'l'
        # merge dataframes
        df2 = df2.merge(df2_std, on=['n', 'l', 'm', 'name'])
        unique_l_values = df2['l'].unique()

        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

        for l_value in unique_l_values:
        # for l_value in [8, 80, 800, 8000]:
            df_filtered = df2[df2['l'] == l_value]
            # logarithmic x axis
            for m in df_filtered['m'].unique():
                df_filtered_m = df_filtered[df_filtered['m']==m]
                id = df_filtered_m['name'].str.split('_').str[-1]
                print(id)
                if 'ctgan' in str(id) or 'tvae' in str(id):
                    df_filtered_m['id'] = 0
                else:
                    df_filtered_m['id'] = (df_filtered_m['name'].str.split('_').str[-1]).astype('int')
                df_filtered_m = df_filtered_m.sort_values(by=['id'])
                plt.errorbar(df_filtered_m['name'], df_filtered_m[' '.join([title,'Discriminator'])],
                             yerr=df_filtered_m['ci'], fmt='o-', capsize=5)

        plt.xlabel('Seeds')
        plt.ylabel(f'{title} Estimation')
        plt.title(title + ' Estimation of different seeds for Different Values of M, L')
        fig_path = results_path + f'/{title}_estimation_vs_m_N_{n}.'
        plt.grid(True, alpha=0.1)
        plt.legend(df_filtered['m'].unique())
        # try:
        #     tikzplotlib.save(fig_path+'tex')
        # except:
        #     print('Error saving to tex')

        plt.savefig(fig_path+'png')

        # plt.savefig(results_path + f'/{title}_error_vs_m_N_{n}.png')
        # plt.show()


def plot_error_mc(df, df_std, title, results_path=''):
    df1 = compute_error(df)
    df_std = df_std.reset_index(level=[0, 1, 2])
    # Drop columns
    df_std = df_std.drop(columns=[title+' MC', title+' p/q'])
    # rename column
    df_std.rename(columns={title+' Discriminator': 'ci'}, inplace=True)
    # select unique values of n
    N = df1['n'].unique()
    df1 = df1[df1['l'] != 5000]
    df_std = df_std[df_std['l'] != 5000]
    # One plot for each value of n, with m on the x axis and error on the y axis.
    for n in N:
        df2 = df1[df1['n'] == n]
        df2_std = df_std[df_std['n']==n]
        # Create a line plot for each value of 'l'
        # merge dataframes
        df2 = df2.merge(df2_std, on=['n', 'l', 'm'])
        unique_l_values = df2['l'].unique()

        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

        for l_value in unique_l_values:
            df_filtered = df2[df2['l'] == l_value]
            plt.errorbar(df_filtered['m'], df_filtered['error'], yerr=df_filtered['ci'], fmt='o-', capsize=5)

        plt.xlabel('m')
        plt.ylabel('error')
        plt.title(title +' Error vs. m for Different Values of l. N=' + str(n))
        fig_path = results_path + f'/{title}_error_vs_m_N_{n}.'
        plt.grid(True, alpha=0.1)
        plt.legend(unique_l_values)
        # tikzplotlib.save(fig_path+'tex')

        plt.savefig(fig_path+'png')
        # plt.show()


def plot_error_real(df, df_std, title='KL', results_path='../results'):
    # Drop rows with l=5000

    df_std.rename(columns={'KL Discriminator': 'ci'}, inplace=True)
    df_std.reset_index(level=[0, 1, 2], inplace=True)

    real_col = title + ' real'
    disc_col = title + ' Discriminator'
    df['error'] = np.abs(df[real_col] - df[disc_col])
    df.reset_index(level=[0, 1, 2], inplace=True)
    df = df.merge(df_std[['n', 'l', 'm', 'ci']], on=['n', 'l', 'm'])
    df = df[df['l'] != 5000]
    N = df['n'].unique()
    for n in N:
        df2 = df[df['n'] == n]
        unique_l_values = df2['l'].unique()
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        for l_value in unique_l_values:
            # for l_value in [8, 80, 800, 8000]:
            df_filtered = df2[df2['l'] == l_value]
            plt.errorbar(df_filtered['m'], df_filtered['error'], yerr=df_filtered['ci'], capsize=5, fmt='o-',
                         ecolor=None)

        plt.xlabel('m')
        plt.ylabel('error')
        true = np.round(df2[real_col].max(), 3)
        plt.title(title + ' Error vs. m for Different Values of l. N=' + str(n) + ' Real divergence = '+str(true))
        plt.grid(True, alpha=0.1)
        fig_path = results_path + f'/{title}_error_vs_m_N_{n}.'
        plt.legend(unique_l_values)
        # tikzplotlib.save(fig_path+'tex')
        plt.savefig(fig_path+'png')
        # plt.show()
        plt.close()


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='ctangana',
        usage='main_2.py [<args>] [-h | --help]'
    )
    parser.add_argument('--config', default=os.path.join(pathlib.Path(__file__).parent.parent.parent, 'config/configuration.yaml'),
                        type=str)
    return parser.parse_args(args)


# def load_config(cfg_file):
#     with open(cfg_file, "r") as fin:
#         raw_text = fin.read()
#
#     configs_save = yaml.safe_load(raw_text)
#     configs = [easydict.EasyDict(configs_save)]
#     configs_save = [configs_save]
#
#     return configs, configs_save


def set_logger(save_path):
    log_file = os.path.join(save_path, 'run.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLog


def plt_tsne(x_real, x_gen, results_path='../results'):

    # check if x_real and x_gen are tensors
    if isinstance(x_real, torch.Tensor):
        # check device
        if x_real.device != 'cpu':
            x_real = x_real.cpu().detach()
            x_gen = x_gen.cpu().detach()
        x = torch.cat((x_real, x_gen), dim=0)
        y = torch.cat((torch.ones(x_real.shape[0]), torch.zeros(x_gen.shape[0])), dim=0)
    else:
        x_real = x_real.dropna()
        # concatenate dataframes
        x = pd.concat([x_real, x_gen], ignore_index=True)
        y = pd.concat([pd.Series(np.ones(x_real.shape[0])), pd.Series(np.zeros(x_gen.shape[0]))], ignore_index=True)

    # concatenate dataframes
    plt.close()
    tsne = TSNE(n_components=2, random_state=42, verbose=1, perplexity=5, n_iter=300)
    print('Computing TSNE')
    x_tsne = tsne.fit_transform(x)
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, alpha=0.5)
    # plt.legend()
    plt.title('TSNE representation of real and generated data')
    fig_path = results_path + '/tsne.'
    plt.legend()

    # tikzplotlib.save(fig_path+'tex')

    plt.savefig(fig_path+'png')
    # plt.show()
    plt.close()
    df = pd.DataFrame(x_tsne)
    df['label'] = y
    sns.histplot(df, x=0, hue='label')
    plt.title('TSNE representation of real and generated data')
    # plt.show()
    plt.close()
    sns.histplot(df, x=1, hue='label')
    plt.title('TSNE representation of real and generated data')
    # plt.show()
    plt.close()


def save_config(cfg, path,i=0):
    # check if path exists
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, 'config.yaml'), 'w') as fo:
        yaml.dump(dict(cfg[i]), fo)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.cuda.manual_seed(seed)


def prepare_result_folder(results_path, N, M, L, l_gt=1000, case='mvn', cfg=None):
    # Create folder for the results

    # check if the results path exists
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)
    else:
        print('Results path already exists')

    os.makedirs(results_path, exist_ok=True)
    # Create folder for each divergence
    os.makedirs(results_path+'/kl', exist_ok=True)
    os.makedirs(results_path+'/js', exist_ok=True)
    # Create a folder for each combination of parameters
    if case in cfg.synthetic_cases:
        L = [l_gt] + L
    for n in N:
        for m in M:
            for l in L:
                os.makedirs(results_path+'/kl'+f'/{n}_{m}_{l}', exist_ok=True)
                os.makedirs(results_path+'/js'+f'/{n}_{m}_{l}', exist_ok=True)


def present_results(results_path, case):
    div = 'kl'
    # Load results
    df = pd.read_csv(results_path + '/' + div + '.csv', index_col=[0, 1, 2])
    df = df.sort_index()
    df_std = pd.read_csv(results_path + '/' + div + '_std.csv', index_col=[0, 1, 2])
    if 'data' in case:
        a = 3
        plot_value(df, df_std, title=div.upper(), results_path=results_path)
        div = 'js'
        df = pd.read_csv(results_path + '/' + div + '.csv', index_col=[0, 1, 2])
        df = df.sort_index()
        df_std = pd.read_csv(results_path + '/' + div + '_std.csv', index_col=[0, 1, 2])
        plot_value(df, df_std, title=div.upper(), results_path=results_path)
    else:
        if 'name' in df.columns:
            df = df.drop(columns=['name', 'val_seed'])
            df_std = df_std.drop(columns=['name', 'val_seed'])
        if 'mvn' in case:
            plot_error_real(df, df_std, title=div.upper(), results_path=results_path)
        elif case == 'gt':
            plot_gt(df, df_std, title='KL', results_path=results_path)
        else:
            plot_error_mc(df, df_std, title=div.upper(), results_path=results_path)
        div = 'js'
        df = pd.read_csv(results_path + '/' + div + '.csv', index_col=[0, 1, 2])
        df = df.sort_index()
        df_std = pd.read_csv(results_path + '/' + div + '_std.csv', index_col=[0, 1, 2])
        if 'name' in df.columns:
            df = df.drop(columns=['name', 'val_seed'])
            df_std = df_std.drop(columns=['name', 'val_seed'])
        if case == 'gt':
            plot_gt(df, df_std, title='JS', results_path=results_path)
        else:
            plot_error_mc(df, df_std, title=div.upper(), results_path=results_path)


def store_results(results_folder, cfg=None):
    kl_folder = results_folder+'/kl'
    js_folder = results_folder+'/js'

    # list of elements in the folder
    param_list = os.listdir(kl_folder)

    df_kl = pd.DataFrame()
    df_js = pd.DataFrame()
    df_kl_std = pd.DataFrame()
    df_kl_ci = pd.DataFrame()
    df_js_std = pd.DataFrame()
    df_js_ci = pd.DataFrame()

    for param in tqdm(param_list):
        # List of elements in the folder
        kl_seeds = os.listdir(kl_folder+'/'+param)
        df_seed_kl = pd.DataFrame()
        df_seed_js = pd.DataFrame()
        # number of elements in kl seeds that contains .csv
        num_seeds = len([seed for seed in kl_seeds if '.csv' in seed])
        feat_dict = {}
        if num_seeds > 0:
            for seed in kl_seeds:
                if '.csv' in seed:
                    # merge dataframe
                    # read csv with header
                    df_i = pd.read_csv(kl_folder+'/'+param+'/'+seed)
                    df_j = pd.read_csv(js_folder+'/'+param+'/'+seed)
                    df_i['val_seed'] = seed.split('_')[1]
                    df_j['val_seed'] = seed.split('_')[1]
                    df_i['name'] = '_'.join(seed.split('_')[2:]).split('.csv')[0]
                    df_j['name'] = '_'.join(seed.split('_')[2:]).split('.csv')[0]
                    # concatenate dataframe df_i to df_seed
                    df_seed_kl = pd.concat([df_seed_kl, df_i])
                    df_seed_js = pd.concat([df_seed_js, df_j])
                    if cfg.print_feat_js:
                        feat = seed
                        js = df_j['JS Discriminator'][0]
                        feat_dict[feat] = js
                        print(f'JS for {feat}: {js}')

            # Sort dictionary by value
            feat_dict = {k: v for k, v in sorted(feat_dict.items(), key=lambda item: item[1])}
            print(feat_dict)
            # Promediate over seeds
            df_aux = df_seed_kl.copy()
            df_aux_js = df_seed_js.copy()

            if 'val_seed' in df_aux.columns:
                # # Keep best seed
                df_seed_kl = pd.DataFrame(df_aux.groupby(['n', 'm', 'l', 'name']).mean()).reset_index()
                df_seed_kl_std = pd.DataFrame(df_aux.groupby(['n', 'm', 'l', 'name']).std()).reset_index()
                df_seed_js = pd.DataFrame(df_aux_js.groupby(['n', 'm', 'l', 'name']).mean()).reset_index()
                df_seed_js_std = pd.DataFrame(df_aux_js.groupby(['n', 'm', 'l', 'name']).std()).reset_index()
            else:
                if len(df_seed_kl) < 1:
                    print('Empty dataframe')
                df_seed_kl = df_seed_kl.groupby(['n', 'm', 'l']).mean()
                df_seed_kl_std = df_aux.groupby(['n', 'm', 'l']).std()
                df_aux_js = df_seed_js.copy()
                df_seed_js = df_seed_js.groupby(['n', 'm', 'l']).mean()
                df_seed_js_std = df_aux_js.groupby(['n', 'm', 'l']).std()

            num_seeds = len(kl_seeds)
            z_score = norm.ppf(1 - 0.05 / 2)
            df_seed_kl_ci = df_seed_kl_std.copy()
            df_seed_kl_ci['KL Discriminator'] = z_score * (df_seed_kl_std['KL Discriminator'] / np.sqrt(num_seeds))

            df_kl = pd.concat([df_kl, df_seed_kl])
            if len(df_kl)<1:
                print('Empty dataframe')
            df_kl_std = pd.concat([df_kl_std, df_seed_kl_std])
            df_kl_ci = pd.concat([df_kl_ci, df_seed_kl_ci])

            # Plot the results using confidence intervals
            df_seed_js_ci = df_seed_js_std.copy()
            df_seed_js_ci['JS Discriminator'] = z_score * (df_seed_js_std['JS Discriminator'] / np.sqrt(num_seeds))

            df_js = pd.concat([df_js, df_seed_js])
            df_js_std = pd.concat([df_js_std, df_seed_js_std])
            df_js_ci = pd.concat([df_js_ci, df_seed_js_ci])
        else:
            print(f'No results for this configuration {param}')
    # Sort df by index
    df_kl = df_kl.sort_index()
    df_kl_std = df_kl_std.sort_index()
    df_kl_ci = df_kl_ci.sort_index()
    # df_js = df_js.sort_index()
    # Save to csv
    df_kl.to_csv(results_folder+'/kl.csv', index=False)
    df_kl_std.to_csv(results_folder+'/kl_std.csv', index=False)
    df_kl_ci.to_csv(results_folder+'/kl_ci.csv', index=False)

    # Sort df by index
    df_js = df_js.sort_index()
    df_js_std = df_js_std.sort_index()
    df_js_ci = df_js_ci.sort_index()
    # Save to csv
    df_js.to_csv(results_folder+'/js.csv', index=False)
    df_js_std.to_csv(results_folder+'/js_std.csv', index=False)
    df_js_ci.to_csv(results_folder+'/js_ci.csv', index=False)