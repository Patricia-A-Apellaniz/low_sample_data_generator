# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 05/09/2023
# https://github.com/sdv-dev/CTGAN
# Modeling Tabular data using Conditional GAN (https://arxiv.org/abs/1907.00503)


# Packages to import
import os
import sys

import pandas as pd

sys.path.insert(0, os.getcwd())

from colorama import Fore, Style
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

def train(real_df, metadata, output_dir, model_name, n_gen=1000, use_pretrained=False, pretrained_dir=None, params=None):

    # Create a synthesizer
    if params is not None:  # Params are for the TVAE only (for fair comparison with the VAE)
        if len(params) > 1:
            print('WARNING: The model ' + model_name + ' will train only with the first parameter combination')
        params = params[0]
        if model_name == 'ctgan':
            synthesizer = CTGANSynthesizer(metadata, epochs=1000, cuda=True, verbose=False)
        elif model_name == 'tvae':
            synthesizer = TVAESynthesizer(metadata, epochs=1000, cuda=True, embedding_dim=params['latent_dim'], compress_dims=[params['hidden_size']], decompress_dims=[params['hidden_size']])
        else:
            raise RuntimeError('[ERROR] State-of-the-art model not recognized')
    else:  # Use the default params of the models
        if model_name == 'ctgan':
            synthesizer = CTGANSynthesizer(metadata, epochs=1000, cuda=True, verbose=False)
        elif model_name == 'tvae':
            synthesizer = TVAESynthesizer(metadata, epochs=1000, cuda=True)
        else:
            raise RuntimeError('[ERROR] State-of-the-art model not recognized')

    # If pretrain selected, pretrain
    if use_pretrained:
        synthesizer.load(pretrained_dir + '/' + model_name + '_synthesizer.pkl')
        print(f"Model {model_name} pretrained loaded")

    # Train synthesizer
    synthesizer.fit(real_df)

    # Save synthesizer
    synthesizer.save(output_dir + model_name + '_synthesizer.pkl')

    # Generate samples
    synthetic_df = synthesizer.sample(num_rows=n_gen)

    # Save dataframe
    synthetic_df.to_csv(output_dir + 'gen_data.csv', index=False)


def main(args=None):
    print('\n\n-------- SYNTHETIC DATA GENERATION - STATE OF THE ART MODELS --------')

    # Environment configuration
    dataset_name = args['dataset_name']
    print('\n\nDataset: ' + Fore.CYAN + dataset_name + Style.RESET_ALL)

    # Sample dataset according to the number of training samples
    model = args['model']
    print('\nModel: ' + Fore.RED + model + Style.RESET_ALL)
    output_dir = args['output_dir'] + '/'
    real_df = pd.read_csv(output_dir + 'real_data.csv')

    # Train model
    if args['train']:
        train(real_df, args['metadata']['metadata'], output_dir, model, n_gen=args['generated_samples'], use_pretrained=args['use_pretrained'], pretrained_dir=args['pretrained_dir'], params=args['param_comb'])

