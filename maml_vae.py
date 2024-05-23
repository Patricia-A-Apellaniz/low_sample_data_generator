# Code adapted from https://github.com/kclip/bayesian_active_meta_learning
import numpy as np
import datetime
from copy import deepcopy
from tqdm import tqdm

# Machine learning packages
from torch import nn
import torch
import torch.optim as optim

from gen_model.data_generation.generator import Generator  # Base VAE model to meta-train
from gen_model.base_model.vae_utils import get_dim_from_type, get_activations_from_types
from gen_model.base_model.vae_modules import LatentSpaceGaussian, LogLikelihoodLoss


class Encoder_Meta(nn.Module):  # Meta-model for VAE Encoder
    # IMPORTANT NOTE: MODULE DESIGNED TO MATCH THE VAE USED IN THE CODE, NOT A GENERAL VAE!!
    def __init__ (self, grad_clip=1000.0, latent_limit=10.0):
        super(Encoder_Meta, self).__init__() # no need to initialize nn since we are given with parameter list
        self.grad_clip = grad_clip
        self.latent_limit = latent_limit  # To limit the latent space values
        self.L = 2  # Encoder is limited to two layers

    def forward(self, net_in, net_params): # net_in is the input and output. net_params is the nn parameters
        for ll in range(self.L):
            curr_layer_weight = net_params[2*ll  ]
            curr_layer_bias = net_params[2*ll+1]
            net_in = torch.nn.functional.linear(net_in, curr_layer_weight, curr_layer_bias)
            if net_in.requires_grad:
                net_in.register_hook(lambda x: x.clamp(min=-self.grad_clip, max=self.grad_clip))
            if ll < self.L-1: # inner layer has ReLU activation, last layer tanh
                net_in = torch.nn.functional.relu(net_in)
            else:
                net_in = torch.nn.functional.tanh(net_in) * self.latent_limit
        return net_in


class Decoder_Meta(nn.Module):  # Meta-model for VAE Decoder
    def __init__(self, feat_dists, max_k=10000.0, dropout_p=0.2, hidden_layers=2):
        super(Decoder_Meta, self).__init__()
        self.L = hidden_layers
        self.feat_dists = feat_dists
        self.out_dim = get_dim_from_type(self.feat_dists)
        self.max_k = max_k
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, net_in, net_params):
        for ll in range(self.L):
            curr_layer_weight = net_params[2*ll  ]
            curr_layer_bias = net_params[2*ll+1]
            net_in = torch.nn.functional.linear(net_in, curr_layer_weight, curr_layer_bias)
            if ll < self.L-1: # inner layer has ReLU activation, last layer depends on the activation of the feature
                net_in = torch.nn.functional.relu(net_in)
                net_in = self.dropout(net_in)
            else:
                net_in = get_activations_from_types(net_in, self.feat_dists, max_k=self.max_k)
        return net_in


class VAE_Meta(nn.Module):  # Variational Autoencoder for meta-learning

    def __init__(self, feat_distributions, latent_dim):
        super(VAE_Meta, self).__init__()
        self.feat_distributions = feat_distributions
        self.latent_dim = latent_dim
        self.encoder = Encoder_Meta()
        self.decoder = Decoder_Meta(feat_distributions)
        self.latent_space = LatentSpaceGaussian(self.latent_dim)  # Latent space
        self.rec_loss = LogLikelihoodLoss(self.feat_distributions)

    def forward(self, net_in, net_params):
        encoder_params = net_params[0: 2 * self.encoder.L]
        decoder_params = net_params[2 * self.encoder.L:]
        # Encoder
        latent_output = self.encoder(net_in, encoder_params)
        latent_params = self.latent_space.get_latent_params(latent_output)
        z = self.latent_space.sample_latent(latent_params)  # Sample the latent space using the reparameterization trick
        # Decoder
        out_params = self.decoder(z, decoder_params)
        out = {'z': z, 'cov_params': out_params, 'latent_params': latent_params}
        return out

    def criterion(self, net_out, target):
        loss_kl = self.latent_space.kl_loss(net_out['latent_params'])
        loss_cov = self.rec_loss(net_out['cov_params'], target, torch.ones_like(target))
        return loss_kl + loss_cov


class MyDataSet(torch.utils.data.Dataset):

    # Constructor
    def __init__(self, dDataset):
        self.x = dDataset['x']  # Inputs
        self.y = dDataset['y']  # Labels
        self.len = len(self.x)

    # Getter
    def __getitem__( self, index):
        return self.x[index], self.y[index]

    # Get length
    def __len__(self):
        return self.len

    def split_into_two_subsets( self, numSamFirstSubSet, bShuffle=True, seed=0):  # including shuffling
        N = self.len
        N0 = numSamFirstSubSet
        N1 = N - N0
        if N0==N:
            return (MyDataSet({'y': self.y, 'x': self.x  }), MyDataSet({'y': [], 'x': []}))
        elif N1==N:
            return (MyDataSet({'y': [], 'x': []}), MyDataSet({'y': self.y, 'x': self.x }))
        else:
            if bShuffle:
                g = torch.Generator()
                g.manual_seed(seed)
                perm = torch.randperm(N, generator=g)
            else:
                perm = torch.arange(0, N)
            return (MyDataSet({'x': self.x[perm[:N0],:], 'y': self.y[perm[:N0]]}),
                    MyDataSet({'x': self.x[perm[-N1 :],:], 'y': self.y[perm[-N1 :]]}))


def FreqMamlMetaTrain(net_init, # instance of nnModule
                      data_set_meta_train,
                      net_meta_intermediate,
                      lr_inner=0.1,  # Inner loop learning rate
                      lr_outer=0.1,  # Outer loop learning rate
                      tasks_batch_size_mtr=20,  # Number of tasks in a batch for meta training, use None for using all tasks
                      numSgdSteps=1,  # Adaptation steps in inner loop
                      numWays=4,  # Tasks points for training
                      waysShuffle=True,  # Whethet to shuffle the ways or using always the first numWays of the task for training
                      numMetaTrainingIter=10000,
                      bFirstOrderApprox=False,  # TRUE = FOMAML; FALSE = MAML with Hessians
                      patience=10,
                      seed=0, # for reproducibility
                      verbose=1):

    startTime = datetime.datetime.now()
    minimal_training_loss_adapt = +np.inf
    xi_adapt = deepcopy(net_init)  # Best net so far
    xi = deepcopy(net_init)  # Net used for meta-training
    optimizer_meta = torch.optim.Adam(xi.parameters(), lr=lr_outer)  # Outer loop optimizer (meta-optimizer)
    vMetaTrainingLossXi = []
    vMetaTrainingLossAdapt = []
    vMetaTrainingAdaptWithUpdates = []
    torch.manual_seed(seed)  # for reproducibility

    patience_counter = 0

    for meta_training_iter in range(numMetaTrainingIter):
        if verbose > 1: print(f"Starting Meta-training iter {meta_training_iter}/{numMetaTrainingIter}; time elapsed { datetime.datetime.now() - startTime }; time per iter {(datetime.datetime.now() - startTime)/(meta_training_iter + 1)}")

        if tasks_batch_size_mtr is None:  # use all tasks
            tasks_batch_size_mtr = len(data_set_meta_train)

        tasks_idx = torch.randperm(len(data_set_meta_train)).numpy()
        lD_val = []
        lengths_lD_tr = torch.zeros(tasks_batch_size_mtr)
        lengths_lD_val = torch.zeros(tasks_batch_size_mtr)
        vLoss_val = torch.zeros(tasks_batch_size_mtr)

        for batch_idx in range(len(data_set_meta_train) // tasks_batch_size_mtr):
            tasks_batch = tasks_idx[batch_idx * tasks_batch_size_mtr: (batch_idx + 1) * tasks_batch_size_mtr]
            xi.zero_grad()
            xi_params_list = list(map(lambda p: p[0], zip(xi.parameters())))
            for w in xi.parameters():
                w.total_grad = torch.zeros_like(w)

            for task_ind, tau in enumerate(tasks_batch):
                xi.zero_grad()
                # Obtain the meta training and validation sets
                [D_tau_tr, D_tau_val] = data_set_meta_train[tau].split_into_two_subsets(numWays, waysShuffle)
                lengths_lD_tr[task_ind] = len(D_tau_tr)
                lengths_lD_val[task_ind] = len(D_tau_val)
                lD_val.append(D_tau_val)
                # local update, stemming from xi
                phi_tau_i_params_list = xi_params_list # initialization, to be reassigned on later updates
                for i in range(numSgdSteps): # m sgd steps over phi_tau_i
                    net_out = net_meta_intermediate(D_tau_tr.x, phi_tau_i_params_list)# locally updated phi_tau
                    local_loss = net_meta_intermediate.criterion(net_out, D_tau_tr.y)
                    local_grad = torch.autograd.grad(local_loss, phi_tau_i_params_list, create_graph= not bFirstOrderApprox) # create_graph= {False: FOMAML , TRUE: MAML}
                    phi_tau_i_params_list = list(map(lambda p: p[1] - lr_inner * p[0], zip(local_grad, phi_tau_i_params_list))) # Note that we always update the inner loop using SGD
                # calculate grad needed for meta-update
                net_out = net_meta_intermediate(D_tau_val.x, phi_tau_i_params_list)
                meta_loss = net_meta_intermediate.criterion(net_out, D_tau_val.y)
                vLoss_val[task_ind] = meta_loss.detach().clone() * lengths_lD_val[task_ind]
                meta_grad = torch.autograd.grad(meta_loss, xi_params_list, create_graph=False)
                # accumulate meta-gradients w.r.t. tasks_batch
                for w,g in zip(xi.parameters(), meta_grad):
                    w.total_grad += g.detach().clone()
            loss_adapt = 1 / torch.sum(lengths_lD_val) * torch.sum(vLoss_val)
            vMetaTrainingLossAdapt.append(loss_adapt.detach().clone()) # make a log of the meta-training loss
            if (loss_adapt <= minimal_training_loss_adapt):  # new loss is better
                if verbose > 1: print(f"iter {meta_training_iter}/{numMetaTrainingIter}: updating xi_adapt having lowest loss {loss_adapt}")
                minimal_training_loss_adapt = loss_adapt.clone()  # update loss
                xi_adapt = deepcopy(xi)  # update hyperparameters to the best set so far
                vMetaTrainingAdaptWithUpdates.append(meta_training_iter)
                patience_counter = 0
            ## actual meta-update
            optimizer_meta.zero_grad()
            for w in xi.parameters():
                w.grad = w.total_grad.clone() #/ torch.sum(lengths_lD_val)  # NOTE: this was present in the original code, it seems not needed? (could be controlled through lr_outer)
            # updating xi
            optimizer_meta.step()
        # evaluating sum L(xi) over minibatch of tasks:
        with torch.no_grad():
            loss_after_iter = torch.tensor(0.0)
            for task_ind, tau in enumerate(tasks_batch):
                net_out = xi(lD_val[task_ind].x)
                loss_after_iter += lengths_lD_val[task_ind] * net_meta_intermediate.criterion(net_out, lD_val[task_ind].y)
            loss_after_iter /= torch.sum(lengths_lD_val)
        vMetaTrainingLossXi.append(loss_after_iter.detach().clone()) # make a log of the meta-training loss
        #if (meta_training_iter % 100 == 0):
        #    if verbose > 1: print(f"meta-loss-xi={loss_after_iter}; min-meta-loss-phi={minimal_training_loss_adapt}; time elapsed { datetime.datetime.now() - startTime }")
        # Early stopping
        patience_counter += 1
        if patience_counter >= patience:
            break
    return xi, xi_adapt, {'vMetaTrainingLossXi': vMetaTrainingLossXi, 'vMetaTrainingLossAdapt': vMetaTrainingLossAdapt,
                          'vMetaTrainingAdaptWithUpdates': vMetaTrainingAdaptWithUpdates}


def main_meta_train(args):
    param_comb = args['param_comb'][0]  # NOTE: We assume a single parameter combination
    latent_dim = param_comb['latent_dim']
    hidden_size = param_comb['hidden_size']
    input_dim = args['real_df'][0].shape[1]
    model_params = {'feat_distributions': args['metadata']['feat_distributions'], 'latent_dim': latent_dim,
                    'hidden_size': hidden_size, 'input_dim': input_dim}

    model_init = Generator(model_params) # Initial model

    data_set_meta_train = [MyDataSet({'x': torch.from_numpy(d.values).float(), 'y': torch.from_numpy(d.values).float()}) for d in args['real_df']]

    net_meta_intermediate = VAE_Meta(model_params['feat_distributions'], latent_dim)
    # Now, we are ready for the meta_training process
    xi, xi_adapt, results = FreqMamlMetaTrain(model_init, # instance of nnModule
                                              data_set_meta_train,
                                              net_meta_intermediate,
                                              lr_inner=1e-3,  # Inner loop learning rate
                                              lr_outer=1e-3,  # Outer loop learning rate
                                              tasks_batch_size_mtr=None,  # Number of tasks in a batch for meta training, use None for using all tasks
                                              numSgdSteps=50,  # Adaptation steps in inner loop
                                              numWays=300,  # Task points for training
                                              waysShuffle=True,  # Whether to shuffle the ways or using always the first numWays of the task for training
                                              numMetaTrainingIter=10000,
                                              bFirstOrderApprox=False,  # TRUE = FOMAML; FALSE = MAML with Hessians
                                              patience=10,
                                              seed=0, # for reproducibility
                                              verbose=2)
    xi_adapt.save(args['output_dir'] + '/maml_model')
