import torch
import scipy
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from typing import Union


class CellDARTModule(pl.LightningModule):

    def __init__(self,
                 adata_sc,
                 adata_st,
                 ctype: Union[pd.Series, str],
                 create_sc_mixture: bool = True,
                 k_sc_mixture: int = 5,
                 N_sc_mixture: int = 5000,
                 scale_data=True,
                 batch_size: int = 128,
                 d_hidden: int = 1024,
                 d_latent: int = 64,
                 d_do: int = 32,
                 alpha: float = 2.0,
                 alpha_lr: float = 10.0,
                 dropout_rate=0.5) -> None:
        super().__init__()

        # TODO: modified for dataset and related d_hidden assign.
        arr_source, ctype_lab, arr_target, ctype_dict = self._settle_dataset(adata_sc, adata_st, ctype,
                                                                             sc_mixture=create_sc_mixture,
                                                                             k_sc_mixture=k_sc_mixture,
                                                                             N_sc_mixture=N_sc_mixture,
                                                                             scale_data=scale_data)
        self.arr_source = arr_source
        self.ctype_lab = ctype_lab
        self.arr_target = arr_target
        self.ctype_dict = ctype_dict
        self.scale_data = scale_data

        # data-specific parameters
        if self.arr_source.shape[0] != self.ctype_lab.shape[0]:
            raise ValueError(
                "Number of samples is not matched between X ({:d} ) and labels ({:d}) for the source data.".format(
                    self.arr_source.shape[0], self.ctype_lab.shape[0]
                ))
        if self.arr_source.shape[1] != self.arr_target.shape[1]:
            raise ValueError(
                "Please maker sure the dimension for source/target data.")

        self.d_ctype_source = self.ctype_lab.shape[1]
        self.d_input = self.arr_source.shape[1]

        # module specific settlements
        self.n_hidden = d_hidden
        self.n_latent = d_latent
        self.alpha = alpha
        self.alpha_lr = alpha_lr
        self.batch_size = batch_size

        # Initial training state
        self.pretrian_phase = True

        # core parts of the modified ADDA method
        self.f_extractor = _make_linear_components(
            self.d_input, [d_hidden, d_latent])
        self.source_classifier = _make_linear_components(
            d_latent, [], self.d_ctype_source)
        self.domain_classifier = _make_linear_components(
            d_latent, d_do, 2, dropout_rate=dropout_rate)

        self.automatic_optimization = False  # enable manual optimization

        # logging lists
        self.log_pre_loss, self.log_adapt_loss = [], []

    def forward(self, x):
        z = self.f_extractor(x)
        y = F.log_softmax(self.source_classifier(
            z), dim=1)  # predicted cell-fraction
        m = self.domain_classifier(z)  # domain label
        return z, y, m

    def training_step(self, batch, batch_idx):
        opt_scm, opt_dcm = self.optimizers()
        if self.pretrain_phase:
            ####################################################
            # Pretraining source domain with pseudo-spots only #
            ####################################################
            Xs, ys, = batch["s"]  # pseudo-spot source data

            # kld for cell fractions TODO: add nograd for insurence
            opt_scm.zero_grad()
            loss_pre = self._get_kld(Xs, ys)
            loss_pre.backward()
            opt_scm.step()

            # TODO: modify logger
            self.log_pre_loss.append(loss_pre.detach().cpu().item())
        else:
            #####################################
            # Adaptation for spot deconvolution #
            #####################################
            (Xs, ys) = batch["s"]
            Xt = batch["t"]

            # optimization1 process
            opt_scm.zero_grad()
            loss_op1, kld_op1, ce_op1 = self._get_loss_op1(Xs, ys, Xt)
            loss_op1.backward()
            opt_scm.step()

            # optimization2 process
            opt_dcm.zero_grad()
            loss_op2 = self._get_loss_op2(Xs, Xt)
            loss_op2.backward()
            opt_dcm.step()

            # TODO: modify logger
            self.log_adapt_loss.append({"op1_loss": loss_op1.detach().cpu().item(),
                                        "op1_kld": kld_op1.detach().cpu().item(),
                                        "op1_ce": ce_op1.detach().cpu().item(),
                                        "op2_loss": loss_op2.detach().cpu().item()})
        return

    def configure_optimizers(self):
        opt_scm = Adam(list(self.f_extractor.parameters()) +
                       list(self.source_classifier.parameters()),
                       lr=1e-3)
        opt_dcm = Adam(self.domain_classifier.parameters(),
                       lr=self.alpha_lr * 1e-3)
        return opt_scm, opt_dcm

    def _get_kld(self, X, y):
        _, pred, _ = self.forward(X)
        return F.kl_div(pred, y, reduction="batchmean")

    def _get_loss_op1(self, Xs, ys, Xt):
        # create adversarial samples
        Xadv = torch.cat([Xs, Xt])
        ya1 = torch.cat([torch.ones(Xs.shape[0]), torch.zeros(
            Xt.shape[0])]).long().to(self.device)
        yc = torch.cat(
            [ys, torch.zeros(Xt.shape[0], self.d_ctype_source).to(self.device)])

        _, pred, dl = self.forward(Xadv)
        # kld_t = 0 for the merged computation
        kld = F.kl_div(pred, yc, reduction="batchmean")
        ce_loss = F.cross_entropy(dl, ya1)
        loss = kld + self.alpha * ce_loss
        return loss, kld, ce_loss

    def _get_loss_op2(self, Xs, Xt):
        Xadv = torch.cat([Xs, Xt])
        ya2 = torch.cat([torch.zeros(Xs.shape[0]), torch.ones(
            Xt.shape[0])]).long().to(self.device)

        _, _, dl = self.forward(Xadv)
        loss = F.cross_entropy(dl, ya2)
        return loss

    def pretrain(self):
        self.pretrain_phase = True

    def adapt(self):
        self.pretrain_phase = False

    def train_dataloader(self):

        dataset_s = TensorDataset(torch.Tensor(
            self.arr_source), torch.Tensor(self.ctype_lab))
        loader_s = DataLoader(dataset_s, batch_size=self.batch_size)
        loader_t = DataLoader(self.arr_target, batch_size=self.batch_size)

        return {"s": loader_s, "t": loader_t}

    def _settle_dataset(self, adata_sc, adata_st, ctype, sc_mixture,
                        k_sc_mixture: int,
                        N_sc_mixture: int,
                        scale_data: bool):
        arr_source = _extract_expression_matrix(adata_sc)
        arr_target = _extract_expression_matrix(adata_st)
        # handle cell-type label
        if isinstance(ctype, str):
            try:
                ctype = adata_sc.obs[ctype]
            except KeyError:
                ctype = adata_sc.obsm[ctype]
        ctype_dict = dict(zip(range(len(set(ctype))), set(ctype)))
        ctype_dict_r = dict((y, x) for x, y in ctype_dict.items())
        ctype_lab = np.asarray(
            [ctype_dict_r[ii] for ii in ctype])

        # create pseudo-spots
        if sc_mixture:
            arr_source, ctype_lab = _random_mix(arr_source, ctype_lab,
                                                nmix=k_sc_mixture,
                                                n_samples=N_sc_mixture)
        else:
            # one-hot encoding
            ctype_lab = np.eye(len(np.unique(ctype_lab)),
                               dtype='uint8')[ctype_lab]

        # scale data
        if scale_data:
            arr_source = _log_minmaxscale(arr_source)
            arr_target = _log_minmaxscale(arr_target)
        return arr_source, ctype_lab, arr_target, ctype_dict


def _log_minmaxscale(arr):
    arrd = len(arr)
    arr = np.log1p(arr)
    return (arr-np.reshape(np.min(arr, axis=1), (arrd, 1)))/np.reshape((np.max(arr, axis=1)-np.min(arr, axis=1)), (arrd, 1))


def _extract_expression_matrix(adata):

    if scipy.sparse.issparse(adata.X):
        return adata.X.todense().A
    else:
        return adata.X


def _make_linear_components(input_dim: int,
                            hidden_units: Union[int, list],
                            output_dim: int = None,
                            dropout_rate: float = 0,
                            batchnorm: bool = True,):
    nn_units = [hidden_units] if isinstance(
        hidden_units, int) else hidden_units
    nn_units.insert(0, input_dim)
    lincomp = OrderedDict()
    for ii in range(len(nn_units) - 1):
        lincomp['dense_{:d}'.format(
            ii + 1)] = nn.Linear(nn_units[ii], nn_units[ii + 1])
        lincomp['activ_{:d}'.format(ii + 1)] = nn.ELU()
        if dropout_rate != 0:
            lincomp['dropout_{:d}'.format(ii + 1)] = nn.Dropout(p=dropout_rate)
        if batchnorm:
            lincomp['batchnm_{:d}'.format(
                ii + 1)] = nn.BatchNorm1d(nn_units[ii + 1])
    if output_dim is not None:
        lincomp['out'] = nn.Linear(nn_units[-1], output_dim)

    return nn.Sequential(lincomp)


def _random_mix(Xs, ys, nmix=5, n_samples=10000):
    nclss = len(set(ys))
    Xs_new, ys_new = [], []
    ys_ = np.eye(len(np.unique(ys)), dtype='uint8')[ys]
    for i in range(n_samples):
        yy = np.zeros(nclss)
        fraction = np.random.rand(nmix)
        fraction = fraction/np.sum(fraction)
        fraction = np.reshape(fraction, (nmix, 1))
        randindex = np.random.randint(len(Xs), size=nmix)
        ymix = ys_[randindex]
        yy = np.sum(ymix*np.reshape(fraction, (nmix, 1)), axis=0)
        XX = Xs[randindex] * fraction
        XX_ = np.sum(XX, axis=0)
        ys_new.append(yy)
        Xs_new.append(XX_)
    Xs_new = np.asarray(Xs_new)
    ys_new = np.asarray(ys_new)
    return Xs_new, ys_new
