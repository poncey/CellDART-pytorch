import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from ._module import CellDARTModule, _log_minmaxscale, _extract_expression_matrix


class CellDARTModel:
    
    def __init__(self,
                 adata_sc,
                 adata_st,
                 ctype,
                 gpus: int = 1,
                 random_seed:int = 0,
                 pretrain_epoch: int = 10,
                 adapt_step: int = 2000,
                 **kwargs):
        pl.seed_everything(random_seed)
        
        self.trainer_pretrain = pl.Trainer(max_epochs=pretrain_epoch, gpus=gpus)
        self.trainer_adapt = pl.Trainer(max_steps=adapt_step, gpus=gpus)

        self.module = CellDARTModule(adata_sc, adata_st, ctype, **kwargs)

    def fit(self):
        # Pretrain phase
        print("#################Pretrain Phase#################")
        self.module.pretrain()
        self.trainer_pretrain.fit(self.module)
        # Adapt phase
        print("#################Adaption Phase#################")
        self.module.adapt()
        self.trainer_adapt.fit(self.module)
        return
    
    def plot_fitting_process(self):
        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(self.module.log_pre_loss)
        ax1.set_title("Loss(KLD) within pre-train phase")
        ax1.set_xlabel("Steps")
        ax2 = fig.add_subplot(1, 2, 2)
        pd.DataFrame(self.module.log_adapt_loss).plot(ax=ax2)
        ax2.set_title("Losses within pre-train phase")
        ax2.set_xlabel("Steps")
    
    def predict(self, adata_spatial=None):
        if adata_spatial is not None:
            arr_spatial = _extract_expression_matrix(adata_spatial)
            if self.module.scale_data is True:
                arr_spatial = _log_minmaxscale(arr_spatial)
        else:
            arr_spatial = self.module.arr_target
        
        if not torch.is_tensor(arr_spatial):
            arr_spatial = torch.Tensor(arr_spatial)

        self.module.eval()
        with torch.no_grad():
            _, arr_output, _ = self.module(arr_spatial)
        return torch.exp(arr_output)
    
    # TODO: save module.
    def save_module(self):
        return
