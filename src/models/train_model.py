import os
import sys
import warnings
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
from omegaconf import DictConfig

import hydra

warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(threshold=sys.maxsize)

from model_def import *

sys.path.append('src/data/')
from my_dataloader import *

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"


Hydra_path = '../conf/'
@hydra.main(config_path= Hydra_path, config_name="train_cfg.yaml")
def main(cfg):

# ====================== load config params from hydra ======================================
    pl_checkpoints_path = os.getcwd() + '/'
    
    # save train_model.py and model_def.py files as part of hydra output
    # shutil.copy(hydra.utils.get_original_cwd() + '/src/models/train_model.py', 'save_train_model.py')
    # shutil.copy(hydra.utils.get_original_cwd() + '/src/models/model_def.py', 'model_def.py')

    if cfg._d: # debug mode
        fast_dev_run = False
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_visible_devices)
    else: # training mode
        fast_dev_run = False
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_visible_devices)

# ======================================== main section ==================================================
    
    dm = LstmSpeechDataModule(cfg)
    model = speech_lstm_PL(cfg)
    

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.ckpt_monitor,
        filename='epoch-{epoch:02d}-val-loss-{val_loss:.4f}',
        save_last = cfg.save_last,
        save_top_k = cfg.save_top_k,
        mode='min',
        verbose=cfg.verbose
    )

    stop_callback = EarlyStopping(
        monitor=cfg.ckpt_monitor,
        patience=cfg.patience
    )

    trainer = Trainer(  
                        gpus=cfg.gpus, 
                        accelerator = 'ddp',
                        fast_dev_run=fast_dev_run, 
                        check_val_every_n_epoch=cfg.check_val_every_n_epoch, 
                        default_root_dir= pl_checkpoints_path,                       
                        callbacks=[stop_callback, checkpoint_callback], 
                        log_gpu_memory=cfg.log_gpu_memory, 
                        progress_bar_refresh_rate=cfg.progress_bar_refresh_rate,
                        precision=cfg.precision,
                        plugins=DDPPlugin(find_unused_parameters=False),
                        num_sanity_val_steps = 0,
                        # resume_from_checkpoint=ckpt_path
                        # auto_lr_find=True
                     )

    trainer.fit(model, dm)
    # trainer.test(model)  
    checkpoint_callback.best_model_path

if __name__ == "__main__":
    main()
