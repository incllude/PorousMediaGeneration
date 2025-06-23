import argparse
from pathlib import Path
from dotenv import load_dotenv

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import LearningRateMonitor
from tensordict import TensorDict
from lightning.pytorch.loggers import WandbLogger
from hydra.utils import instantiate
from monai.data import DataLoader
from utils.utils import set_seed, SaveRangeEpochsCallback, custom_collate_fn


def get_config_name():
    """Parse command line arguments to get config name."""

    config_dir = Path("configs")
    available_configs = [f.stem for f in config_dir.glob("*.yaml") if f.is_file()]
    
    parser = argparse.ArgumentParser(description='Training script with configurable config name')
    parser.add_argument('--config', '-c', type=str, default='segmentation_settings', 
                       choices=available_configs,
                       help=f'Name of the config file (without .yaml extension). Available: {available_configs}')
    
    args, _ = parser.parse_known_args()
    
    return args.config


@hydra.main(version_base=None, config_path="configs", config_name=get_config_name())
def train(cfg: DictConfig) -> None:
    """Main training function."""
    
    set_seed(cfg.seed)
    push_config = OmegaConf.to_object(cfg)
    log_epochs = range(cfg.log_epochs[0], cfg.log_epochs[1], cfg.log_epochs[2])
    char2func = OmegaConf.to_object(instantiate(cfg.characteristics, _recursive_=True))
    characteristics = lambda x: TensorDict(**{k: v(x) for k, v in char2func.items()}, batch_size=(x.size(0)))
    
    dataset = instantiate(cfg.dataset, characteristics=characteristics)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size//cfg.trainer.accumulate_grad_batches,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn
    )

    model = instantiate(cfg.model, contexts=list(char2func.keys()))
    generation = instantiate(
        cfg.generation,
        model=model,
        volume_shape=OmegaConf.to_container(cfg.dataset.volume_shape),
        log_epochs=log_epochs,
        characteristics=characteristics
    )
    generation.set_training_settings(cfg.optimizer, cfg.scheduler)
    
    logger = WandbLogger(
        project=cfg.logger.project,
        name=cfg.logger.name,
        save_dir=cfg.logger.save_dir,
        config=push_config
    )
    callbacks = [
        SaveRangeEpochsCallback(
            range_epochs=log_epochs,
            prefix=cfg.model_name,
            dirpath=cfg.checkpoint_dir
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    trainer = instantiate(cfg.trainer,
                          accumulate_grad_batches=cfg.trainer.accumulate_grad_batches//cfg.trainer.devices,
                          logger=logger,
                          callbacks=callbacks,
                          # enable_checkpointing=True)
                          enable_checkpointing=False)
    
    trainer.fit(generation, dataloader)
    
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":

    load_dotenv()    

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    train()
