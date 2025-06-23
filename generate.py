import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

import hydra
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from utils.utils import set_seed, visualize_3d_slices
from tqdm import tqdm
import json


def get_config_name():
    """Parse command line arguments to get config name."""

    config_dir = Path("configs")
    available_configs = [f.stem for f in config_dir.glob("*.yaml") if f.is_file()]
    
    parser = argparse.ArgumentParser(description='Generation script with configurable config name')
    parser.add_argument('--config', '-c', type=str, default='segmentation_settings', 
                       choices=available_configs,
                       help=f'Name of the config file (without .yaml extension). Available: {available_configs}')
    
    args, _ = parser.parse_known_args()
    
    return args.config


@hydra.main(version_base=None, config_path="configs", config_name=get_config_name())
def generate(cfg: DictConfig) -> None:
    """Main training function."""
    
    set_seed(cfg.seed)
    dataset = instantiate(cfg.dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    segmentation_model = instantiate(cfg.segmentation.model)
    segmentation_model.load_state_dict(torch.load(cfg.segmentation_ckpt, map_location=torch.device(device)))
    segmentation_model.eval()
    segmentation_generator = instantiate(
        cfg.segmentation.generation,
        model=segmentation_model,
        volume_shape=OmegaConf.to_container(cfg.volume_shape)
    ).to(device)

    tomogram_model = instantiate(cfg.tomogram.model)
    tomogram_model.load_state_dict(torch.load(cfg.tomogram_ckpt, map_location=torch.device(device)))
    tomogram_model.eval()
    tomogram_generator = instantiate(
        cfg.tomogram.generation,
        model=tomogram_model,
        volume_shape=OmegaConf.to_container(cfg.volume_shape)
    ).to(device)

    porosity_distribution = []
    os.makedirs(f"{cfg.output_path}/original", exist_ok=True)
    for i in tqdm(range(cfg.num_samples)):
        os.makedirs(f"{cfg.output_path}/original/{i}", exist_ok=True)
        orig_tomo, orig_segm = dataset[i]
        fig = visualize_3d_slices(orig_segm, orig_tomo, num_slices=5)
        porosity_distribution.append(orig_segm[2].mean().item())
        orig_segm = torch.argmax(orig_segm, dim=0)
        orig_tomo = (orig_tomo + 1) / 2
        orig_tomo = orig_tomo * (58091 - 1679) + 1697
        np.save(f"{cfg.output_path}/original/{i}/tomogram.npy", orig_tomo.numpy())
        np.save(f"{cfg.output_path}/original/{i}/segmentation.npy", orig_segm.numpy())
        fig.savefig(f"{cfg.output_path}/original/{i}/slice_visualization.png")
        

    os.makedirs(f"{cfg.output_path}/generation", exist_ok=True)
    for i in tqdm(range(cfg.num_samples)):
        os.makedirs(f"{cfg.output_path}/generation/{i}", exist_ok=True)
        pushed_porosity = porosity_distribution[i]
        segm = segmentation_generator.sample(size=1, context=torch.tensor([pushed_porosity]))
        tomo = tomogram_generator.sample(segm=segm, clip_denoised=False)
        segm, tomo = segm[0].cpu(), tomo[0].cpu()
        fig = visualize_3d_slices(segm, tomo, num_slices=5)
        segm = torch.argmax(segm, dim=0)
        output_porosity = (segm == 2).to(torch.float).mean().item()
        tomo = (tomo + 1) / 2
        tomo = np.clip(tomo * (58091 - 1679) + 1697, a_min=1697, a_max=58091)
        np.save(f"{cfg.output_path}/generation/{i}/tomogram.npy", tomo.numpy())
        np.save(f"{cfg.output_path}/generation/{i}/segmentation.npy", segm.numpy())
        fig.savefig(f"{cfg.output_path}/generation/{i}/slice_visualization.png")
        info = {
            "pushed_porosity": pushed_porosity,
            "output_porosity": output_porosity
        }
        with open(f"{cfg.output_path}/generation/{i}/info.json", 'w') as json_file:
            json.dump(info, json_file)


if __name__ == "__main__":

    load_dotenv()
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    generate()
