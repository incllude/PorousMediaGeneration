# Porous Media Generation

A deep learning framework for synthetic generation of porous media structures using diffusion models and flow matching techniques. This project enables controllable generation of both segmentation masks and tomographic images of porous materials with specified porosity characteristics.

## Overview

This repository implements a two-stage generative pipeline for porous media synthesis:

1. **Segmentation Generation**: Generation of 3D porous structure segmentation masks with controllable porosity
2. **Tomogram Translation**: Translation of segmentation masks to realistic tomographic images

The framework supports multiple generative algorithms including Probabilistic Flow Matching (PFM), Segmentation Guided Diffusion Models (SGDM), and Discrete Denoising Diffusion Probabilistic Models (D3PM).

## Project Structure

```
PorousMediaGeneration/
├── configs/                    # Hydra configuration files
│   ├── dataset/               # Dataset configurations
│   ├── model/                 # Model architecture configurations
│   ├── generation/            # Generation algorithm configurations
│   ├── optimizer/             # Optimizer configurations
│   ├── scheduler/             # Learning rate scheduler configurations
│   ├── trainer/               # Training configurations
│   ├── logger/                # Logging configurations
│   ├── characteristics/       # Porous media characteristic configurations
│   ├── segmentation_settings.yaml  # Segmentation training configuration
│   ├── tomogram_settings.yaml      # Tomogram training configuration
│   └── generation_settings.yaml    # Generation pipeline configuration
├── models/                    # Model implementations
│   ├── unet.py               # U-Net architecture
│   ├── dit.py                # Diffusion Transformer
│   ├── flow_matching.py      # Probabilistic Flow Matching
│   ├── sgdpm.py              # Segmentation Guided Diffusion Model
│   └── d3pm.py               # Discrete Denoising Diffusion
├── dataset/                   # Dataset handling
│   └── datasets.py           # Dataset classes
├── utils/                     # Utility functions
│   └── utils.py              # Helper functions and callbacks
├── train.py                  # Training script
├── generate.py               # Generation script
└── requirements.txt          # Python dependencies
```

## Configuration System

The project utilizes Hydra for hierarchical configuration management. The configuration system allows for modular composition of training and generation pipelines through the following structure:

### Core Configuration Components

- **Models**: `unet`, `unet_att`, `dit` - Neural network architectures
- **Generation Algorithms**: `pfm`, `sgdm`, `d3pm` - Generative model types
- **Datasets**: `segmentation`, `tomogram` - Data loading configurations
- **Characteristics**: `full` - Porous media property computation

### Configuration Composition

The framework uses Hydra's package directive to create flexible configuration hierarchies:

```yaml
defaults:
  - model@segmentation.model: unet
  - generation@segmentation.generation: pfm
  - model@tomogram.model: unet_att
  - generation@tomogram.generation: sgdm
  - dataset: tomogram
```

This approach enables independent configuration of segmentation and tomogram generation components without requiring separate configuration directories.

## Installation

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd PorousMediaGeneration
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables (optional):
Create a `.env` file with required API keys and paths.

## Training

### Configuration Selection

The training script supports dynamic configuration selection through command-line arguments:

```bash
# Train with specific configuration
python train.py --config segmentation_settings

# Alternative configuration
python train.py --config tomogram_settings
```

### Available Training Configurations

- **segmentation_settings**: Training configuration for porous structure segmentation generation
- **tomogram_settings**: Training configuration for tomographic image generation

### Training Parameters

Key training parameters can be configured through YAML files:

- `batch_size`: Training batch size
- `seed`: Random seed for reproducibility
- `log_epochs`: Epoch range for model checkpointing
- `model_name`: Prefix for saved checkpoints
- `checkpoint_dir`: Directory for saving model checkpoints

### Example Training Commands

```bash
# Train segmentation generator
python train.py --config segmentation_settings

# Train tomogram translator
python train.py --config tomogram_settings
```

## Data Generation

### Configuration

Generation is controlled through the `generation_settings.yaml` configuration file, which specifies:

- Model checkpoints for segmentation and tomogram generators
- Number of samples to generate
- Output directory structure
- Volume dimensions (default: 64×64×64)

### Generation Pipeline

The generation process consists of two stages:

1. **Segmentation Generation**: Creates 3D porous structure masks with specified porosity
2. **Tomogram Translation**: Converts segmentation masks to realistic tomographic images

### Generation Commands

```bash
# Generate with default configuration
python generate.py --config generation_settings

# Generate using the generation configuration
python generate.py --config generation_settings
```

### Output Structure

Generated data is organized as follows:

```
output_path/
├── original/              # Reference data (if available)
│   └── {sample_id}/
│       ├── tomogram.npy
│       ├── segmentation.npy
│       └── slice_visualization.png
└── generation/            # Generated samples
    └── {sample_id}/
        ├── tomogram.npy
        ├── segmentation.npy
        ├── slice_visualization.png
        └── info.json      # Generation metadata
```

### Output Formats

- **Tomograms**: 3D NumPy arrays with voxel intensity values
- **Segmentations**: 3D NumPy arrays with class labels (0: background, 1: polevoy feldspar, 2: pore)
- **Visualizations**: PNG images showing representative 2D slices
- **Metadata**: JSON files containing porosity information and generation parameters

## Model Architectures

### Supported Models

- **U-Net**: Standard U-Net architecture for 3D volume processing
- **U-Net with Attention**: Enhanced U-Net with attention mechanisms
- **Diffusion Transformer (DiT)**: Transformer-based diffusion model

### Generation Algorithms

- **Probabilistic Flow Matching (PFM)**: Continuous normalizing flow approach
- **Segmentation-Guided Diffusion Model (SGDM)**: DDPM-style diffusion
- **Discrete Denoising Diffusion (D3PM)**: Discrete state space diffusion

## Porous Media Characteristics

The framework computes and utilizes the following porous media properties:

- **Porosity**: Volume fraction of pore space
- **Specific Surface Area**: Surface area per unit volume of solid material

These characteristics enable conditional generation of porous structures with desired properties.

## License

This project is licensed under the MIT License. See LICENSE file for details.
