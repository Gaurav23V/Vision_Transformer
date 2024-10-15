
# Vision Transformer (ViT) Implementation

## Overview

This repository contains a PyTorch-based implementation of the **Vision Transformer (ViT)** architecture, as introduced by Dosovitskiy et al. in their seminal research on applying Transformer models to computer vision tasks. The Vision Transformer leverages the Transformer architecture, originally designed for Natural Language Processing (NLP), to perform image classification by treating image patches as tokens analogous to words in a sentence.

## Features

- **Modular Architecture**: Clean separation of components such as patch embedding, Transformer encoder blocks, and the classification head.
- **Configurable Hyperparameters**: Easily adjust settings like image size, patch size, number of Transformer layers, embedding dimensions, and more.
- **Training and Evaluation Scripts**: Dedicated scripts for setting up data loaders, training the model, and evaluating performance.
- **Jupyter Notebook**: An interactive notebook detailing the replication process of the Vision Transformer from scratch.
- **Comprehensive Documentation**: Clear instructions for installation, usage, and understanding the project structure.

## Directory Structure

```
vision_transformer/
├── notebooks/
│   └── ViT_Demo.ipynb          # Interactive notebook explaining the ViT implementation
├── scripts/
│   ├── data_setup.py            # Script for creating train and test dataloaders
│   ├── engine.py                # Contains training and testing functions
│   ├── train.py                 # Trains the ViT model using device-agnostic code
│   └── utils.py                 # Utility functions for model training and saving
├── src/
│   ├── __init__.py
│   ├── mlp_block.py             # Implementation of the MLP block in Transformer
│   ├── msa_block.py             # Implementation of the Multi-Head Self-Attention block
│   ├── patch_embedding.py       # Patch Embedding layer implementation
│   ├── transformer_encoder.py   # Transformer Encoder Block implementation
│   └── vit.py                   # Vision Transformer model implementation
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
```

## Installation

### Prerequisites

- **Python 3.7+**
- **pip** (Python package installer)
- **Git**

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your_username/vision_transformer.git
   cd vision_transformer
   ```

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv env
   source env/bin/activate      # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

### Data Setup

Before training the model, set up the data by running the `data_setup.py` script. This script prepares the train and test dataloaders for image classification tasks.

```bash
python scripts/data_setup.py --data_dir path/to/your/data --batch_size 32
```

**Parameters:**

- `--data_dir`: Path to the directory containing the dataset.
- `--batch_size`: Number of samples per batch.

### Training the Model

Train the Vision Transformer using the `train.py` script. This script utilizes device-agnostic code, allowing training on CPU or GPU seamlessly.

```bash
python scripts/train.py --config config/train_config.yaml
```

**Parameters:**

- `--config`: Path to the YAML configuration file containing training parameters (e.g., learning rate, number of epochs).

### Evaluating the Model

Evaluate the trained model's performance on the test dataset using the `engine.py` script.

```bash
python scripts/evaluate.py --model_path checkpoints/vit_checkpoint.pth --config config/eval_config.yaml
```

**Parameters:**

- `--model_path`: Path to the saved model checkpoint.
- `--config`: Path to the YAML configuration file for evaluation settings.

### Using the Jupyter Notebook

For an interactive exploration of the Vision Transformer, refer to the [ViT_Demo.ipynb](notebooks/ViT_Demo.ipynb) notebook. This notebook provides a step-by-step guide to training and evaluating the model, complete with visualizations and insights.

```bash
jupyter notebook notebooks/ViT_Demo.ipynb
```

## Project Components

### Scripts

- **`data_setup.py`**: Prepares the training and testing dataloaders for image classification tasks.
- **`engine.py`**: Contains functions for training and testing the PyTorch model.
- **`train.py`**: Orchestrates the training process using device-agnostic code to utilize available hardware.
- **`utils.py`**: Provides utility functions for model training, saving checkpoints, and other helper tasks.

### Source (`src/`) Modules

- **`mlp_block.py`**: Implements the Multi-Layer Perceptron (MLP) block used within the Transformer encoder.
- **`msa_block.py`**: Implements the Multi-Head Self-Attention (MSA) block.
- **`patch_embedding.py`**: Handles the division of images into patches and their embedding into the Transformer.
- **`transformer_encoder.py`**: Defines the Transformer Encoder Block, combining MSA and MLP blocks with residual connections.
- **`vit.py`**: Integrates all components to form the complete Vision Transformer model.


### `requirements.txt`

```plaintext
torch>=1.7.1
torchvision>=0.8.2
matplotlib>=3.3.2
numpy>=1.19.2
pandas>=1.1.3
tqdm>=4.50.2
PyYAML>=5.3.1
```

**Description:**

- **`torch` and `torchvision`**: Core libraries for building and training the Vision Transformer.
- **`matplotlib`**: Used for plotting and visualization in notebooks and scripts.
- **`numpy`**: Fundamental package for numerical computations.
- **`pandas`**: Utilized for data manipulation and analysis.
- **`tqdm`**: Provides progress bars for loops and training processes.
- **`PyYAML`**: Enables parsing of YAML configuration files.

**Note:** Ensure that you add any additional packages used in your project to the `requirements.txt` file to maintain environment consistency.

---
