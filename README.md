# Comparative Profiling of Latent Diffusion Model Training

This README provides instructions on how to replicate the experiments detailed in the paper "Comparative Profiling: Insights Into Latent Diffusion Model Training" by Bradley Aldous and Ahmed M. Abdelmoniem, presented at the 4th Workshop on Machine Learning and Systems (EuroMLSys '24). The experiments involve training and profiling the AudioLDM and Stable Diffusion models.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Training Commands](#training-commands)
  - [Profiling with PyTorch Profiler](#profiling-with-pytorch-profiler)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Generative AI models, particularly latent diffusion models (LDMs), have shown remarkable capabilities in generating high-fidelity audio and images. This README provides the steps to set up environments, train the models, and profile their performance using the repositories:

- AudioLDM: [AudioLDM Training and Finetuning](https://github.com/haoheliu/AudioLDM-training-finetuning/tree/main)
- Stable Diffusion: [Stable Diffusion Training](https://github.com/CompVis/latent-diffusion/tree/main)

## Installation

To replicate the experiments, you will need to set up the environments for both AudioLDM and Stable Diffusion models according to the instructions provided in their respective repositories.

### AudioLDM Environment Setup

1. Clone the AudioLDM repository:
    ```bash
    git clone https://github.com/haoheliu/AudioLDM-training-finetuning.git
    cd AudioLDM-training-finetuning
    ```

2. Follow the installation instructions provided in the [AudioLDM repository](https://github.com/haoheliu/AudioLDM-training-finetuning/tree/main) to set up the environment and install dependencies.

### Stable Diffusion Environment Setup

1. Clone the Stable Diffusion repository:
    ```bash
    git clone https://github.com/CompVis/latent-diffusion.git
    cd latent-diffusion
    ```

2. Follow the installation instructions provided in the [Stable Diffusion repository](https://github.com/CompVis/latent-diffusion/tree/main) to set up the environment and install dependencies.

## Usage

### Training Commands

The training commands for each model are as follows. These commands assume that you have set up the environments as described above.

#### AudioLDM

To train the AudioLDM model, use the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py --config configs/audioldm.yaml --gpus 2 --batch_size 16
```

In this command:
- `CUDA_VISIBLE_DEVICES=0,1` specifies that GPUs 0 and 1 will be used.
- `--gpus 2` indicates that two GPUs are being used for training.
- `--batch_size 16` sets the batch size to 16.

#### Stable Diffusion

To train the Stable Diffusion model, use the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py --base configs/stable-diffusion.yaml --gpus 2 --batch_size 16
```

In this command:
- `CUDA_VISIBLE_DEVICES=0,1` specifies that GPUs 0 and 1 will be used.
- `--gpus 2` indicates that two GPUs are being used for training.
- `--batch_size 16` sets the batch size to 16.

These flags can be changed based on your needs and training setups.

### Profiling with PyTorch Profiler

Both models can be profiled using Weights & Biases and PyTorch Profiler to monitor GPU utilisation, execution time, and memory usage. Here is an example of how to integrate the PyTorch Profiler into your training script:

```python
from pytorch_lightning import Trainer
from pytorch_lightning.profiler import PyTorchProfiler

profiler = PyTorchProfiler(
    dirpath="profiler_logs",
    filename="profiler_output",
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('log_dir'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
)

trainer = Trainer(
    profiler=profiler,
    ...
)
```

In your training script, you can add the profiler to the `trainer_kwargs` dictionary as shown below:

```python
# Trainer and callbacks
trainer_kwargs = dict()

# Add PyTorch profiler to trainer_kwargs
profiler = PyTorchProfiler(
    dirpath="profiler_logs",
    filename="profiler_output",
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('log_dir'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
)
trainer_kwargs["profiler"] = profiler

trainer = Trainer(**trainer_kwargs)
```

## Contributing

We welcome contributions to this project. Please fork the repository, make your changes, and create a pull request with a description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
